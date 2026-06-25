# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2019 .. 2022
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using a local solver
accessed through a shared library (.dll on Windows, .so on Linux).
Interface with library is done using ctypes module.
See https://docs.python.org/2/library/ctypes.html for details.
"""

from docplex.cp.solution import *
from docplex.cp.solution import CpoSolveResult
from docplex.cp.utils import compare_natural
from docplex.cp.solver.solver import CpoSolver, CpoSolverAgent, CpoSolverException
from docplex.cp.blackbox import BLACKBOX_ARGUMENT_TYPES_ENCODING

import ctypes
from ctypes.util import find_library
import json
import sys
import time
import os
import traceback
import threading


#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

# Version of this client
#CLIENT_VERSION = 6 # last OO's version
CLIENT_VERSION = 7

# Events received from library
_EVENT_SOLVER_INFO     = 1  # Information on solver as JSON document
_EVENT_JSON_RESULT     = 2  # Solve result expressed as a JSON string
_EVENT_LOG_OUTPUT      = 3  # Log data on output stream
_EVENT_LOG_WARNING     = 4  # Log data on warning stream
_EVENT_LOG_ERROR       = 5  # Log data on error stream
_EVENT_SOLVER_ERROR    = 6  # Solver error. Details are in event associated string.
_EVENT_CPO_CONFLICT    = 7  # Conflict in CPO format

# Event notifier callback prototype
_EVENT_NOTIF_PROTOTYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)

# CPO callback prototype (event name, json data)
_CPO_CALLBACK_PROTOTYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

# Blackbox evaluation callback prototype (json call request, nb_result (out), result values (out))
_BLACKBOX_EVAL_PROTOTYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_char))

# Function prototypes (mandatory, return type, args_type)
_LIB_FUNCTION_PROTYTYPES = \
{
    'createSession'        : (True,  ctypes.c_void_p, (ctypes.c_void_p,)),
    'deleteSession'        : (True,  ctypes.c_int,    (ctypes.c_void_p,)),
    'setCpoModel'          : (True,  ctypes.c_int,    (ctypes.c_void_p, ctypes.c_char_p)),
    'solve'                : (True,  ctypes.c_int,    (ctypes.c_void_p,)),
    'startSearch'          : (True,  ctypes.c_int,    (ctypes.c_void_p,)),
    'searchNext'           : (True,  ctypes.c_int,    (ctypes.c_void_p,)),
    'endSearch'            : (True,  ctypes.c_int,    (ctypes.c_void_p,)),
    'abortSearch'          : (True,  ctypes.c_int,    (ctypes.c_void_p,)),
    'propagate'            : (True,  ctypes.c_int,    (ctypes.c_void_p,)),
    'refineConflict'       : (True,  ctypes.c_int,    (ctypes.c_void_p,)),
    'runSeeds'             : (True,  ctypes.c_int,    (ctypes.c_void_p, ctypes.c_int,)),
    'refineConflictWithCpo': (False, ctypes.c_int,    (ctypes.c_void_p, ctypes.c_bool)),
    'setExplainFailureTags': (False, ctypes.c_int,    (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int))),
    'setCpoCallback'       : (False, ctypes.c_int,    (ctypes.c_void_p, ctypes.c_void_p,)),
    'addBlackBoxFunction'  : (False, ctypes.c_int,    (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, )),
    'addBlackBoxFunction2' : (False, ctypes.c_int,    (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, )),
    'setClientVersion'     : (False, ctypes.c_int,    (ctypes.c_int,)),
}

# Dictionary of loaded libraries. Key is lib file, value is a tuple (lib handler, set of available function names)
# This is neeeded as library file can be changed dynamically when solving, and multiple libraries may then be used.
_LOADED_LIBRARIES = {}

# Lock to protect creation of lib handler
_LOADED_LIBRARIES_LOCK = threading.Lock()


#-----------------------------------------------------------------------------
#  Public classes
#-----------------------------------------------------------------------------

class CpoSolverLib(CpoSolverAgent):
    """ Interface to a local solver through a shared library """
    __slots__ = ('lib_handler',          # Lib handler
                 'session',              # Solve session in the library
                 'notify_event_proto',   # Prototype of the event callback
                 'first_log_error',      # First line of error
                 'first_solver_error',   # First error (exception) thrown by solver
                 'callback_proto',       # Prototype of the CPO callback function
                 'blackbox_eval_proto',  # Prototype of the blackbox function evaluation function
                 'present_funs',         # Set lib functions that are available
                 'last_conflict_cpo',    # Last conflict in CPO format
                 'proxy_version',        # Library version number
                 )

    def __init__(self, solver, context):
        """ Create a new solver using shared library.

        Args:
            solver:   Parent solver
            context:  Solver agent context
        Raises:
            CpoSolverException if library is not found
        """
        # Call super
        super(CpoSolverLib, self).__init__(solver, context)

        # Initialize attributes
        self.first_log_error = None
        self.first_solver_error = None
        self.lib_handler = None # (to not block end() in case of init failure)
        self.last_conflict_cpo = None
        self.process_infos['ClientVersion'] = CLIENT_VERSION

        # Connect to library
        self._access_library()
        self.context.log(2, "Solving library: '{}'".format(self.context.lib))

        # Set client version
        if 'setClientVersion' in self.present_funs:
            self.lib_handler.setClientVersion(CLIENT_VERSION)

        # Create session
        # CAUTION: storing callback prototype is mandatory. Otherwise, it is garbaged and the callback fails.
        self.notify_event_proto = _EVENT_NOTIF_PROTOTYPE(self._notify_event)
        self.session = self.lib_handler.createSession(self.notify_event_proto)
        self.context.log(5, "Solve session: {}".format(self.session))

        # Get library version
        self.proxy_version = self.version_info.get('ProxyVersion', 0)

        # Transfer all solver infos in process info
        self.process_infos.update(self.version_info)

        # Check solver version if any
        sver = self.version_info.get('SolverVersion')
        mver = solver.get_model_format_version()
        if sver and mver and compare_natural(mver, sver) > 0:
            raise CpoSolverException("Solver version {} is lower than model format version {}.".format(sver, mver))

        # Initialize other attributes
        self.callback_proto = None
        self.blackbox_eval_proto = None


    def solve(self):
        """ Solve the model

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: shared library file
         * 3: Main solving steps
         * 5: Every individual event got from library

        Returns:
            Model solve result (object of class CpoSolveResult)
        Raises:
            CpoException if error occurs
        """
        # Initialize model if needed
        self._init_solver()

        # Solve the model
        self._call_lib_function('solve', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def start_search(self):
        """ Start a new search. Solutions are retrieved using method search_next().
        """
        # Initialize model if needed
        self._init_solver()

        self._call_lib_function('startSearch', False)


    def search_next(self):
        """ Get the next available solution.

        Returns:
            Next model result (type CpoSolveResult)
        """
        # Call library function
        self._call_lib_function('searchNext', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) model solution with last solve information (type CpoSolveResult)
        """
        # Call library function
        self._call_lib_function('endSearch', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def abort_search(self):
        """ Abort current search.
        This method is designed to be called by a different thread than the one currently solving.
        """
        self._call_lib_function('abortSearch', False)


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        See documentation of CpoSolver.refine_conflict() for details.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        """
        # Initialize model if needed
        self._init_solver()

        # Check if cpo format required
        self.last_conflict_cpo = None
        if self.context.add_conflict_as_cpo and ('refineConflictWithCpo' in self.present_funs):
            # Request refine conflict with CPO format
            self._call_lib_function('refineConflictWithCpo', True, True)
        else:
            # Call library function
            self._call_lib_function('refineConflict', True)

        # Build result object
        result = self._create_result_object(CpoRefineConflictResult, self.last_json_result)
        result.cpo_conflict = self.last_conflict_cpo
        self.last_conflict_cpo = None
        return result


    def propagate(self):
        """ This method invokes the propagation on the current model.

        See documentation of CpoSolver.propagate() for details.

        Returns:
            Propagation result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        # Initialize model if needed
        self._init_solver()

        # Call library function
        self._call_lib_function('propagate', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def run_seeds(self, nbrun):
        """ This method runs *nbrun* times the CP optimizer search with different random seeds
        and computes statistics from the result of these runs.

        This method does not return anything. Result statistics are displayed on the log output
        that should be activated.

        Each run of the solver is stopped according to single solve conditions (TimeLimit for example).
        Total run time is then expected to take *nbruns* times the duration of a single run.

        Args:
            nbrun: Number of runs with different seeds.
        Returns:
            Run result, object of class :class:`~docplex.cp.solution.CpoRunResult`.
        """
        # Initialize model if needed
        self._init_solver()

        # Call library function
        self._call_lib_function('runSeeds', False, nbrun)

        # Build result object
        self.last_json_result = None
        return self._create_result_object(CpoRunResult)


    def set_explain_failure_tags(self, ltags=None):
        """ This method allows to set the list of failure tags to explain in the next solve.

        The failure tags are displayed in the log when the parameter :attr:`~docplex.cp.CpoParameters.LogSearchTags`
        is set to 'On'.
        All existing failure tags previously set are cleared prior to set the new ones.
        Calling this method with an empty list is then equivalent to just clear tags.

        Args:
            ltags:  List of tag ids to explain
        """
        # Initialize model if needed
        self._init_solver()

        # Build list of tags
        if ltags is None:
            ltags = []
        elif not is_array(ltags):
            ltags = (ltags,)
        nbtags = len(ltags)

        # Call the function
        self._call_lib_function('setExplainFailureTags', False, nbtags, (ctypes.c_int * nbtags)(*ltags))


    def end(self):
        """ End solver and release all resources.
        """
        if self.lib_handler is not None:
            self.lib_handler.deleteSession(self.session)
            self.lib_handler = None
            self.session = None
            super(CpoSolverLib, self).end()


    def _is_abort_search_supported(self):
        """ Check if this agent supports an actual abort_search() instead of killing the solver

        Return:
            True if this agent supports actual abort_search()
        """
        return self.proxy_version >= 9


    def _init_solver(self):
        """ Initialize solver
        """
        # Reset last errors
        self.first_log_error = None
        self.first_solver_error = None
        # Initialize model if needed
        self._init_model_in_solver()


    def _call_lib_function(self, dfname, json, *args):
        """ Call a library function
        Args:
            dfname:  Name of the function to be called
            json:    Indicate if a JSON result is expected
            *args:   Optional arguments (after session)
        Raises:
            CpoDllException if function call fails or if expected JSON is absent
        """
        # Log call
        if self.context.is_log_enabled(5):
            if args:
                self.context.log(5, "Call lib function: ", dfname, "(", ", ".join(str(a) for a in args))
            else:
                self.context.log(5, "Call lib function: ", dfname, "()")

        # Reset JSON result if JSON required
        if json:
            self.last_json_result = None

        # Get the library function
        fun = getattr(self.lib_handler, dfname, None)
        if fun is None:
            raise CpoNotSupportedException("The function '{}' is not found in the library. Try with a most recent version.".format(dfname))

        # Call the library function
        try:
            rc = fun(self.session, *args)
        except Exception as e:
            if self.context.log_exceptions:
                traceback.print_exc()
            raise CpoSolverException("Error while calling function '{}': {}.".format(dfname, e))
        if rc != 0:
            errmsg = "Call to '{}' failure (rc={})".format(dfname, rc)
            errext = self.first_solver_error if self.first_solver_error else self.first_log_error
            if errext:
                errmsg += ": {}".format(errext)
            raise CpoSolverException(errmsg)

        # Check if JSON result is present
        if json:
            if self.last_json_result is None:
               raise CpoSolverException("No JSON result provided by function '{}'".format(dfname))
            self._log_received_message(dfname, self.last_json_result)


    def _notify_event(self, event, data):
        """ Callback called by the library to notify Python of an event (log, error, etc)
        Args:
            event:  Event id (integer)
            data:   Event data string
        """
        # Process event
        if event == _EVENT_LOG_OUTPUT or event == _EVENT_LOG_WARNING:
            # Store log if required
            if self.log_enabled:
                self._add_log_data(data.decode('utf-8'))

        elif event == _EVENT_JSON_RESULT:
            self.last_json_result = data.decode('utf-8')

        elif event == _EVENT_SOLVER_INFO:
            self.version_info = verinf = json.loads(data.decode('utf-8'))
            # Update information
            verinf['AgentModule'] = __name__
            verinf['BlackboxEvalMutex'] = verinf.get('BlackboxEvalMutex', 0) > 0
            self.context.log(3, "Local solver info: '", verinf, "'")

        elif event == _EVENT_LOG_ERROR:
            ldata = data.decode('utf-8')
            if self.first_log_error is None:
                self.first_log_error = ldata.replace('\n', '')
            out = self.log_output if self.log_output is not None else sys.stdout
            out.write("ERROR: {}\n".format(ldata))
            out.flush()

        elif event == _EVENT_SOLVER_ERROR:
            errmsg = data.decode('utf-8')
            if self.first_log_error is not None:
                errmsg += " (" + self.first_log_error + ")"
            if self.first_log_error is None:
                self.first_log_error = errmsg
            out = self.log_output if self.log_output is not None else sys.stdout
            out.write("SOLVER ERROR: {}\n".format(errmsg))
            out.flush()

        elif event == _EVENT_CPO_CONFLICT:
            self.last_conflict_cpo = data.decode('utf-8')


    def _cpo_callback(self, event, data):
        """ Callback called by the library to notify Python of a CPO solver callback event
        Args:
            event:  Event name (string)
            data:   JSON data (string)
        """
        # Decode all data
        stime = time.time()
        event = event.decode('utf-8')
        data = data.decode('utf-8')
        self.process_infos.incr(CpoProcessInfos.TOTAL_UTF8_DECODE_TIME, time.time() - stime)

        # Build result data and notify solver
        res = self._create_result_object(CpoSolveResult, data)
        self.solver._notify_callback_event(event, res)


    def _blackbox_eval_callback(self, jeval, nbres, result, szexcpt, excpt):
        """ Callback called by the library to evaluate a blackbox function
        Args:
            jeval:   Evaluation context expressed as JSON data
            nbres:   Pointer on number of results
            result:  Pointer on results array
            szexcpt: Max size of error string buffer
            excpt:   Exception string buffer (length 500)
        Returns:
            Array of double values
        """
        # Decode JSON string
        stime = time.time()
        jeval = jeval.decode('utf-8', errors='ignore')
        self.process_infos.incr(CpoProcessInfos.TOTAL_UTF8_DECODE_TIME, time.time() - stime)
        self.context.log(5, "JSON blackbox evaluation request: ", jeval)

        # Process blackbox function evaluation request
        try:
            # Retrieve evaluation elements
            bbf, args, bnds = self.solver._get_blackbox_function_eval_context(jeval)

            # Evaluate function
            lck = bbf.eval_mutex
            if lck is not None:
                with bbf.eval_mutex:
                    res = bbf._eval_function(bnds, *args)
            else:
                res = bbf._eval_function(bnds, *args)
            # Store result
            if res:
                nbres[0] = len(res)
                for i, v in enumerate(res):
                    result[i] = v
        except Exception as e:
            if self.context.log_exceptions:
                traceback.print_exc()
            # Build error message
            orig = traceback.extract_tb(sys.exc_info()[2])[-1]
            err = "({}, {}) {}: {}".format(orig[0], orig[1], type(e).__name__, e)
            # Set it in response buffer
            err = err.encode('utf8')
            mlen = min(szexcpt - 1, len(err))
            for i in range(mlen):
                excpt[i] = err[i]
            # Add ending zero
            try:
                excpt[mlen] = 0
            except:
                excpt[mlen] = '\x00'


    def _send_model_to_solver(self, cpostr):
        """ Send the model to the solver.
        This method must be extended by agent implementations to actually do the operation.
        Args:
            copstr:  String containing the model in CPO format
        """
        # Encode model
        stime = time.time()
        cpostr = cpostr.encode('utf-8')
        self.process_infos.incr(CpoProcessInfos.TOTAL_UTF8_ENCODE_TIME, time.time() - stime)

        # Send CPO model to process
        stime = time.time()
        self._call_lib_function('setCpoModel', True, cpostr)
        self.process_infos.incr(CpoProcessInfos.TOTAL_DATA_SEND_TIME, time.time() - stime)


    def _add_callback_processing(self):
        """ Add the processing of solver callback.
        """
        # CAUTION: storing callback prototype is mandatory. Otherwise, it is garbaged and the callback fails.
        self.callback_proto = _CPO_CALLBACK_PROTOTYPE(self._cpo_callback)
        self._call_lib_function('setCpoCallback', False, self.callback_proto)


    def _register_blackbox_function(self, name, bbf):
        """ Register a blackbox function in the solver
        This method must be extended by agent implementations to actually do the operation.

        Args:
            name: Name of the blackbox function in the model (may differ from the declared one)
            bbf: Blackbox function descriptor, object of class :class:`~docplex.cp.blackbox.CpoBlackboxFunction`
        """
        # Check lib version
        ver = self.version_info.get('LibVersion', 0)
        if ver < 7:
            raise CpoSolverException("This version of the CPO library ({}) does not support blackbox functions.".format(ver))
        if ver < 8:
            raise CpoSolverException("This version of the CPO library ({}) does not support blackbox cache configuration.".format(ver))

        # Encode list of argument types
        atypes = [BLACKBOX_ARGUMENT_TYPES_ENCODING[t] for t in bbf.get_arg_types()]

        # Set blackbox callback if not already done
        if self.blackbox_eval_proto is None:
            self.blackbox_eval_proto = _BLACKBOX_EVAL_PROTOTYPE(self._blackbox_eval_callback)

        # Register blackbox function
        name = name.encode('utf-8')
        dimension = bbf.get_dimension()
        nbargs = len(atypes)
        if (ver < 8):
            self._call_lib_function('addBlackBoxFunction', False, name, dimension, nbargs, (ctypes.c_int * nbargs)(*atypes),
                                    self.blackbox_eval_proto)
        else:
            self._call_lib_function('addBlackBoxFunction2', False, name, dimension, nbargs, (ctypes.c_int * nbargs)(*atypes),
                                    bbf.get_cache_size(), bbf.is_global_cache(), self.blackbox_eval_proto)


    def _access_library(self):
        """ Access the CPO library.
        This method uses a library cache to optimize connection to the library in case of multiple solves.
        This method sets local attributes lib_handler and present_funs.
        Raises:
            CpoSolverException if library is not found
        """
        # Access library
        libf = self.context.libfile
        if not libf:
            msg = "CPO library file should be given in 'solver.lib.libfile' context attribute."
            self._notify_libfile_not_found(msg)
            raise CpoSolverException(msg)

        # Lock to access loaded libraries cache
        with _LOADED_LIBRARIES_LOCK:
            t = _LOADED_LIBRARIES.get(libf)
            if t is None:
                # Search for library file
                if not os.path.isfile(libf):
                    lf = find_library(libf)
                    if lf is None:
                        msg = "Can not find library '{}'".format(libf)
                        self._notify_libfile_not_found(msg)
                        raise CpoSolverException(msg)
                    libf = lf
                # Check library is executable
                if not is_exe_file(libf):
                    msg = "Library file '{}' is not executable".format(libf)
                    self._notify_libfile_not_found(msg)
                    raise CpoSolverException(msg)
                # Load library
                try:
                    lib = ctypes.CDLL(libf)
                except Exception as e:
                    msg = "Can not load library '{}': {}".format(libf, e)
                    self._notify_libfile_not_found(msg)
                    raise CpoSolverException(msg)

                # Define function prototypes
                prfuns = set()
                for name, proto in _LIB_FUNCTION_PROTYTYPES.items():
                    mand, rtype, argtypes = proto
                    try:
                        f = getattr(lib, name)
                        f.restype = rtype
                        f.argtypes = argtypes
                        prfuns.add(name)
                    except:
                        if mand:
                            raise CpoSolverException("Function '{}' not found in the library {}".format(name, lib))

                # Add lib into the cache
                t = (lib, prfuns)
                _LOADED_LIBRARIES[libf] = t

        # Set local attributes
        self.lib_handler, self.present_funs = t


    def _notify_libfile_not_found(self, msg):
        """ Print an error message and display warning about library file
        Args:
            msg:  Error message to print
        """
        #traceback.print_stack()
        out = self.context.get_log_output()
        if out is None:
            out = sys.stdout
        banner = "#" * 79
        out.write(banner + '\n')
        out.write("# {}\n".format(msg))
        out.write("# Please check that:\n")
        out.write("#  - you have installed IBM ILOG CPLEX Optimization Studio on your computer,\n")
        out.write("#    (see https://ibmdecisionoptimization.github.io/docplex-doc/getting_started.html for details),\n")
        out.write("#  - your system path includes a reference to the directory where the library file 'lib_cpo_solver_*(.lib or .so)' is located,\n")
        out.write("#  - the context attribute 'context.solver.lib.libfile' is properly set to 'lib_cpo_solver_*(.lib or .so)',\n")
        out.write("#  - or that it is set to an absolute path to this file.\n")
        out.write(banner + '\n')
        out.flush()
