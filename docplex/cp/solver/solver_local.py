# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2016 .. 2022
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using
a local CP Optimizer Interactive (cpoptimizer(.exe)).
"""

from docplex.cp.solution import *
from docplex.cp.utils import CpoException
from docplex.cp.solver.solver import CpoSolver, CpoSolverAgent, CpoSolverException
from docplex.cp.catalog import *
from docplex.cp.blackbox import BLACKBOX_ARGUMENT_TYPES_ENCODING

import subprocess
import sys
import time
import threading
import json
import os
import traceback


#-----------------------------------------------------------------------------
#  Private constants
#-----------------------------------------------------------------------------

# Version of this client
#CLIENT_VERSION = 8 # last OO's version
CLIENT_VERSION = 9

# List of command ids that can be sent to solver
CMD_EXIT                = "Exit"              # End process (no data)
CMD_SET_CPO_MODEL       = "SetCpoModel"       # CPO model as string
CMD_SOLVE_MODEL         = "SolveModel"        # Complete solve of the model (no data)
CMD_START_SEARCH        = "StartSearch"       # Start search (no data)
CMD_SEARCH_NEXT         = "SearchNext"        # Get next solution (no data)
CMD_END_SEARCH          = "EndSearch"         # End search (no data)
CMD_ABORT_SEARCH        = "AbortSearch"       # Abort search (no data)
CMD_REFINE_CONFLICT     = "RefineConflict"    # Refine conflict (no data)
CMD_PROPAGATE           = "Propagate"         # Propagate (no data)
CMD_RUN_SEEDS           = "RunSeeds"          # Run with multiple seeds.
CMD_ADD_CALLBACK        = "AddCallback"       # Add callback proxy to the solver
CMD_SET_FAILURE_TAGS    = "SetFailureTags"    # Give list of failure tags to explain.
CMD_ADD_BLACKBOX_FUN    = "AddBlackboxFun"    # Add/register a blackbox function
CMD_SET_CLIENT_VERSION  = "SetClientVersion"  # Set the version number of the client

# List of events received from solver
EVT_VERSION_INFO        = "VersionInfo"        # Local solver version info (String in JSON format)
EVT_SUCCESS             = "Success"            # Success in last command execution
EVT_ERROR               = "Error"              # Error (data is error string)
EVT_TRACE               = "DebugTrace"         # Debugging trace
EVT_SOLVER_OUT_STREAM   = "OutStream"          # Solver output stream
EVT_SOLVER_WARN_STREAM  = "WarningStream"      # Solver warning stream
EVT_SOLVER_ERR_STREAM   = "ErrorStream"        # Solver error stream
EVT_SOLVE_RESULT        = "SolveResult"        # Solver result in JSON format
EVT_CONFLICT_RESULT     = "ConflictResult"     # Conflict refiner result in JSON format
EVT_CONFLICT_RESULT_CPO = "ConflictResultCpo"  # Conflict refiner result in CPO format
EVT_PROPAGATE_RESULT    = "PropagateResult"    # Propagate result in JSON format
EVT_RUN_SEEDS_RESULT    = "RunSeedsResult"     # Run seeds result (no data, all is in log)
EVT_CALLBACK_EVENT      = "CallbackEvent"      # Callback event. Data is event name.
EVT_CALLBACK_DATA       = "CallbackData"       # Callback data, following event. Data is JSON document.
EVT_BLACKBOX_FUNCALL    = "BlackboxFunCall"    # Call to a blackbox function for evaluation.

# Events sent back to solver
EVT_BLACKBOX_RESULT     = "BlackboxResult"     # Result of a blackbox function call

# Max possible received data size in one message
_MAX_RECEIVED_DATA_SIZE = 1000000

# Python 2 indicator
IS_PYTHON_2 = (sys.version_info[0] == 2)


#-----------------------------------------------------------------------------
#  Public classes
#-----------------------------------------------------------------------------

class CpoSolverLocal(CpoSolverAgent):
    """ Interface to a local solver through an external process """
    __slots__ = ('process',             # Sub-process handler
                 'active',              # Agent active indicator
                 'pout',                # Sub-process output stream
                 'pin',                 # Sub-process input stream
                 'available_commands',  # List of available commands
                 'timeout_kill',        # Indicates process have been killed by timeout
                 'out_lock',            # Lock to protect output stream
                 'proxy_version',       # Version of the remote proxy
                 'os',                  # Hack for Python 38 to avoid display of "OSError: [WinError 6] The handle is invalid"
                                        # when this object is deleted. (Don't ask why this work, it is a mystery)
                )

    def __init__(self, solver, context):
        """ Create a new solver that solves locally with CP Optimizer Interactive.

        Args:
            solver:  Parent solver
            context: Solver context
        Raises:
            CpoException if proxy executable does not exists
        """
        # Call super
        self.process = None
        self.active = True
        self.timeout_kill = False
        super(CpoSolverLocal, self).__init__(solver, context)
        self.process_infos['ClientVersion'] = CLIENT_VERSION

        # Check if executable file exists
        xfile = context.execfile
        if xfile is None or not is_string(xfile):
            self._notify_execfile_not_found()
            raise CpoException("Executable file should be given in 'execfile' context attribute.")
        if not os.path.isfile(xfile):
            self._notify_execfile_not_found()
            raise CpoException("Executable file '{}' does not exists".format(xfile))
        if not is_exe_file(xfile):
            self._notify_execfile_not_found()
            raise CpoException("Executable file '{}' is not executable".format(xfile))

        # Create solving process
        cmd = [context.execfile]
        if context.parameters is not None:
            cmd.extend(context.parameters)
        context.log(2, "Solver exec command: '", ' '.join(cmd), "'")
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, universal_newlines=False)
        except:
            self._notify_execfile_not_found()
            raise CpoException("Can not execute command '{}'. Please check availability of required executable file.".format(' '.join(cmd)))
        self.pout = self.process.stdin
        self.pin = self.process.stdout
        self.out_lock = threading.Lock()

        # Read initial version info from process
        timeout = context.process_start_timeout
        timer = threading.Timer(timeout, self._process_start_timeout)
        timer.start()
        try:
            evt, data = self._read_message()
        except Exception as e:
            if context.log_exceptions:
                traceback.print_exc()
            if self.timeout_kill:
                raise CpoSolverException("Solver process was too long to start and respond ({} seconds). Process has been killed.".format(timeout))
            raise CpoSolverException("Solver sub-process start failure: {}".format(e))
        timer.cancel()

        # Check received message
        if evt != EVT_VERSION_INFO:
            raise CpoSolverException("Unexpected event {} received instead of version info event {}.".format(evt, EVT_VERSION_INFO))
        self.version_info = verinf = json.loads(data)
        self.available_commands = self.version_info['AvailableCommands']
        self.proxy_version = self.version_info.get('ProxyVersion', 0)
        # Normalize information
        verinf['AgentModule'] = __name__

        context.log(3, "Local solver info: '", verinf, "'")

        # Transfer all solver infos in process info
        self.process_infos.update(self.version_info)

        # Check solver version if any
        sver = self.version_info.get('SolverVersion')
        mver = solver.get_model_format_version()
        if sver and mver and compare_natural(mver, sver) > 0:
            raise CpoSolverException("Solver version {} is lower than model format version {}.".format(sver, mver))

        # Send client version if accepted by server
        avcmds = verinf.get('AvailableCommands')
        if (avcmds is not None) and (CMD_SET_CLIENT_VERSION in avcmds):
            nbfrm = bytearray(4)
            encode_integer_big_endian_4(CLIENT_VERSION, nbfrm, 0)
            self._write_message(CMD_SET_CLIENT_VERSION, data=nbfrm)
            self._wait_event(EVT_SUCCESS)


    def _process_start_timeout(self):
        """ Process the raise of start timeout timer """
        # Check if version info has been read
        if not self.version_info:
            # Kill sub-process
            self.timeout_kill = True
            self.process.kill()


    def solve(self):
        """ Solve the model

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: The process exec file
         * 3: Content of the JSON response
         * 4: Solver traces (if any)
         * 5: Messages sent/receive to/from process

        Returns:
            Model solve result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Start solve
        self._write_message(CMD_SOLVE_MODEL)

        # Wait JSON result
        jsol = self._wait_json_result(EVT_SOLVE_RESULT)

        # Build result object
        return self._create_result_object(CpoSolveResult, jsol)


    def start_search(self):
        """ Start a new search. Solutions are retrieved using method search_next().
        """
        # Initialize model if needed
        self._init_model_in_solver()

        self._write_message(CMD_START_SEARCH)

        # Wait for answer message, only for version > 14 (for compatibility)
        if self.proxy_version >= 14:
            self._wait_event(EVT_SUCCESS)


    def search_next(self):
        """ Get the next available solution.

        Returns:
            Next model result (type CpoSolveResult)
        """

        # Request next solution
        self._write_message(CMD_SEARCH_NEXT)

        # Wait JSON result
        jsol = self._wait_json_result(EVT_SOLVE_RESULT)

        # Build result object
        return self._create_result_object(CpoSolveResult, jsol)


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) solve result with last solve information (type CpoSolveResult)
        """

        # Request end search
        self._write_message(CMD_END_SEARCH)

        # Wait JSON result
        jsol = self._wait_json_result(EVT_SOLVE_RESULT)

        # Build result object
        return self._create_result_object(CpoSolveResult, jsol)


    def abort_search(self):
        """ Abort current search.
        This method is designed to be called by a different thread than the one currently solving.
        """
        # Check if remote angel is capable of processing abort_search
        if self._is_abort_search_supported():
            # Request abort search
            self._write_message(CMD_ABORT_SEARCH)
            # No event returned back because solve is supposed to wait for another
        else:
            # Old fashion, just kill the sub-process
            try:
                self.process.kill()
            except:
                pass


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        See documentation of :meth:`~docplex.cp.solver.solver.CpoSolver.refine_conflict` for details.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Check if cpo format required
        pver = self.version_info.get('ProxyVersion')
        if self.context.add_conflict_as_cpo and pver and (int(pver) >= 9):
            # Request refine conflict with CPO format
            self._write_message(CMD_REFINE_CONFLICT, bytearray([1]))
            # Wait JSON result
            jsol = self._wait_json_result(EVT_CONFLICT_RESULT)
            result = self._create_result_object(CpoRefineConflictResult, jsol)
            if result.is_conflict():
                # Wait for CPO conflict
                cposol = self._wait_event(EVT_CONFLICT_RESULT_CPO)
                result.cpo_conflict = cposol
        else:
            # Request refine conflict
            self._write_message(CMD_REFINE_CONFLICT)
            # Wait JSON result
            jsol = self._wait_json_result(EVT_CONFLICT_RESULT)
            result = self._create_result_object(CpoRefineConflictResult, jsol)

        # Return result object
        return result


    def propagate(self):
        """ This method invokes the propagation on the current model.

        See documentation of :meth:`~docplex.cp.solver.solver.CpoSolver.propagate` for details.

        Returns:
            Propagation result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Request propagation
        self._write_message(CMD_PROPAGATE)

        # Wait JSON result
        jsol = self._wait_json_result(EVT_PROPAGATE_RESULT)

        # Build result object
        return self._create_result_object(CpoSolveResult, jsol)


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
        Raises:
            CpoNotSupportedException: method not available in local solver.
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Check command availability
        if CMD_RUN_SEEDS not in self.available_commands:
            raise CpoNotSupportedException("Method 'run_seeds' is not available in local solver '{}'".format(self.context.execfile))

        # Request run seeds
        nbfrm = bytearray(4)
        encode_integer_big_endian_4(nbrun, nbfrm, 0)

        self._write_message(CMD_RUN_SEEDS, data=nbfrm)

        # Wait result
        self._wait_event(EVT_RUN_SEEDS_RESULT)

        # Build result object
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
        self._init_model_in_solver()

        # Check command availability
        if CMD_SET_FAILURE_TAGS not in self.available_commands:
            raise CpoNotSupportedException("Method 'set_explain_failure_tags' is not available in local solver '{}'".format(self.context.execfile))

        # Build list of tags
        if ltags is None:
            ltags = []
        elif not is_array(ltags):
            ltags = (ltags,)
        nbtags = len(ltags)
        tagfrm = bytearray(4 * (nbtags + 1))
        encode_integer_big_endian_4(nbtags, tagfrm, 0)
        for i, t in enumerate(ltags):
            encode_integer_big_endian_4(t, tagfrm, 4 * (i + 1))

        # Send command
        self._write_message(CMD_SET_FAILURE_TAGS, data=tagfrm)
        self._wait_event(EVT_SUCCESS)


    def end(self):
        """ End solver and release all resources.
        """
        if self.active:
            self.active = False
            try:
                #traceback.print_stack()
                self._write_message(CMD_EXIT)
            except BaseException:
                pass
            try:
                self.process.kill()
            except BaseException:
                pass
            try:
                self.process.wait()
            except BaseException:
                pass
            self.process = None
            try:
                self.pout.close()
            except BaseException:
                pass
            try:
                self.pin.close()
            except BaseException:
                pass
            super(CpoSolverLocal, self).end()


    def _is_abort_search_supported(self):
        """ Check if this agent supports an actual abort_search() instead of killing the solver

        Return:
            True if this agent supports actual abort_search()
        """
        return self.proxy_version >= 17


    def _wait_event(self, xevt):
        """ Wait for a particular event while forwarding logs if any.
        Args:
            xevt: Expected event
        Returns:
            Message data
        Raises:
            SolverException if an error occurs
        """
        self.context.log(5, "Wait event: ", xevt)

        # Initialize first error string to enrich exception if any
        firsterror = None

        # There is a bug in the messages sent by older versions of the C++ library with multiple workers and Python callbacks.
        # The bug that may happen is that an EVT_CALLBACK_EVENT may be not followed immediately by its corresponding EVT_CALLBACK_DATA.
        # The workaround here memorizes the type of callback event, and as long as there are other events or EVT_CALLBACK_EVENT of the same type,
        # we can wait for the EVT_CALLBACK_DATA, and process it when it arrives.
        # If there is an EVT_CALLBACK_EVENT of another type before the EVT_CALLBACK_DATA, then we throw all the callbacks
        # events waiting (as we are not able to match the EVT_CALLBACK_DATA with the good type of EVT_CALLBACK_EVENT).
        # (In practice we have seen only an EVT_BLACKBOX_FUNCALL sent between an EVT_CALLBACK_EVENT and its corresponding EVT_CALLBACK_DATA.)
        numOpeningEvents = 0
        typeOfOpeningEvents = None
        stackingOpeningEvent = True        

        # Read events
        while True:
            # Read and process next message
            evt, data = self._read_message()
 
            if evt == xevt:
                return data
                
            elif evt == EVT_CALLBACK_EVENT:
                # store the callback event
                if numOpeningEvents == 0 :
                    stackingOpeningEvent = True
                    typeOfOpeningEvents = data
                elif data != typeOfOpeningEvents:
                    stackingOpeningEvent = False
                numOpeningEvents = numOpeningEvents+1
            elif evt == EVT_CALLBACK_DATA:
                assert numOpeningEvents>0
                numOpeningEvents = numOpeningEvents-1
                if stackingOpeningEvent: 
                    res = self._create_result_object(CpoSolveResult, data)
                    self.solver._notify_callback_event(typeOfOpeningEvents, res)

            elif evt == EVT_BLACKBOX_FUNCALL:
                # Retrieve dialog id
                sx = data.find(':')
                if sx < 0:
                    raise CpoException("Dialog id not found in blackbox function evaluation request '{}'".format(data))
                did = data[:sx]
                data = data[sx + 1:]
                # Retrieve evaluation elements
                try:
                    bbf, args, bnds = self.solver._get_blackbox_function_eval_context(data)
                    # Process blackbox function evaluation request
                    if bbf.is_parallel_eval():
                        # Evaluate in a separate thread
                        thrd = threading.Thread(target=lambda: self._eval_blackbox(did, bbf, args, bnds))
                        thrd.start()
                    else:
                        # Evaluate synchronously (ensure mutual exclusion)
                        self._eval_blackbox(did, bbf, args, bnds)
                except Exception as e:
                    if self.context.log_exceptions:
                        traceback.print_exc()
                    # Build and send error message
                    orig = traceback.extract_tb(sys.exc_info()[2])[-1]
                    rdata = "{} ERROR ({}, {}) {}: {}".format(did, orig[0], orig[1], type(e).__name__, e)
                    # Send back response
                    self._write_message(EVT_BLACKBOX_RESULT, rdata)

            elif evt in (EVT_SOLVER_OUT_STREAM, EVT_SOLVER_WARN_STREAM):
                if data:
                    # Store log if required
                    if self.log_enabled:
                        self._add_log_data(data)

            elif evt == EVT_SOLVER_ERR_STREAM:
                if data:
                    if firsterror is None:
                        firsterror = data.replace('\n', '')
                    out = self.log_output if self.log_output is not None else sys.stdout
                    out.write("ERROR: {}\n".format(data))
                    out.flush()

            elif evt == EVT_TRACE:
                self.context.log(None, "ANGEL TRACE: " + data)

            elif evt == EVT_ERROR:
                if firsterror is not None:
                    data += " (" + firsterror + ")"
                self.end()
                raise CpoSolverException("Solver error: " + data)

            else:
                self.end()
                raise CpoSolverException("Unknown event received from local solver: " + str(evt))


    def _eval_blackbox(self, did, bbf, args, bnds):
        """ Evaluate a blackbox function
        Args:
            did:  Dialog id
            bbf:  Blackbox function descriptor
            args: Argument values
            bnds: Result bounds
        """
        # Process blackbox function evaluation request
        try:
            res = bbf._eval_function(bnds, *args)
            # Build response
            if res:
                rdata = did + ' ' + (str(len(res)) if res else "0") + ' ' + (' '.join(str(v) for v in res))
            else:
                rdata = did + ' 0'
        except Exception as e:
            if self.context.log_exceptions:
                traceback.print_exc()
            # Build error message
            orig = traceback.extract_tb(sys.exc_info()[2])[-1]
            rdata = "{} ERROR ({}, {}) {}: {}".format(did, orig[0], orig[1], type(e).__name__, e)
        # Send back response
        #print("Response data: {}".format(rdata))
        self._write_message(EVT_BLACKBOX_RESULT, rdata)


    def _wait_json_result(self, evt):
        """ Wait for a JSON result while forwarding logs if any.
        Args:
            evt: Event to wait for
        Returns:
            JSON solution string, decoded from UTF8
        """

        # Wait JSON result
        data = self._wait_event(evt)

        # Store last json result
        self._set_last_json_result_string(data)
        #self.context.log(3, "JSON result:\n", data)

        return self.last_json_result


    def _write_message(self, cid, data=None):
        """ Write a message to the solver process
        Args:
            cid:   Command name
            data:  Data to write, already encoded in UTF8 if required
        """
        # Encode elements
        stime = time.time()
        cid = cid.encode('utf-8')
        if is_string(data):
            data = data.encode('utf-8')
        nstime = time.time()
        self.process_infos.incr(CpoProcessInfos.TOTAL_UTF8_ENCODE_TIME, nstime - stime)

        # Build header
        tlen = len(cid)
        if data is not None:
            tlen += len(data) + 1
        if tlen > 0xffffffff:
            raise CpoSolverException("Try to send a message with length {}, greater than {}.".format(tlen, 0xffffffff))
        frame = bytearray(6)
        frame[0] = 0xCA
        frame[1] = 0xFE
        encode_integer_big_endian_4(tlen, frame, 2)

        # Log message to send
        self.context.log(5, "Send message: cmd=", cid, ", tsize=", tlen)

        # Add data if any
        if data is None:
            frame = frame + cid
        else:
            frame = frame + cid + bytearray(1) + data

        # Write message frame
        with self.out_lock:
           self.pout.write(frame)
           self.pout.flush()

        # Update statistics
        self.process_infos.incr(CpoProcessInfos.TOTAL_DATA_SEND_TIME, time.time() - nstime)
        self.process_infos.incr(CpoProcessInfos.TOTAL_DATA_SEND_SIZE, len(frame))


    def _read_message(self):
        """ Read a message from the solver process
        Returns:
            Tuple (evt, data)
        """
        # Read message header
        frame = self._read_frame(6)
        if (frame[0] != 0xCA) or (frame[1] != 0xFE):
            erline = frame + self._read_error_message()
            erline = erline.decode()
            self.end()
            raise CpoSolverException("Invalid message header. Possible error written by solver directly on stdout: " + erline)

        # Read message data
        tsize = decode_integer_big_endian_4(frame, 2)
        data = self._read_frame(tsize)

        # Split name and data
        ename = 0
        while (ename < tsize) and (data[ename] != 0):
            ename += 1

        # Decode name and data
        stime = time.time()
        if ename == tsize:
            # Command only, no data
            evt = data.decode('utf-8')
            data = None
        else:
            # Split command and data
            evt = data[0:ename].decode('utf-8')
            data = data[ename+1:].decode('utf-8')

        # Update statistics
        self.process_infos.incr(CpoProcessInfos.TOTAL_UTF8_DECODE_TIME, time.time() - stime)
        self.process_infos.incr(CpoProcessInfos.TOTAL_DATA_RECEIVE_SIZE, tsize + 6)

        # Log received message
        self._log_received_message(evt, data)

        return evt, data


    def _read_frame(self, nbb):
        """ Read a byte frame from input stream
        Args:
            nbb:  Number of bytes to read
        Returns:
            Byte array
        """
        # Read data
        data = self.pin.read(nbb)
        if len(data) != nbb:
            if len(data) == 0:
                # Check if first read of data
                if self.process_infos.get(CpoProcessInfos.TOTAL_DATA_RECEIVE_SIZE, 0) == 0:
                    if IS_WINDOWS:
                        raise CpoSolverException("Nothing to read from local solver process. Possibly not started because cplex dll is not accessible.")
                    else:
                        raise CpoSolverException("Nothing to read from local solver process. Check its availability.")
                else:
                    try:
                        self.process.wait()
                        rc = self.process.returncode
                    except:
                        rc = "unknown"
                    raise CpoSolverException("Nothing to read from local solver process. Process seems to have been stopped (rc={}).".format(rc))
            else:
                raise CpoSolverException("Read only {} bytes when {} was expected.".format(len(data), nbb))

        # Return
        if IS_PYTHON_2:
            data = bytearray(data)
        # print("Read frame: {}".format(', '.join([str(x) for x in data])))

        return data


    def _read_error_message(self):
        """ Read stream to search for error line end. Called when wrong input is detected,
        to try to read an "Assertion failed" message for example.
        Returns:
            Byte array
        """
        data = []
        bv = self.pin.read(1)
        if IS_PYTHON_2:
            while (bv != '') and (bv != '\n'):
                data.append(ord(bv))
                bv = self.pin.read(1)
                data = bytearray(data)
        else:
            while (bv != b'') and (bv != b'\n'):
                data.append(ord(bv))
                bv = self.pin.read(1)

        return bytearray(data)


    def _send_model_to_solver(self, cpostr):
        """ Send the model to the solver.

        Args:
            cpostr:  String containing the model in CPO format
        """
        self._write_message(CMD_SET_CPO_MODEL, cpostr)
        self._wait_json_result(EVT_SUCCESS)  # JSON stored


    def _add_callback_processing(self):
        """ Add the processing of solver callback.
        """
        # Check angel version
        aver = self.version_info.get('AngelVersion', 0)
        if aver < 8:
            raise CpoSolverException("This version of the CPO solver angel ({}) does not support solver callbacks.".format(aver))
        self._write_message(CMD_ADD_CALLBACK)
        self._wait_event(EVT_SUCCESS)


    def _register_blackbox_function(self, name, bbf):
        """ Register a blackbox function in the solver

        Args:
            bbf: Blackbox function descriptor, object of class :class:`~docplex.cp.blackbox.CpoBlackboxFunction`
        """
        # Check angel version
        aver = self.version_info.get('AngelVersion', 0)
        if aver < 14:
            raise CpoSolverException("This version of the CPO solver angel ({}) does not support blackbox functions.".format(aver))

        # Get blackbox function parameters
        name = name.encode('utf8')
        dimension = bbf.get_dimension()
        argtypes = bbf.get_arg_types()
        nbargs = len(argtypes)

        # Build message frame
        frm = bytearray(2 + len(argtypes) + 5 + len(name) + 1)

        # Add dimension and parameters
        frm[0] = dimension
        frm[1] = nbargs
        for i in range(nbargs):
            frm[2 + i] = BLACKBOX_ARGUMENT_TYPES_ENCODING[argtypes[i]]

        # Add cache attributes
        encode_integer_big_endian_4(bbf.cachesize, frm, 2 + nbargs)
        frm[6 + nbargs] = 1 if bbf.globalcache else 0

        # Add name
        for i in range(len(name)):
            frm[7 + nbargs + i] = name[i]
        frm[7 + nbargs + len(name)] = 0

        # Send message
        self._write_message(CMD_ADD_BLACKBOX_FUN, data=frm)
        self._wait_event(EVT_SUCCESS)


    def _notify_execfile_not_found(self):
        """ Print an error message if solver is not found """
        out = self.context.get_log_output()
        if out is None:
            out = sys.stdout
        banner = "#" * 79
        out.write(banner + '\n')
        out.write("# Solver executable file is not found !\n")
        out.write("# Please check that:\n")
        out.write("#  - you have installed IBM ILOG CPLEX Optimization Studio on your computer,\n")
        out.write("#    (see https://ibmdecisionoptimization.github.io/docplex-doc/getting_started.html for details),\n")
        out.write("#  - your system path includes a reference to the directory where the executable file 'cpoptimizer(.exe)' is located,\n")
        out.write("#  - the context attribute 'context.solver.local.execfile' is properly set to 'cpoptimizer(.exe)',\n")
        out.write("#  - or that it is set to an absolute path to this file.\n")
        out.write(banner + '\n')
        out.flush()
