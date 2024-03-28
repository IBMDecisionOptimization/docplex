# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015 .. 2022
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis
# Author: Christiane BRACCHI, IBM Decision Optimization, France Lab, Saclay

"""
This module implements appropriate software to solve a CPO model represented by a
:class:`docplex.cp.model.CpoModel` object.

It implements the following object classes:

 * :class:`CpoSolver` contains the public interface allowing to make solving requests with a model.
 * :class:`CpoSolverAgent` is an abstract class that is extended by the actual implementation(s) of
   the solving functions.

The :class:`CpoSolver` identifies and creates the required :class:`CpoSolverAgent` depending on the configuration
parameter *context.solver.agent* that contains the name of the agent to be used. This name is used to
access the configuration context *context.solver.<agent>* that contains the details about this agent.

For example, the default configuration refers to *local* as default solver agent, to solve model using local process
*CP Optimizer Interactive*.
This means that at least following configuration elements must be set:
::

   context.solver.agent = 'local'
   context.solver.local.execfile = <Name or path of the process 'CP Optimizer Interactive'>

The different methods that can be called on a CpoSolver object are:

 * :meth:`~CpoSolver.solve` simply solve the model and returns a solve result, if any.
   For convenience reason, this method is also directly available on the CpoModel object (:meth:`docplex.cp.model.CpoModel.solve`).
 * :meth:`~CpoSolver.search_next` and :meth:`~CpoSolver.end_search` allows to iterate on different solutions of the model.
 * :meth:`~CpoSolver.refine_conflict` calls the conflict refiner that identifies a minimal conflict for the infeasibility of
   the model.
 * :meth:`~CpoSolver.propagate` calls the propagation that communicates the domain reduction of a decision variable to
   all of the constraints that are stated over this variable.

Except :meth:`solve`, these functions are only available with a local solver with release greater or equal to 12.7.0.0
When a method is not available, an exception *CpoNotSupportedException* is raised.

If the methods :meth:`~CpoSolver.search_next` and :meth:`~CpoSolver.end_search` are available in the underlying
solver agent, the :class:`CpoSolver` object acts as an iterator. All solutions are retrieved using a loop like:
::

   solver = CpoSolver(mdl)
   for sol in solver:
       sol.write()

A such solution iteration can be interrupted at any time by calling :meth:`~CpoSolver.end_search` that returns
a fail solution including the last solve status.


Detailed description
--------------------
"""

import docplex.cp.config as config
import docplex.cp.utils as utils
from docplex.cp.cpo.cpo_compiler import CpoCompiler
from docplex.cp.solution import *
import docplex.cp.solution as solution
from docplex.cp.solver.solver_listener import CpoSolverListener
import docplex.cp.solver.solver_listener as listener
from docplex.cp.solver.cpo_callback import CpoCallback
from docplex.cp.blackbox import *

import importlib
import inspect
import threading
import time
import traceback
import copy


###############################################################################
##  Public constants
###############################################################################

# Solver statuses
STATUS_IDLE              = "Idle"             # Solver created but inactive
STATUS_RELEASED          = "Released"         # Solver stopped with resources released.
STATUS_ABORTED           = "Aborted"          # Solver has been aborted.
STATUS_SOLVING           = "SolveRunning"     # Simple solve in progress
STATUS_SEARCH_WAITING    = "SearchWaiting"    # Search started or waiting to call next
STATUS_SEARCH_RUNNING    = "NextRunning"      # Search of next solution in progress
STATUS_REFINING_CONFLICT = "RefiningConflict" # Solver refine conflict in progress
STATUS_PROPAGATING       = "Propagating"      # Propagation in progress
STATUS_RUNNING_SEEDS     = "RunningSeeds"     # Run seeds in progress

# Set of statuses that end solver
_ENDING_STATUSES = frozenset((STATUS_RELEASED, STATUS_ABORTED))


###############################################################################
##  Public classes
###############################################################################

class CpoSolverException(CpoException):
    """ Exceptions raised for problems related to solver.
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(CpoSolverException, self).__init__(msg)


class CpoSolverAgent(object):
    """ This class is an abstract class that must be extended by every solver agent that intend
    to be called by :class:`CpoSolver` to solve a CPO model.
    """
    __slots__ = ('name',             # Agent name
                 'solver',           # Parent solver
                 'model',            # Source model
                 'params',           # Solver parameters
                 'context',          # Solve context
                 'last_json_result', # String of the last received JSON result
                 'version_info',     # Solver version information (dict)
                 'process_infos',    # Processing information
                 'log_output',       # Log output stream
                 'log_print',        # Print log indicator
                 'log_data',         # Log data buffer (list of strings)
                 'log_enabled',      # Global log enabled indicator
                )

    def __init__(self, solver, context):
        """ **Constructor**

        Args:
            solver:   Parent solver, object of type CpoSolver
            context:  Solver agent context
        Raises:
            CpoException if agent can not be created properly.
        """
        super(CpoSolverAgent, self).__init__()
        self.solver = solver
        self.model = solver.get_model()
        self.params = context.params
        self.context = context
        self.last_json_result = None
        self.version_info = {}
        self.process_infos = solver.process_infos

        # Initialize log
        self.log_output = context.get_log_output()
        self.log_print = context.trace_log and (self.log_output is not None)
        self.log_data = [] if context.add_log_to_solution else None
        self.log_enabled = self.log_print or (self.log_data is not None)


    def __del__(self):
        # End solver
        self.end()


    def solve(self):
        """ Solve the model

        Returns:
            Model solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def start_search(self):
        """ Start a new search. Solutions are retrieved using method search_next().

        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def search_next(self):
        """ Search the next available solution.

        Returns:
            Next solve result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) solve result with last solve information,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def abort_search(self):
        """ Abort current search.
        This method is designed to be called by a different thread than the one currently solving.

        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        Given an infeasible model, the conflict refiner can identify conflicting constraints and variable domains
        within the model to help you identify the causes of the infeasibility.
        In this context, a conflict is a subset of the constraints and/or variable domains of the model
        which are mutually contradictory.
        Since the conflict is minimal, removal of any one of these constraints will remove that
        particular cause for infeasibility.
        There may be other conflicts in the model; consequently, repair of a given conflict
        does not guarantee feasibility of the remaining model.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def propagate(self):
        """ This method invokes the propagation on the current model.

        Constraint propagation is the process of communicating the domain reduction of a decision variable to
        all of the constraints that are stated over this variable.
        This process can result in more domain reductions.
        These domain reductions, in turn, are communicated to the appropriate constraints.
        This process continues until no more variable domains can be reduced or when a domain becomes empty
        and a failure occurs.
        An empty domain during the initial constraint propagation means that the model has no solution.

        The result is a object of class CpoSolveResult, the same than the one returned by solve() method.
        However, in this case, variable domains may not be completely defined.

        Returns:
            Propagation result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def run_seeds(self, nbrun):
        """ This method runs *nbrun* times the CP optimizer search with different random seeds
        and computes statistics from the result of these runs.

        Result statistics are displayed on the log output that should be activated.
        If the appropriate configuration variable *context.solver.add_log_to_solution* is set to True (default),
        log is also available in the *CpoRunResult* result object, accessible as a string using the method
        :meth:`~docplex.cp.solution.CpoRunResult.get_solver_log`

        Each run of the solver is stopped according to single solve conditions (TimeLimit for example).
        Total run time is then expected to take *nbruns* times the duration of a single run.

        Args:
            nbrun: Number of runs with different seeds.
        Returns:
            Run result, object of class :class:`~docplex.cp.solution.CpoRunResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def set_explain_failure_tags(self, ltags):
        """ This method allows to set the list of failure tags to explain in the next solve.

        The failure tags are displayed in the log when the parameter :attr:`~docplex.cp.CpoParameters.LogSearchTags`
        is set to 'On'.
        All existing failure tags previously set are cleared prior to set the new ones.
        Calling this method with an empty list is then equivalent to just clear tags.

        Args:
            ltags:  List of tag ids to explain
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        self._raise_not_supported()


    def end(self):
        """ End solver agent and release all resources.
        """
        self.solver = None
        # Other resources not released because can be called after the end
        #self.model = None
        #self.params = None
        #self.context = None

    def _is_abort_search_supported(self):
        """ Check if this agent supports an actual abort_search() instead of killing the solver

        Return:
            True if this agent supports actual abort_search()
        """
        return False

    def _log_received_message(self, evt, data):
        """ Log a received message with json data length depending on log level
        Args:
            evt:   Received event name
            data:  JSON data
        """
        # Check enough log level
        if not self.context.is_log_enabled(5):
            return
        # Check no data
        if data is None:
            self.context.log(5, "Received: ", evt)
        else:
            # Log full data if log level 6
            if self.context.is_log_enabled(6):
                # Full version if log level 6
                self.context.log(6, "Received: ", evt, ", data: '", data, "'")
            else:
                # Cut at cpInfos if any
                sx = data.find('"cpInfo" : {')
                ndata = data if sx < 0 else data[:sx]
                self.context.log(5, "Received: ", evt, ", data: '", ndata, "...(skipped)...'")


    def _send_model_to_solver(self, cpostr):
        """ Send the model to the solver.
        This method must be extended by agent implementations to actually do the operation.
        Args:
            copstr:  String containing the model in CPO format
        """
        pass


    def _add_callback_processing(self):
        """ Add the processing of solver callback.
        This method must be extended by agent implementations to actually do the operation.
        """
        pass


    def _register_blackbox_function(self, name, bbf):
        """ Register a blackbox function in the solver
        This method must be extended by agent implementations to actually do the operation.

        Args:
            name: Name of the blackbox function in the model (may differ from the declared one)
            bbf: Blackbox function descriptor, object of class :class:`~docplex.cp.blackbox.CpoBlackboxFunction`
        """
        pass


    def _init_model_in_solver(self):
        """ Send the model to the solver if not already done. """
        if self.solver.model_sent:
            return

        # Get solver version for checking
        sver = self.version_info.get('SolverVersion', "1")

        # Get model string (to force identification of blackbox functions)
        mstr = self.solver.get_cpo_model_string()

        # Register callback and blackbox functions if any and not already done
        # Indicator is useful if the same model is solved and then refine conflict with the same CpoSolver
        # as model is then sent twice (first normally, and secondly with all named constraints)
        if not self.solver.callbacks_registered:
            # Register callback
            if self.solver.callbacks:
                # Check solver version
                if compare_natural(sver, "12.10") < 0:
                    raise CpoSolverException(
                        "This version of the CPO solver ({}) does not support solver callbacks.".format(sver))
                self._add_callback_processing()
                self.context.log(3, "CPO callback created.")
            # Register blackbox functions
            bbfs = self.solver.blackbox_map
            if bbfs:
                # Check solver version
                # if compare_natural(sver, "20.10") <= 0:
                #     raise CpoSolverException("This version of the CPO solver ({}) does not support blackbox functions.".format(sver))
                for name, bbf in bbfs.items():
                    # Check that bbf has an implementation
                    if not bbf.has_implementation():
                        raise CpoSolverException("Blackbox function '{}' has no implementation".format(name))
                    self._register_blackbox_function(name, bbf)
                self.context.log(3, "Blackbox function(s) registered.")
            # Set registration indicator
            self.solver.callbacks_registered = True

        # Send model to solver
        stime = time.time()
        self._send_model_to_solver(mstr)
        self.process_infos.incr(CpoProcessInfos.MODEL_SUBMIT_TIME, time.time() - stime)
        self.context.log(3, "Model sent to solver.")
        self.solver.model_sent = True


    def _add_log_data(self, data):
        """ Add new log data
        Args:
            data:  Data to log (String)
        """
        self.solver._notify_new_log(data)
        if self.log_enabled:
            if self.log_print:
                write_checking_unicode_errors(self.log_output, data)
                self.log_output.flush()
            if self.log_data is not None:
                self.log_data.append(data)
        # Update statistics
        self.process_infos.incr(CpoProcessInfos.TOTAL_LOG_DATA_SIZE, len(data))


    def _set_last_json_result_string(self, json):
        """ Set the string containing last received JSON result

        Args:
            json: JSON result string
        """
        self.last_json_result = json


    def _get_last_json_result_string(self):
        """ Get the string containing last received JSON result

        Return:
            Last JSON result string, None if none
        """
        return self.last_json_result


    def _create_result_object(self, rclass, jsol=None):
        """ Create a new result object and fill it with necessary data
        Args:
            rclass:            Result object class
            jsol (optional):   JSON solution string, not decoded
        Returns:
            New result object preinitialized
        """
        res = rclass(self.model)

        # Process JSON solution
        #self.context.log(3, "JSON data:\n", jsol)
        self.last_json_result = jsol

        # Parse JSON solution
        if jsol:
            # Parse JSON
            stime = time.time()
            jsol = parse_json_string(jsol)
            self.process_infos.incr(CpoProcessInfos.TOTAL_JSON_PARSE_TIME, time.time() - stime)
            # Build result structure
            res._add_json_solution(jsol, self.solver.expr_map)

        # Process Log
        if self.log_data is not None:
            res._set_solver_log(''.join(self.log_data))
            self.log_data = []
        res.process_infos.update(self.process_infos)
        return res


    def _raise_not_supported(self):
        """ Raise an exception indicating that the calling method is not supported.
        """
        raise CpoNotSupportedException("Method '{}' is not available in solver agent '{}' ({})."
                                       .format(inspect.stack()[1][3], self.context.agent, type(self)))


class CpoSolver(object):
    """ This class represents the public API of the object allowing to solve a CPO model.

    It create the appropriate :class:`CpoSolverAgent` that actually implements solving functions, depending
    on the value of the configuration parameter *context.solver.agent*.
    """
    __slots__ = ('model',           # Model to solve
                 'context',         # Solving context
                 'cpostr',          # CPO model string
                 'expr_map',        # Map of expressions to rebuild result
                 'blackbox_map',    # Set of blackbox functions used in the model
                 'process_infos',   # Processing information
                 'agent',           # Solver agent
                 'status',          # Current solver status
                 'operation',       # Current running operation
                 'last_result',     # Last returned solution
                 'listeners',       # List of solve listeners
                 'status_lock',     # Lock protecting status change
                 'callbacks',       # List of CPO solver callbacks
                 'abort_supported', # Indicates that abort is supported by agent
                 'model_published', # Indicates if CPO model has been published
                 'model_sent',      # Indicates if CPO model has been sent to the solver
                 'callbacks_registered',  # Indicate if callbacks (CPO callback and blackboxes) have been registered
                 )

    def __init__(self, model, **kwargs):
        """ **Constructor**

        All necessary solving parameters are taken from the solving context that is constructed from the following list
        of sources, each one overwriting the previous:

           - the default solving context that is defined in the module :mod:`~docplex.cp.config`
           - the user-specific customizations of the context that may be defined (see :mod:`~docplex.cp.config` for details),
           - the parameters that are set in the model itself,
           - the optional arguments of this method.

        If an optional argument other than `context` or `params` is given to this method, it is searched in the
        context where its value is replaced by the new one.
        If not found, it is then considered as a solver parameter.
        In this case, only public parameters are allowed, except if the context attribute `solver.enable_undocumented_params`
        is set to True. This can be done directly when creating the solver, as for example:
        ::

            slvr = CpoSolver(mdl, enable_undocumented_params=True, MyPrivateParam=MyValue)

        Args:
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
            (listeners) (Optional): List of solution listeners
        """
        super(CpoSolver, self).__init__()
        self.agent = None
        self.process_infos = CpoProcessInfos()
        self.cpostr = None
        self.expr_map = None
        self.blackbox_map = None
        self.last_result = None
        self.status = STATUS_IDLE
        self.status_lock = threading.Lock()
        self.listeners = []
        self.callbacks = []
        self.operation = None
        self.abort_supported = False
        self.model_published = False
        self.model_sent = False
        self.callbacks_registered = False

        # Build effective context from args
        # OO's version
        # context = config._get_effective_context(**kwargs)
        # context.params = model.merge_with_parameters(context.params)
        ## trying to fix CP#303
        ctx = config._get_effective_context()
        if model.parameters:
            ctx.params.set_other(model.parameters)
        ctx = config._get_effective_context(context=ctx, **kwargs)

        # If defined, limit the number of threads
        mxt = ctx.solver.max_threads
        if isinstance(mxt, int):
            # Maximize number of workers
            nbw = ctx.params.Workers
            try:
                # nbw can be None
                nbw = int(nbw) if nbw else None
            except ValueError:
                # treat badly formatted strings as None
                nbw = None
            if (nbw is None) or (nbw > mxt):
                ctx.params.Workers = mxt
                print("WARNING: Number of workers has been reduced to " + str(mxt) + " to comply with platform limitations.")

        # Save attributes
        self.model = model
        self.context = ctx

        # Determine appropriate solver agent
        self.agent = self._get_solver_agent()
        self.abort_supported = self.agent._is_abort_search_supported()

        # Add configured default listeners if any
        # Note: calling solver_created() is not required as it is done by add_listener().
        lstnrs = ctx.solver.listeners
        if lstnrs is not None:
            if is_array(lstnrs):
                for lstnr in lstnrs:
                    self._add_listener_from_class(lstnr)
            else:
                self._add_listener_from_class(lstnrs)


    def __iter__(self):
        """  Define solver as an iterator """
        return self


    def __del__(self):
        # End solver
        self.end()


    def __enter__(self):
        # For usage in with
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        # End solver
        self.end()


    def set_solve_with_search_next(self, swsn):
        """ Set the flag indicating to solve with a search_next sequence instead of a single solve.

        If this indicator is set, a call to method :meth:`~CpoSolver.solve` will automatically call method
        :meth:`~CpoSolver.solve_with_search_next` instead,
        allowing listeners to be warned about all intermediate solutions.

        The same behavior is also obtained if the configuration attribute context.solver.solve_with_search_next
        is set to True.

        Args:
            swsn:  Solve wist start-next indicator
        """
        self.context.solver.solve_with_search_next = swsn


    def get_model(self):
        """ Returns the model solved by this solver.

        Returns:
            Model solved by this solver
        """
        return self.model


    def get_model_format_version(self):
        """ If defined, returns the format version of the model.

        Returns:
            Model format version, None if not defined.
        """
        return None if self.model is None else self.model.get_format_version()


    def get_solver_version(self):
        """ Returns, if available, the version of the underlying solver.

        Returns:
            Solver version, None if not defined.
        """
        return self.agent.version_info.get("SolverVersion")


    def get_parameters(self):
        """ Get the actual solver parameters, as modified by configuration or arguments.

        Returns:
            Solver parameters, object of class :class:`~docplex.cp.parameters.CpoParameters`.
        """
        return self.context.params


    def solve(self):
        """ Solve the model

        This function solves the model using CP Optimizer's built-in strategy.
        The built-in strategy is determined by setting the parameter SearchType (see docplex.cp.parameters).
        If the model contains an objective, then the optimal solution with respect to the objective will be calculated.
        Otherwise, a solution satisfying all problem constraints will be calculated.

        The function returns an object of the class CpoSolveResult (see docplex.cp.solution) that contains the solution
        if exists, plus different information on the solving process.

        If the context parameter *solve_with_search_next* (or config parameter *context.solver.solve_with_search_next*)
        is set to True, the call to solve() is replaced by loop of search_next() calls which returns the last
        solution found.
        Difference is that, if a solver listener has been added to the solver, it is warned of all intermediate solutions.

        Returns:
            Solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoException: (or derived) if error.
        """
        self.last_result = None

        # Check solve with start/next
        if self.context.solver.solve_with_search_next:
            return self.solve_with_search_next()

        # Notify listeners
        self._notify_listeners_start_operation(listener.OPERATION_SOLVE)

        # Solve model
        self._check_status(STATUS_IDLE)
        self._set_status(STATUS_SOLVING)
        stime = time.time()
        try:
            msol = self.agent.solve()
        except Exception as e:
            # Check if aborted in the mean time
            if self._check_status_aborted():
                return self.last_result
            if self.context.log_exceptions:
                traceback.print_exc()
            raise e
        self._set_status(STATUS_IDLE)
        stime = time.time() - stime
        self.context.solver.log(1, "Model '", self.model.get_name(), "' solved in ", round(stime, 2), " sec.")
        msol.process_infos[CpoProcessInfos.SOLVE_TOTAL_TIME] = stime

        # Store last solution
        self.last_result = msol

        # Notify listeners
        for lstnr in self.listeners:
            lstnr.new_result(self, msol)
        self._notify_listeners_end_operation()

        # Return solution
        return msol
        
     
    def solve_with_search_next(self):
        """ Solve the model using a start/next loop instead of standard solve.

        Return:
            Last solve result
        """
        # Loop on all new solutions
        last_res = None
        while True:
            # Search for next solution
            sres = self.search_next()

            # Check old-style abort
            if (not self.abort_supported) and (sres.get_solve_status() == SOLVE_STATUS_JOB_ABORTED):
                return last_res if last_res is not None else self.last_result

            # Check successful search
            if sres.is_new_solution():
                # Store solution as currently best one to be returned
                last_res = sres
            else:
                break

        # Process end of search
        #print("msol: {}, is_sol: {}, isoptimal: {}".format(msol,  msol.is_solution(), msol.is_solution_optimal()))
        #print("last_sol: {}".format(last_sol))
        if sres.is_solution_optimal():
            # Force optimal solution to be new even if already given
            last_res = sres
            last_res.new_solution = True
        elif sres.get_stop_cause() == STOP_CAUSE_ABORT:
            # Force last solution to be last one to get the abort status
            last_res = sres
        elif last_res is None:
            last_res = sres
        else:
            # Update last solution with last solver infos
            last_res.solver_infos = sres.solver_infos
        self.end_search()
        return last_res


    def search_next(self):
        """ Get the next available solution.

        This method returns an object of class :class:`~docplex.cp.solution.CpoSolveResult` whose method
        :meth:`~docplex.cp.solution.CpoSolveResult.is_solution` returns True if a new solution is found.
        This method returns False if there is no solution, or if the solution is the same than the previous
        one but the solve status has moved from Feasible to Optimal.
        In this last case, the optimality of the last solution can be checked using the following code:
        ::

            optimal = slvr.get_last_result().is_solution_optimal()

        This function is available with local CPO solver for release number greater or equal to 12.7.0.

        Returns:
            Next solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoNotSupportedException: if method not available in the solver agent.
        """
        # Initiate search if needed
        if self.status == STATUS_IDLE:
            self.last_result = None
            self.agent.start_search()
            self._set_status(STATUS_SEARCH_WAITING)
            # Notify listeners about start of search
            self._notify_listeners_start_operation(listener.OPERATION_SOLVE)

        # Check if status is aborted in the mean time (may be caused by listener)
        if self._check_status_aborted():
            return self.last_result

        self._check_status(STATUS_SEARCH_WAITING)

        # Search next
        stime = time.time()
        self._set_status(STATUS_SEARCH_RUNNING)
        try:
            sres = self.agent.search_next()
        except BaseException as e:
            sys.stdout.flush()
            # Check if aborted in the mean time
            if self._check_status_aborted():
                return self.last_result
            if self.context.log_exceptions:
                traceback.print_exc()
            raise CpoSolverException("Exception caught from CP solver: {}".format(e))
        if self.abort_supported and sres.get_search_status() == SEARCH_STATUS_STOPPED:
            self._set_status(STATUS_IDLE)
        else:
            self._set_status(STATUS_SEARCH_WAITING)
        stime = time.time() - stime
        self.context.solver.log(1, "Model '", self.model.get_name(), "' next solution in ", round(stime, 2), " sec.")

        # Special case for old solvers where last optimal solution is empty
        if sres.is_solution_optimal and (sres.solution is None or sres.solution.is_empty()) and (self.last_result is not None):
            sres.solution = self.last_result.solution

        # Store last solution
        self.last_result = sres

        # Notify listeners
        for lstnr in self.listeners:
            lstnr.new_result(self, sres)

        # Return solution
        return sres


    def end_search(self):
        """ End current search.

        This function is available only with local CPO solver with release number greater or equal to 12.7.0.

        Returns:
            Last model solution with last solve information,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoNotSupportedException: if method not available in the solver agent.
        """
        # Check solve aborted
        if self.status in (STATUS_RELEASED, STATUS_ABORTED):
            self._notify_listeners_end_operation()
            self._set_status(STATUS_IDLE)
            return self.last_result

        # Check old fashion abort
        if not self.abort_supported:
            if self.status == STATUS_IDLE:
                self._notify_listeners_end_operation()
                self._set_status(STATUS_IDLE)
                return self.last_result
            self._check_status(STATUS_SEARCH_WAITING)

        # Normal case
        msol = self.agent.end_search()
        self._set_status(STATUS_IDLE)
        self.last_result = msol
        self._notify_listeners_end_operation()
        return msol


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        Given an infeasible model, the conflict refiner can identify conflicting constraints and variable domains
        within the model to help you identify the causes of the infeasibility.
        In this context, a conflict is a subset of the constraints and/or variable domains of the model
        which are mutually contradictory.
        Since the conflict is minimal, removal of any one of these constraints will remove that
        particular cause for infeasibility.
        There may be other conflicts in the model; consequently, repair of a given conflict
        does not guarantee feasibility of the remaining model.

        Conflict refiner is controlled by the following parameters (that can be set at CpoSolver creation):

         * ConflictRefinerBranchLimit
         * ConflictRefinerFailLimit
         * ConflictRefinerIterationLimit
         * ConflictRefinerOnVariables
         * ConflictRefinerTimeLimit

        that are described in module :mod:`docplex.cp.parameters`.

        Note that the general *TimeLimit* parameter is used as a limiter for each conflict refiner iteration, but the
        global limitation in time must be set using *ConflictRefinerTimeLimit* that is infinite by default.

        This function is available only with local CPO solver with release number greater or equal to 12.7.0.

        Returns:
            List of constraints that cause the conflict,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        Raises:
            CpoNotSupportedException: if method not available in the solver agent.
        """
        # Start refine conflict
        self._check_status(STATUS_IDLE)
        self._set_status(STATUS_REFINING_CONFLICT)
        self._notify_listeners_start_operation(listener.OPERATION_REFINE_CONFLICT)

        # Ensure cpo model is generated with all constraints named
        namecstrs = self.context.model.name_all_constraints
        if not namecstrs:
            self.context.model.name_all_constraints = True
            self.cpostr = None
            self.agent.solver.model_sent = False

        # Refine conflict
        msol = self.agent.refine_conflict()

        # Restore previous name constraints indicator
        self.context.model.name_all_constraints = namecstrs

        # Call listeners with conflict result
        for lstnr in self.listeners:
            lstnr.new_result(self, msol)

        # End refine conflict
        self._set_status(STATUS_IDLE)
        self._notify_listeners_end_operation()

        return msol


    def propagate(self):
        """ This method invokes the propagation on the current model.

        Constraint propagation is the process of communicating the domain reduction of a decision variable to
        all of the constraints that are stated over this variable.
        This process can result in more domain reductions.
        These domain reductions, in turn, are communicated to the appropriate constraints.
        This process continues until no more variable domains can be reduced or when a domain becomes empty
        and a failure occurs.
        An empty domain during the initial constraint propagation means that the model has no solution.

        The result is a object of class CpoSolveResult, the same than the one returned by solve() method.
        However, variable domains may not be completely defined.

        This function is available only with local CPO solver with release number greater or equal to 12.7.0.

        Returns:
            Propagation result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoNotSupportedException: method not available in configured solver agent.
        """
        self._check_status(STATUS_IDLE)

        # Notify listeners
        self._notify_listeners_start_operation(listener.OPERATION_PROPAGATE)

        # Propagate model
        self._set_status(STATUS_PROPAGATING)
        psol = self.agent.propagate()
        self._set_status(STATUS_IDLE)

        # Notify listeners
        for lstnr in self.listeners:
            lstnr.new_result(self, psol)
        self._notify_listeners_end_operation()

        return psol


    def run_seeds(self, nbrun):
        """ This method runs *nbrun* times the CP optimizer search with different random seeds
        and computes statistics from the result of these runs.

        Result statistics are displayed on the log output that should be activated.
        If the appropriate configuration variable *context.solver.add_log_to_solution* is set to True (default),
        log is also available in the *CpoRunResult* result object, accessible as a string using the method
        :meth:`~docplex.cp.solution.CpoRunResult.get_solver_log`

        Each run of the solver is stopped according to single solve conditions (TimeLimit for example).
        Total run time is then expected to take *nbruns* times the duration of a single run.

        Args:
            nbrun: Number of runs with different seeds.
        Returns:
            Run result, object of class :class:`~docplex.cp.solution.CpoRunResult`.
        Raises:
            CpoNotSupportedException: method not available in configured solver agent.
        """
        self._check_status(STATUS_IDLE)
        self._set_status(STATUS_RUNNING_SEEDS)
        self._notify_listeners_start_operation(listener.OPERATION_RUN_SEEDS)
        rsol = self.agent.run_seeds(nbrun)
        self._set_status(STATUS_IDLE)
        self._notify_listeners_end_operation()
        return rsol


    def explain_failure(self, ltags=None):
        """ This method allows to explain solve failures.

        If called with no arguments, this method invokes a solve of the model with appropriate parameters
        that enable, in the log, the print of a number tag for each solve failure.

        If called with a list of failure tag to explain, the solver is invoked again in a way that it explains,
        in the log, the reason for the failure of the required failure tags.

        This method sets the following solve parameters before calling the solver:

         * :attr:`~docplex.cp.CpoParameters.LogSearchTags` = 'On'
         * :attr:`~docplex.cp.CpoParameters.Workers` = 1
         * :attr:`~docplex.cp.CpoParameters.LogPeriod` = 1
         * :attr:`~docplex.cp.CpoParameters.SearchType` = 'DepthFirst'

        Args:
            ltags:  List of tag ids to explain. If empty or None, the solver is just invoked with appropriate
                    solve parameters to make failure tags displayed in the log.
        Returns:
            Solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        # Set solver parameters
        params = self.agent.params
        params.LogSearchTags = 'On'
        params.Workers = 1
        params.LogPeriod = 1
        params.SearchType = 'DepthFirst'

        # Add failure tags if any
        if ltags:
            self.agent.set_explain_failure_tags(ltags)

        # Solve the model
        msol = self.solve()

        # Remove failure tags if any
        if ltags:
            self.agent.set_explain_failure_tags()

        # Return
        return msol


    def abort_search(self):
        if self.abort_supported and self.agent is not None:
            # Abort search is implemented, just call it.
            self.agent.abort_search()
        else:
            # Abort current search
            self._set_status(STATUS_ABORTED)
            agt = self.agent
            self.agent = None
            if agt is not None:
                agt.abort_search()


    def end(self):
        # End this solver and release associated resources
        agt = self.agent
        self.agent = None
        self._set_status(STATUS_RELEASED)
        if agt is not None:
            agt.end()


    def next(self):
        # """ For solution iteration, get the next available solution.
        #
        # This function is available only with local CPO solver with release number greater or equal to 12.7.0.
        #
        # Returns:
        #     Next solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        # """
        # Check if last solution was optimal
        if not self.last_result or self.last_result.fail_status != FAIL_STATUS_SEARCH_COMPLETED:
            # Get next solution
            msol = self.search_next()
            if msol:
                return msol
        self.end_search()
        raise StopIteration()


    def __next__(self):
        # """ Get the next available solution (same as next() for compatibility with Python 3)
        #
        # This function is available only with local CPO solver with release number greater or equal to 12.7.0.
        #
        # Returns:
        #     Next solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        # """
        return self.next()


    def get_last_solution(self):
        """ Get the last result returned by this solver.

        DEPRECATED. Use get_last_result instead.

        Returns:
            Solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        return self.last_result


    def get_last_result(self):
        """ Get the last result returned by this solver.

        Calling this method can be useful to determine, for example, if the last solution returned
        by a sequence of start_search() and search_next(), or by a solution iterator, is optimal.

        Returns:
            Solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        return self.last_result


    def add_listener(self, lstnr):
        """ Add a solver listener.

        A solver listener is an object extending the class :class:`~docplex.cp.solver.solver_listener.CpoSolverListener`
        which provides multiple functions that are called to notify about the different solving steps.

        Args:
            lstnr:  Solver listener
        """
        assert isinstance(lstnr, CpoSolverListener), \
            "Listener should be an object of class docplex.cp.solver.solver_listener.CpoSolverListener"
        self.listeners.append(lstnr)
        # Notify listener
        lstnr.solver_created(self)


    def _add_listener_from_class(self, lstnr):
        """ Add a solver listener from its class (instance is created).

        Args:
            lstnr:  Solver listener class, or string identifying the class
        """
        if is_string(lstnr):
            # Get listener class from string
            try:
                lclass = utils.get_module_element_from_path(lstnr)
            except Exception as e:
                raise CpoException("Unable to retrieve solver listener class '{}': {}".format(lstnr, e))
            if not inspect.isclass(lclass):
                raise CpoException("Solver listener '{}' is not a class.".format(lstnr))
            if not issubclass(lclass, CpoSolverListener):
                raise CpoException("Solver listener class '{}' should extend CpoSolverListener.".format(lstnr))
        else:
            # Listener is assumed to directly be a class
            lclass = lstnr
            if not inspect.isclass(lclass):
                raise CpoException("Solver listener '{}' is not a class.".format(lclass))
            if not issubclass(lclass, CpoSolverListener):
                raise CpoException("Solver listener class '{}' should extend CpoSolverListener.".format(lclass))
        # Add listener
        self.add_listener(lclass())


    def remove_listener(self, lstnr):
        """ Remove a solver listener previously added with :meth:`~docplex.cp.solver.solver.CpoSolver.add_listener`.

        Args:
            lstnr:  Listener to remove.
        """
        self.listeners.remove(lstnr)


    def add_callback(self, cback):
        """ Add a CPO solver callback.

        A solver callback is an object extending the class :class:`~docplex.cp.solver.cpo_callback.CpoCallback`
        which provides multiple functions that are called to notify about the different solving steps.

        Args:
            cback:  Solver callback
        """
        assert isinstance(cback, CpoCallback), \
            "CPO callback should be an object of class docplex.cp.solver.cpo_callback.CpoCallback"
        self.callbacks.append(cback)


    def remove_callback(self, cback):
        """ Remove a CPO solver callback. previously added with :meth:`~docplex.cp.solver.solver.CpoSolver.add_callback`.

        Args:
            cback:  Callback to remove.
        """
        self.callbacks.remove(cback)


    def _set_status(self, status):
        """ Change solve status, only if allowed

        Args:
            status: New solve status
        """
        with self.status_lock:
            if (status in _ENDING_STATUSES) or (not self.status in _ENDING_STATUSES):
                self.status = status


    def _notify_new_log(self, data):
        """ Notify new log data (called by agent)

        Args:
            data: Log data as a string
        """
        # Notify listeners
        for lstnr in self.listeners:
            lstnr.new_log_data(self, data)


    def _notify_callback_event(self, event, data):
        """ Notify a CPO callback event (called by agent)

        Args:
            event:  Event id
            data:   JSON document associated to this event
        """
        # Notify callbacks
        for cback in self.callbacks:
            cback.invoke(self, event, data)


    def _check_status(self, ests):
        """ Throws an exception if solver status is not the expected one

        Args:
            ests:  Expected status, or list of expected statuses
        Raise:
            CpoException if solver is not in the right status
        """
        if self.status != ests:
           raise CpoException("Unexpected solver status. Should be '{}' instead of '{}'".format(ests, self.status))


    def _notify_listeners_start_operation(self, op):
        """ Call all listeners with operation start

        Args:
            op:  Operation that is started
        """
        self.operation = op
        for lstnr in self.listeners:
            lstnr.start_operation(self, op)


    def _notify_listeners_end_operation(self):
        """ Call all listeners with last operation end
        """
        op = self.operation
        for lstnr in self.listeners:
            lstnr.end_operation(self, op)
        self.operation = None


    def _check_status_aborted(self):
        """ Check if the solve status has been changed to aborted by another thread or a listener.
        If so, an Aborted solve result is stored in last_result.

        Returns: true if status was aborted, false otherwise.
        """
        if self.status != STATUS_ABORTED:
            return False

        if self.abort_supported:
            self.status = STATUS_IDLE
            return False

        self._set_status(STATUS_RELEASED)
        self.last_result = self._create_result_aborted()
        for lstnr in self.listeners:
            lstnr.new_result(self, self.last_result)
        self._notify_listeners_end_operation()
        return True


    def _create_result_aborted(self):
        """ Create a solve result with aborted status
        """
        # Clone last result if any
        if self.last_result is None:
            res = CpoSolveResult(self.model)
            res.solve_status = SOLVE_STATUS_UNKNOWN
        else:
            res = copy.copy(self.last_result)
        res.fail_status = FAIL_STATUS_ABORT
        res.search_status = SEARCH_STATUS_STOPPED
        res.stop_cause = STOP_CAUSE_ABORT
        res.new_solution = False
        return res


    def _get_solver_agent(self):
        """ Get the solver agent instance that is used to solve the model.

        Returns:
            Solver agent instance
        Raises:
            CpoException:  Agent creation error
        """
        # Determine selectable agent(s)
        sctx = self.context.solver

        alist = sctx.agent
        if alist is None:
            # Return empty solver agent
            return CpoSolverAgent(self, sctx.params, sctx)
        elif not (is_string(alist) or is_array(alist)):
            raise CpoException("Agent identifier in config.context.solver.agent should be a string or a list of strings.")

        # Create agent
        if is_string(alist):
            aname = alist
            agent = self._create_solver_agent(alist)
        else:
            # Search first available agent in the list
            agent = None
            aname = None
            errors = []
            for aname in alist:
                try:
                    agent = self._create_solver_agent(aname)
                    break
                except Exception as e:
                    errors.append((aname, str(e)))
                # Agent not found
                errstr = ', '.join(a + ": " + str(e) for (a, e) in errors)
                raise CpoException("Agent creation error: " + errstr)

        # Log solver agent
        sctx.log(1, "Solve model '", self.model.get_name(), "' with agent '", aname, "'")
        agent.process_infos[CpoProcessInfos.SOLVER_AGENT] = aname
        return agent


    def _create_solver_agent(self, aname):
        """ Create a new solver agent from its name.

        Args:
            name: Name of the agent
        Returns:
            Solver agent instance
        Raises:
            CpoException: Agent creation error
        """
        # Get agent context
        sctx = self.context.solver.get(aname)
        if not isinstance(sctx, Context):
            raise CpoSolverException("Unknown solving agent '" + aname + "'. Check config.context.solver.agent parameter.")
        if sctx.is_log_enabled(3):
            sctx.log(3, "Context for solving agent '", aname, "':")
            sctx.write(out=sctx.get_log_output())
        cpath = sctx.class_name
        if cpath is None:
            raise CpoSolverException("Solving agent '" + aname + "' context should contain an attribute 'class_name'")

        # Retrieve solver agent class
        try:
            sclass = utils.get_module_element_from_path(cpath)
        except Exception as e:
            raise CpoSolverException("Unable to retrieve solver agent class '{}': {}".format(cpath, e))
        if not inspect.isclass(sclass):
            raise CpoSolverException("Solver agent '{}' is not a class.".format(cpath))
        if not issubclass(sclass, CpoSolverAgent):
            raise CpoSolverException("Solver agent class '{}' should extend CpoSolverAgent.".format(cpath))

        # Create agent instance
        agent = sclass(self, sctx)
        return agent


    def _build_cpo_model_string(self):
        """ Build the CPO model string and store data required to retrieve model elements
        """
        if self.cpostr is None:
            stime = time.time()
            cplr = CpoCompiler(self.model, params=self.context.params, context=self.context)
            self.cpostr = cplr.get_as_string()
            self.expr_map = cplr.get_expr_map()
            self.blackbox_map = cplr.get_blackbox_map()
            self.process_infos[CpoProcessInfos.MODEL_COMPILE_TIME] = time.time() - stime
            self.process_infos[CpoProcessInfos.MODEL_DATA_SIZE] = len(self.cpostr)


    def _publish_model(self):
        """ Publish the CPO model on console and in file if required by configuration
        """
        # Check if already published
        if self.model_published:
            return

        # Trace CPO model if required
        ctx = self.context
        lout = ctx.get_log_output()
        if lout and ctx.solver.trace_cpo:
            stime = time.time()
            lout.write("Model '" + str(self.model.get_name()) + "' in CPO format:\n")
            lout.write(self.cpostr)
            lout.write("\n")
            self.model.write_information(lout)
            lout.write("\n")
            lout.flush()
            self.process_infos.incr(CpoProcessInfos.MODEL_DUMP_TIME, time.time() - stime)

        # Dump in dump directory if required
        if ctx.model.dump_directory:
            stime = time.time()
            make_directories(ctx.model.dump_directory)
            mname = self.model.get_name()
            if mname is None:
                mname = "Anonymous"
            else:
                # Remove special characters introduced by Jupyter
                mname = mname.replace('<', '').replace('>', '')
            file = ctx.model.dump_directory + "/" + mname + ".cpo"
            with utils.open_utf8(file, 'w') as f:
                f.write(self.cpostr)
            self.process_infos.incr(CpoProcessInfos.MODEL_DUMP_TIME, time.time() - stime)

        # Set published indicator
        self.model_published = True


    def get_cpo_model_string(self):
        """ Get the CPO model as a string, as it is sent to the solver.

        Result string is the exact string that is sent to solver, including all variations caused by configuration
        or changes made when creating the solver.

        Return:
            String containing the CPO model in CPO file format
        """
        # Build string
        self._build_cpo_model_string()

        # Publish model
        self._publish_model()

        # Return
        return self.cpostr


    def _get_blackbox_function_eval_context(self, jdata):
        """ Get the evaluation context of a blackbox function

        Args:
            jdata: JSON data containing function evaluation context
        Returns:
            tuple (blackbox function descriptor, argument values, bound values)
        """
        #print("Enter in _evaluate_blackbox_function: {}".format(jdata))

        # Parse JSON data
        stime = time.time()
        fcall = parse_json_string(jdata)
        self.process_infos.incr(CpoProcessInfos.TOTAL_JSON_PARSE_TIME, time.time() - stime)

        # Retrieve blackbox descriptor from its name
        name = fcall.get('name')
        bbf = self.blackbox_map.get(name)
        #bbf = self.model.get_blackbox_function(name)
        if bbf is None:
            raise CpoException("Try to evaluate a blackbox function {} that does not exists".format(name))
        if not isinstance(bbf, CpoBlackboxFunction):
            raise CpoException("Expression named '{}' is not a blackbox function".format(name))

        # Build arguments values
        params = fcall.get('parameters', ())
        ptypes = bbf.get_arg_types()
        if len(ptypes) != len(params):
           raise CpoException("Blackbox function call to '{}' contains a wrong number of parameters.".format(name))
        incvars = bbf.args_with_vars
        argvalues = [self._build_arg_value(t, v, incvars) for t, v in zip(ptypes, params)]

        # Build bound values
        bnds = fcall.get('returnValuesBounds', ())
        if len(bnds) != bbf.dimension:
           raise CpoException("Blackbox function call to '{}' contains a wrong number of bounds.".format(name))
        bndsvalues = [tuple(solution._get_num_value(v) for v in b) for b in bnds]

        return bbf, argvalues, bndsvalues


    def _build_arg_value(self, tp, vl, adv):
        """ Build blackbox function call argument value
        Args:
            tp:  Parameter type
            vl:  Parameter JSON value
            adv: Include vars indicator
        Returns:
            Parameter value to be passed to evaluation
        """
        n = vl.get('name')
        t = vl.get('type')
        v = vl.get('value')

        if tp in (Type_Int, Type_IntExpr,):
            return solution._get_num_value(v)

        if tp in (Type_Float, Type_FloatExpr,):
            return float(v)

        if tp is Type_IntVar:
            if adv:
                iv = self.expr_map.get(n)
                return CpoIntVarSolution._create_from_json(iv, v)
            else:
                return solution._get_num_value(v)

        if tp is Type_IntervalVar:
            iv = self.expr_map.get(n)
            r = CpoIntervalVarSolution._create_from_json(iv, v)
            return r if adv else r.get_value()

        if tp is Type_IntArray:
            return [solution._get_num_value(e) for e in v]

        if tp is Type_IntExprArray:
            return [solution._get_num_value(e.get('value')) for e in v]

        if tp is Type_IntVarArray:
            if adv:
                return [CpoIntVarSolution._create_from_json(self.expr_map.get(e.get('name')), e.get('value')) for e in v]
            else:
                return [solution._get_num_value(e.get('value')) for e in v]

        if tp is Type_FloatArray:
            return v

        if tp is Type_FloatExprArray:
            return [float(e.get('value')) for e in v]

        if tp is Type_IntervalVarArray:
            res = [CpoIntervalVarSolution._create_from_json(self.expr_map.get(e.get('name')), e.get('value')) for e in v]
            return res if adv else [e.get_value() for e in res]

        if tp is Type_SequenceVar:
            # Retrieve original variable
            sv = self.expr_map.get(n)
            assert sv is not None, "Sequence variable '{}' not found in the model".format(n)
            vars = sv.get_interval_variables()
            # Build actual list of variables from list of indexes
            lv = [vars[i] for i in v]
            return CpoSequenceVarSolution(self.expr_map.get(n), lv) if adv else lv

        if tp is Type_SequenceVarArray:
            res = []
            for jsv in v:
                svn = jsv.get('name')
                sv = self.expr_map.get(svn)
                assert sv is not None, "Sequence variable '{}' not found in the model".format(svn)
                vars = sv.get_interval_variables()
                lv = [vars[i] for i in jsv.get('value')]
                if adv:
                    res.append(CpoSequenceVarSolution(sv, lv))
                else:
                    res.append(lv)
            return res

        raise CpoException("INTERNAL ERROR: Unknown blackbox argument type {}".format(tp))


###############################################################################
##  Public functions
###############################################################################

def get_version_info():
    """ If the solver agent defined in the configuration enables this function,
    this method returns solver version information.

    This method creates a CP solver using the default configuration parameters.
    It then retrieves this information, and close the solver.
    The returned value is a dictionary with various information, as in the
    following example:
    ::

    {
       "ProxyVersion" : 5,
       "SourceDate" : "Sep 12 2017",
       "SolverVersion" : "12.8.0.0",
       "IntMin" : -9007199254740991,
       "IntMax" : 9007199254740991,
       "IntervalMin" : -4503599627370494,
       "IntervalMax" : 4503599627370494,
    }

    Returns:
        Solver information dictionary, or empty dictionary if not available.
    """
    from docplex.cp.model import CpoModel
    try:
        with CpoSolver(CpoModel()) as slvr:
            return slvr.agent.version_info
    except:
        if config.context.log_exceptions:
            traceback.print_exc()
        pass
    return {}


def get_solver_version():
    """ If the solver agent defined in the configuration enables this function,
    this method returns solver version number.

    Returns:
        Solver version string, or None if not available.
    """
    vinfo = get_version_info()
    return vinfo.get('SolverVersion') if vinfo else None


###############################################################################
##  Private Functions
###############################################################################

def _get_solver_agent_class(aname, sctx):
    """ Get a solver agent class from its name

    Args:
        aname:  Solver agent name
        sctx:   Candidate solver context
    Returns:
        Solver agent class
    """
    # Check for solver agent context
    if not isinstance(sctx, Context):
        raise CpoException("Unknown solving agent '" + aname + "'. Check config.context.solver.agent parameter.")
    cpath = sctx.class_name
    if cpath is None:
        raise CpoSolverException("Solving agent '" + aname + "' context does not contain attribute 'class_name'")

    # Split class name
    pnx = cpath.rfind('.')
    if pnx < 0:
        raise CpoSolverException("Invalid class name '" + cpath + "' for solving agent '" + aname + "'. Should be <package>.<module>.<class>.")
    mname = cpath[:pnx]
    cname = cpath[pnx + 1:]

    # Load module
    try:
        module = importlib.import_module(mname)
    except Exception as e:
        raise CpoSolverException("Module '" + mname + "' import error: " + str(e))

    # Create and check class
    sclass = getattr(module, cname, None)
    if sclass is None:
        raise CpoSolverException("Module '" + mname + "' does not contain a class '" + cname + "'")
    if not inspect.isclass(sclass):
        raise CpoSolverException("Agent class '" + cpath + "' is not a class.")
    if not issubclass(sclass, CpoSolverAgent):
        raise CpoSolverException("Solver agent class '" + cpath + "' does not extend CpoSolverAgent.")

    # Return
    return sclass


def _replace_names_in_json_dict(jdict, renmap):
    """ Replace keys that has been renamed in a JSON result directory
    Args:
        jdict:  Json result dictionary
        renmap: Renaming map, key is name to replace, value is name to use instead
    """
    if jdict:
        for k in list(jdict.keys()):
            nk = renmap.get(k)
            if nk:
                jdict[nk] = jdict[k]
                del jdict[k]

