# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015 - 2022
# --------------------------------------------------------------------------

"""
Configuration of the CP Optimizer Python API

This module is the top-level handler of the configuration parameters for
the CP Optimizer Python API. It contains the default values of the different
configuration parameters.

It should NOT be changed directly.
The preferable way is to add at least one of the following files that contain the changes
to be performed:

 * *cpo_config.py*, a local set of changes on these parameters,
 * *cpo_config_<hostname>.py*, a hostname dependent set of changes.

Each of these files is searched first in the directory where the Pyton main file is located,
then in the current directory, and finally in the *PYTHONPATH*. The first one that is found is read.
Final set of parameters is obtained by reading first this module, and then those listed above.

If called as main, this module prints the actual configuration on standard output, including
all customizations made using the mechanism described above.

Following sections describe the most important parameters that can be easily modified to customize
the behavior of the Python API.
All available parameters are available by consulting the source code of this module.

General parameters
------------------

*context.log_output = sys.stdout*

    This parameter contains the default log stream.
    By default it is set to the standard output.
    A value of *None* can be used to disable all logs.

*context.verbose = 0*

    This parameter controls the verbosity level of the log, between 0 and 9, if *log_output* is not None.
    The default value of 0 means no log.

*context.model.add_source_location = True*

    This parameter indicates that when the model is transformed into CPO format, additional information is added
    to correlate expressions with the Python file and line where it has been generated.
    If any error is raised by the solver during the solve, this information is provided in the
    error description, which allows for easier debugging.

*context.model.length_for_alias = None*

    This parameter allows to associate a shorter alias to variables whose name is longer than the given length.
    In the CPO representation of the model, variable is declared with its original name and an alias is created
    to use it with a shorter name in model expressions, allowing to reduce the size of the generated CPO format.

    In the returned solution, variable can be still retrieved with their original names.

    By default, the value is None, which indicates to always keep original variable names.

*context.model.name_all_constraints = False*

    This parameter enables the naming of all constraints when the model is generated in CPO format.
    It is mandatory only if the *refine conflict* function is called.
    Anyway, if the *refine conflict* function is called, and if the CPO format of the model has already been generated,
    it is generated again with this option set in order to allow proper completion of the request.
    Setting it to *True* is preferable only if *refine conflict* function is called on a big model.

*context.model.factorize_expressions = True*

   This parameter indicates to factorize common expressions when generating the model in CPO file format.

*context.model.dump_directory = None*

    This parameter gives the name of a directory where the CPO files that are generated for solving models are stored
    for logging purpose.

    If not None, the directory is created and generated models are stored in files named `<model_name>.cpo`.

*context.model.sort_names = None*

    This parameter precise how the variables are sorted when declared in the CPO file.

    The value can be None for no sort, 'alphabetical' to sort in alphabetical order, or 'natural' to sort in natural order
    (meaning for example that X11 will be declared after X2, which is not the case in alphabetical order).

*context.model.cache.size = 10000*

    This parameter gives the maximum capacity of the internal cache used to speed-up conversion of Python expressions
    into CPO expressions.

*context.model.cache.active = True*

    This parameter allows to enable or disable the expression cache mechanism.
    Value os a boolean (True or False). Default value is True.

*context.params.xxx*

    The parameter `context.params` is an instance of the class
    :class:`~docplex.cp.parameters.CpoParameters` (in :doc:`parameters.py</docplex.cp.parameters.py>`)
    which describes all of the public solver parameters as properties.


Configuration of the model solving
----------------------------------

*context.solver.trace_log = False*

    This parameter indicates to trace the log generated by the solver when solving the CPO model.
    The log is printed on the `context.log_output` stream, if given.

    The default value of this parameter is True for a local solve, but is set to False if remote solve or if the
    Python interpreter is considered as running in a notebook (if module `ipykernel` is detected in system modules).

*context.solver.trace_cpo = False*

    This parameter indicates to trace the CPO model that is generated before submitting it for solving.
    The model is printed on the `context.log_output stream`, if given.

*context.solver.enable_undocumented_params = False*

    This parameter allows to enable the possibility to set solving parameters that are not in the public parameters
    detailed in the class
    :class:`~docplex.cp.parameters.CpoParameters` (in :doc:`parameters.py</docplex.cp.parameters.py>`).

*context.solver.add_log_to_solution = True*

    This parameter indicates to add the solver log content to the solution object.
    By default, this parameter is True but it can be set to False if the log is very big or of no interest.

*context.solver.add_conflict_as_cpo = True*

    This parameter indicates to include the conflict in CPO format in the conflict refiner result
    By default, this parameter is True.

*context.solver.agent = 'local'*

    This parameter specifies the name of the solver agent that is used to solve the model.
    The value of this parameter is the name of a child context of `context.solver`, which contains necessary attributes
    that allow to create and run the required agent.

*context.solver.log_prefix = "[Solver] "*

    Prefix that is added to every message that is logged by the solver component.


Configuration of the `local` solving agent
------------------------------------------

*context.solver.local.execfile*

    Name or full path of the CP Optimizer Interactive executable file.
    By default, it is set to *"cpoptimizer(.exe)"*, and is searched in the file system using the strategy described below.


Configuration of the `lib` solving agent
----------------------------------------

*context.solver.lib.libfile*

    Name or full path of the CP Optimizer library file.
    By default, it is set to *"lib_cpo_solver_\*(.dll)"*, and is searched in the file system using the strategy described below.


Search for executable or library files
--------------------------------------

If an executable or library file is not given with its full path, it is searched in the file system in a list of
directories that is built in this order:

   * the directory where the main Python file is located,
   * the current directory,
   * all directories of the environment variables 'LD_LIBRARY_PATH' or 'DYLD_LIBRARY_PATH' if defined,
   * all directories of the environment variable 'PATH'.

If the file name contains an "\*", and if there are multiple files matching the pattern, the most recent is selected.


Configuration for best performances
-----------------------------------

To configure the CP Python API for best performances, the following configuration settings may be used.
Obviously, this performance is won at the cost of the loss of some features that may be useful in other cases.
::

    context.verbose = 0
    context.model.add_source_location = False
    context.model.length_for_alias = 10
    context.model.name_all_constraints = False
    context.model.dump_directory = None
    context.model.sort_names = None
    context.solver.trace_cpo = False
    context.solver.trace_log = False
    context.solver.add_log_to_solution = False

Detailed description
--------------------
"""
from docplex.cp.utils import *
from docplex.cp.parameters import CpoParameters, ALL_PARAMETER_NAMES

import sys
import socket
import os
import traceback
import platform
import fnmatch

# Check if running in a worker environment
try:
    from docplex_wml.version import *
    #from docplex_wml.worker.environment import WorkerEnvironment
    #import docplex.util.environment as runenv
    from docplex_wml import _in_ws_nb
    if _in_ws_nb is False:
        IS_IN_WORKER = True
    else:
        IS_IN_WORKER = False
except:
    IS_IN_WORKER = False

# CP Optimizer Interactive executable name
CPO_EXEC_INTERACTIVE = "cpoptimizer" + (".exe" if IS_WINDOWS else "")

# CP Optimizer Interactive executable name
CPO_LIBRARY = "lib_cpo_solver_*" + (".dll" if IS_WINDOWS else ".so")

# Environment variable for context changes
CONTEXT_ENVIRONMENT = "DOCPLEX_CP_CONTEXT"


#=============================================================================
# Definition of the default context
#=============================================================================

#-----------------------------------------------------------------------------
# Global context

# Create default context infrastructure
context = Context()

# Default log output
context.log_output = sys.stdout

# Default log verbosity
context.verbose = 0

# Visu enable indicator (internal, can be disabled for testing purpose)
context.visu_enabled = True

# Indicator to log catched exceptions
context.log_exceptions = False


#-----------------------------------------------------------------------------
# CPLEX Optimization Studio context

context.cos = Context()

# Location of CPLEX Optimization Studio, used to search for executables if needed.
# Default value is None, meaning that configuration searches for the most recent
# CPLEX_STUDIO_DIRxxx environment variable
context.cos.location = None


#-----------------------------------------------------------------------------
# Modeling context

context.model = Context()

# Indicate to add source location in model
context.model.add_source_location = True

# Minimal variable name length that trigger use of shorter alias. None for no alias.
context.model.length_for_alias = None

# Automatically add a name to every top-level constraint
context.model.name_all_constraints = False

# Model format generation version that is used by default if no format version is given in the model.
# If None, latest format is used without specifying it explicitly.
context.model.version = None

# Name of the directory where store copy of the generated CPO files. None for no dump.
context.model.dump_directory = None

# Flag to indicate to the compiler to factorize expressions used more than ones
context.model.factorize_expressions = True

# Flag to generate short model output (internal)
context.model.short_output = False

# Type of sort for model variables. Value is in {None, 'alphabetical', 'natural')
context.model.sort_names = None

# Expression cache, global to all models
context.model.cache = Context()
context.model.cache.size = 10000
context.model.cache.active = True


#-----------------------------------------------------------------------------
# Parsing context

context.parser = Context()

# Enable CPO parser to automatically generate blackboxes for unknown functions
context.parser.auto_blackbox = False

# Indicate to print warning messages (for example for unknown blackbox functions) at the end of parsing
context.parser.print_warnings = True

# Indicate to check format version and raise an exception if out of range
context.parser.check_format_version = True

# Indicate to FZN parser to reduce model when possible
context.parser.fzn_reduce = False


#-----------------------------------------------------------------------------
# Solving parameters

context.params = CpoParameters()

# Default time limit in seconds (None for no limit)
context.params.TimeLimit = None

# Workers count (None for number of cores)
context.params.Workers = None


#-----------------------------------------------------------------------------
# Solving context

context.solver = Context()

# Indicate to trace CPO model before solving
context.solver.trace_cpo = False

# Indicate to trace solver log on log_output.
context.solver.trace_log = True
if is_in_notebook():
    # In notebooks, no traces by default because it may crash the environment
    context.solver.trace_log = False

# Enable undocumented parameters
context.solver.enable_undocumented_params = False

# Max number of threads allowed for model solving
context.solver.max_threads = None

# Indicate to add solver log to the solution
context.solver.add_log_to_solution = True

# Indicate to add the conflict in CPO format to conflict refiner result
context.solver.add_conflict_as_cpo = True

# Indicate to replace simple solve by a search_next loop
context.solver.solve_with_search_next = False

# Log prefix
context.solver.log_prefix = "[Solver] "

# Name of the agent to be used for solving. Value is name of one of this context child context.
context.solver.agent = 'local'

# Auto-publish parameters
context.solver.auto_publish = Context()

# Indicate to auto-publish solve details in environment
context.solver.auto_publish.solve_details = True

# Indicate to auto-publish results in environment
context.solver.auto_publish.result_output = "solution.json"

# Indicate to auto-publish kpis in environment
context.solver.auto_publish.kpis_output = "kpis.csv"

# For KPIs output, name of the kpi name column
context.solver.auto_publish.kpis_output_field_name = "Name"

# For KPIs output, name of the kpi value column
context.solver.auto_publish.kpis_output_field_value = "Value"

# Indicate to auto-publish conflicts in environment
context.solver.auto_publish.conflicts_output = "conflicts.csv"

# Indicate to enable auto-publish also with local environment
context.solver.auto_publish.local_publish = False

# Default solver listeners
context.solver.listeners = ["docplex.cp.solver.environment_client.EnvSolverListener"]


#-----------------------------------------------------------------------------
# Local solving using CP Interactive executable

context.solver.local = Context()

# Python class implementing the agent
context.solver.local.class_name = "docplex.cp.solver.solver_local.CpoSolverLocal"

# Name or path of the CP Optimizer Interactive program
context.solver.local.execfile = CPO_EXEC_INTERACTIVE

# Parameters of the exec file (mandatory, do not change)
context.solver.local.parameters = ['-angel']

# Agent log prefix
context.solver.local.log_prefix = "[Local] "

# Local sub-process start timeout in seconds
context.solver.local.process_start_timeout = 5

#-----------------------------------------------------------------------------
# Local solving with CPO library (internal)

context.solver.lib = Context()

# Python class implementing the agent
context.solver.lib.class_name = "docplex.cp.solver.solver_lib.CpoSolverLib"

# Name or path of the CPO library
context.solver.lib.libfile = CPO_LIBRARY

# Agent log prefix
context.solver.lib.log_prefix = "[PyLib] "

# Check if library has been installed with workers specific package
try:
    from docplex_cpo_solver import get_library_path
    lfile = get_library_path()
    if lfile:
        # Force solver to use lib by default
        context.solver.lib.libfile = lfile
        context.solver.agent = 'lib'
except:
    pass


#-----------------------------------------------------------------------------
# Solver simulators

# Solver simulator that always fails
context.solver.simulatorfail = Context()
context.solver.simulatorfail.class_name = "docplex.cp.solver.solver_simulator.CpoSolverSimulatorFail"
context.solver.simulatorfail.log_prefix = "[SimulatorFail] "

# Solver simulator generating a random solution
context.solver.simulatorrandom = Context()
context.solver.class_name = "docplex.cp.solver.solver_simulator.CpoSolverSimulatorRandom"
context.solver.log_prefix = "[SimulatorRandom] "


#-----------------------------------------------------------------------------
# Parameters for solver interactive monitor

context.interactive = Context()

# Confirm solver monitor window closing
context.interactive.user_preferences_file = "DocplexCpUserPreferences.prf"

# Confirm solver monitor window closing
context.interactive.window_confirm_exit = True

# Abort solve when closing window
context.interactive.solver_abort_on_exit = True

# Daemon display thread (avoid tk exception if listener)
context.interactive.daemon_thread = True


#-----------------------------------------------------------------------------
# Apply special changes if running in a worker

if IS_IN_WORKER:
    try:
        context.solver.max_threads = int(os.environ["CORE_NUMBERS"])
    except:
        context.solver.max_threads = 1
    #context.solver.max_threads = runenv.get_environment().get_available_core_count()


#-----------------------------------------------------------------------------
# Create context variable

# Create a local context variable
# Now useless, comes from a time where there was LOCAL_CONTEXT and DOCLOUD_CONTEXT.
# Kept for ascending compatibility
LOCAL_CONTEXT = context


#=============================================================================
# Public functions
#=============================================================================

def get_default():
    """ Get the default context

    Default context is also accessible with the global variable 'context' in this module.

    Returns:
        Current default context
    """
    return context


def set_default(ctx):
    """ Set the default context.

    Default context becomes accessible in the global variable 'context' in this module.

    Args:
        ctx: New default context
    """
    if ctx is None:
        ctx = Context()
    else:
        assert isinstance(ctx, Context), "Context object must be of class Context"
    sys.modules[__name__].context = ctx


#=============================================================================
# Private functions
#=============================================================================

def _get_port_name():
    """ Get the COS port name for the calling environment

    Returns:
        COS port name, None if not found
    """
    sstm = platform.system()
    if sstm == 'Windows': return 'x64_win64'
    if sstm == 'Darwin':  return 'x86-64_osx'
    if sstm == 'Linux':
        machine = platform.machine()
        if machine == 'x86_64':
            machine = 'x86-64'
        return machine + '_linux'
    return None


# List of system properties identifying candidate COS root directories
_COS_ROOT_DIRS_ENV_VARS = ("CPLEX_STUDIO_DIR201", "CPLEX_STUDIO_DIR1210", "CPLEX_STUDIO_DIR129", "CPLEX_STUDIO_DIR128")


def _build_search_path (ctx):
    """ Build the path where search for executables

    Args:
        ctx:  Context to get information from
    """
    # Initialize search path
    path = []

    # Set it to COS location if provided
    cosloc = ctx.get_by_path('cos.location')
    if cosloc:
        # Check existence of the directory
        if not os.path.isdir(cosloc):
            raise Exception("COS location directory '{}' does not exists.".format(cosloc))
        # As explicitly defined, add it in front of search path
        path.append(cosloc + "/bin/" + _get_port_name())

    # Add directory where python main is located
    mpfile = get_main_file()
    if mpfile:
        path.append(os.path.dirname(mpfile))

    # Add current directory
    path.append('.')

    # Add all system path
    path.extend(get_system_path())

    # Append docplex install directory
    python_home = os.path.dirname(os.path.abspath(sys.executable))
    if IS_WINDOWS:
        path.append(os.path.join(python_home, "Scripts"))
        appdata = os.environ.get('APPDATA')
        if appdata is not None:
            path.append(os.path.join(appdata, os.path.join('Python', 'Scripts')))
    else:
        path.append("~/.local/bin")
        path.append(os.path.join(python_home, "bin"))

    # Add all COS locations if not already forced
    if not cosloc:
        # Add all existing installed COS starting from the most recent
        for cnv in _COS_ROOT_DIRS_ENV_VARS:
            cdir = os.getenv(cnv)
            if cdir:
                path.append(cdir + "/cpoptimizer/bin/" + _get_port_name())

    return path


def _search_exec_file(file, ctx):
    """ Search the first occurrence of an executable.

    Args:
        file:  Executable file name
        ctx:   Context to get information from
    Returns:
        Full path of the first executable file found, given file if not found
    """
    # Check null
    if not file:
       return None
    # Check if given file is directly executable
    if is_exe_file(file):
        return file
    # Check if file contains a path
    fname = os.path.basename(file)
    if fname == file:
        # Build full search path
        path = _build_search_path(ctx)
    else:
        # Keep file path as single path
        path = [os.path.dirname(file)]
    # Check if file contains a pattern
    if "*" in fname:
        # Check in the path
        for d in path:
            if not os.path.isdir(d):
                continue
            lf = [os.path.join(d, f) for f in os.listdir(d) if fnmatch.fnmatch(f, fname)]
            if lf:
                # Take most recent executable file
                lf = [f for f in lf if is_exe_file(f)]
                if lf:
                    # Skip "ai" files if any
                    nlf = [f for f in lf if not os.path.splitext(f)[0].endswith("ai")]
                    if nlf:
                        lf = nlf
                    lf.sort(key=lambda x: os.path.getmtime(x))
                    return lf[-1]
    else:
        # Check directly
        for d in path:
            if not os.path.isdir(d):
                continue
            nf = os.path.join(d, fname)
            if is_exe_file(nf):
                return nf

    return file


# Attribute values denoting a default value
DEFAULT_VALUES = ("ENTER YOUR KEY HERE", "ENTER YOUR URL HERE", "default")

def _is_defined(arg, kwargs):
    return (arg in kwargs) and kwargs[arg] and (kwargs[arg] not in DEFAULT_VALUES)

def _change_context_attribute(ctx, key, value):
    rp = ctx.search_and_replace_attribute(key, value)
    # If not found, set in solving parameters
    if rp is None:
        params = ctx.params
        if not isinstance(params, CpoParameters):
            raise CpoException("Invalid configuration attribute '{}' (no 'params' section where put it)".format(key))
        if key in ALL_PARAMETER_NAMES or ctx.solver.enable_undocumented_params:
            setattr(params, key, value)
        else:
            raise CpoException("CPO solver does not accept a parameter named '{}'".format(key))

def _get_effective_context(**kwargs):
    """ Build a effective context from a variable list of arguments that may specify changes to default.

    Args:
        context (optional):   Source context, if not default.
        params (optional):    Solving parameters (CpoParameters) that overwrite those in the solving context
        (others) (optional):  All other context parameters that can be changed.
    Returns:
        Updated (cloned) context
    """
    # Determine source context
    ctx = kwargs.get('context')
    if (ctx is None) or (ctx in DEFAULT_VALUES):
        ctx = context
    ctx = ctx.clone()
    # print("\n*** Source context");
    # ctx.write()

    # First set parameters if given
    prms = kwargs.get('params')
    if prms is not None:
        ctx.params.set_other(prms)

    # Process other changes, check first if undocumented params are enabled (or not)
    uk = 'enable_undocumented_params'
    if uk in kwargs:
        _change_context_attribute(ctx, uk, kwargs[uk])
    for k, v in kwargs.items():
        # Replace obsolete solve_with_start_next attribute
        if k == 'solve_with_start_next':
            k = 'solve_with_search_next'
        if (k != 'context') and (k != 'params') and (k != uk) and (v not in DEFAULT_VALUES):
            _change_context_attribute(ctx, k, v)

    # Get solver execfile
    try:
        if ctx.solver.agent == 'local':
            ctx.solver.local.execfile = _search_exec_file(ctx.solver.local.execfile, ctx)
        elif ctx.solver.agent == 'lib':
            ctx.solver.lib.libfile = _search_exec_file(ctx.solver.lib.libfile, ctx)
    except:
        pass

    # Return
    return ctx


#=============================================================================
# Overload default context with optional customizations
#=============================================================================

# Compute config files search path
_CONFIG_SEARCH_PATH = []
mpfile = get_main_file()     # Add directory of the main file if any
if mpfile:
    _CONFIG_SEARCH_PATH.append(os.path.dirname(mpfile))
_CONFIG_SEARCH_PATH.append('.')       # Add current directory
_CONFIG_SEARCH_PATH.extend(sys.path)  # Add all PYTHONPATH


def _eval_file(file):
    """ If found in python path, evaluate the content of a python module in this module.
    Args:
        file: Python file to evaluate
    """
    for dir in _CONFIG_SEARCH_PATH:
        f = dir + "/" + file if dir else file
        if os.path.isfile(f):
            # print("Loading configuration file '{}'".format(f))
            try:
                with open(f) as fin:
                    fcont = fin.read()
                exec(fcont)
                return
            except Exception as e:
                if context.log_exceptions:
                    traceback.print_exc()
                raise Exception("Error while loading config file {}: {}".format(f, str(e)))

# Load all config changes
for f in ("cpo_config.py", "cpo_config_" + socket.gethostname() + ".py",):
    _eval_file(f)

# Check particular case of renamed solve_with_start_next
ov = context.solver.solve_with_start_next
if ov is not None:
    context.solver.solve_with_search_next = ov

# Overwrite with environment definitions if any
envchg = os.environ.get(CONTEXT_ENVIRONMENT)
if envchg:
    for chg in envchg.split(';'):
        cx = chg.find('=')
        if cx < 0:
            raise Exception("Invalid context change in environment variable {}: '{}'".format(CONTEXT_ENVIRONMENT, chg))
        try:
            context.set_by_path(chg[:cx].strip(), chg[cx+1:].strip())
        except:
            raise Exception("Error when applying change from environment variable {}: '{}'".format(CONTEXT_ENVIRONMENT, chg))


#=============================================================================
# Print configuration when called as main
#=============================================================================

if __name__ == "__main__":
    # Instanciate executable files if needed
    ctx = _get_effective_context()
    ctx.write()
