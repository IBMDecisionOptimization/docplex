# --------------------------------------------------------------------------
# File: _parameter_classes.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Parameters for the CPLEX Python API.

This module defines classes for parameters, groups of parameters, and
parameter constants used in the CPLEX Python API.  For more detail, see also
the corresponding commands of the Interactive Optimizer documented in the
CPLEX Parameters Reference Manual.
"""
import functools
import weakref

from ._aux_functions import init_list_args
from . import _procedural as CPX_PROC
from . import _constants
from ..exceptions import CplexError, CplexSolverError, error_codes
from ..paramset import ParameterSet
from ..constant_class import ConstantClass


def _get_info_wrapper(func):
    """Decorator to lazily load parameter info.

    We have to lazily load parameter info in several methods of the
    Parameter class. This decorator makes it easy and consistent to do
    this.
    """
    @functools.wraps(func)
    def wrap(self, *args, **kwargs):
        # First, lazily load parameter info if necc.
        self._get_info()
        # Then call the function.
        return func(self, *args, **kwargs)
    return wrap


class Parameter():
    """Base class for Cplex parameters.

    """

    def __init__(self, env, about, parent, name, constants=None):
        """non-public"""
        self._env = weakref.proxy(env)
        self._id, self._help, self._type = about
        self._parent = parent
        self._name = name
        if constants is not None:
            self.values = constants()
        self._has_info = False
        # self._defval gets set lazily by self._get_info().
        self._defval = None

    def __repr__(self):
        """Returns the name of the parameter within the hierarchy."""
        return "".join([self._parent.__repr__(), '.', self._name])

    def _check_value(self, value):
        """Checks the validity of the parameter value."""
        raise NotImplementedError

    def set(self, value):
        """Sets the parameter to value."""
        self._check_value(value)
        try:
            self._env.parameters._set(self._id, value, self._type)
        except TypeError:
            # Replace ugly TypeError message from the SWIG layer with
            # something more informative.
            raise TypeError("invalid parameter value: {0}".format(value)) from None

    def get(self):
        """Returns the current value of the parameter."""
        return self._env.parameters._get(self._id, self._type)

    def reset(self):
        """Sets the parameter to its default value."""
        try:
            self.set(self.default())
        except CplexSolverError as cse:
            if ((self._id == _constants.CPX_PARAM_CPUMASK) and
                    cse.args[2] == error_codes.CPXERR_UNSUPPORTED_OPERATION):
                pass
            else:
                raise

    def _get_info(self):
        """Lazily load the default, min, and max values."""
        raise NotImplementedError

    @_get_info_wrapper
    def default(self):
        """Returns the default value of the parameter."""
        return self._defval

    def type(self):
        """Returns the type of the parameter.

        Allowed types are float, int, and str.
        """
        return type(self.default())

    def help(self):
        """Returns the documentation for the parameter."""
        return self._help


class NumParameter(Parameter):
    """Class for integer and float parameters.

    """

    @_get_info_wrapper
    def _check_value(self, value):
        """Checks the validity of the parameter value."""
        # As we define a special min value for CPX_PARAM_CLONELOG in the Python API
        # we have to have special handling for it.
        if (self._id == _constants.CPX_PARAM_CLONELOG and
                value < self._minval):
            raise ValueError("invalid {0} parameter value: {1}".format(
                self._name, value))

    def _get_info(self):
        """Lazily load the default, min, and max values."""
        if self._has_info:
            return
        self._has_info = True
        (self._defval,
         self._minval,
         self._maxval) = self._env.parameters._get_info(self._id, self._type)
        # Override some default values for the Python API.
        if self._id == _constants.CPX_PARAM_CLONELOG:
            self._minval = 0
        elif self._id == _constants.CPX_PARAM_DATACHECK:
            self._defval = _constants.CPX_DATACHECK_WARN

    @_get_info_wrapper
    def min(self):
        """Returns the minimum value for the parameter."""
        return self._minval

    @_get_info_wrapper
    def max(self):
        """Returns the maximum value for the parameter."""
        return self._maxval


class StrParameter(Parameter):
    """Class for string parameters.

    """

    def _check_value(self, value):
        """Checks the validity of the parameter value."""

    def _get_info(self):
        """Lazily load the default value.

        Note
          For string parameters there is no min and max value.
        """
        if self._has_info:
            return
        self._has_info = True
        self._defval = self._env.parameters._get_info(self._id, self._type)


class ParameterGroup():
    """Class containing a group of Cplex parameters.

    """

    def __init__(self, env, members, parent):
        """non-public"""
        self._env = weakref.proxy(env)
        self._parent = parent
        # self._name gets set dynamically below when we call
        # self.__dict__.update() (see _parameter_hierarchy.py).
        self._name = None
        self.__dict__.update(members(env, self))

    def __repr__(self):
        """Returns the name of the parameter group within the hierarchy."""
        return "".join([self._parent.__repr__(), '.', self._name])

    def reset(self):
        """Sets the parameters in the group to their default values."""
        for member in self.__dict__.values():
            if (isinstance(member, (ParameterGroup, Parameter)) and
                    member != self._parent):
                member.reset()

    def _get_params(self, filterfunc):
        """non-public"""
        retval = []
        for member in self.__dict__.values():
            if isinstance(member, ParameterGroup) and member != self._parent:
                retval.extend(member._get_params(filterfunc))
            if isinstance(member, Parameter):
                if filterfunc(member):
                    retval.append((member, member.get()))
        return retval

    def get_changed(self):
        """Returns a list of the changed parameters in the group.

        Returns a list of (parameter, value) pairs.  Each parameter is
        an instance of the Parameter class, and thus the parameter
        value can be changed via its set method, or this object can be
        passed to the tuning functions.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.parameters.read.datacheck.set(
        ...     c.parameters.read.datacheck.values.assist)
        >>> for parameter, value in c.parameters.get_changed():
        ...     pass  # do something
        """
        return self._get_params(lambda x: x.get() != x.default())

    def get_all(self):
        """Returns a list of all the parameters in the group.

        Returns a list of (parameter, value) pairs.  Each parameter is
        an instance of the Parameter class, and thus the parameter
        value can be changed via its set method, or this object can be
        passed to the tuning functions.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> for parameter, value in c.parameters.get_all():
        ...     pass  # do something
        """
        return self._get_params(lambda x: True)


class TuningConstants(ConstantClass):
    """Status codes returned by tuning methods.

    For an explanation of tuning, see that topic in
    the CPLEX User's Manual.
    """

    completed = 0  # There is no constant for this.
    abort = _constants.CPX_TUNE_ABORT
    time_limit = _constants.CPX_TUNE_TILIM
    dettime_limit = _constants.CPX_TUNE_DETTILIM


class RootParameterGroup(ParameterGroup):
    """Class containing all the Cplex parameters.

    """

    tuning_status = TuningConstants()
    """See `TuningConstants()`"""

    def __init__(self, env, members):
        if env is None and members is None:
            return
        env.parameters = self
        super().__init__(env, members, None)
        # At the C-level, the apiencoding parameter is always UTF-8 in
        # the Python API.
        self._set(_constants.CPX_PARAM_APIENCODING, "UTF-8",
                  _constants.CPX_PARAMTYPE_STRING)
        CPX_PROC.fixparam(self._env._e, _constants.CPX_PARAM_APIENCODING)
        # Turn off access to presolved problem in callbacks in the Python API.
        # CPX_PARAM_MIPCBREDLP is hidden so we have to set it via the
        # parameter ID.
        self._set(_constants.CPX_PARAM_MIPCBREDLP, 0,
                  _constants.CPX_PARAMTYPE_INT)
        CPX_PROC.fixparam(self._env._e, _constants.CPX_PARAM_MIPCBREDLP)
        # Fix CPX_PARAM_SCRIND to "off" (see RTC-36832).
        self._set(_constants.CPX_PARAM_SCRIND, _constants.CPX_OFF,
                  _constants.CPX_PARAMTYPE_INT)
        CPX_PROC.fixparam(self._env._e, _constants.CPX_PARAM_SCRIND)
        # By default, the datacheck parameter is "on" in the Python API.
        self.read.datacheck.set(_constants.CPX_DATACHECK_WARN)

    def reset(self):
        """Sets the parameters in the group to their default values."""
        # Rather than calling ParameterGroup.reset(self), we can just
        # reset using CPXXsetdefaults, which should be much faster. We
        # still allow users to call reset() on individual parameters or
        # parameter groups, though.
        CPX_PROC.setdefaults(self._env._e)
        # By default, the datacheck parameter is "on" in the Python API.
        self.read.datacheck.set(_constants.CPX_DATACHECK_WARN)

    def __repr__(self):
        """Return 'parameters'."""
        return self._name

    def _set(self, which_parameter, value, paramtype=None):
        # RTC-34595
        if paramtype is None:
            paramtype = CPX_PROC.getparamtype(self._env._e,
                                              which_parameter)
        if paramtype == _constants.CPX_PARAMTYPE_INT:
            if isinstance(value, float):
                value = int(value)  # will upconvert to long, if necc.
            CPX_PROC.setintparam(self._env._e, which_parameter, value)
        elif paramtype == _constants.CPX_PARAMTYPE_DOUBLE:
            if isinstance(value, int):
                value = float(value)
            CPX_PROC.setdblparam(self._env._e, which_parameter, value)
        elif paramtype == _constants.CPX_PARAMTYPE_STRING:
            CPX_PROC.setstrparam(self._env._e, which_parameter, value)
        else:
            assert paramtype == _constants.CPX_PARAMTYPE_LONG
            if isinstance(value, float):
                value = int(value)  # will upconvert to long, if necc.
            CPX_PROC.setlongparam(self._env._e, which_parameter, value)

    def _get(self, which_parameter, paramtype=None):
        # RTC-34595
        if paramtype is None:
            paramtype = CPX_PROC.getparamtype(self._env._e,
                                              which_parameter)
        switcher = {
            _constants.CPX_PARAMTYPE_INT: CPX_PROC.getintparam,
            _constants.CPX_PARAMTYPE_DOUBLE: CPX_PROC.getdblparam,
            _constants.CPX_PARAMTYPE_STRING: CPX_PROC.getstrparam,
            _constants.CPX_PARAMTYPE_LONG: CPX_PROC.getlongparam
        }
        func = switcher[paramtype]
        return func(self._env._e, which_parameter)

    def _get_info(self, which_parameter, paramtype=None):
        # RTC-34595
        if paramtype is None:
            paramtype = CPX_PROC.getparamtype(self._env._e,
                                              which_parameter)
        switcher = {
            _constants.CPX_PARAMTYPE_INT: CPX_PROC.infointparam,
            _constants.CPX_PARAMTYPE_DOUBLE: CPX_PROC.infodblparam,
            _constants.CPX_PARAMTYPE_STRING: CPX_PROC.infostrparam,
            _constants.CPX_PARAMTYPE_LONG: CPX_PROC.infolongparam
        }
        func = switcher[paramtype]
        return func(self._env._e, which_parameter)

    def _validate_fixed_args(self, fixed_parameters_and_values):
        if isinstance(fixed_parameters_and_values, ParameterSet):
            if fixed_parameters_and_values not in self._cplex._pslst:
                raise ValueError("parameter set must have been created"
                                 " by this CPLEX problem object")
            else:
                return  # done checking
        valid = False  # guilty until proven innocent
        try:
            paramset = set()
            for (param, _) in fixed_parameters_and_values:
                param_id, _ = param._id, param._type
                if param_id in paramset:
                    raise CplexError("duplicate parameters detected")
                else:
                    paramset.add(param_id)
            # If we can iterate over fixed_parameters_and_values and
            # access the _id and _type attributes of the parameters,
            # then it's considered valid.
            valid = True
        except (AttributeError, TypeError):
            pass
        if not valid:
            raise TypeError("invalid fixed_parameters_and_values arg detected")

    def _get_fixed_args_iter(self, arg):
        if isinstance(arg, ParameterSet):
            for param_id in arg.get_ids():
                param_type = CPX_PROC.getparamtype(self._env._e, param_id)
                param_value = arg.get(param_id)
                yield param_id, param_type, param_value
        else:
            for (param, value) in arg:
                yield param._id, param._type, value

    def _process_fixed_args(self, fixed_parameters_and_values):
        """non-public"""
        if __debug__:
            self._validate_fixed_args(fixed_parameters_and_values)
        int_params_and_values = []
        dbl_params_and_values = []
        str_params_and_values = []
        has_datacheck = False
        for (param_id, param_type, value) in self._get_fixed_args_iter(
                fixed_parameters_and_values):
            if param_id == _constants.CPX_PARAM_DATACHECK:
                has_datacheck = True
            if param_type in (_constants.CPX_PARAMTYPE_INT,
                              _constants.CPX_PARAMTYPE_LONG):
                int_params_and_values.append((param_id, value))
            elif param_type == _constants.CPX_PARAMTYPE_DOUBLE:
                dbl_params_and_values.append((param_id, value))
            else:
                assert param_type == _constants.CPX_PARAMTYPE_STRING, \
                    "unexpected parameter type"
                str_params_and_values.append((param_id, value))
        # In the Python API, the datacheck parameter defaults to "on".
        # When calling the tuning functions the datacheck parameter can
        # be changed as a side effect. Here, we ensure that the value of
        # the datacheck parameter is the same before and after. That is,
        # _unless_ the user overrides it here, explicitly, by passing the
        # datacheck parameter in as a fixed parameter.
        if not has_datacheck:
            int_params_and_values.append(
                (_constants.CPX_PARAM_DATACHECK,
                 self.read.datacheck.get()))
        return (int_params_and_values, dbl_params_and_values,
                str_params_and_values)

    def tune_problem_set(self, filenames, filetypes=None,
                         fixed_parameters_and_values=None):
        """Tunes parameters for a set of problems.

        filenames must be a sequence of strings specifying a set of
        problems to tune.

        If filetypes is given, it must be a sequence of the same
        length as filenames also consisting of strings that specify
        the types of the corresponding files.

        If fixed_parameters_and_values is given, it may be either a
        ParameterSet instance or a sequence of sequences of length 2
        containing instances of the Parameter class that are to be fixed
        during the tuning process and the values at which they are to be
        fixed.

        tune_problem_set returns the status of the tuning procedure,
        which is an attribute of parameters.tuning_status.

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.lpmethod,
        ...        c.parameters.lpmethod.values.auto)
        >>> status = c.parameters.tune_problem_set(
        ...     filenames=["lpex.mps", "example.mps"],
        ...     fixed_parameters_and_values=ps)
        >>> c.parameters.tuning_status[status]
        'completed'
        >>> status = c.parameters.tune_problem_set(
        ...     filenames=["lpex.mps", "example.mps"],
        ...     fixed_parameters_and_values=[
        ...         (c.parameters.lpmethod,
        ...          c.parameters.lpmethod.values.auto)])
        >>> c.parameters.tuning_status[status]
        'completed'
        >>> status = c.parameters.tune_problem_set(
        ...     filenames=["lpex.mps", "example.mps"])
        >>> c.parameters.tuning_status[status]
        'completed'
        """
        filetypes, fixed_parameters_and_values = init_list_args(
            filetypes, fixed_parameters_and_values)
        (int_params_and_values,
         dbl_params_and_values,
         str_params_and_values) = self._process_fixed_args(
             fixed_parameters_and_values)
        return CPX_PROC.tuneparamprobset(self._env._e,
                                         filenames, filetypes,
                                         int_params_and_values,
                                         dbl_params_and_values,
                                         str_params_and_values)

    def tune_problem(self, fixed_parameters_and_values=None):
        """Tunes parameters for a Cplex problem.

        If fixed_parameters_and_values is given, it may be either a
        ParameterSet instance or a sequence of sequences of length 2
        containing instances of the Parameter class that are to be fixed
        during the tuning process and the values at which they are to be
        fixed.

        tune_problem returns the status of the tuning procedure, which
        is an attribute of parameters.tuning_status.

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.lpmethod,
        ...        c.parameters.lpmethod.values.auto)
        >>> status = c.parameters.tune_problem(ps)
        >>> c.parameters.tuning_status[status]
        'completed'
        >>> status = c.parameters.tune_problem([
        ...     (c.parameters.lpmethod,
        ...      c.parameters.lpmethod.values.auto)])
        >>> c.parameters.tuning_status[status]
        'completed'
        >>> status = c.parameters.tune_problem()
        >>> c.parameters.tuning_status[status]
        'completed'
        """
        (fixed_parameters_and_values,) = init_list_args(
            fixed_parameters_and_values)
        (int_params_and_values,
         dbl_params_and_values,
         str_params_and_values) = self._process_fixed_args(
             fixed_parameters_and_values)
        return CPX_PROC.tuneparam(self._env._e, self._cplex._lp,
                                  int_params_and_values,
                                  dbl_params_and_values,
                                  str_params_and_values)

    def read_file(self, filename):
        """Reads a set of parameters from the file filename."""
        CPX_PROC.readcopyparam(self._env._e, filename)

    def write_file(self, filename):
        """Writes a set of parameters to the file filename."""
        CPX_PROC.writeparam(self._env._e, filename)


class off_on_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    off = _constants.CPX_OFF
    on = _constants.CPX_ON


class auto_off_on_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_AUTO
    off = _constants.CPX_OFF
    on = _constants.CPX_ON


class writelevel_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_WRITELEVEL_AUTO
    all_variables = _constants.CPX_WRITELEVEL_ALLVARS
    discrete_variables = _constants.CPX_WRITELEVEL_DISCRETEVARS
    nonzero_variables = _constants.CPX_WRITELEVEL_NONZEROVARS
    nonzero_discrete_variables = _constants.CPX_WRITELEVEL_NONZERODISCRETEVARS


class scale_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = -1
    equilibration = 0
    aggressive = 1


class mip_emph_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    balanced = _constants.CPX_MIPEMPHASIS_BALANCED
    optimality = _constants.CPX_MIPEMPHASIS_OPTIMALITY
    feasibility = _constants.CPX_MIPEMPHASIS_FEASIBILITY
    best_bound = _constants.CPX_MIPEMPHASIS_BESTBOUND
    hidden_feasibility = _constants.CPX_MIPEMPHASIS_HIDDENFEAS
    heuristic = _constants.CPX_MIPEMPHASIS_HEURISTIC


class brdir_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    down = _constants.CPX_BRDIR_DOWN
    auto = _constants.CPX_BRDIR_AUTO
    up = _constants.CPX_BRDIR_UP


class search_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_MIPSEARCH_AUTO
    traditional = _constants.CPX_MIPSEARCH_TRADITIONAL
    dynamic = _constants.CPX_MIPSEARCH_DYNAMIC


class subalg_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_ALG_AUTOMATIC
    primal = _constants.CPX_ALG_PRIMAL
    dual = _constants.CPX_ALG_DUAL
    network = _constants.CPX_ALG_NET
    barrier = _constants.CPX_ALG_BARRIER
    sifting = _constants.CPX_ALG_SIFTING


class nodesel_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    depth_first = _constants.CPX_NODESEL_DFS
    best_bound = _constants.CPX_NODESEL_BESTBOUND
    best_estimate = _constants.CPX_NODESEL_BESTEST
    best_estimate_alt = _constants.CPX_NODESEL_BESTEST_ALT


class alg_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_ALG_AUTOMATIC
    primal = _constants.CPX_ALG_PRIMAL
    dual = _constants.CPX_ALG_DUAL
    barrier = _constants.CPX_ALG_BARRIER
    sifting = _constants.CPX_ALG_SIFTING
    network = _constants.CPX_ALG_NET
    concurrent = _constants.CPX_ALG_CONCURRENT


class varsel_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    min_infeasibility = _constants.CPX_VARSEL_MININFEAS
    default = _constants.CPX_VARSEL_DEFAULT
    max_infeasibility = _constants.CPX_VARSEL_MAXINFEAS
    pseudo_costs = _constants.CPX_VARSEL_PSEUDO
    strong_branching = _constants.CPX_VARSEL_STRONG
    pseudo_reduced_costs = _constants.CPX_VARSEL_PSEUDOREDUCED


class dive_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = 0
    traditional = 1
    probing = 2
    guided = 3


class file_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = 0
    memory = 1
    disk = 2
    disk_compressed = 3


class fpheur_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = -1
    auto = 0
    feas = 1
    obj_and_feas = 2


class cardls_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    disabled = -1
    auto = 0
    at_root = 1
    at_all_nodes = 2


class miqcp_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = 0
    QCP_at_node = 1
    LP_at_node = 2


class presolve_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = -1
    auto = 0
    force = 1
    probe = 2


class v_agg_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = -1
    auto = 0
    moderate = 1
    aggressive = 2
    very_aggressive = 3


class kappastats_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = -1
    auto = 0
    sample = 1
    full = 2


class agg_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = -1
    auto = 0
    moderate = 1
    aggressive = 2


class replace_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    firstin_firstout = _constants.CPX_SOLNPOOL_FIFO
    worst_objective = _constants.CPX_SOLNPOOL_OBJ
    diversity = _constants.CPX_SOLNPOOL_DIV


class ordertype_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    default = 0
    cost = _constants.CPX_MIPORDER_COST
    bounds = _constants.CPX_MIPORDER_BOUNDS
    scaled_cost = _constants.CPX_MIPORDER_SCALEDCOST


class mip_display_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = 0
    integer_feasible = 1
    mip_interval_nodes = 2
    node_cuts = 3
    LP_root = 4
    LP_all = 5


class conflict_algorithm_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_CONFLICTALG_AUTO
    fast = _constants.CPX_CONFLICTALG_FAST
    propagate = _constants.CPX_CONFLICTALG_PROPAGATE
    presolve = _constants.CPX_CONFLICTALG_PRESOLVE
    iis = _constants.CPX_CONFLICTALG_IIS
    limitedsolve = _constants.CPX_CONFLICTALG_LIMITSOLVE
    solve = _constants.CPX_CONFLICTALG_SOLVE


class dual_pricing_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_DPRIIND_AUTO
    full = _constants.CPX_DPRIIND_FULL
    steep = _constants.CPX_DPRIIND_STEEP
    full_steep = _constants.CPX_DPRIIND_FULLSTEEP
    steep_Q_start = _constants.CPX_DPRIIND_STEEPQSTART
    devex = _constants.CPX_DPRIIND_DEVEX


class primal_pricing_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    partial = _constants.CPX_PPRIIND_PARTIAL
    auto = _constants.CPX_PPRIIND_AUTO
    devex = _constants.CPX_PPRIIND_DEVEX
    steep = _constants.CPX_PPRIIND_STEEP
    steep_Q_start = _constants.CPX_PPRIIND_STEEPQSTART
    full = _constants.CPX_PPRIIND_FULL


class display_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = 0
    normal = 1
    detailed = 2


class prered_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = _constants.CPX_PREREDUCE_NOPRIMALORDUAL
    primal = _constants.CPX_PREREDUCE_PRIMALONLY
    dual = _constants.CPX_PREREDUCE_DUALONLY
    primal_and_dual = _constants.CPX_PREREDUCE_PRIMALANDDUAL

class prereform_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = _constants.CPX_PREREFORM_NONE
    interfere_crush = _constants.CPX_PREREFORM_INTERFERE_CRUSH
    interfere_uncrush = _constants.CPX_PREREFORM_INTERFERE_UNCRUSH
    all = _constants.CPX_PREREFORM_ALL


class sos1reform_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = -1
    auto = 0
    logarithmic = 1


class sos2reform_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = -1
    auto = 0
    logarithmic = 1


class coeffreduce_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = 0
    integral = 1
    any = 2


class dependency_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = -1
    off = 0
    begin = 1
    end = 2
    begin_and_end = 3


class dual_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    no = -1
    auto = 0
    yes = 1


class linear_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    only_linear = 0
    full = 1


class repeatpre_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = -1
    off = 0
    without_cuts = 1
    with_cuts = 2
    new_root_cuts = 3


class sym_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = -1
    off = 0
    mild = 1
    moderate = 2
    aggressive = 3
    more_aggressive = 4
    very_aggressive = 5


class qcpduals_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    no = 0
    if_possible = 1
    force = 2


class sift_alg_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_ALG_AUTOMATIC
    primal = _constants.CPX_ALG_PRIMAL
    dual = _constants.CPX_ALG_DUAL
    barrier = _constants.CPX_ALG_BARRIER
    network = _constants.CPX_ALG_NET


class feasopt_mode_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    min_sum = _constants.CPX_FEASOPT_MIN_SUM
    opt_sum = _constants.CPX_FEASOPT_OPT_SUM
    min_inf = _constants.CPX_FEASOPT_MIN_INF
    opt_inf = _constants.CPX_FEASOPT_OPT_INF
    min_quad = _constants.CPX_FEASOPT_MIN_QUAD
    opt_quad = _constants.CPX_FEASOPT_OPT_QUAD


class measure_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    average = _constants.CPX_TUNE_AVERAGE
    minmax = _constants.CPX_TUNE_MINMAX


class tune_display_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = 0
    minimal = 1
    settings = 2
    settings_and_logs = 3


class bar_order_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    approx_min_degree = _constants.CPX_BARORDER_AMD
    approx_min_fill = _constants.CPX_BARORDER_AMF
    nested_dissection = _constants.CPX_BARORDER_ND


class crossover_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = _constants.CPX_ALG_NONE
    auto = _constants.CPX_ALG_AUTOMATIC
    primal = _constants.CPX_ALG_PRIMAL
    dual = _constants.CPX_ALG_DUAL


class bar_alg_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    default = 0
    infeas_estimate = 1
    infeas_constant = 2
    standard = 3


class bar_start_alg_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    zero_dual = 1
    estimated_dual = 2
    average_primal_zero_dual = 3
    average_primal_estimated_dual = 4


class par_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    opportunistic = _constants.CPX_PARALLEL_OPPORTUNISTIC
    auto = _constants.CPX_PARALLEL_AUTO
    deterministic = _constants.CPX_PARALLEL_DETERMINISTIC


class qp_alg_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_ALG_AUTOMATIC
    primal = _constants.CPX_ALG_PRIMAL
    dual = _constants.CPX_ALG_DUAL
    network = _constants.CPX_ALG_NET
    barrier = _constants.CPX_ALG_BARRIER


class advance_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = 0
    standard = 1
    alternate = 2


class clocktype_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = 0
    CPU = 1
    wall = 2


class solutiontype_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPX_AUTO_SOLN
    basic = _constants.CPX_BASIC_SOLN
    non_basic = _constants.CPX_NONBASIC_SOLN


class optimalitytarget_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = 0
    optimal_convex = 1
    first_order = 2
    optimal_global = 3


class datacheck_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    off = _constants.CPX_DATACHECK_OFF
    warn = _constants.CPX_DATACHECK_WARN
    assist = _constants.CPX_DATACHECK_ASSIST


class benders_strategy_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = _constants.CPX_BENDERSSTRATEGY_OFF
    auto = _constants.CPX_BENDERSSTRATEGY_AUTO
    user = _constants.CPX_BENDERSSTRATEGY_USER
    workers = _constants.CPX_BENDERSSTRATEGY_WORKERS
    full = _constants.CPX_BENDERSSTRATEGY_FULL


class network_display_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    none = _constants.CPXNET_NO_DISPLAY_OBJECTIVE
    true_objective_values = _constants.CPXNET_TRUE_OBJECTIVE
    penalized_objective_values = _constants.CPXNET_PENALIZED_OBJECTIVE


class network_netfind_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    pure = _constants.CPX_NETFIND_PURE
    reflection_scaling = _constants.CPX_NETFIND_REFLECT
    general_scaling = _constants.CPX_NETFIND_SCALE


class network_pricing_constants(ConstantClass):
    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    # pylint: disable=too-few-public-methods
    auto = _constants.CPXNET_PRICE_AUTO
    partial = _constants.CPXNET_PRICE_PARTIAL
    multiple_partial = _constants.CPXNET_PRICE_MULT_PART
    multiple_partial_with_sorting = _constants.CPXNET_PRICE_SORT_MULT_PART
