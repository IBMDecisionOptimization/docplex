# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2022, 2022
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
A blackbox expression is a numerical expression for which the analytical form is not known or cannot be
formulated using CP Optimizer's classical expressions.
A blackbox expression is specified by giving a function that evaluates the expression at given points.

This feature is accessible from Python only for CPLEX Studio versions 22.1 and later.


Defining a blackbox function
----------------------------

A blackbox function is defined by an instance of the class :class:`~docplex.cp.blackbox.CpoBlackboxFunction`
that contains:

 * the name of the blackbox function, auto-allocated if not given,
 * the number of values that are returned by the evaluation of the function, one by default,
 * the list of argument types, auto-determined if not given,
 * the implementation of the function,
 * optionally an argument to pass known result bounds when evaluating the function,
 * an indicator allowing parallel evaluation of the blackbox function, False by default.

Each argument type can be given using its symbolic name string, or using its corresponding
type descriptor constant object of class :class:`~docplex.cp.catalog.CpoType` listed in module
:mod:`~docplex.cp.catalog`. Allowed argument types are:

 * 'Int' or :const:`~docplex.cp.catalog.Type_Int`
 * 'IntVar' or :const:`~docplex.cp.catalog.Type_IntVar`
 * 'IntExpr' or :const:`~docplex.cp.catalog.Type_IntExpr`
 * 'Float' or :const:`~docplex.cp.catalog.Type_Float`
 * 'FloatExpr' or :const:`~docplex.cp.catalog.Type_FloatExpr`
 * 'IntervalVar' or :const:`~docplex.cp.catalog.Type_IntervalVar`
 * 'SequenceVar' or :const:`~docplex.cp.catalog.Type_SequenceVar`
 * 'IntArray' or :const:`~docplex.cp.catalog.Type_IntArray`
 * 'IntVarArray' or :const:`~docplex.cp.catalog.Type_IntVarArray`
 * 'IntExprArray' or :const:`~docplex.cp.catalog.Type_IntExprArray`
 * 'FloatArray' or :const:`~docplex.cp.catalog.Type_FloatArray`
 * 'FloatExprArray' or :const:`~docplex.cp.catalog.Type_FloatExprArray`
 * 'IntervalVarArray' or :const:`~docplex.cp.catalog.Type_IntervalVarArray`
 * 'SequenceVarArray' or :const:`~docplex.cp.catalog.Type_SequenceVarArray`

If not given, the list of argument types is automatically computed as the smallest list of types
that are common to all the references to the blackbox function in the model.


Evaluating a blackbox function
------------------------------

The implementation of the function is a function or a lambda expression that takes as many parameters as declared
in the list of argument types.
If the blackbox definition parameter *args_with_vars* is not set to True (default value is False), the
arguments are limited to their values.
If it is set to True, then the argument is an object of class CpoXxxVarSolution every time the argument type explicitly
refers to a variable. This allows to access to the model variable, that constitute the argument, if available.
In such case:

 * argument variable can be accessed with method *get_var()*,
 * argument value can be accessed with method *get_value()*.

Here is what is given as argument value for each argument type if *args_with_vars* is set to False (default):

 * 'Int': integer constant.
 * 'IntVar': integer constant.
 * 'IntExpr': integer constant.
 * 'Float': float constant.
 * 'FloatExpr': float constant.
 * 'IntervalVar': interval variable solution value, named tuple containing start, end and size of the variable.
 * 'SequenceVar': sequence variable solution value, ordered list of model interval variables in the sequence.
 * 'IntArray': list of integer constants.
 * 'IntVarArray': list of integer constants.
 * 'IntExprArray': list of integer constants.
 * 'FloatArray': list of float constants.
 * 'FloatExprArray': list of float constants.
 * 'IntervalVarArray': list of interval variable solution value (see 'IntervalVar').
 * 'SequenceVarArray': list of sequence variable solution value (see 'SequenceVar').

Here is what is given as argument value for each argument type if *args_with_vars* is explicitly set to True:

 * 'Int': integer constant.
 * 'IntVar': object of :class:`~docplex.cp.solution.CpoIntVarSolution`.
 * 'IntExpr': integer constant.
 * 'Float': float constant.
 * 'FloatExpr': float constant.
 * 'IntervalVar': object of :class:`~docplex.cp.solution.CpoIntervalVarSolution`.
 * 'SequenceVar': object of :class:`~docplex.cp.solution.CpoSequenceVarSolution`.
 * 'IntArray': list of integer constants.
 * 'IntVarArray': list of objects of :class:`~docplex.cp.solution.CpoIntVarSolution`.
 * 'IntExprArray': list of integer constants.
 * 'FloatArray': list of float constants.
 * 'FloatExprArray': list of float constants.
 * 'IntervalVarArray': list of objects of class :class:`~docplex.cp.solution.CpoIntervalVarSolution`.
 * 'SequenceVarArray': list of objects of class :class:`~docplex.cp.solution.CpoSequenceVarSolution`, with value as list of objects of class :class:`~docplex.cp.expression.CpoSequenceVar`.

The function may return:

 * one or several float results in a list,
 * a single number value, automatically converted in a list with this single value,
 * a boolean value, converted as an integer 0 or 1 put in a single value list,
 * *None*, if the function has no solution for these arguments.

If an error occurs during the function evaluation, the exception is forwarded to the solver that will fail with this error.

If an exception is thrown, it is propagated to the solver that rethrow an exception to exit from the solve.

As the evaluation of the blackbox function is required by the different CPO workers, multiple evaluation requests
may happen concurrently.
As Python does not support real multi-threading (see Global Interpreter Lock here: https://wiki.python.org/moin/GlobalInterpreterLock),
concurrent processing may introduce overhead, or computation problems if the blackbox evaluation uses services of
libraries that are not designed to run concurrently.
By default, blackbox function evaluation is then executed in mutual exclusion, but this can be changed by setting
the parameter *parallel* to True, or using method :meth:`~CpoBlackboxFunction.set_parallel_eval`

To avoid calling the blackbox function multiple times with the same parameters, the solver can use a cache that may be
configured at the declaration of the blackbox.
This cache is by default local to each call instance of the blackbox function in the model, in case the evaluation
of the function depends on the calling context and may return different results with the same parameters depending
where the call is placed in the model.


Using a blackbox function in a model
------------------------------------

Once defined, the blackbox function can be used in a model simply by calling the blackbox function descriptor
with appropriate model expressions as arguments.
Following is a simple example that shows how to use a blackbox function with 2 parameters:
::

    bbf = CpoBlackboxFunction(lambda x, y: x + y)
    mdl = CpoModel()
    v1 = integer_var(0, 10, "V1")
    v2 = integer_var(0, 10, "V2")
    mdl.add(bbf(v1, v2) == 5)

If the blackbox function returns multiple values, using the function in a model is done in two steps, as follows:
::

    bbf = CpoBlackboxFunction(impl=lambda x, y: (x + y, x - y), dimension=2)
    mdl = CpoModel()
    v1 = integer_var(0, 10, "V1")
    v2 = integer_var(0, 10, "V2")
    a, b = bbf(v1, v2)
    mdl.add(a + b == 4)

Note that not all returned expressions need to be used in the model.


Detailed description
--------------------
"""

from docplex.cp.catalog import *
from docplex.cp.expression import build_cpo_expr, CpoFunctionCall
from docplex.cp.utils import *
from itertools import count

#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

# List of possible argument types (DO NOT CHANGE AS ENCODING DEPENDS ON IT)
ARGUMENT_TYPES = (Type_Int, Type_IntVar, Type_IntExpr,
                  Type_Float, Type_FloatExpr,
                  Type_IntervalVar, Type_SequenceVar,
                  Type_IntArray, Type_IntVarArray, Type_IntExprArray,
                  Type_FloatArray, Type_FloatExprArray,
                  Type_IntervalVarArray, Type_SequenceVarArray,)

# Encoding of blackbox types into integer (zero reserved)
BLACKBOX_ARGUMENT_TYPES_ENCODING = {t: (i + 1) for i, t in enumerate(ARGUMENT_TYPES)}

# Set of all argument types
_ARG_TYPES_SET = set(ARGUMENT_TYPES)

# Build allowed types per name, ignoring case. Key is type name in lower case, value is type descriptor.
_ARG_TYPES_DICT = {t.get_name().lower(): t for t in ARGUMENT_TYPES}


#-----------------------------------------------------------------------------
# Classes
#-----------------------------------------------------------------------------

class CpoBlackboxFunction(object):
    """ This class represents the descriptor of a blackbox function.

    A blackbox function is defined by:

     * a name, that must not be equal to an existing modeling operation,
     * the number of float values it returns (the dimension),
     * the list argument types,
     * the implementation of the function, that evaluates the result from a list of
       fully evaluated arguments.
     * a parameter allowing to pass the known bounds of the result.

    Each argument type can be given using its symbolic name string, or using its corresponding
    type descriptor constant object of class :class:`~docplex.cp.catalog.CpoType` listed in module
    :mod:`~docplex.cp.catalog`.

    Allowed argument types are:

     * 'Int' or :const:`~docplex.cp.catalog.Type_Int`
     * 'IntVar' or :const:`~docplex.cp.catalog.Type_IntVar`
     * 'IntExpr' or :const:`~docplex.cp.catalog.Type_IntExpr`
     * 'Float' or :const:`~docplex.cp.catalog.Type_Float`
     * 'FloatExpr' or :const:`~docplex.cp.catalog.Type_FloatExpr`
     * 'IntervalVar' or :const:`~docplex.cp.catalog.Type_IntervalVar`
     * 'SequenceVar' or :const:`~docplex.cp.catalog.Type_SequenceVar`
     * 'IntArray' or :const:`~docplex.cp.catalog.Type_IntArray`
     * 'IntVarArray' or :const:`~docplex.cp.catalog.Type_IntVarArray`
     * 'IntExprArray' or :const:`~docplex.cp.catalog.Type_IntExprArray`
     * 'FloatArray' or :const:`~docplex.cp.catalog.Type_FloatArray`
     * 'FloatExprArray' or :const:`~docplex.cp.catalog.Type_FloatExprArray`
     * 'IntervalVarArray' or :const:`~docplex.cp.catalog.Type_IntervalVarArray`
     * 'SequenceVarArray' or :const:`~docplex.cp.catalog.Type_SequenceVarArray`
    """
    __slots__ = ('name',           # Name of the blackbox function, None for auto-allocation
                 'dimension',      # Number of result values
                 'argtypes',       # List of argument types
                 'atypes_given',   # Indicates that argument types where given at function declaration
                 'impl',           # Implementation of the function
                 'bounds_param',   # Name of the bounds parameter
                 'args_with_vars', # Indicates to include variables in arguments
                 'cachesize',      # Size of the function call cache
                 'globalcache',    # Global cache indicator
                 'operation',      # Corresponding operation descriptor
                 'eval_count',     # Number of evaluation resuests
                 'auto',           # Auto-created blackbox (for smart parsing)
                 'eval_mutex',     # Lock to ensure mutual exclusion of function evaluation
                )

    def __init__(self, impl=None, dimension=1, argtypes=None, name=None, parallel=False, bounds_parameter=None,
                 args_with_vars=False, cachesize=-1, globalcache=False):
        """ **Constructor**

        The list of function argument types is optional.
        If not given, it is automatically determined as the most common types of the expression arguments used
        in its different references in the model.

        Function implementation is optional.
        This allows to parse a CPO file that contains references to blackbox functions(s), which requires
        to register them in the model prior to parse it.
        However, the model will not be able to be solved.

        The name of the function is optional.
        If not given, a name is automatically allocated when solving the model.
        If given, the name of the function must be a symbol (only letters and digits, starting by a letter)
        that is not already used as the name of an existing modeling function.

        Bound parameter is optional.
        If defined, the known bounds of the function result is passed to the function implementation using this named
        parameter.
        If the dimension of the function is 1, the bounds is a simple tuple containing lower and upper bounds.
        Otherwise, the bounds is a list of tuples, one for each returned value.
        If unknown, bounds are set to -inf or inf (float("inf") in Python).

        By default, arguments passed when evaluating the function are only values.
        If the parameter *include_vars* is set to True, when arguments are variables, an object of type
        CpoXxxVarSolution, that include reference to the model variable, is used instead.

        A cache can be used by the solver to avoid calling the blackbox function multiple times with the same values.
        By default (cachesize=-1), the size of the cache is automatically determined by the solver, but it can be
        forced to a given value, or zero for no cache at all.

        By default, this cache is local to each call instance of the blackbox function in the model, in case the
        evaluation of the function depends on the calling context and may return different results with the same
        parameters depending where the call is placed in the model.
        The parameter *globalcache* can be set to *True* if the same cache can be used for all call instances.

        Args:
            impl:             (Optional) Implementation of the function
            dimension:        (Optional) Number of float values that are returned by the function. Default is 1.
            argtypes:         (Optional) List of argument types or type names.
            name:             (Optional) Name of the function, restricted to symbol.
            parallel:         (Optional) Indicates that the blackbox function evaluation is allowed concurrently.
                              Default is False.
            bounds_parameter: (Optional) Name of the parameter in which known return values bounds can be set.
                              Default is None.
            args_with_vars:   (Optional) Enable passing an object of type CpoXxxVarSolution instead of value only.
                              Default is False.
            cachesize:        (Optional) Indicates that the blackbox function evaluation is allowed concurrently.
                              Default value is -1, indicating that the cache is managed by the solver with default settings.
            globalcache:      (Optional) Indicates that the same cache can be used for all blackbox function call instances
                              in the model.
                              Default is False.
        """
        # Check dimension
        if dimension is not None:
            assert is_int(dimension) and dimension >= 1, "Blackbox function dimension should be greater than zero"
        self.dimension = dimension
        self.bounds_param = bounds_parameter
        self.args_with_vars = args_with_vars

        # Check argument types
        if argtypes is None:
            self.atypes_given = False
            self.argtypes = None
        else:
            self.atypes_given = True
            self.argtypes = []
            for t in argtypes:
                if t in _ARG_TYPES_SET:
                    at = t
                else:
                    # Consider t as a name and search in types map
                    at = _ARG_TYPES_DICT.get(str(t).lower())
                    if at is None:
                        raise AssertionError("Argument type '{}' is not allowed as blackbox function argument".format(t))
                self.argtypes.append(at)

        # Check function name
        if name is not None:
            if name in ALL_OPERATIONS_PER_NAME:
                raise AssertionError("Function name {} is already used for a standard modeling operation".format(name))
        self.name = name

        # Store attributes
        self.impl = impl
        self.cachesize = cachesize
        self.globalcache = globalcache

        self.eval_count = 0
        self.auto = (dimension is None)
        self.set_parallel_eval(parallel)

        # Build operation descriptor
        if self.atypes_given:
            self.operation = CpoOperation(name, name, None, -1, (CpoSignature(Type_FloatExprArray, self.argtypes),) )
        else:
            self.operation = CpoOperation(name, name, None, -1, () )


    def set_name(self, nm):
        """ Set the name of the blackbox function

        Args:
            nm: Name of the blackbox function
        """
        self.name = nm


    def get_name(self):
        """ Get the name of the blackbox function

        Returns:
            Name of the blackbox function
        """
        return self.name


    def set_dimension(self, dim):
        """ Set the dimension of this blackbox function, i.e. the number of values that it returns.

        Args:
            dim: Number of result values (size of the result list or array)
        """
        self.dimension = dim


    def get_dimension(self):
        """ Get the dimension of this blackbox function

        Returns:
            Number of result values (size of the result list or array)
        """
        return self.dimension


    def set_implementation(self, impl):
        """ Set the blackbox function implementation

        Args:
            impl:  Blackbox function implementation
        """
        self.impl = impl


    def get_implementation(self):
        """ Get the blackbox function implementation

        Returns:
            Blackbox function implementation
        """
        return self.impl


    def has_implementation(self):
        """ Get if the blackbox function has an implementation

        Returns:
            True if the blackbox function has an implementation
        """
        return self.impl is not None


    def get_arg_types(self):
        """ Get the list of argument types

        Returns:
            List of argument types, objects of class :class:`~docplex.cp.catalog.CpoType`
        """
        return self.argtypes


    def set_parallel_eval(self, par):
        """ Set parallel evaluation enablement indicator.

        Args:
            par:  Parallel evaluation indicator.
        """
        self.eval_mutex = None if par else threading.Lock()


    def is_parallel_eval(self):
        """ Check if parallel evaluation is allowed.

        Returns:
            True if parallel evaluation is allowed, false oherwise
        """
        return self.eval_mutex is None


    def set_cache_size(self, size):
        """ Set the size of evaluation cache.

        Args:
            size: Cache size, -1 for default, 0 for none.
        """
        self.cachesize = size


    def get_cache_size(self):
        """ Get the size of the evaluation cache

        Returns:
            Evaluation cache size, -1 for default, 0 for none.
        """
        return self.cachesize


    def set_global_cache(self, glob):
        """ Set the global cache indicator.

        When set, there is a single evaluation cache for all the blackbox function call instances.

        Args:
            glob: Global cache indicator
        """
        self.globalcache = glob


    def is_global_cache(self):
        """ Check if a global cache has been set.

        Returns:
            True if there is a global cache for this function.
        """
        return self.globalcache


    def get_eval_count(self):
        """ Get the number of times this blackbox function has been evaluated

        Returns:
            number of times this blackbox function has been evaluated
        """
        return self.eval_count


    def reset_eval_count(self):
        """ Reset the number of times this blackbox function has been evaluated
        """
        self.eval_count = 0


    def build_model_call(self, *args):
        """ Build a model expression representing a call to this blackbox function.

        Args:
            *args:  List of expressions that are arguments of the function
        Returns:
            List model expressions representing access to the unitary result values,
            or single result if dimension is 1.
        """
        # Build argument expressions
        argexprs = [build_cpo_expr(a) for a in args]

        # Update/check argument types
        self._update_check_arg_types(argexprs)

        # Build function call expression
        expr = CpoBlackboxFunctionCall(self, argexprs)

        # Build list of result access expressions
        res = tuple(CpoFunctionCall(Oper_eval, Type_FloatExpr, (expr, build_cpo_expr(i))) for i in range(self.dimension))
        return res if self.dimension > 1 else res[0]


    def __call__(self, *args):
        """ Build a model expression representing a call to this blackbox function.

        Args:
            *args:  List of expressions that are arguments of the function
        Returns:
            List model expressions representing access to the unitary result values,
            or single result if dimension is 1.
        """
        return self.build_model_call(*args)


    def __str__(self):
        """ Build a string representing this blackbox function.

        Returns:
            String representing this blackbox function.
        """
        name = "Anonymous" if self.name is None else self.name
        argtypes = "..." if self.argtypes is None else ', '.join(t.get_name() for t in self.argtypes)
        return "{}({}): {}".format(self.name, argtypes, self.dimension)


    def _eval_function(self, rbnds, *args):
        """ Evaluate the function from the list of parameter values

        Args:
            rbnds: Known result bounds, None in unknown
            *args: List of parameter values
        Returns:
            List of result float values
        """
        #print("Evaluate blackbox function {}{}".format(self.name, args))
        # Increment number of evaluation requests
        self.eval_count += 1

        # Get and check arguments
        assert self.argtypes is not None, "Blackbox function '{}' argument types are unknown".format(self.name)
        if len(args) != len(self.argtypes):
            raise CpoException("Evaluation of blackbox function '{}' with wrong number of parameters {} when {} are expected.".format(self.name, len(args), len(self.argtypes)))

        # Build associated arguments
        kwargs = {}
        if (rbnds is not None) and (self.bounds_param is not None):
            if self.dimension == 1:
                rbnds = rbnds[0]
            kwargs[self.bounds_param] = rbnds

        # Evaluate function
        if self.impl is None:
            raise CpoException("Blackbox function '{}' implementation is not provided".format(self.name))

        res = self.impl(*args, **kwargs)

        # Check single result (separatly to process the case of zero)
        if is_number(res):
            assert self.dimension == 1,  "Evaluation of blackbox function '{}' returned 1 result values instead of {} that have been declared.".format(self.name, self.dimension)
            res = (res,)
        elif is_bool(res):
            assert self.dimension == 1,  "Evaluation of blackbox function '{}' returned 1 result values instead of {} that have been declared.".format(self.name, self.dimension)
            res = (int(res),)
        # Check result (None is allowed)
        elif res:
            assert is_array(res), "Evaluation of blackbox function '{}' should return a tuple or a list, not {}.".format(self.name, type(res))
            assert len(res) == self.dimension, "Evaluation of blackbox function '{}' returned {} result values instead of {} that have been declared.".format(self.name, len(res), self.dimension)
            assert all(is_number(v) for v in res), "Evaluation of blackbox function '{}' result should contain only numbers.".format(self.name)

        #print("{}{} = {}".format(self.name, args, res))
        return res


    def eval(self, *args):
        """ Evaluate the function from the list of parameter values

        This function evaluates the blackbox function without providing bounds.

        Args:
            *args: List of parameter values
        Returns:
            List of result float values
        """
        return self._eval_function(None, *args)


    def _update_check_arg_types(self, argexprs):
        """ Update function argument types from a list of argument expressions

        Args:
            argexprs:  List of expressions that are arguments of the function
        """
        # Check if argument types already known
        if self.argtypes is None:
            # Retrieve argument types from expressions
            self.argtypes = [_get_argument_type(a) for a in argexprs]
            # Set new signature in operation
            self.operation.signatures = (CpoSignature(Type_FloatExprArray, self.argtypes),)
        else:
            # Build and check list or arguments
            assert len(argexprs) == len(self.argtypes), "This blackbox function should be called with {} arguments".format(len(self.argtypes))
            if self.atypes_given:
                for i, a, t in zip(count(), argexprs, self.argtypes):
                    assert _get_argument_type(a).is_kind_of(t), "The argument {} of blackbox function '{}' should be a {}".format(i + 1, self.name, t.get_public_name())
            else:
                tchanged = False
                for i, a, t in zip(count(), argexprs, self.argtypes):
                    # Determine most common type
                    ct = _get_argument_type(a).get_common_type(t)
                    assert ct is not None, "Argument type {} is not compatible with already used type {}".format(a.type, t)
                    assert ct in _ARG_TYPES_SET, "Common expression type {} is not allowed as blackbox function argument".format(ct)
                    if ct != t:
                        self.argtypes[i] = ct
                        tchanged = True
                # Set new signature in operation if type changed
                if tchanged:
                    self.operation.signatures = (CpoSignature(Type_FloatExprArray, self.argtypes),)


    def _update_dimension(self, anx):
        """ Update blackbox dimension with an evaluation index

        Args:
            anx:  Evaluation index
        """
        if (self.dimension is None) or (self.dimension <= anx):
            self.dimension = anx + 1


class CpoBlackboxFunctionCall(CpoFunctionCall):
    """ This class represent all model expression nodes that call a blackbox function.
    """
    __slots__ = ('blackbox',  # Blackbox function descriptor
                )

    def __init__(self, bbf, oprnds):
        """ **Constructor**
        Args:
            bbf:    Blackbox function descriptor
            oprnds: List of operand expressions.
        """
        assert isinstance(bbf, CpoBlackboxFunction), "Argument 'bbf' should be a CpoBlackboxFunction"
        super(CpoBlackboxFunctionCall, self).__init__(bbf.operation, Type_Blackbox, oprnds)
        self.blackbox = bbf

    def _equals(self, other):
        """ Checks the equality of this expression with another object.

        This particular method just checks local attributes, but does not check recursively children if any.
        Recursion is implemented by method equals() that uses a self-managed stack to avoid too many
        recursive calls that may lead to an exception 'RuntimeError: maximum recursion depth exceeded'.

        Args:
            other: Other object to compare with.
        Returns:
            True if 'other' is semantically identical to this object, False otherwise.
        """
        return super(CpoBlackboxFunctionCall, self)._equals(other) and (self.blackbox == other.blackbox)

#-----------------------------------------------------------------------------
# Private functions
#-----------------------------------------------------------------------------

# Dictionary of type mapping to accepted type
_ARG_TYPE_MAPPING = {t: t for t in ARGUMENT_TYPES}
_ARG_TYPE_MAPPING.update(
{
    Type_Bool:           Type_IntExpr,
    Type_BoolExpr:       Type_IntExpr,
    Type_BoolArray:      Type_IntArray,
    Type_BoolExprArray:  Type_IntExprArray,
})


def _get_argument_type(a):
    """ Get the blackbox argument type corresponding to a given argument type

    Args:
        a: Argument value
    Returns:
        Authorized blackbox argument type
    Raises:
        CpoException if given argument type is not supported.
    """
    at = a.type
    nt = _ARG_TYPE_MAPPING.get(at)
    if nt is None:
        raise CpoException("Expression type {} is not allowed as blackbox function argument".format(at))
    return nt


