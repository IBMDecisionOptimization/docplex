# --------------------------------------------------------------------------
# File: _aux_functions.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Internal auxiliary functions."""
try:
    import collections.abc as collections_abc  # For Python >= 3.3
except ImportError:
    import collections as collections_abc
import functools
import inspect
import itertools
import os
import warnings

from ..exceptions import CplexError, WrongNumberOfArgumentsError

CPLEX_PY_DISABLE_NAME_CONV = os.getenv("CPLEX_PY_DISABLE_NAME_CONV")


class deprecated():
    """A decorator that marks methods/functions as deprecated."""

    def __init__(self, version):
        self.version = version

    def __call__(self, cls_or_func):
        if (inspect.isfunction(cls_or_func) or
                inspect.ismethod(cls_or_func)):
            fmt = "{0} function or method"
        # NOTE: Doesn't work for classes .. haven't figured that out yet.
        #       Specifically, when a decorated class is used as a base
        #       class.
        # elif inspect.isclass(cls_or_func):
        #     fmt = "{0} class"
        else:
            raise TypeError(type(cls_or_func))
        msg = _getdeprecatedmsg(fmt.format(cls_or_func.__name__),
                                self.version)

        @functools.wraps(cls_or_func)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return cls_or_func(*args, **kwargs)
        return wrapped


def deprecated_class(name, version, stacklevel=3):
    """Emits a warning for a deprecated class.

    This should be called in __init__.

    name - the name of the class (e.g., PresolveCallback).

    version - the version at which the class was deprecated (e.g.,
              "V12.7.1").

    stacklevel - indicates how many levels up the stack is the caller.
    """
    msg = _getdeprecatedmsg("{0} class".format(name), version)
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


def _getdeprecatedmsg(item, version):
    return "the {0} is deprecated since {1}".format(item, version)


def max_arg_length(arg_list):
    """Returns the max length of the arguments in arg_list."""
    return max([len(x) for x in arg_list])


if __debug__:

    # With non-optimzied bytecode, validate_arg_lengths actually does
    # something.
    def validate_arg_lengths(arg_list, allow_empty=True, extra_msg=""):
        """Checks for equivalent argument lengths.

        If allow_empty is True (the default), then empty arguments are not
        checked against the max length of non-empty arguments. Some functions
        allow NULL arguments in the Callable Library, for example.
        """
        arg_lengths = [len(x) for x in arg_list]
        if allow_empty:
            arg_lengths = [x for x in arg_lengths if x > 0]
        if not arg_lengths:
            return
        max_length = max(arg_lengths)
        for arg_length in arg_lengths:
            if arg_length != max_length:
                raise CplexError("inconsistent argument lengths" + extra_msg)

else:

    def validate_arg_lengths(
            arg_list,
            allow_empty=True,
            extra_msg=""
    ):  # pylint: disable=unused-argument
        """A no-op.

        A no-op if using python -O or the PYTHONOPTIMIZE environment
        variable is defined as a non-empty string.
        """
        pass


def make_ranges(indices):
    """non-public"""
    i = 0
    j = 0
    while i < len(indices):
        while j < len(indices) - 1 and indices[j + 1] == indices[j] + 1:
            j += 1
        yield (indices[i], indices[j])
        i = j + 1
        j = i


def identity(x):
    """Simple identity function."""
    return x


def apply_freeform_two_args(fn, conv, args, unpack_single=True):
    """non-public"""
    if conv is None:
        conv = identity
    nargs = len(args)
    if nargs == 2:
        conarg0, conarg1 = (conv(args[0]), conv(args[1]))
        if (isinstance(conarg0, int) and isinstance(conarg1, int)):
            return fn(conarg0, conarg1)
        raise TypeError("expecting names or indices")
    elif nargs == 1:
        if isinstance(args[0], (list, tuple)):
            return list(itertools.chain.from_iterable(
                fn(i, j) for i, j in make_ranges(conv(args[0]))))
        conarg0 = conv(args[0])
        if isinstance(conarg0, int):
            result = fn(conarg0, conarg0)
            if unpack_single:
                return result[0]
            return result
        raise TypeError("expecting name or index")
    elif nargs == 0:
        return fn(0)
    raise WrongNumberOfArgumentsError()


def apply_freeform_one_arg(fn, conv, maxval, args):
    """non-public"""
    if conv is None:
        conv = identity
    nargs = len(args)
    if nargs == 2:
        conarg0, conarg1 = (conv(args[0]), conv(args[1]))
        if (isinstance(conarg0, int) and isinstance(conarg1, int)):
            return [fn(x) for x in range(conarg0, conarg1 + 1)]
        raise TypeError("expecting names or indices")
    elif nargs == 1:
        if isinstance(args[0], (list, tuple)):
            return [fn(x) for x in conv(args[0])]
        conarg0 = conv(args[0])
        if isinstance(conarg0, int):
            return fn(conarg0)
        raise TypeError("expecting name or index")
    elif nargs == 0:
        return apply_freeform_one_arg(fn, conv, 0,
                                      (list(range(maxval)),))
    raise WrongNumberOfArgumentsError()


def apply_pairs(fn, conv, *args):
    """non-public"""
    nargs = len(args)
    if nargs == 2:
        fn([conv(args[0])], [args[1]])
        return
    if nargs == 1:
        pair = unzip(args[0])
        # NB: If pair is empty, then we do nothing.
        if pair:
            fn(conv(pair[0]), list(pair[1]))
        return
    raise WrongNumberOfArgumentsError(nargs)


def delete_set_by_range(fn, conv, max_num, *args):
    """non-public"""
    nargs = len(args)
    if nargs == 0:
        # Delete All:
        if max_num > 0:
            fn(0, max_num - 1)
    elif nargs == 1:
        # Delete all items from a possibly unordered list of mixed types:
        args = listify(conv(args[0]))
        ranges = make_ranges(list(sorted(args)))
        for i, j in reversed(list(ranges)):
            fn(i, j)
    elif nargs == 2:
        # Delete range from arg[0] to arg[1]:
        fn(conv(args[0]), conv(args[1]))
    else:
        raise WrongNumberOfArgumentsError()


class _group():
    """Object to contain constraint groups"""

    def __init__(self, gp):
        """Constructor for the _group object

        gp is a list of tuples of length two (the first entry of which
        is the preference for the group (a float), the second of which
        is a tuple of pairs (type, id), where type is an attribute of
        conflict.constraint_type and id is either an index or a valid
        name for the type).

        Example input: [(1.0, ((2, 0),)), (1.0, ((3, 0), (3, 1)))]
        """
        self._gp = gp

    def __str__(self):
        return str(self._gp)


def make_group(conv, max_num, c_type, *args):
    """Returns a _group object

    input:
    conv    - a function that will convert names to indices
    max_num - number of existing constraints of a given type
    c_type  - constraint type
    args    - arbitrarily many arguments (see description below)

    If args is empty, every constraint/bound is assigned weight 1.0.

    If args is of length one or more, every constraint/bound is assigned
    a weight equal to the float passed in as the first item.

    If args contains additional items, they determine a subset of
    constraints/bounds to be included.  If one index or name is
    specified, it is the only one that will be included.  If two indices
    or names are specified, all constraints between the first and the
    second, inclusive, will be included.  If a sequence of names or
    indices is passed in, all of their constraints/bounds will be
    included.

    See example usage in _subinterfaces.ConflictInterface.
    """
    nargs = len(args)
    if nargs <= 1:
        cons = list(range(max_num))
    if nargs == 0:
        weight = 1.0
    else:
        weight = args[0]
    if nargs == 2:
        weight = args[0]
        cons = listify(conv(args[1]))
    elif nargs == 3:
        cons = list(range(conv(args[1]), conv(args[2]) + 1))
    return _group([(weight, ((c_type, i),)) for i in cons])


def init_list_args(*args):
    """Initialize default arguments with empty lists if necessary."""
    return tuple([] if a is None else a for a in args)


def listify(x):
    """Returns [x] if x isn't already a list.

    This is used to wrap arguments for functions that require lists.
    """
    # Assumes name to index conversions already done.
    assert not isinstance(x, str)
    try:
        iter(x)
        return x
    except TypeError:
        return [x]


def _cachelookup(item, getindexfunc, cache):
    try:
        idx = cache[item]
    except KeyError:
        idx = getindexfunc(item)
        cache[item] = idx
    return idx


# If the CPLEX_PY_DISABLE_NAME_CONV environment variable is defined,
# we will skip name-to-index conversion (i.e., these functions become
# no-ops), which can improve performance.
if CPLEX_PY_DISABLE_NAME_CONV:

    def convert_sequence(
            seq,
            getindexfunc,
            cache=None
    ):  # pylint: disable=unused-argument
        """Returns seq immediately.

        See comments about CPLEX_PY_DISABLE_NAME_CONV.
        """
        return seq

    def convert(
            name,
            getindexfunc,
            cache=None
    ):  # pylint: disable=unused-argument
        """Returns name immediately.

        See comments about CPLEX_PY_DISABLE_NAME_CONV.
        """
        return name

else:

    # By default (i.e., if the CPLEX_PY_DISABLE_NAME_CONV environment
    # variable is not defined), these functions perform name-to-index
    # conversion, which can hurt performance.

    def convert_sequence(seq, getindexfunc, cache=None):
        """Converts a sequence of names to indices as necessary.

        If you are calling `convert` (see below) in a tight loop, but you
        know that you are always working with a sequence, then it can be
        more efficient to call this method directly (there is no overhead
        checking if it is a sequence).
        """
        if cache is None:
            cache = {}
        results = []
        for item in seq:
            if isinstance(item, str):
                idx = _cachelookup(item, getindexfunc, cache)
                results.append(idx)
            else:
                results.append(item)
        return results

    def convert(name, getindexfunc, cache=None):
        """Converts from names to indices as necessary.

        If name is a string, an index is returned.

        If name is a sequence, a sequence of indices is returned.

        If name is neither (i.e., it's an integer), then that is returned
        as is.

        getindexfunc is a function that takes a name and returns an index.

        The optional cache argument allows for further localized
        caching (e.g., within a loop).
        """
        # In some cases, it can be benficial to cache lookups.
        if cache is None:
            cache = {}
        if isinstance(name, str):
            return _cachelookup(name, getindexfunc, cache)
        if isinstance(name, collections_abc.Sequence):
            # It's tempting to use a recursive solution here, but that kills
            # performance for the case where all indices are passed in (i.e.,
            # no names). This is due to the fact that we end up doing the
            # extra check for sequence types over and over (above).
            return convert_sequence(name, getindexfunc, cache)
        return name


def unzip(iterable=None):
    """Inverse of the zip function.

    Example usage:

    >>> z = list(zip([1, 2, 3], [4, 5, 6]))
    >>> unzip(z)
    [(1, 2, 3), (4, 5, 6)]
    """
    if iterable is None:
        iterable = []
    return list(zip(*iterable))
