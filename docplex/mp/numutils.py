# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

# gendoc: ignore

import math
from docplex.mp.sttck import StaticTypeChecker as sttck


def round_nearest_halfway_from_zero(x, infinity=1e+20):
    """ Rounds the argument to the nearest integer.

    For values like 1.5 the intetger with greater absolute value is returned.
    This treats positive and negative values in a symmetric manner.
    This is called "round half away from zero"


    Args:
        x: the value to round
        infinity: the model's infinity value. All values above infinity are set to +INF

    Returns:
        an integer value

    Example:
        round_nearest(0) = 0
        round_nearest(1.1) = 1
        round_nearest(1.5) = 2
        round_nearest(1.49) = 1
    """
    if x == 0:
        return 0
    elif x >= infinity:
        return infinity
    elif x <= -infinity:
        return -infinity
    else:
        raw_nearest = my_round_even(x)  # math.floor(x + 0.5)
        return int(raw_nearest)


def my_round_even(number):
    """
    Simplified version from future
    """
    from decimal import Decimal, ROUND_HALF_EVEN

    d = Decimal.from_float(number).quantize(1, rounding=ROUND_HALF_EVEN)
    return int(d)


def round_nearest_towards_infinity(x, infinity=1e+20):
    """ Rounds the argument to the nearest integer.

    For ties like 1.5 the ceiling integer is returned.
    This is called "round towards infinity"

    Args:
        x: the value to round
        infinity: the model's infinity value. All values above infinity are set to +INF

    Returns:
        an integer value

    Example:
        round_nearest(0) = 0
        round_nearest(1.1) = 1
        round_nearest(1.5) = 2
        round_nearest(1.49) = 1
    """
    if x == 0:
        return 0
    elif x >= infinity:
        return infinity
    elif x <= -infinity:
        return -infinity
    else:
        raw_nearest = math.floor(x + 0.5)
        return int(raw_nearest)

def round_nearest_towards_infinity1(x):
    return round_nearest_towards_infinity(x)

class _NumPrinter(object):
    """
    INTERNAL.
    """

    def __init__(self, nb_digits_for_floats, num_infinity=1e+20, pinf="+inf", ninf="-inf"):
        assert (nb_digits_for_floats >= 0)
        assert (isinstance(pinf, str))
        assert (isinstance(ninf, str))
        self.true_infinity = num_infinity
        self.precision = nb_digits_for_floats
        self.__positive_infinity = pinf
        self.__negative_infinity = ninf
        # coin the format from the nb of digits
        # 2 -> %.2f
        self._double_format = "%." + ('%df' % nb_digits_for_floats)

    def to_string(self, num):
        if num >= self.true_infinity:
            return self.__positive_infinity
        elif num <= - self.true_infinity:
            return self.__negative_infinity
        else:
            try:
                if num.is_integer():  # the is_integer() function is faster than testing: num == int(num)
                    return '%d' % num
                else:
                    return self._double_format % num
            except AttributeError:
                return '%d' % num

def compute_tolerance(baseline: float, abstol: float, reltol: float) -> float:
    """ Computes effective tolerance from a baseline value and relative and absolute tolerances.

    :param baseline: the input value
    :param abstol: absolute tolerance
    :param reltol: relative tolerance
    :return: tolerance to use for th einput value


    Example:
        >> compute_tolerance(1000, 3, 0.01)
        >> 10
        >> compute_tolerance(1000, 1, 0.002)
        >> 2
    """
    assert abstol >= 0
    assert reltol >= 0
    assert reltol < 1
    return max(abstol, reltol * abs(baseline))


def resolve_abs_rel_tolerances(logger, abstols, reltols, size, accept_none=True,
                               default_abstol=1e-6,
                               default_reltol=1e-4,
                               caller=None):
    """ Resolve candidate tolerances.

    Takes as asrguments either a list of numbers, a number or None. When a list is passed, it is
    assumed to be of length size.

    :param logger:
    :param abstols:
    :param reltols:
    :param size: the desired length of the two returtned lists.
    :param accept_none: a flag indicating whether the resolve procedure accepts None. If True,
        returns a list of length 'size' using the default values.
    :param default_abstol: the default absolute tolerance (1e-6)
    :param default_reltol: the default relative tolerance (1e-4)
    :param caller: A string indicating the name of the caller method for error messages, or None.


    :return: a tuple of two lists of length `size` with valid tolerances.
    """
    assert default_abstol >= 0, "Absolute tolerance must be positive"
    assert default_reltol >= 0, "Relative tolerance must be positive"
    assert default_reltol < 1, "Relative tolerance cannot be greater than 100%"
    assert size >= 1

    if abstols is not None:
        abstols_ = sttck.typecheck_optional_num_seq(logger, abstols, expected_size=size, accept_none=accept_none, caller=caller)
    else:
        abstols_ = [default_abstol] * size
    if reltols is not None:
        reltols_ = sttck.typecheck_optional_num_seq(logger, reltols, expected_size=size, accept_none=accept_none, caller=caller)
    else:
        reltols_ = [default_reltol] * size

    return abstols_, reltols_
