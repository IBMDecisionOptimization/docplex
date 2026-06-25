# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------
from enum import Enum

from docplex.mp.utils import is_number, is_string


class Priority(Enum):
    """
    This enumerated class defines the priorities: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH, MANDATORY.
    """

    def __new__(cls, value, print_name):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = value
        obj._print_name = print_name
        return obj

    VERY_LOW = 100, 'Very Low'
    LOW = 200, 'Low'
    MEDIUM = 300, 'Medium'
    HIGH = 400, 'High'
    VERY_HIGH = 500, 'Very High'
    MANDATORY = 999999999, 'Mandatory'

    def __repr__(self):
        return 'Priority<{}>'.format(self.name)

    @property
    def cplex_preference(self):
        return self._get_geometric_preference_factor(base=10.0)

    def _get_geometric_preference_factor(self, base):
        # INTERNAL: returns a CPLEX preference factor as a power of "base"
        # MEDIUM priority is the balance point with a preference of 1.
        assert is_number(base)

        if self.is_mandatory():
            return 1e+20
        else:
            # noinspection PyTypeChecker
            medium_index = Priority.MEDIUM.value / 100
            # pylint complains about no value member but is wrong!
            diff = self.value / 100 - medium_index
            factor = 1.0
            pdiff = diff if diff >= 0 else -diff
            for _ in range(0, int(pdiff)):
                factor *= base
            return factor if diff >= 0 else 1.0 / factor

    def less_than(self, other):
        assert isinstance(other, Priority)
        return self.value < other.value

    def __lt__(self, other):
        return self.less_than(other)

    def __gt__(self, other):
        return other.less_than(self)

    def is_mandatory(self):
        return self == Priority.MANDATORY

    @classmethod
    def parse(cls, arg, logger, accept_none=True, do_raise=True):
        ''' Converts its argument to a ``Priority`` object.

        Returns `default_priority` if the text is not a string, empty, or does not match.

        Args;
            arg: The argument to convert.

            logger: An error logger

            accept_none: True if None is a possible value. Typically,
                Constraint.set_priority accepts None as a way to
                remove the constraint's own priority.

            do_raise: A Boolean flag indicating if an exception is to be raised if the value
                is not recognized.

        Returns:
            A Priority enumerated object.
        '''
        if isinstance(arg, cls):
            return arg
        elif is_string(arg):
            key = arg.lower()
            # noinspection PyTypeChecker
            for p in cls:
                if key == p.name.lower() or key == str(p.value):
                    return p
            if do_raise:
                logger.fatal('String does not match priority type: {}', arg)
            else:
                logger.error('String does not match priority type: {}', arg)
                return None
            return None
        elif accept_none and arg is None:
            return None
        else:
            logger.fatal('Cannot convert to a priority: {0!s}'.format(arg))


class UserPriority(object):

    def __init__(self, pref, name=None):
        assert pref >= 0
        self._preference = pref
        self._name = name

    @property
    def cplex_preference(self):
        return self._preference

    # noinspection PyMethodMayBeStatic
    def is_mandatory(self):
        return False

    @property
    def name(self):
        return self._name or '_user_'

    @property
    def value(self):
        return self._preference

    def __str__(self):
        name = self._name
        sname = '%s: ' % name if name else ''
        return 'UserPriority({0}{1})'.format(sname, self._preference)
