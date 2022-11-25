# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

from abc import abstractmethod, ABCMeta

from docplex.mp.operand import Operand
from docplex.mp.sttck import StaticTypeChecker


# noinspection PyUnusedLocal,PyPropertyAccess

class _AbstractModelObject(metaclass=ABCMeta):
    """
    Abstract API for all classes which have a "model" property.
    """

    @property
    @abstractmethod
    def model(self):  # pragma: no cover
        raise NotImplementedError

    def is_in_model(self, mdl):
        return self.model is mdl

    def get_linear_factory(self):
        return self.model._lfactory

    @property
    def lfactory(self):
        return self.model._lfactory

    @property
    def qfactory(self):
        return self.model._qfactory

    def _check_model_has_solution(self):
        self.model._check_has_solution()

    @property
    def error_handler(self):
        return self.logger

    @property
    def logger(self):
        return self.model.error_handler

    def fatal(self, msg, *args):
        self.logger.fatal(msg, args)

    def error(self, msg, *args):
        self.logger.error(msg, args)

    def warning(self, msg, *args):
        self.logger.warning(msg, args)


class _AbstractValuable(_AbstractModelObject):
    # abstract API for all objects which can be evaluated from a solution.

    __slots__ = ()

    def _round_if_discrete(self, raw_value):
        return self.model._round_element_value_if_necessary(self, raw_value)

    @abstractmethod
    def _raw_solution_value(self, s=None):
        # INTERNAL: compute raw solution value, no rounding, no checking
        raise NotImplementedError  # pragma: no cover

    @property
    def solution_value(self):
        self.model._check_has_solution()
        raw = self._raw_solution_value()
        return self._round_if_discrete(raw)

    @property
    def raw_solution_value(self):
        self.model._check_has_solution()
        return self._raw_solution_value()

    @property
    def sv(self):
        return self.solution_value

    @property
    def rsv(self):
        return self.raw_solution_value


class _SubscriptionMixin(object):
    __slots__ = ()

    # INTERNAL:
    # This class is absolutely not meant to be directly instantiated
    # but used as a mixin

    @classmethod
    def _new_empty_subscribers(cls):
        return []

    def notify_used(self, user):
        # INTERNAL
        self._subscribers.append(user)

    notify_subscribed = notify_used

    def notify_unsubscribed(self, subscriber):
        # 1 find index
        for s, sc in enumerate(self._subscribers):
            if sc is subscriber:
                del self._subscribers[s]
                break

    def clear_subscribers(self):
        self._subscribers = []

    def is_in_use(self):
        return bool(self._subscribers)

    @property
    def nb_subscribers(self):
        return len(self._subscribers)

    def is_shared(self):
        return self.nb_subscribers >= 2

    def is_used_by(self, obj):
        # lists are not optimal here, but we favor insertion: append is faster than set.add
        return any(obj is sc for sc in self.iter_subscribers())

    def notify_modified(self, event):
        for s in self._subscribers:
            s.notify_expr_modified(self, event)

    def iter_subscribers(self):
        return iter(self._subscribers)

    def notify_replaced(self, new_expr):
        for s in self._subscribers:
            s.notify_expr_replaced(self, new_expr)

    def grab_subscribers(self, other):
        # grab subscribers from another expression
        # typically when an expression is replaced by another.
        for s in other.iter_subscribers():
            self._subscribers.append(s)
        # delete all subscriptions on old
        other.clear_subscribers()


class _AbstractBendersAnnotated(_AbstractModelObject):
    # a maxin class to group all benders-related code.
    __slots__ = ()

    def set_benders_annotation(self, group):
        self.model.set_benders_annotation(self, group)

    def get_benders_annotation(self):
        return self.model.get_benders_annotation(self)


class _AbstractNamable(metaclass=ABCMeta):

    # abstract name API across all modeling objects.

    @property
    @abstractmethod
    def name(self):  # pragma: no cover
        raise NotImplemented

    @abstractmethod
    def _set_name(self, new_name):  # pragma: no cover
        raise NotImplementedError

    def check_name(self, new_name):
        pass

    def get_name(self):
        # deprecate
        return self.name

    def set_name(self, new_name):
        self.check_name(new_name)
        self._set_name(new_name)

    @property
    def safe_name(self):
        return self.name or ''

    def check_lp_name(self, qualifier, new_name, accept_empty, accept_none):
        return StaticTypeChecker.check_lp_name(logger=self, qualifier=qualifier, obj=self, new_name=new_name,
                                               accept_empty=accept_empty, accept_none=accept_none)

    def has_name(self):
        return self.name is not None

    def has_user_name(self):
        return self.has_name()


class ModelObject(_AbstractModelObject):
    # base for all model objects
    __array_priority__ = 100

    __slots__ = ('_model',)

    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def repr_str(self):
        # INTERNAL
        try:
            return self.to_string(use_space=False)
        except (TypeError, AttributeError):
            return str(self)

    def zero_expr(self):
        # INTERNAL
        return self._model._lfactory.new_zero_expr()

    def _unsupported_binary_operation(self, lhs, op, rhs):
        self.fatal("Unsupported operation: {0!s} {1:s} {2!s}", lhs, op, rhs)

    def __str__(self):
        return self.to_string(use_space=self._model.str_use_space)

    # def to_string(self):
    #     raise NotImplementedError


class ModelingObjectBase(ModelObject, _AbstractNamable):
    """ModelingObjectBase()

    Parent class for all modeling objects (variables and constraints).

    This class is not intended to be instantiated directly.
    """

    __array_priority__ = 100

    __slots__ = ('_name',)

    # noinspection PyMissingConstructor
    def __init__(self, model, name=None):
        self._name = name
        self._model = model

    @property
    def name(self):
        """ This property is used to get or set the name of the modeling object.

        """
        return self._name

    @name.setter
    def name(self, new_name):
        self.set_name(new_name)

    def _set_name(self, name):
        self._name = name

    def has_name(self):
        """ Checks whether the object has a name.

        Returns:
            True if the object has a name.

        """
        return super().has_name()

    def has_user_name(self):
        """ Checks whether the object has a valid name given by the user.

        Returns:
            True if the object has a valid name given by the user.

        """
        return self.has_name()

    @property
    def model(self):
        """
        This property returns the :class:`docplex.mp.model.Model` to which the object belongs.
        """
        return super().model


class IndexableObject(ModelingObjectBase):
    __slots__ = ("_index",)

    @staticmethod
    def is_valid_index(idx):
        # INTERNAL: This is where the valid index check is performed
        return idx >= 0

    _invalid_index = -2

    # noinspection PyMissingConstructor
    def __init__(self, model, name=None, index=_invalid_index):
        #  ModelingObjectBase.__init__(self, model, name)
        self._model = model
        self._name = name
        self._index = index

    def is_generated(self):
        """ Checks whether this object has been generated by another modeling object.

        If so, the origin object is stored in the ``_origin`` attribute.

        Returns:
            True if the objects has been generated.
        """
        return self.origin is not None

    @property
    def origin(self):
        return self.model.get_obj_origin(self)

    @origin.setter
    def origin(self, origin):
        self.model.set_obj_origin(self, origin)

    def __hash__(self):
        return id(self)

    @property
    def model(self):
        return self._model

    @property
    def index(self):
        return self._index

    @property
    def index1(self):
        raw = self._index
        return raw if raw == self._invalid_index else raw + 1

    def _set_index(self, idx):
        self._index = idx

    def has_valid_index(self):
        return self._index >= 0

    def _set_invalid_index(self):
        self._index = self._invalid_index

    @property
    def safe_index(self):
        if not self.has_valid_index():
            self.fatal("Modeling object {0!s} has invalid index: {1:d}", self, self._index)  # pragma: no cover
        return self._index

    @property
    def container(self):
        return self.model.get_var_container(self)

    @container.setter
    def container(self, ctn):
        self._model.set_var_container(self, ctn)

    @property
    @abstractmethod
    def cplex_scope(self) -> int:
        return -1  # crash

    def get_scope(self):
        try:
            cpx_scope = self.cplex_scope
            return self.model._get_obj_scope(cpx_scope, error='ignore')
        except AttributeError:
            return None

    @property
    def scope(self):
        return self.get_scope()


class Expr(ModelObject, Operand, _AbstractValuable):
    """Expr()

    Parent class for all expression classes.
    """
    __slots__ = ()

    @property
    def name(self):
        return None

    def clone(self):  # pragma: no cover
        raise NotImplementedError  # pragma: no cover

    def iter_variables(self):
        # internal
        raise NotImplementedError  # pragma: no cover

    def copy(self, target_model, var_mapping):
        # internal
        raise NotImplementedError  # pragma: no cover

    def number_of_variables(self):
        """
        Returns:
            integer: The number of variables in the expression.
        """
        return sum(1 for _ in self.iter_variables())  # pragma: no cover

    def contains_var(self, dvar):
        """ Checks whether a variable is present in the expression.

        :param: dvar (:class:`docplex.mp.dvar.Var`): A decision variable.

        Returns:
            Boolean: True if the variable is present in the expression, else False.
        """
        return any(dvar is v for v in self.iter_variables())

    def to_string(self, nb_digits=None, use_space=False):
        from io import StringIO
        oss = StringIO()
        if nb_digits is None:
            nb_digits = self.model.float_precision
        self.to_stringio(oss, nb_digits=nb_digits, use_space=use_space)
        return oss.getvalue()

    def to_readable_string(self):
        return self.to_string(use_space=True)[:self.model.readable_str_len]

    def to_stringio(self, oss, nb_digits, use_space, var_namer=lambda v: v.name):
        raise NotImplementedError  # pragma: no cover

    def _num_to_stringio(self, oss, num, ndigits=None, print_sign=False, force_plus=False, use_space=False):
        k = num
        if print_sign:
            if k < 0:
                sign = u'-'
                k = -k
            elif k > 0 and force_plus:
                # force a plus
                sign = u'+'
            else:
                sign = None
            if use_space:
                oss.write(u' ')
            if sign:
                oss.write(sign)
            if use_space:
                oss.write(u' ')
        # INTERNAL
        ndigits = ndigits or self.model.float_precision
        try:
            if k == int(k):
                oss.write(u'%d' % k)
            else:
                # use second arg as nb digits:
                oss.write(u"{0:.{1}f}".format(k, ndigits))
        except ValueError:
            # possibly a nan
            oss.write('?')

    # def __pos__(self):
    #     # + e is identical to e
    #     return self

    def is_discrete(self):
        raise NotImplementedError  # pragma: no cover

    def is_quad_expr(self):
        """ Returns True if the expression is quadratic

        """
        return False

    def get_linear_part(self):
        return self  # should be not implemented...

    def is_zero(self):
        return False

    constant = property(Operand.get_constant)

    @property
    def float_precision(self):
        return 0 if self.is_discrete() else self.model.float_precision

    def __pow__(self, power):
        # INTERNAL
        if 0 == power:
            return 1
        elif 1 == power:
            return self
        elif 2 == power:
            return self.square()
        else:
            self.model.unsupported_power_error(self, power)

    def square(self):
        # redefine for each class of expression
        return None  # pragma: no cover

    def __gt__(self, e):
        """ The strict > operator is not supported
        """
        self.model.unsupported_relational_operator_error(self, ">", e)

    def __lt__(self, e):
        """ The strict < operator is not supported
        """
        self.model.unsupported_relational_operator_error(self, "<", e)

# ---
