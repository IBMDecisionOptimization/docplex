# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2020
# --------------------------------------------------------------------------

# gendoc: ignore


from os.path import dirname

from docplex.mp.environment import Environment
from docplex.mp.error_handler import is_docplex_debug


# returns a _SafeCplexWrapper class for cplex 12.7
def _get_safe_cplex_wrapper(cplex_module):
    class _SafeCplexWrapper(cplex_module.Cplex):  # pragma: no cover
        # INTERNAL
        # safe wrapping for pwl issue (cf RTC 31149, 31154, 31155)
        # to be removed in 12.7.1
        # noinspection PyArgumentList
        def __init__(self, *args):
            cplex_module.Cplex.__init__(self, *args)
            try:
                PWLConstraintInterface = cplex_module._internal._pwl.PWLConstraintInterface
                self.pwl_constraints = PWLConstraintInterface()
                self.pwl_constraints._setup(self)
            except AttributeError:
                pass

            try:
                LongAnnotationInterface = cplex_module._internal._anno.LongAnnotationInterface
                self.long_annotations = LongAnnotationInterface()
                self.long_annotations._setup(self)
            except AttributeError:
                pass

    return _SafeCplexWrapper


class CplexAdapter(object):
    '''
    This class acts as an adapter to the CPLEX module, depending on a COS location.
    It also contains some convenience methods.

    We want to proxy most used CPLEX methods as members of this adapter for quick access
    (the Adapter works as a proxy namespace)
    '''
    def __init__(self, coslocation=None, procedural=True):
        self.procedural = procedural
        env = Environment()
        # get the CPLEX module from specified location
        cplex_module = env.get_cplex_module(coslocation)
        if cplex_module is None:
            location = coslocation if coslocation is not None else "default path"
            raise ImportError("Cannot import cplex from %s" % location)

        self.cplex_module = cplex_module
        self.cplex_location = dirname(dirname(cplex_module.__file__))
        cpx = self.cplex_module.Cplex()

        cpxv = cpx.get_version()
        if cpxv.startswith('12.7.0'):  # pragma: no cover
            del cpx
            # create a safe wrapper for RTC-31555
            cpx = _get_safe_cplex_wrapper(self.cplex_module)
        self.cpx = cpx
        # quick access to constants
        cpx_cst = self.cplex_module._internal._constants
        self.cpx_cst = cpx_cst
        # methods
        cpxproc = self.cplex_module._internal._procedural
        self.chgcoeflist = cpxproc.chgcoeflist
        self.chgobj = cpxproc.chgobj
        self.chgrhs = cpxproc.chgrhs
        self.chgqpcoef = cpxproc.chgqpcoef
        self.newcols = cpxproc.newcols
        self.setintparam = cpxproc.setintparam
        self.addindconstr = cpxproc.addindconstr
        self.addrows = cpxproc.addrows
        self.chgrngval = cpxproc.chgrngval
        self.addpwl = cpxproc.addpwl
        self.getnumpwl = cpxproc.getnumpwl
        self.chgcolname = cpxproc.chgcolname
        self.getx = cpxproc.getx
        self.chgctype = cpxproc.chgctype
        self.getprobtype = cpxproc.getprobtype
        self.getnumcols = cpxproc.getnumcols
        self.getnumrows = cpxproc.getnumrows
        self.getrows = cpxproc.getrows
        self.getcolname = cpxproc.getcolname
        self.getlb = cpxproc.getlb
        self.getub = cpxproc.getub
        self.getsense = cpxproc.getsense

        # allocators
        try:
            self.allocate_int_c_array = self.cplex_module._internal._list_array_utils.allocate_int_C_array
            self.allocate_long_c_array = self.cplex_module._internal._list_array_utils.allocate_long_C_array
            self.allocate_double_c_array = self.cplex_module._internal._list_array_utils.allocate_double_C_array
            self.fast_get_row_tuples = cpxproc.fast_getrows
            self.fast_getobj = cpxproc.fast_getobj
            self._use_fast_C_interface = True
        except AttributeError:
            self.allocate_int_c_array = None
            self.allocate_long_c_array = None
            self.allocate_double_c_array = None
            self.fast_get_row_tuples = None
            self.fast_getobj = None
            self._use_fast_C_interface = False


        try:
            # needs cplex > 12.9
            self.multiobjsetobj = cpxproc.multiobjsetobj
        except AttributeError:
            self.multiobjsetobj = None

        try:
            # fast multiobj API:
            # check one to None to see if it's present.
            self.fast_multiobj_getweight = cpxproc.fast_multiobjgetweight
            self.fast_multiobj_getprio   = cpxproc.fast_multiobjgetpriority
            self.fast_multiobj_getoffset = cpxproc.fast_multiobjgetoffset
            self.fast_multiobj_getabstol = cpxproc.fast_multiobjgetabstol
            self.fast_multiobj_getreltol = cpxproc.fast_multiobjgetreltol
            self.fast_multiobj_getobj    = cpxproc.fast_multiobjgetobj
            self.has_non_default_lb      = cpxproc.has_non_default_lb
            self.has_non_default_ub      = cpxproc.has_non_default_ub
            self.has_name                = cpxproc.has_name

            if is_docplex_debug():
                print(f"-- using has_xxx methods in C")

        except AttributeError:
            self.fast_multiobj_getweight = None
            self.fast_multiobj_getprio   = None
            self.fast_multiobj_getoffset = None
            self.fast_multiobj_getabstol = None
            self.fast_multiobj_getreltol = None
            self.fast_multiobj_getobj    = None
            self.has_non_default_lb      = None
            self.has_non_default_ub      = None
            self.has_name                = None

        subinterfaces = self.cplex_module._internal._subinterfaces
        self.ct_linear = subinterfaces.FeasoptConstraintType.linear
        self.ct_quadratic = subinterfaces.FeasoptConstraintType.quadratic
        self.ct_indicator = subinterfaces.FeasoptConstraintType.indicator

        # initialize constants etc...
        try:
            cpxv = self.cplex_module.__version__
            cpxv_as_tuples = cpxv.split('.')
            cpxvt = tuple(int(x) for x in cpxv_as_tuples)
            self.is_post1210 = cpxvt >= (12, 10)
        except AttributeError:  # pragma: no cover
            self.is_post1210 = False
        # chbmatrix
        try:
            # from 12.7.1 up
            self.chbmatrix = cpxproc.chbmatrix
        except AttributeError:  # pragma: no cover
            # up to 12.7.0
            try:
                self.chbmatrix = self.cplex_module._internal._matrices.chbmatrix
            except AttributeError:
                self.chbmatrix = None

        # indicator types
        try:
            self.cpx_indicator_type_ifthen = cpx_cst.CPX_INDICATOR_IF
            self.cpx_indicator_type_equiv = cpx_cst.CPX_INDICATOR_IFANDONLYIF
        except AttributeError:  # pragma: no cover
            # handle previous versions without indicator type
            self.cpx_indicator_type_ifthen = None
            self.cpx_indicator_type_equiv = None

        if self.cpx_indicator_type_equiv is None:
            self.supports_typed_indicators = False  # pragma: no cover
        else:
            try:
                IndicatorConstraintInterface = subinterfaces.IndicatorConstraintInterface
                # noinspection PyStatementEffect
                IndicatorConstraintInterface.type_
                self.supports_typed_indicators = True
            except AttributeError:  # pragma: no cover
                self.supports_typed_indicators = False

        # fast methods
        self.fast_add_linear = self.fast_add_linear_1210 if self.is_post1210 else self.fast_add_linear1290

        # exception
        self.CplexError = self.cplex_module.exceptions.CplexError
        self.CplexSolverError = self.cplex_module.exceptions.CplexSolverError

        self._monkeypatch_LAU()

        # ---- END OF INIT -----


    @classmethod
    def _monkeypatch_LAU(cls):
        try:
            from cplex._internal._list_array_utils import fast_array_to_list
            # set "array_to_list" as "fast_array_to_list
            import cplex._internal._list_array_utils as lau
            lau.__dict__['array_to_list'] = fast_array_to_list
            if is_docplex_debug():
                print(f"-- patched array to list")
        except ImportError:
            pass


    @property
    def use_fast_C_interface(self):
        # TODO: add a yes/no flag in model.
        return self._use_fast_C_interface

    def fast_get_solution(self, cpx, nb_vars):
        return self.getx(cpx._env._e, cpx._lp, 0, nb_vars-1)

    def _fast_get_var_names1290(self, cpx):
        cpxenv = cpx._env
        nb_vars = self.getnumcols(cpxenv._e, cpx._lp)
        # noinspection PyArgumentList
        return self.getcolname(cpxenv._e, cpx._lp, 0, nb_vars-1, cpxenv._apienc)

    def _fast_get_var_names12100(self, cpx):
        cpxenv = cpx._env
        nb_vars = self.getnumcols(cpxenv._e, cpxenv._lp)
        return self.getcolname(cpxenv._e, cpx._lp, 0, nb_vars-1)

    def fast_get_rows(self, cpx):
        cpxenv = cpx._env._e
        cpxlp = cpx._lp
        num_rows = self.getnumrows(cpxenv, cpxlp)
        matbeg, matind, matval = self.getrows(cpxenv, cpxlp, 0, num_rows - 1)
        size = len(matbeg)

        def make_tuple(k):
            begin = matbeg[k]
            if k == size - 1:
                end = len(matind)
            else:
                end = matbeg[k + 1]
            return matind[begin:end], matval[begin:end]

        return [make_tuple(i) for i in range(size)]

    def fast_add_linear1290(self, cpx, lin_expr, cpx_senses, rhs, names, ranges=None):
        # INTERNAL
        # BEWARE: expects a string for senses, not a list
        cpx_linearcts = cpx.linear_constraints
        num_old_rows = cpx_linearcts.get_num()
        num_new_rows = len(rhs)
        cpxenv = cpx._env
        # noinspection PyArgumentList
        with self.chbmatrix(lin_expr, cpx._env_lp_ptr, 0,
                            cpxenv._apienc) as (rmat, nnz):
            # noinspection PyArgumentList
            self.addrows(cpxenv._e, cpx._lp, 0,
                         len(rhs), nnz, rhs, cpx_senses,
                         rmat, [], names, cpxenv._apienc)
        if ranges:
            self.chgrngval(cpxenv._e, cpx._lp,
                           list(range(num_old_rows, num_old_rows + num_new_rows)),
                           ranges)

        return range(num_old_rows, cpx_linearcts.get_num())

    def fast_add_linear_1210(self, cpx, lin_expr, cpx_senses, rhs, names, ranges=None):
        # INTERNAL
        # BEWARE: expects a string for senses, not a list
        cpx_linearcts = cpx.linear_constraints
        num_old_rows = cpx_linearcts.get_num()
        num_new_rows = len(rhs)
        cpxenv = cpx._env
        # noinspection PyArgumentList
        with self.chbmatrix(lin_expr, cpx._env_lp_ptr, 0) as (rmat, nnz):
            self.addrows(cpxenv._e, cpx._lp, 0,
                         len(rhs), nnz, rhs, cpx_senses,
                         rmat, [], names)
        if ranges:
            self.chgrngval(
                cpxenv._e, cpx._lp,
                list(range(num_old_rows, num_old_rows + num_new_rows)),
                ranges)

        return range(num_old_rows, cpx_linearcts.get_num())

    def static_fast_set_linear_obj(self, cpx, indices, obj_coefs):
        self.chgobj(cpx._env._e, cpx._lp, indices, obj_coefs)


    # def add_linear(self, cpx, lin_expr, cpx_senses, rhs, names, ranges=None):
    #     if not self.is_post1210 and self.chbmatrix:
    #         # BEWARE: expects a string for senses, not a list
    #         cpx_linearcts = cpx.linear_constraints
    #         num_old_rows = cpx_linearcts.get_num()
    #         num_new_rows = len(rhs)
    #         cpxenv = cpx._env
    #         # noinspection PyArgumentList
    #         with self.chbmatrix(lin_expr, cpx._env_lp_ptr, 0,
    #                             cpxenv._apienc) as (rmat, nnz):
    #             # noinspection PyArgumentList
    #             self.addrows(cpxenv._e, cpx._lp, 0,
    #                          len(rhs), nnz, rhs, cpx_senses,
    #                          rmat, [], names, cpxenv._apienc)
    #         if ranges:
    #             self.chgrngval(
    #                 cpxenv._e, cpx._lp,
    #                 list(range(num_old_rows, num_old_rows + num_new_rows)),
    #                 ranges)
    #
    #         return range(num_old_rows, cpx_linearcts.get_num())
    #     else:  # pragma: no cover
    #         return cpx.linear_constraints.add(lin_expr=lin_expr,
    #                                           senses=cpx_senses,
    #                                           rhs=rhs,
    #                                           names=names,
    #                                           range_values=ranges)

