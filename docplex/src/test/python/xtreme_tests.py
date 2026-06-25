from __future__ import print_function

import unittest
import six

from docplex.mp.utils import DOcplexException
from docplex.mp.model import Model

from testutils import DocplexAbstractTest


class Dumb(object):
    def __repr__(self):
        return 'Dumb()'


class DocplexErrorTests(DocplexAbstractTest):

    def _try_var_bad_vartype(self):
        self.model.var(vartype="foo")

    def test_bad_vartype(self):
        self.assertRaises(DOcplexException, self._try_var_bad_vartype)

    def _try_var_list_float_size(self):
        m = self.model
        m.continuous_var_list(keys=3.14)

    def test_var_list_float_size(self):
        self.assertRaises(DOcplexException, self._try_var_list_float_size)

    def _try_var_cube_empty_subseq(self):
        m = self.model
        keys1 = [1, 2]
        keys2 = []
        keys3 = [1.1, 2.2]
        m.var_hypercube(m.continuous_vartype, [keys1, keys2, keys3])

    def test_var_cube_empty_subseq(self):
        self.assertRaises(DOcplexException, self._try_var_cube_empty_subseq)

    def test_var_cube_empty_seq(self):
        self.assertRaises(DOcplexException, lambda m_: m_.var_hypercube('C', []), self.model)

    def test_str_anonymous(self):
        mdl = self.model
        anonymous_int_var = mdl.integer_var(3, 7)
        self.assertTrue(str(anonymous_int_var))

    def test_hash_vars(self):
        mdl = self.model
        x = mdl.binary_var()
        y = mdl.binary_var()
        self.assertTrue(hash(x) != hash(y))

    def test_value_unsolved(self):
        mdl = self.model
        xvar = mdl.continuous_var(3, 7)
        self.assertRaises(DOcplexException, lambda x: x.solution_value, xvar)

    def test_float_unsolved(self):
        mdl = self.model
        xvar = mdl.continuous_var(3, 7)
        self.assertRaises(DOcplexException, xvar.__float__)

    def test_int_unsolved(self):
        mdl = self.model
        ivar = mdl.integer_var()
        self.assertRaises(DOcplexException, ivar.__int__)

    def test_int_continuous(self):
        mdl = self.model
        xvar = mdl.continuous_var(3, 7)
        mdl.minimize(xvar)
        self.assertTrue(mdl.solve())
        self.assertRaises(DOcplexException, xvar.__int__)

    @staticmethod
    def _try_add_var_string(mdl, x):
        e = mdl.linear_expr(x)
        e += "foo"

    def test_add_var_string(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, self._try_add_var_string, mdl, x)

    @staticmethod
    def _try_sub_var_string(mdl, x):
        e = mdl.linear_expr(x)
        e -= "foo"

    def test_sub_var_string(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, self._try_sub_var_string, mdl, x)

    @staticmethod
    def _try_mul(x, y):
        return x * y

    def try_mul_none(self, arg1):
        e = arg1 * None

    def test_mul_var_none(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, self.try_mul_none, x)

    def test_mul_expr_none(self):
        mdl = self.model
        x = mdl.continuous_var(name='x').to_linear_expr()
        self.assertRaises(DOcplexException, self.try_mul_none, x)

    @staticmethod
    def try_inverse(mdl, e):
        one = mdl.linear_expr(1.0)
        return one / e

    def test_div_expr_by_var(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, self.try_inverse, mdl, x)

    @staticmethod
    def try_num_inverse(e):
        return 1.0 / e

    def test_div_num_by_var(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, self.try_num_inverse, x)

    def test_div_by_expr(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        # expr = x+1
        expr = x + 1
        self.assertRaises(DOcplexException, self.try_inverse, mdl, expr)

    def check_zero_divide(self, e):
        # INTERNAL
        try:
            e / 0
            msg = None
        except DOcplexException as dox:
            msg = str(dox)
        self.assertIsNotNone(msg)
        self.assertTrue(msg.lower().find('zero divide') >= 0)

    def test_var_divide_zero(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.check_zero_divide(x)

    def test_mnm_divide_zero(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.check_zero_divide(7 * x)

    def test_linexpr_divide_zero(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        y = mdl.continuous_var(name='y')
        self.check_zero_divide(3 * x + 7 * y + 11)

    def _try_create_domain_var(self, vartype, lb, ub):
        self.model.var(vartype, lb, ub)

    def test_create_empty_integer_domain(self):
        self.assertRaises(DOcplexException, self._try_create_domain_var, self.model.integer_vartype, 4, 3)

    def test_create_empty_float_domain(self):
        self.assertRaises(DOcplexException, self._try_create_domain_var, self.model.continuous_vartype, 7.7, 3.3)

    def test_create_empty_domain_integer_varlist(self):
        self.assertRaises(DOcplexException, lambda m: m.integer_var_list(3, lb=3, ub=2), self.model)

    def _try_build_expr_from_string(self):
        self.model.linear_expr("foo")

    def test_bad_expr_arg(self):
        self.assertRaises(DOcplexException, self._try_build_expr_from_string)

    def try_get_coeff(self, expr, not_a_var):
        return expr.get_coef(not_a_var)

    def test_expr_bad_get_coeff1(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        e1 = mdl.linear_expr(7) + x
        e2 = mdl.linear_expr(11) + x
        self.assertRaises(DOcplexException, self.try_get_coeff, e1, e2)

    def test_expr_bad_get_coeff2(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        e1 = mdl.linear_expr(7) + x
        self.assertRaises(DOcplexException, self.try_get_coeff, e1, 'foo')

    @staticmethod
    def try_expr_bad_tuple(mdl):
        return mdl.linear_expr(["x", "y"])

    @staticmethod
    def try_expr_tuple3(mdl, x):
        return mdl.linear_expr((x, 1, 1))

    def test_expr_bad_tuple(self):
        m = self.model
        self.assertRaises(DOcplexException, self.try_expr_bad_tuple, m)

    def test_expr_tuple3(self):
        m = self.model
        x = m.continuous_var(name='x')
        self.assertRaises(ValueError, self.try_expr_tuple3, m, x)

    @staticmethod
    def _try_expr_bad_tuple4(mdl):
        return mdl.linear_expr(("foo", 3.14))

    @staticmethod
    def try_expr_from_string_list(mdl):
        mdl.linear_expr(['x', 'y'])

    def test_expr_bad_list(self):
        m = self.model
        self.assertRaises(DOcplexException, self.try_expr_from_string_list, m)

    def try_duplicated_constraint_names(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        x_ge_1 = m.add_constraint(x >= 1, "ct")
        y_ge_2 = m.add_constraint(y >= 1, "ct")
        self.assertEqual(0, x_ge_1.safe_index)
        self.assertEqual(1, y_ge_2.safe_index)

    def _try_duplicate_var_names(self):
        m = self.model
        m.continuous_var(name='xxx')
        m.continuous_var(name='xxx')  # this should raise an ex

    @unittest.skip('now a warning')
    def test_duplicate_var_names(self):
        self.assertRaises(DOcplexException, self._try_duplicate_var_names)

    def _try_export_unsupported_extension(self):
        m = self.model
        m.export(format_spec="zorglub")

    def test_export_unsupported_extension(self):
        self.assertRaises(ValueError, self._try_export_unsupported_extension)

    @staticmethod
    def _try_boolean_conversion_continuous(dvar):
        if bool(dvar):
            pass

    def test_boolean_conversion_continuous(self):
        mdl = self.model
        x = mdl.continuous_var(lb=0, ub=1, name='x')
        mdl.maximize(x)
        self.assertTrue(mdl.solve())
        # self.assertRaises(IloException, self._try_boolean_conversion_continuous, x)

    def test_non_numeric_lb(self):
        six.assertRaisesRegex(self, DOcplexException, 'Var.lb', lambda m: m.continuous_var(lb='foo'), self.model)

    def test_non_numeric_ub(self):
        six.assertRaisesRegex(self, DOcplexException, 'Var.ub', lambda m: m.continuous_var(ub='foo'), self.model)

    def test_varlist_w_string_bounds(self):
        mdl = self.model
        six.assertRaisesRegex(self, DOcplexException, 'Variable lb expect numbers',
                              lambda m: m.continuous_var_list(3, lb=['a', 'b', 'c']), mdl)

    def test_custom_var_lb_unexpected_type(self):
        self.assertRaises(DOcplexException, lambda mm: mm.continuous_var_list(3, lb='foo'), self.model)

    def test_create_vars_custom_lbs_dict_bad_bound(self):
        m = self.model
        size = 3
        custom_lbs = {k: (k+1)**3 for k in range(size)}
        custom_lbs[0] = 'foo'
        six.assertRaisesRegex(self, DOcplexException,
                              'Variable lb expect numbers',
                              lambda m_: m_.continuous_var_list(3, lb=custom_lbs), m)

    @staticmethod
    def _try_indicator(mdl, b, ct, active=1):
        mdl.add_indicator(binary_var=b, linear_ct=ct, active_value=active)

    # indicators
    def test_indicator_not_a_binary_var(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_indicator, mdl, x + 1, x <= y)

    def test_indicator_not_a_linear_ct(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_indicator, mdl, x, y)

    def test_indicator_bad_active_value(self):
        mdl = self.model
        x = mdl.binary_var(name='b')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_indicator, mdl, x, x <= y, active=2)

    @staticmethod
    def _try_lt_op_on_cts(mdl, ct1, ct2):
        mdl.add_constraint(ct1 < ct2)

    @staticmethod
    def _try_gt_op_on_cts(mdl, ct1, ct2):
        mdl.add_constraint(ct1 > ct2)

    def test_lt_cts(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        ct1 = x <= 3
        ct2 = y >= 5
        self.assertRaises(DOcplexException, self._try_lt_op_on_cts, m, ct1, ct2)

    def test_gt_cts(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        ct1 = x <= 3
        ct2 = y >= 5
        self.assertRaises(DOcplexException, self._try_gt_op_on_cts, m, ct1, ct2)

    @staticmethod
    def _try_mixing_variables_from_two_models():
        x = Model(name="m1").integer_var(0, 1, 'x')
        y = Model(name="m2").integer_var(0, 1, 'y')
        c = (x <= y)

    def test_mixup_variables_from_different_models(self):
        self.assertRaises(DOcplexException, self._try_mixing_variables_from_two_models)

    # an expression with a strict < or > raises an exception
    @staticmethod
    def _try_strict_less_op(mdl, expr1, expr2):
        mdl.add_constraint(expr1 < expr2)

    def test_strict_less_op_var_var(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_less_op, mdl, x, y)

    def test_strict_less_op_var_num(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        self.assertRaises(DOcplexException, self._try_strict_less_op, mdl, x, 0)

    def test_strict_less_op_mon_var(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_less_op, mdl, 2 * x, y)

    def test_strict_less_op_expr_var(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_less_op, mdl, 2 * x + 3 * y + 7, y)

    def test_strict_less_op_expr_mon(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_less_op, mdl, 2 * x + 3 * y + 7, 6 * y)

    def test_strict_less_op_expr_num(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_less_op, mdl, 2 * x + 3 * y + 7, 6)

    def test_strict_less_op_num_expr(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_less_op, mdl, 6, 2 * x + 3 * y + 7)

    @staticmethod
    def _try_strict_greater_op(mdl, expr1, expr2):
        mdl.add_constraint(expr1 > expr2)

    def test_strict_greater_op_var_var(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_greater_op, mdl, x, y)

    def test_strict_gt_op_var_num(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        self.assertRaises(DOcplexException, self._try_strict_greater_op, mdl, x, 0)

    def test_strict_gt_op_mon_var(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_greater_op, mdl, 2 * x, y)

    def test_strict_greater_op_expr_var(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_greater_op, mdl, 2 * x + 3 * y + 7, y)

    def test_strict_greater_op_expr_mon(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_greater_op, mdl, 2 * x + 3 * y + 7, 6 * y)

    def test_strict_greater_op_expr_num(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_greater_op, mdl, 2 * x + 3 * y + 7, 6)

    def test_strict_greater_op_num_expr(self):
        mdl = self.model
        x = mdl.binary_var(name='x')
        y = mdl.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_strict_greater_op, mdl, 12, 2 * x + 3 * y + 7)

    @staticmethod
    def _try_bool_conversion(dvar):
        return dvar.to_bool()

    def test_convert_continuous_var_to_bool(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, self._try_bool_conversion, x)

    def _try_negative_time_limit(self):
        self.model.set_time_limit(-12)

    def test_timelimit_negative(self):
        self.assertRaises(DOcplexException, self._try_negative_time_limit)

    def test_timelimit_too_low(self):
        self.model.set_time_limit(0.9)
        self.assertEqual(1, self.model.number_of_warnings)

    @staticmethod
    def _try_rename(renamed, new_name):
        renamed.name = new_name

    def test_var_set_name_none(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, self._try_rename, x, None)

    def test_var_set_name_emptystring(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, self._try_rename, x, "")

    def test_model_set_name_none(self):
        self.assertRaises(DOcplexException, self._try_rename, self.model, None)

    def test_model_set_name_emptystring(self):
        self.assertRaises(DOcplexException, self._try_rename, self.model, "")

    def test_remove_ct_by_unknown_name(self):
        self.model.remove_constraint("zorglub")

    def test_float_precision_negative(self):
        self.model.float_precision = -4
        self.assertEqual(1, self.model.number_of_warnings)
        self.assertEqual(0, self.model.float_precision)

    def test_float_precision_too_large(self):
        mdl = self.model
        mdl.float_precision = 777
        self.assertEqual(1, mdl.number_of_warnings)
        self.assertEqual(mdl.environment.max_nb_digits, self.model.float_precision)

    def _try_mix_exprs_in_ct(self):
        m1 = Model("m1")
        m2 = Model("m2")
        x_m1 = m1.continuous_var(name="x1")
        x_m2 = m2.continuous_var(name="x2")
        self.model.add_constraint(x_m1 <= x_m2)

    def test_expression_mixup(self):
        try:
            self._try_mix_exprs_in_ct()
            msg = None
        except DOcplexException as docx:
            msg = docx.message
        self.assertIsNotNone(msg, "exception not raised")
        self.assertTrue(msg.find("mix") >= 0)

    def _try_quadratic(self, first, second=None):
        if second is None:
            e = first * first
        else:
            e = first * second
        return e

    @staticmethod
    def _try_not_equal(arg1, arg2):
        e = arg1 != arg2

    def test_ne_var_var(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, x, y)

    def test_ne_var_mnm(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, x, 2 * y)

    def test_ne_var_expr(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, x, 2 * y + 5)

    def test_ne_mnm_var(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, 2 * x, y)

    def test_ne_mnm_mnm(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, 2 * x, 2 * y)

    def test_ne_mnm_expr(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, 2 * x, 2 * y + x + 7)

    def test_ne_expr_var(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, x + 1, y)

    def test_ne_expr_mnm(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, x + 1, 2 * y)

    def test_ne_expr_expr(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        self.assertRaises(DOcplexException, self._try_not_equal, x + 1, y + 2)

    @staticmethod
    def _try_dot(mdl, terms, koefs):
        e = mdl.dot(terms, koefs)

    def test_dot_dict(self):
        m = self.model
        xd = m.continuous_var_dict(keys=['x', 'y', 'z'])  # 3 variables
        ks = [1, 3, 5]
        six.assertRaisesRegex(self, DOcplexException, "requires a list of", self._try_dot, m, xd, ks)

    def test_dot_set(self):
        m = self.model
        set_of_xs = set(m.continuous_var_list(keys=['x', 'y', 'z']))  # 3 variables
        ks = [1, 3, 5]
        six.assertRaisesRegex(self, DOcplexException, "requires a list of", self._try_dot, m, set_of_xs, ks)

    def test_constraints_non_iterable(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        self.assertRaises(DOcplexException, lambda m: m.add_constraints(x >= 1), mdl)

    def test_add_constraints_mixed(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.integer_var(name='iy')
        self.assertRaises(DOcplexException,
                          lambda m: m.add_constraints([(x >= 1, 'ct1'), (y >= 3, 'ct2')], ['foo', 'bar']),
                          m)

    def test_add_constraints_tuple3(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.integer_var(name='iy')
        self.assertRaises(DOcplexException,
                          lambda m: m.add_constraints([(x >= 1, 'ct1', 1), (y >= 3, 'ct2', 2)]),
                          m)

    def test_add_constraints_tuple1(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.integer_var(name='iy')
        self.assertRaises(DOcplexException,
                          lambda m: m.add_constraints([(x >= 1,), (y >= 3,)]), m)

    def test_add_constraints_mixing_tupkles_w_cts(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.integer_var(name='iy')
        self.assertRaises(DOcplexException,
                          lambda m: m.add_constraints([(x >= 1), (y >= 3, 'foo')]), m)

    def test_rc_unsolved(self):
        m = self.model
        x = m.continuous_var(name='x')
        self.assertRaises(DOcplexException, lambda m: m.reduced_costs([x]), m)


class DOcplexTypecheckerTests(DocplexAbstractTest):
    def test_ct_by_index_bad_index(self):
        six.assertRaisesRegex(self, DOcplexException, 'Invalid index',
                              lambda m: m.get_constraint_by_index(-3), self.model)

    def test_bad_cts_seq(self):
        m = self.model
        x = m.continuous_var(name='x')
        ct = (x >= 1)
        cts = [ct, 'foo']
        six.assertRaisesRegex(self, DOcplexException,
                              'Expecting sequence of constraints',
                              lambda m: m.add_constraints(cts), self.model)

    def test_bad_var_keys(self):
        bad_keys = [1, None]
        six.assertRaisesRegex(self, DOcplexException,
                              'Variable keys cannot be None',
                              lambda m: m.binary_var_list(bad_keys), self.model)

    def test_bad_scalprod_coef(self):
        m = self.model
        x = m.continuous_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, 'Expecting valid float number,',
                              lambda m: m.scal_prod([x], ['foo']), self.model)



    def test_bad_progress_listener(self):
        six.assertRaisesRegex(self, DOcplexException, 'Expecting ProgressListener instance',
                              lambda mm: mm.add_progress_listener('foo'), self.model)

    def test_not_in_model(self):
        with Model(name='other') as other:
            other_x = other.continuous_var(name='xx')
            other_c = other.add(other_x <= 1)
            six.assertRaisesRegex(self, DOcplexException, 'not in model',
                                  lambda mm: mm.remove_constraint(other_c), self.model)

    def test_bad_solve_params(self):
        with Model(name='other') as m:
            other_x = m.continuous_var(name='xx')
            other_c = m.add(other_x <= 1)
            six.assertRaisesRegex(self, DOcplexException, 'Expecting CPLEX parameters',
                                  lambda mm: mm.solve(cplex_parameters='foo'), m)


class NumericTypecheckErrorTests(unittest.TestCase):
    def setUp(self):
        self.model = Model(checker='numeric')
        self.x = self.model.continuous_var(name='x')

    def test_numeric_not_a_number(self):
        not_num = Dumb()
        six.assertRaisesRegex(self, DOcplexException, 'Not a number',
                              lambda mm: mm.scal_prod([self.x], [not_num]), self.model)

    def test_numeric_nan(self):
        anan = float('nan')  # well over 1e+20..
        six.assertRaisesRegex(self, DOcplexException, 'NaN value found', lambda z: z * anan, self.x)

    def test_numeric_beyond_pinf(self):
        bigp = 3e+30  # well over 1e+20..
        xx = self.x
        mn = xx * bigp
        self.assertEqual(1e+20, mn.coef)

    def test_numeric_beyond_ninf(self):
        bign = -3e+30  # well over 1e+20..
        xx = self.x
        mn = xx * bign
        self.assertEqual(-1e+20, mn.coef)

    def test_numeric_pinf_times_var(self):
        pinf = float('inf')
        six.assertRaisesRegex(self, DOcplexException, 'Infinite value detected', lambda z: z * pinf, self.x)

    def test_numeric_pinf_plus_var(self):
        pinf = float('inf')
        six.assertRaisesRegex(self, DOcplexException, 'Infinite value detected', lambda z: z + pinf, self.x)

    def test_numeric_pinf_minus_var(self):
        pinf = float('inf')
        six.assertRaisesRegex(self, DOcplexException, 'Infinite value detected', lambda z: z - pinf, self.x)

    def test_numeric_ninf_times_var(self):
        pinf = float('-inf')
        six.assertRaisesRegex(self, DOcplexException, 'Infinite value detected', lambda z: z * pinf, self.x)

    def tearDown(self):
        self.model.end()
        self.model = None


if __name__ == "__main__":
    unittest.main()
