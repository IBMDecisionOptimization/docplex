import unittest
import six

from testutils import DocplexAbstractTest
from docplex.mp.utils import DOcplexException
from docplex.mp.model import Model

infinite_detected_msg = "Infinite value detected"


class DocplexNumericTests(unittest.TestCase):

    def setUp(self):
        self.model = Model(checker="numeric")
        self.zenan = float('nan')
        self.pinf = float('inf')
        self.cpx_inf = 1e+20
        self.ix = self.model.integer_var(name='ix')

    def tearDown(self):
        self.model.end()
        self.model = None

    def test_nan_rhs(self):
        six.assertRaisesRegex(self, DOcplexException, "NaN value found in expression",
                              lambda m: m.add(self.ix <= self.zenan), self.model)

    def test_nan_lhs(self):
        six.assertRaisesRegex(self, DOcplexException, "NaN value found in expression",
                              lambda m: m.add(self.zenan <= self.ix), self.model)

    def test_nan_range_lb(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "Model.range_constraint: NaN value detected",
                              lambda m: m.add_range(self.zenan, x, 100),
                              nm)

    def test_nan_range_ub(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "Model.range_constraint: NaN value detected",
                              lambda m: m.add_range(1, x, self.zenan),
                              nm)

    def test_nan_coef_expr(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "NaN value found in expression",
                              lambda m: m.add(self.zenan * x <= 1), nm)

    def test_nan_constant_expr(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "NaN value found in expression",
                              lambda m: m.add(2 * x + self.zenan <= 1), nm)

    def test_nan_var_lb(self):
        nm = self.model
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m: m.integer_var(name='nanlb', lb=self.zenan), nm)

    def test_nan_var_ub(self):
        nm = self.model
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m: m.integer_var(name='nanlb', ub=self.zenan), nm)

    def test_nan_varlist_lb(self):
        nm = self.model
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m: m.integer_var_list(3, lb=self.zenan), nm)

    def test_nan_varlist_ub(self):
        nm = self.model
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m: m.integer_var_list(3, ub=self.zenan), nm)

    def test_nan_inside_sum(self):
        nm = self.model
        x = nm.integer_var(name='x')
        sum_args = [1, 2, x, x + 1, 3 * x, self.zenan]
        six.assertRaisesRegex(self, DOcplexException, "NaN", lambda m: m.sum(sum_args), nm)

    def test_nan_inside_scalprod(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "NaN", lambda m: m.dot([x], [self.zenan]), nm)

    def test_pinf_rhs(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, infinite_detected_msg, lambda m: m.add(x <= self.pinf),
                              nm)

    def test_pinf_lhs(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, infinite_detected_msg, lambda m: m.add(self.pinf <= x),
                              nm)

    def test_pinf_range_lb(self):
        six.assertRaisesRegex(self, DOcplexException, "Model.range_constraint: Infinite value detected",
                              lambda m: m.add_range(self.pinf, self.ix, 100),
                              self.model)

    def test_pinf_range_ub(self):
        six.assertRaisesRegex(self, DOcplexException, "Model.range_constraint: Infinite value detected",
                              lambda m: m.add_range(1, self.ix, self.pinf),
                              self.model)

    def test_1e20_lhs(self):
        # no exception
        self.model.add(self.cpx_inf <= self.ix)

    def test_1e20_rhs(self):
        # no exception
        self.model.add(self.ix <= self.cpx_inf)

    def test_num_seq_list(self):
        ll = [1, self.zenan, 3]
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m_: m_._checker.typecheck_num_seq(ll),
                              self.model)

    def test_check_constraint_numeric_pass_ok(self):
        # should not raise
        self.model._checker.typecheck_constraint("foo")

    def test_check_var_numeric_pass_ok(self):
        self.model._checker.typecheck_var("foo")

    def test_num_seq_list_raise(self):
        m = self.model
        lbs = [1, self.zenan, 3]
        ubs = [10, 20, 30]
        xs = m.continuous_var_list(3, name="x")
        six.assertRaisesRegex(self, DOcplexException, "Model.add_ranges.lbs, pos 1: NaN value",
                              lambda m_: m_.add_ranges(lbs, exprs=xs, ubs=ubs), m)


class DocplexStandardCheckerTests(DocplexAbstractTest):

    def setUp(self):
        DocplexAbstractTest.setUp(self)
        self.zenan = float('nan')
        self.pinf = float('inf')

    def test_pinf_rhs(self):
        mdl = self.model
        x = mdl.integer_var(name='x')
        ct = mdl.add(x <= self.pinf)
        self.assertEqual("x <= 1e+20", str(ct))
        mdl.minimize(x)
        sol = mdl.solve()
        self.assertIsNotNone(sol)
        self.assertAlmostEqual(0, x.solution_value, delta=1e-6)

    def test_linf_lhs(self):
        mdl = self.model
        x = mdl.integer_var(name='x', lb=-1e+20, ub=-3)
        ct = mdl.add(-self.pinf <= x)
        self.assertEqual("x >= -1e+20", str(ct))
        mdl.maximize(x)
        sol = mdl.solve()
        self.assertIsNotNone(sol)
        self.assertAlmostEqual(x.ub, x.solution_value, delta=1e-6)

    def check_large_rhs(self, rhs, ok=True, expected=90):
        with Model(name="this one still not working, but almost") as m:
            v1 = m.continuous_var(ub=10, name="v1")
            v2 = m.continuous_var(ub=10, name="v2")
            m.add_constraint(3 * v1 + 2 * v2 <= rhs)
            m.maximize(5 * v1 + 4 * v2)
            s = m.solve()
            if ok:
                self.assertIsNotNone(s)
                self.assertEqual(m.objective_value, expected)
            else:
                self.assertIsNone(s)

    def test_pinf_rhs1(self):
        self.check_large_rhs(1e+20)

    def test_pinf_rhs2(self):
        self.check_large_rhs(self.pinf)

    def test_pinf_rhs3(self):
        self.check_large_rhs(min(1e+20, self.pinf))

    def test_pinf_rhs4(self):
        self.check_large_rhs(max(1e+20, self.pinf))

    def test_pinf_rhs5(self):
        self.check_large_rhs(min(0.999 * 1e+20, self.pinf))

    def test_ninf_rhs6(self):
        # with -oo , no solve
        self.check_large_rhs(-self.pinf, ok=False)

    def test_num_seq_list(self):
        ll = [1, self.zenan, 3]
        # no raise
        self.assertEqual(ll, self.model._checker.typecheck_num_seq(ll))

    def test_check_constraint_ko_raise(self):
        six.assertRaisesRegex(self, DOcplexException, "Expecting constraint",
                              lambda m_: m_._checker.typecheck_constraint('foo'), self.model)

    def test_check_var_ko_raise(self):
        six.assertRaisesRegex(self, DOcplexException, "Expecting decision variable",
                              lambda m_: m_._checker.typecheck_var('foo'), self.model)


class DocplexFullCheckerTests(unittest.TestCase):

    def setUp(self):
        self.model = Model(checker="full")
        self.zenan = float('nan')
        self.pinf = float('inf')
        self.cpx_inf = 1e+20
        self.ix = self.model.integer_var(name='ix')

    def tearDown(self):
        self.model.end()
        self.model = None

    def test_standard_checker_name(self):
        self.assertEqual(self.model._checker.name, "full")

    def test_nan_rhs(self):
        six.assertRaisesRegex(self, DOcplexException, "NaN value found in expression",
                              lambda m: m.add(self.ix <= self.zenan), self.model)

    def test_nan_lhs(self):
        six.assertRaisesRegex(self, DOcplexException, "NaN value found in expression",
                              lambda m: m.add(self.zenan <= self.ix), self.model)

    def test_nan_range_lb(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "Model.range_constraint: NaN value detected",
                              lambda m: m.add_range(self.zenan, x, 100),
                              nm)

    def test_nan_range_ub(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "Model.range_constraint: NaN value detected",
                              lambda m: m.add_range(1, x, self.zenan),
                              nm)

    def test_nan_coef_expr(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "NaN value found in expression",
                              lambda m: m.add(self.zenan * x <= 1), nm)

    def test_nan_constant_expr(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "NaN value found in expression",
                              lambda m: m.add(2 * x + self.zenan <= 1), nm)

    def test_nan_var_lb(self):
        nm = self.model
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m: m.integer_var(name='nanlb', lb=self.zenan), nm)

    def test_nan_var_ub(self):
        nm = self.model
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m: m.integer_var(name='nanlb', ub=self.zenan), nm)

    def test_nan_varlist_lb(self):
        nm = self.model
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m: m.integer_var_list(3, lb=self.zenan), nm)

    def test_nan_varlist_ub(self):
        nm = self.model
        six.assertRaisesRegex(self, DOcplexException, "NaN value",
                              lambda m: m.integer_var_list(3, ub=self.zenan), nm)

    def test_nan_inside_sum(self):
        nm = self.model
        x = nm.integer_var(name='x')
        sum_args = [1, 2, x, x + 1, 3 * x, self.zenan]
        six.assertRaisesRegex(self, DOcplexException, "NaN", lambda m: m.sum(sum_args), nm)

    def test_nan_inside_scalprod(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, "NaN", lambda m: m.dot([x], [self.zenan]), nm)

    def test_pinf_rhs(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, infinite_detected_msg, lambda m: m.add(x <= self.pinf),
                              nm)

    def test_pinf_lhs(self):
        nm = self.model
        x = nm.integer_var(name='x')
        six.assertRaisesRegex(self, DOcplexException, infinite_detected_msg, lambda m: m.add(self.pinf <= x),
                              nm)

    def test_pinf_range_lb(self):
        six.assertRaisesRegex(self, DOcplexException, "Model.range_constraint: Infinite value detected",
                              lambda m: m.add_range(self.pinf, self.ix, 100),
                              self.model)

    def test_pinf_range_ub(self):
        six.assertRaisesRegex(self, DOcplexException, "Model.range_constraint: Infinite value detected",
                              lambda m: m.add_range(1, self.ix, self.pinf),
                              self.model)

    def test_1e20_lhs(self):
        # no exception
        self.model.add(self.cpx_inf <= self.ix)

    def test_1e20_rhs(self):
        # no exception
        self.model.add(self.ix <= self.cpx_inf)

    def test_num_seq_list(self):
        m = self.model
        lbs = [1, self.zenan, 3]
        ubs = [10, 20, 30]
        xs = m.continuous_var_list(3, name="x")
        six.assertRaisesRegex(self, DOcplexException, "Model.add_ranges.lbs, pos 1: NaN value",
                              lambda m_: m_.add_ranges(lbs, exprs=xs, ubs=ubs), m)

    def test_check_constraint_full_ko_raise(self):
        six.assertRaisesRegex(self, DOcplexException, "Expecting constraint",
                              lambda m_: m_._checker.typecheck_constraint('foo'), self.model)

    def test_check_var_full_ko_raise(self):
        six.assertRaisesRegex(self, DOcplexException, "Expecting decision variable",
                              lambda m_: m_._checker.typecheck_var('foo'), self.model)


class DocplexDummyCheckerTests(unittest.TestCase):

    def setUp(self):
        self.model = Model(checker="off")
        self.zenan = float('nan')
        self.pinf = float('inf')
        self.cpx_inf = 1e+20
        self.ix = self.model.integer_var(name='ix')

    def tearDown(self):
        self.model.end()
        self.model = None

    def test_checker_name(self):
        self.assertEqual(self.model._checker.name, "off")

    def test_check_constraint_numeric_pass_ok(self):
        # should not raise
        self.model._checker.typecheck_constraint("foo")

    def test_check_var_numeric_pass_ok(self):
        # should not raise
        self.model._checker.typecheck_var("foo")

    def test_num_seq_list_nan_pass_ok(self):
        ll = [1, self.zenan, 3]
        self.model._checker.typecheck_num_seq(ll)

    def test_num_seq_list_pinf_pass_ok(self):
        ll = [1, self.pinf, 3]
        self.model._checker.typecheck_num_seq(ll)


class DOcplexSwitchCheckerTests(unittest.TestCase):

    def test_checker_swoitch_on(self):
        with Model(checker="off") as m:
            self.assertEqual(m._checker.name, "off")
            # do nothing
            m._checker.typecheck_var("foo")
            m.set_checker("on")
            six.assertRaisesRegex(self, DOcplexException, "Expecting decision variable",
                                  lambda m_: m_._checker.typecheck_var('foo'), m)

    def test_checker_switch_off(self):
        with Model(checker="on") as m:
            self.assertEqual(m._checker.name, "std")

            six.assertRaisesRegex(self, DOcplexException, "Expecting decision variable",
                                  lambda m_: m_._checker.typecheck_var('foo'), m)

            m.set_checker("off")
            # do nothing
            m._checker.typecheck_var("foo")

    def test_model_bad_checker_key(self):
        with Model(checker="spirou") as m:
            self.assertEqual(m._checker.name, "std")


if __name__ == "__main__":
    unittest.main()
