import unittest
import six

from docplex.mp.functional import *
from docplex.mp.utils import DOcplexException
from testutils import RedirectedOutputToStringContext

from docplex.mp.model import Model
from docplex.mp.format import LPFormat


class NonLinearExprTests(unittest.TestCase):
    def setUp(self):
        self.model = Model(keep_ordering=True, name_functional_vars=True)
        self.x = self.model.continuous_var(name='x')
        self.y = self.model.continuous_var(name='y')

    def tearDown(self):
        self.model.end()
        self.model = None

    def check_expr(self, e, expected_str):
        # utility
        self.assertIsInstance(e, Expr)
        self.assertEqual(expected_str, str(e))

    def test_create_absexpr(self):
        m = self.model
        abs_expr = m.abs(self.x)
        self.assertIsInstance(abs_expr, LinearOperand)

    def test_two_absexprs(self):
        # create two absexprs with same expr
        m = self.model
        try:
            abs_expr1 = m.abs(self.x)
            abs_expr2 = m.abs(self.x)
            raised = False
        except DOcplexException:
            raised = True
        self.assertFalse(raised)

    def test_absexpr_clone(self):
        m = self.model
        abs_expr = m.abs(self.x)
        abs_expr2 = abs_expr.clone()
        self.assertIsInstance(abs_expr2, AbsExpr)
        self.assertIsNot(abs_expr2, abs_expr)  # different
        self.assertTrue(abs_expr2.argument_expr, abs_expr.argument_expr)

    def test_absexpr_str(self):
        m = self.model
        abs_expr = m.abs(self.x + 7 * self.y)
        self.check_expr(abs_expr, "abs(x+7y)")

    def test_absexpr_repr(self):
        m = self.model
        abs_expr = m.abs(self.x + 7 * self.y)
        self.assertEqual("docplex.mp.AbsExpr(x+7y)", repr(abs_expr))

    def test_absexpr_name(self):
        m = self.model
        abs_expr = m.abs(self.x + 7 * self.y)
        self.assertEqual("_abs3", abs_expr._get_resolved_f_var().name)

    def test_absexpr_num(self):
        m = self.model
        some_num = 3.14
        abs_nump = m.abs(some_num)
        self.assertEqual(some_num, abs_nump)
        abs_numn = m.abs(-some_num)
        self.assertEqual(some_num, abs_numn)
        self.assertEqual(0, m.abs(0))

    def test_var_minus_abs(self):
        m = self.model
        e = self.y - m.abs(self.x - 1)
        self.assertEqual('y-_abs3', str(e))

    def test_absexpr_negate(self):
        m = self.model
        z = - m.abs(self.x)
        self.check_expr(z, "-_abs2")

    def test_absexpr_add_coef(self):
        m = self.model
        z = m.abs(self.x) + 1
        self.check_expr(z, "_abs2+1")

    def test_absexpr_radd_coef(self):
        m = self.model
        z = 3 + m.abs(self.x)
        self.check_expr(z, "_abs2+3")

    def test_absexpr_add_mn(self):
        m = self.model
        z = m.abs(self.x - 1) + 7 * self.x
        self.check_expr(z, "_abs3+7x")

    def test_absexpr_radd_mn(self):
        m = self.model
        z = 7 * self.x + m.abs(self.x - 1)
        self.check_expr(z, "7x+_abs3")

    def test_absexpr_add_linexp(self):
        m = self.model
        e = 3 * self.x + 5 * self.y + 7
        z = m.abs(self.x - 1) + e
        self.check_expr(z, "_abs3+3x+5y+7")

    def test_absexpr_radd_linexp(self):
        m = self.model
        e = 3 * self.x + 5 * self.y + 7
        z = e + m.abs(self.x - 1)
        self.check_expr(z, "3x+5y+_abs3+7")

    def test_absexpr_sub_coef(self):
        m = self.model
        z = m.abs(self.x) - 1
        self.check_expr(z, "_abs2-1")

    def test_absexpr_rsub_coef(self):
        m = self.model
        z = 3 - m.abs(self.x)
        self.check_expr(z, "-_abs2+3")

    def test_absexpr_sub_mnm(self):
        m = self.model
        my = 4 * self.y
        z = m.abs(self.x - 1) - my
        self.check_expr(z, "_abs3-4y")

    def test_absexpr_rsub_mnm(self):
        m = self.model
        my = 4 * self.y
        z = my - m.abs(self.x - 1)
        self.check_expr(z, "4y-_abs3")

    def test_absexpr_sub_lin(self):
        m = self.model
        lin = 4 * self.y + self.x + 7
        z = m.abs(self.x - 1) - lin
        self.check_expr(z, "_abs3-4y-x-7")

    def test_absexpr_rsub_lin(self):
        m = self.model
        lin = 4 * self.y + self.x + 7
        z = lin - m.abs(self.x - 1)
        self.check_expr(z, "4y+x-_abs3+7")

    def test_absexpr_mul_coef(self):
        m = self.model
        z = 2 * m.abs(self.x)
        self.assertIsInstance(z, Expr)

    def test_absexpr_rmul_coef(self):
        m = self.model
        z = m.abs(self.x) * 2
        self.assertIsInstance(z, Expr)
        self.assertEqual(str(z), "2_abs2")

    def test_absexpr_as_obj(self):
        m = self.model
        ij = m.integer_var(lb=-10, ub=11, name='ij')
        m.maximize(m.abs(ij))
        # m.export_as_lp(basename='absexpr_obj')
        if m._can_solve():
            m.solve()
            self.assertAlmostEqual(11, m.objective_value, delta=1e-4)

    def test_absexpr_square(self):
        m = self.model
        z = m.abs(self.x)
        z2 = z ** 2
        self.assertEqual("_abs2^2", str(z2))

    def test_absexpr_quadproduct(self):
        m = self.model
        xx = m.continuous_var(name="xx")
        yy = m.continuous_var(name="yy")
        zz = m.abs(xx) * m.abs(yy)
        self.assertEqual("_abs4*_abs7", str(zz))

    def test_absexpr_div_coef(self):
        m = self.model
        z = m.abs(self.x) / 0.8
        self.assertIsInstance(z, Expr)
        self.assertEqual(str(z), "1.250_abs2")

    def test_absexpr_sum(self):
        m = self.model
        size = 5
        xs = m.integer_var_list(size, name='xv', lb=1, ub=9)
        ecarts = m.sum(m.abs(xs[i] - 2.5) for i in range(size))
        raw = m.sum(xs)
        # m.add_constraint(xs[0] - xs[1] >= 4)
        # we want to be as close to 2.5 as possible w/integer values: 2 or 3
        # the raw part of th eobjective is here to choose 2
        # so we expect all variables to be equal to 2
        m.minimize(ecarts + 0.01 * raw)
        s = m.solve()
        self.assertIsNotNone(s)
        # self.assertEqual(ecarts.solution_value, 2.5)
        for i in range(size):
            self.assertEqual(xs[i].solution_value, 2)

    @unittest.skip(' do not check for internals')
    def test_absexpr_stats(self):
        m = self.model
        z = m.abs(self.x) * 2
        tstats = m.statistics.as_tuple()
        self.assertEqual(tstats, (2, 0, 5, 0, 0, 0, 3, 0, 2, 0))

    def test_absexpr_export_lp(self):
        m = self.model
        m.add_constraint(2 * m.abs(self.x) <= 47)
        try:
            lps = m.lp_string
        except:
            lps = ""
        self.assertTrue(len(lps) > 0)
        self.assertFalse("(" in lps)

    def test_absexpr_lp_names(self):
        m = self.model
        x = m.continuous_var(lb=-20, ub=15, name='xx')
        z = m.continuous_var(name='z')
        m.add(z == m.abs(x), name='zIsAbsOfx')
        absvar = m.get_var_by_index(4)
        self.assertTrue(absvar.is_generated())
        absvar_name = absvar.name
        self.assertEqual("_abs4", absvar_name)
        self.assertTrue(LPFormat.is_lp_compliant(absvar_name))
        # lps = m.lp_string
        # wait for generated in LPs
        # self.assertIn(absvar_name, lps)

    def test_absexpr_as_objective(self):
        m = self.model
        x = m.integer_var(name='ij1to9', lb=1, ub=9)
        dist_to_five = m.abs(x - 5)
        m.maximize(dist_to_five + x)
        if m._can_solve():
            s = m.solve()
            self.assertIsNotNone(s)
            self.assertEqual(dist_to_five.solution_value, 4)
            self.assertEqual(9, x.solution_value)

    def test_absexpr_value_if_unused(self):
        m = self.model
        x = m.integer_var(name='xv', lb=1, ub=9)
        dist_to_five = m.abs(x - 5)
        m.maximize(x)
        if m._can_solve():
            m.solve()
            # dist_to_five has a solution value vene if not used.
            self.assertEqual(dist_to_five.solution_value, 4)

    def test_abs_expr_linear_lhs(self):
        m = self.model
        x = m.integer_var(name='xv', lb=1, ub=9)
        dist_to_five = m.abs(x - 5)
        act = m.add(dist_to_five >= 1)
        self.assertEqual('abs(xv-5) >= 1', str(act))

    @staticmethod
    def _try_absexpr(mdl, e):
        mdl.abs(e)

    def test_absexpr_none(self):
        self.assertRaises(DOcplexException, self._try_absexpr, self.model, None)

    def test_absexpr_from_str(self):
        self.assertRaises(DOcplexException, self._try_absexpr, self.model, "fooo")

    def test_absexpr_list(self):
        m = self.model
        xs = m.integer_var_list(10, name='xxx')
        six.assertRaisesRegex(self, DOcplexException, "abs", self._try_absexpr, m, xs)

    def test_absexpr_iter(self):
        m = self.model
        xs = m.integer_var_list(10, name='xxx')
        six.assertRaisesRegex(self, DOcplexException, "abs", self._try_absexpr, m, iter(xs))

    def test_absexpr_comp(self):
        m = self.model
        xs = m.integer_var_list(10, name='xxx')
        six.assertRaisesRegex(self, DOcplexException, "abs", self._try_absexpr, m, (xs[i] for i in range(5)))

    def test_absexpr_dict(self):
        m = self.model
        xd = m.integer_var_dict(10, name='xxx')
        six.assertRaisesRegex(self, DOcplexException, "abs", self._try_absexpr, m, xd)

    def test_abs_expr_matrix(self):
        m = self.model
        xm = m.continuous_var_matrix(keys1=range(5), keys2=range(3), name='xxx')
        six.assertRaisesRegex(self, DOcplexException, "abs", self._try_absexpr, m, xm)

    # ----- min
    def test_min_emptyset(self):
        m = self.model
        self.assertEqual(m.infinity, m.min())

    # @unittest.skip("min^2 has to wait for quadratic factory")
    def test_min_square(self):
        m = self.model
        xs = m.integer_var_list(2, lb=-21, ub=20, name=["a", "b"])
        z = m.min(xs)
        z2 = z ** 2
        self.assertEqual("_min4^2", str(z2))

    def test_min_one_expr(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        emm = m.min(x1 - y1)
        e = emm.to_linear_expr()
        self.assertIsInstance(e, Expr)
        # self.assertEqual("min(ix-iy)", str(e))
        m.maximize(e)
        if m._can_solve():
            min_sol = m.solve()
            self.assertIsNotNone(min_sol)
            self.assertEqual(min_sol[x1], 13)
            self.assertEqual(min_sol[y1], 3)

    def test_min_two_exprs(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        expr_min = m.min(x1, y1)
        m.minimize(2 * x1 + y1)
        if m._can_solve():
            sol = m.solve()
            self.assertIsNotNone(sol)
            self.assertEqual(1, expr_min.solution_value)
            # sol.display()

    def test_min_clone(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        min1 = m.min(x1, y1, x1 + y1)
        min2 = min1.clone()
        self.assertIsNot(min1, min2)
        self.assertIsInstance(min2, MinimumExpr)
        exprs1 = list(min1.iter_exprs())
        exprs2 = list(min2.iter_exprs())
        self.assertEqual(len(exprs1), len(exprs2))
        for e in range(len(exprs1)):
            self.assertIs(exprs1[e], exprs2[e])

    def test_min_repr(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        emin = m.min(x1, y1, x1 + y1)
        self.assertEqual("docplex.mp.MinExpr(ix,iy,ix+iy)", repr(emin))

    def test_min_value_unused(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        emin = m.min(x1, y1, x1 + y1)
        m.minimize(x1 + y1)
        if m._can_solve():
            m.solve()
            self.assertEqual(emin.solution_value, 1)

    def test_min_sequence(self):
        mdl = Model("max_in_obj")
        xs = mdl.integer_var_list(2, ub=20, name="x")
        min_xs_list = mdl.min(xs)
        self.assertIsInstance(min_xs_list, Expr)

    def test_min_iter(self):
        mdl = Model("max_in_obj")
        xs = mdl.integer_var_list(2, ub=20, name="x")
        xs_it = iter(xs)
        min_xs_list = mdl.min(xs_it)
        self.assertIsInstance(min_xs_list, Expr)

    def test_min_as_objective(self):
        mdl = Model("max_in_obj")
        x = mdl.integer_var_list(2, ub=20, name="x")
        obj = mdl.min(*x)
        mdl.maximize(obj)
        mdl.add_constraint(mdl.sum(x) == 30)
        mdl.add_constraint(x[0] == 2 * x[1])  # make sure there is a unique solution
        if mdl._can_solve():
            mdl.solve()
            with RedirectedOutputToStringContext() as out:
                mdl.print_solution()
            sol_out = out.get_str()
            self.assertIn("objective: 10\n", sol_out)
            self.assertIn("x_0=20", sol_out)
            self.assertIn("x_1=10", sol_out)

    # --- below test argument checking
    @staticmethod
    def _try_min(mdl, *args):
        return mdl.min(*args)

    def test_min_none(self):
        self.assertRaises(DOcplexException, self._try_min, self.model, None)

    def test_min_str(self):
        self.assertRaises(DOcplexException, self._try_min, self.model, "fooo")

    def test_min_bad_list(self):
        args = ["foo", "bar"]
        self.assertRaises(DOcplexException, self._try_min, self.model, args)

    def test_min_mixed_list(self):
        # send a list with: a var, an expr, a monomial, and a num
        m = self.model
        x = m.integer_var(name="xx", lb=1, ub=13)
        y = m.integer_var(name="yy", lb=3, ub=11)
        args = [x, x + y + 3, 4 * y, 7]
        expr = m.min(args)
        self.assertEqual("min(xx,xx+yy+3,4yy,7)", str(expr))

    def test_min_non_unique_list(self):
        m = self.model
        x = m.integer_var(name="ix", lb=1, ub=13)
        y = m.integer_var(name="iy", lb=3, ub=11)
        self.assertRaises(DOcplexException, self._try_min, m, [x, y], 1)

    def test_min_bad_list1(self):
        l = ["foo", "bar"]
        self.assertRaises(DOcplexException, self._try_min, self.model, l)

    def test_min_bad_list2(self):
        m = self.model
        x = m.integer_var(name="ix", lb=1, ub=13)
        y = m.integer_var(name="iy", lb=3, ub=11)
        self.assertRaises(DOcplexException, self._try_min, m, x, y, "foo")

    def test_min_emptylist(self):
        m = self.model
        self.assertEqual(m.infinity, m.min([]))

    def test_min_emptydict(self):
        m = self.model
        self.assertEqual(m.infinity, m.min({}))

    def test_min_vardict(self):
        m = self.model
        vardict = m.integer_var_dict(3, name='z', lb=7)
        e = m.min(vardict)
        self.assertEqual("min(z_0,z_1,z_2)", str(e))

    def test_min_div_coef(self):
        m = self.model
        vardict = m.integer_var_list(3, name=['a', 'b', 'c'], lb=7)
        e = m.min(vardict)
        z = e / 0.8
        self.assertIsInstance(z, Expr)
        self.assertEqual(str(z), "1.250_min5")

    def test_min_no_generated_vars_in_print_solution(self):
        mdl = self.model
        x = mdl.integer_var_list(10, ub=20, name="x#")
        mdl.minimize(mdl.min(x))
        mdl.add_constraint(mdl.sum(x) == 30)
        sol = mdl.solve()
        self.assertIsNotNone(sol)
        with RedirectedOutputToStringContext() as out:
            mdl.print_solution()
        self.assertNotIn("_x", out.get_str())

    def test_abs_min(self):
        m = Model("max_in_obj")
        xs = m.integer_var_list(2, lb=-21, ub=20, name="x#")
        min_xsl = m.min(xs)
        absmin = m.abs(min_xsl)
        m.maximize(absmin)
        if m._can_solve():
            s = m.solve()
            self.assertIsNotNone(s)
            self.assertEqual(absmin.solution_value, 21)

    def test_min_one_num(self):
        m = Model("min_one_num")
        m17 = m.min(17)
        self.assertEqual(17, m17)

    # @unittest.skip('rtc-39075')
    def test_min_3_num(self):
        m = Model("min_one_num")
        m17 = m.min(32, 17, 43)
        self.assertEqual(17, m17)

    def test_min_infeasible(self):
        mdl = self.model
        xs = mdl.integer_var_list(3, lb=11, ub=20, name="x#")
        z = mdl.integer_var(name='z', lb=33)
        mdl.add(z == mdl.min(xs))
        s = mdl.solve()
        self.assertIsNone(s)
        self.assertIn('infeasible', mdl.solve_details.status)

    # --- max
    def test_max_emptyset(self):
        m = self.model
        self.assertEqual(-m.infinity, m.max())

    def test_max_one_num(self):
        m = Model("min_one_num")
        m17 = m.max(17)
        self.assertEqual(17, m17)

    # @unittest.skip('rtc-39075')
    def test_max_3_num(self):
        m = Model("max_three_num")
        m43 = m.max(17, 23, 43)
        self.assertEqual(43, m43)

    # @unittest.skip("max^2 has to wait for quadratic factory")
    def test_max_square(self):
        m = self.model
        xs = m.integer_var_list(2, lb=-21, ub=20, name=["a", "b"])
        z = m.max(xs)
        z2 = z ** 2
        self.assertEqual("_max4^2", str(z2))

    def test_max_one_expr(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        emm = m.max(x1 - y1)
        self.assertTrue(emm.is_discrete())
        e = emm.to_linear_expr()
        self.assertIsInstance(e, Expr)
        self.assertEqual("ix-iy", str(e))
        m.minimize(e)
        if m._can_solve():
            max_sol = m.solve()  # expected result is -10
            # min_sol.display()
            self.assertIsNotNone(max_sol)
            self.assertEqual(max_sol[x1], 1)
            self.assertEqual(max_sol[y1], 11)

    # @unittest.skip("wait for solve of RTC 29404")
    def test_max_discrete(self):
        # RTC29404
        m = self.model
        x = m.integer_var_list(keys=20, lb=0, ub=20, name="x#")
        xmax = m.max(*x)
        m.maximize(xmax)
        self.assertTrue(m.objective_expr.is_discrete())

    def test_max_two_exprs(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        expr_max = m.max(x1, y1)
        # max(x,y) <= 7
        m.add_constraint(expr_max <= 7)
        # we maximize with preference to x1, so x1=7, y1=7, obj=21
        m.maximize(2 * x1 + y1)
        if m._can_solve():
            sol = m.solve()
            # sol.display()
            self.assertIsNotNone(sol)
            self.assertEqual(21, sol.objective_value)
            self.assertAlmostEqual(7, expr_max.solution_value, delta=1e-6)
            self.assertAlmostEqual(7, sol[x1], delta=1e-6)
            self.assertAlmostEqual(7, sol[y1], delta=1e-6)

    def test_max_clone(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        max1 = m.max(x1, y1, x1 + y1)
        max2 = max1.clone()
        self.assertIsNot(max1, max2)
        self.assertIsInstance(max2, MaximumExpr)
        exprs1 = list(max1.iter_exprs())
        exprs2 = list(max2.iter_exprs())
        self.assertEqual(len(exprs1), len(exprs2))
        for e in range(len(exprs1)):
            self.assertIs(exprs1[e], exprs2[e])

    def test_max_repr(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        emax = m.max(x1, y1, x1 + y1)
        self.assertEqual("docplex.mp.MaxExpr(ix,iy,ix+iy)", repr(emax))

    def test_max_in_ct(self):
        mdl = Model("test")
        xs = mdl.binary_var_list(keys=2, name='binVar#')
        z = mdl.binary_var(name='z')
        max_xs = mdl.max(*xs)
        mdl.add_constraint(z <= max_xs)
        mdl.add_constraint(max_xs <= 2 * z)
        mdl.maximize(z)
        if mdl._can_solve():
            mdl.solve()
            self.assertAlmostEqual(1, mdl.objective_value, delta=1e-4)
            self.assertAlmostEqual(1, max_xs.solution_value, delta=1e-6)

    def test_max_as_objective(self):
        mdl = Model("max_in_obj")
        x = mdl.integer_var_list(2, ub=20, name="x")
        obj = mdl.max(*x)
        mdl.maximize(obj)
        mdl.add_constraint(mdl.sum(x) == 30)
        mdl.add_constraint(x[0] == 2 * x[1])  # make sure there is a unique solution
        if mdl._can_solve():
            mdl.solve()
            with RedirectedOutputToStringContext() as out:
                mdl.print_solution()
            sol_out = out.get_str()
            self.assertIn("objective: 20\n", sol_out)
            self.assertIn("x_0=20", sol_out)
            self.assertIn("x_1=10", sol_out)

    def test_max_div_coef(self):
        m = self.model
        vardict = m.integer_var_list(3, name=['a', 'b', 'c'], lb=7)
        e = m.max(vardict)
        z = e / 0.8
        self.assertIsInstance(z, Expr)
        self.assertEqual(str(z), "1.250_max5")

    def test_max_two_identical_maxs(self):
        with Model("max_in_obj", name_functional_vars=True) as mdl:
            xs = mdl.integer_var_list(2, ub=20, name="x")
            max_xs1 = mdl.max(*xs)
            max_xs2 = mdl.max(*xs)
            # force resolution
            mdl.add_constraint(max_xs1 <= 3)
            mdl.add_constraint(max_xs2 <= 4)
            lps = mdl.lp_string
            self.assertIn('_max2', lps)
            self.assertIn('_max5', lps)


    def test_abs_two_identical_abs(self):
        with Model("max_in_obj", name_functional_vars=True) as mdl:
            xs = mdl.integer_var_list(2, ub=20, name="x")
            abs_xs1 = mdl.abs(mdl.sum(xs) - 1)
            abs_xs2 = mdl.abs(mdl.sum(xs) - 1)
            # force resolution
            mdl.add_constraint(abs_xs1 <= 3)
            mdl.add_constraint(abs_xs2 <= 4)
            lps = mdl.lp_string
            self.assertIn('_abs', lps)

    def test_max_value_unused(self):
        m = self.model
        x1 = m.integer_var(name="ix", lb=1, ub=13)
        y1 = m.integer_var(name="iy", lb=3, ub=11)
        emax = m.max(x1, y1, x1 + y1)
        m.maximize(x1 + y1)
        if m._can_solve():
            s = m.solve()
            # s.display()
            self.assertEqual(emax.solution_value, 24)  # max is 13+11 here

    def test_max_sequence(self):
        mdl = self.model
        xs = mdl.integer_var_list(2, ub=20, name="x")
        max_xs_list = mdl.max(xs)
        self.assertIsInstance(max_xs_list, Expr)

    def test_max_var_set(self):
        mdl = self.model
        xl = mdl.integer_var_list(2, ub=20, name="x")
        xs = {x for x in xl}
        max_x_set = mdl.max(xs)
        mdl.minimize(max_x_set)
        if mdl._can_solve():
            s = mdl.solve()
            for v in xl:
                self.assertEqual(0, s[v])

    def test_max_iter(self):
        mdl = Model("max_in_obj")
        xs = mdl.integer_var_list(2, ub=20, name="x")
        xs_it = iter(xs)
        max_xs_list = mdl.max(xs_it)
        self.assertIsInstance(max_xs_list, Expr)

    # --- below test argument checking
    @staticmethod
    def _try_max(mdl, *args):
        return mdl.max(*args)

    def test_max_none(self):
        self.assertRaises(DOcplexException, self._try_max, self.model, None)

    def test_max_str(self):
        self.assertRaises(DOcplexException, self._try_max, self.model, "fooo")

    def test_max_bad_list(self):
        l = ["foo", "bar"]
        six.assertRaisesRegex(self, DOcplexException, "Model.max", self._try_max, self.model, l)

    def test_max_mixed_list(self):
        # send a list with: a var, an expr, a monomial, and a num
        m = self.model
        x = m.integer_var(name="xx", lb=1, ub=13)
        y = m.integer_var(name="yy", lb=3, ub=11)
        l = [x, x + y + 3, 4 * y, 7]
        expr = m.max(l)
        self.assertEqual("max(xx,xx+yy+3,4yy,7)", str(expr))

    def test_max_non_unique_list(self):
        m = self.model
        x = m.integer_var(name="ix", lb=1, ub=13)
        y = m.integer_var(name="iy", lb=3, ub=11)
        six.assertRaisesRegex(self, DOcplexException, "Model.max", self._try_max, m, [x, y], 1)

    def test_max_bad_list2(self):
        m = self.model
        x = m.integer_var(name="ix", lb=1, ub=13)
        y = m.integer_var(name="iy", lb=3, ub=11)
        six.assertRaisesRegex(self, DOcplexException, "Model.max", self._try_max, m, x, y, "foo")

    def test_max_set_strings(self):
        m = self.model
        # m.max( {"foo", "bar"})
        self.assertRaises(DOcplexException, self._try_max, m, {"foo", "bar"})

    def test_max_vardict(self):
        m = self.model
        vardict = m.binary_var_dict(3, name='z')
        e = m.max(vardict)
        self.assertEqual("max(z_0,z_1,z_2)", str(e))
        # max_vars = set(e.iter_variables())
        # self.assertEqual(4, len(max_vars))

    def test_abs_max(self):
        m = Model("max_in_obj")
        xs = m.integer_var_list(2, lb=-20, ub=19, name="x#")
        max_xsl = m.max(xs)
        absmax = m.abs(max_xsl)
        m.maximize(absmax)
        if m._can_solve():
            s = m.solve()
            self.assertIsNotNone(s)
            self.assertEqual(20, abs(max(x.solution_value for x in xs)))

    def test_max_abs(self):
        m = Model("max_in_obj")
        xs = m.integer_var_list(2, lb=-20, ub=19, name="x#")
        max_abs = m.max(m.abs(x) for x in xs)
        m.maximize(max_abs)
        if m._can_solve():
            s = m.solve()
            self.assertIsNotNone(s)
            self.assertEqual(20, max_abs.solution_value)

    def test_max_emptylist(self):
        m = self.model
        self.assertEqual(-m.infinity, m.max([]))

    def test_max_emptydict(self):
        m = self.model
        self.assertEqual(-m.infinity, m.max({}))

    def test_max_no_generated_vars_in_print_solution(self):
        mdl = self.model
        x = mdl.integer_var_list(10, ub=20, name="x#")
        mdl.minimize(mdl.max(x))
        mdl.add_constraint(mdl.sum(x) == 30)
        sol = mdl.solve()
        self.assertIsNotNone(sol)
        with RedirectedOutputToStringContext() as out:
            mdl.print_solution()
        self.assertNotIn("_x", out.get_str())

    def test_max_infeasible(self):
        mdl = self.model
        xs = mdl.integer_var_list(3, lb=11, ub=20, name="x#")
        z = mdl.integer_var(name='z', ub=9)
        mdl.add(z == mdl.max(xs))
        s = mdl.solve()
        self.assertIsNone(s)
        self.assertIn('infeasible', mdl.solve_details.status)

    # ----

    def test_abs_no_switch_to_hidden_names(self):
        # check that LP printer does not switch to hidden names for all names
        # if the bas expr is itself noncompliant (e.g. has a '+') in it.
        mdl = self.model
        user_varname = 'zorglub'
        x = mdl.integer_var_list(3, ub=20, name=user_varname)
        mdl.add_constraint(mdl.sum(x) == 1)
        mdl.minimize(mdl.abs(x[0] + mdl.max(x) + 3 * mdl.min(x)))
        lps = mdl.lp_string
        self.assertIn(user_varname, lps)
        # print(lps)

    def test_model_sum_abs_min_max(self):
        mdl = self.model
        xs = mdl.integer_var_list(3, lb=[-11, -12, -13], ub=[11, 12, 13], name=["a", "b", "c"])
        xsum = mdl.sum(xs)
        xabs = mdl.abs(xsum)
        xmin = mdl.min(xs)
        xmax = mdl.max(xs)
        bigsum = mdl.sum([xabs, xmin, xmax])
        mdl.maximize(bigsum)
        if mdl._can_solve():
            s = mdl.solve()
            self.assertIsNotNone(s)
            # a=11, b=12,c=13
            #   -> max is 13, min is 11, abs(sum) is 36: total is 60
            # s.display()
            self.assertEqual(60, s.objective_value)


class LogicalExprTests(unittest.TestCase):

    def setUp(self):
        self.model = Model(name='logicals')
        self.a = self.model.binary_var(name='a')
        self.b = self.model.binary_var(name='b')
        self.c = self.model.binary_var(name='c')
        self.z = self.model.binary_var(name='z')

    def tearDown(self):
        self.model.end()
        self.model = None

    def test_and_empty(self):
        and0 = self.model.logical_and()
        self.assertEqual('1', str(and0))

    def test_and_one_var(self):
        m = self.model
        and1 = m.logical_and(self.a)
        self.assertIs(and1, self.a)

    def test_and_two_free_vars(self):
        m = self.model
        m.add(1 == m.logical_and(self.a, self.b))
        m.minimize(self.a + self.b)

        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(1, self.a.solution_value)
        self.assertEqual(1, self.b.solution_value)

    def test_and_bound_vars(self):
        m = self.model
        z = self.z
        m.add(z == m.logical_and(self.a, self.b, self.c))
        m.add(self.b == 0)
        m.minimize(self.a + self.b)
        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(0, z.solution_value)

    def check_truth_table(self, xa, xb, xc):
        s = self.model.solve()
        self.assertIsNotNone(s)
        self.assertEqual(xa, self.a.solution_value, "wrong value for a")
        self.assertEqual(xb, self.b.solution_value, "wrong value for b")
        self.assertEqual(xc, self.c.solution_value, "wrong value for c")

    def test_and_from_rhs_00(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(a == 0)
        m.add(b == 0)
        self.check_truth_table(0, 0, 0)

    def test_and_from_rhs_01(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(a == 0)
        m.add(b == 1)
        self.check_truth_table(0, 1, 0)

    def test_and_from_rhs_10(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(a == 1)
        m.add(b == 0)
        self.check_truth_table(1, 0, 0)

    def test_and_from_rhs_11(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(a == 1)
        m.add(b == 1)
        self.check_truth_table(1, 1, 1)

    def test_and_from_lhs1_free(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(c == 1)
        self.check_truth_table(1, 1, 1)

    def test_and_from_lhs1_rhs10_ko(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(c == 1)
        m.add(a == 0)
        self.assertIsNone(m.solve())

    def test_and_from_lhs1_rhs01_ko(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(c == 1)
        m.add(b == 0)
        self.assertIsNone(m.solve())

    def test_and_from_lhs0_free(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(c == 0)
        self.assertIsNotNone(m.solve())
        self.assertTrue(a.solution_value + b.solution_value <= 1)

    def test_and_from_lhs0_rhs10_ok(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(c == 0)
        m.add(a == 0)
        m.add(b == 1)
        self.assertIsNotNone(m.solve())

    def test_and_from_lhs0_rhs01_ko(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_and(a, b))
        m.add(c == 0)
        m.add(b == 0)
        m.add(a == 1)
        self.assertIsNotNone(m.solve())

    def test_and_from_lhs0_multiple(self):
        m = self.model
        size = 17
        bs = m.binary_var_list(size, name='bb')
        z = self.z
        m.add(z == m.logical_and(*bs))
        m.add(z == 0)
        for i in range(size - 1):
            m.add(bs[i] == 1)
        # last one must be 0
        self.assertIsNotNone(m.solve())
        self.assertEqual(0, bs[size - 1].solution_value)
        # m.print_solution(print_zeros=True)

    def test_and_from_lhs1_multiple(self):
        m = self.model
        size = 17
        bs = m.binary_var_list(size, name='bb')
        z = self.z
        m.add(z == m.logical_and(*bs))
        m.add(z == 1)
        self.assertIsNotNone(m.solve())
        for i in range(size):
            self.assertEqual(1, bs[i].solution_value)

    def test_or_empty(self):
        or0 = self.model.logical_or()
        self.assertEqual('0', str(or0))

    def test_or_one_var(self):
        m = self.model
        or1 = m.logical_or(self.a)
        self.assertIs(or1, self.a)

    def test_or_continuous_ko(self):
        m = self.model
        xx = m.continuous_var(name='xx')
        six.assertRaisesRegex(self, DOcplexException,
                              'Not a logical operand: docplex.mp.Var\(type=C',
                              lambda m_: m_.logical_or(self.a, xx), m)

    def test_or_float_ko(self):
        m = self.model
        six.assertRaisesRegex(self, DOcplexException,
                              'Not a logical operand: 3.14',
                              lambda m_: m_.logical_or(self.a, 3.14), m)

    def test_or_two_free_vars(self):
        m = self.model
        c_or = (m.logical_or(self.a, self.b) >= 1)
        m.minimize(self.a + self.b)

        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(0, s.objective_value)
        m.add(c_or)
        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(1, s.objective_value)

    def test_or_non_free_vars(self):
        m = self.model
        z = self.z
        m.add(z == m.logical_or(self.a, self.b, self.c))
        m.maximize(z)

        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(1, s[z])
        m.add(self.a + self.b + self.c == 0)

        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(0, s[z])

    def test_or_from_rhs_00(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(a == 0)
        m.add(b == 0)
        self.check_truth_table(0, 0, 0)

    def test_or_from_rhs_10(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(a == 1)
        m.add(b == 0)
        self.check_truth_table(1, 0, 1)

    def test_or_from_rhs_01(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(a == 0)
        m.add(b == 1)
        self.check_truth_table(0, 1, 1)

    def test_or_from_rhs_11(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(a == 1)
        m.add(b == 1)
        self.check_truth_table(1, 1, 1)

    def test_or_from_lhs1_free(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(c == 1)
        self.assertIsNotNone(m.solve())
        self.assertTrue(a.solution_value + b.solution_value >= 1)

    def test_or_from_lhs1_rhs00_ko(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(c == 1)
        m.add(a == 0)
        m.add(b == 0)
        self.assertIsNone(m.solve())

    def test_or_from_lhs1_rhs01_ok(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(c == 1)
        m.add(a == 0)
        m.add(b == 1)
        self.assertIsNotNone(m.solve())

    def test_or_from_lhs1_rhs10_ok(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(c == 1)
        m.add(a == 1)
        m.add(b == 0)
        self.assertIsNotNone(m.solve())

    def test_or_from_lhs0_free(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(c == 0)
        self.assertIsNotNone(m.solve())
        self.assertTrue(a.solution_value + b.solution_value <= 0)

    def test_and_from_lhs0_rhs10_ko(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(c == 0)
        m.add(a == 0)
        m.add(b == 1)
        self.assertIsNone(m.solve())

    def test_or_from_lhs0_rhs01_ko(self):
        m = self.model
        a, b, c = self.a, self.b, self.c
        m.add(c == m.logical_or(a, b))
        m.add(c == 0)
        m.add(b == 0)
        m.add(a == 1)
        self.assertIsNone(m.solve())


class NestedLogicalExprTests(unittest.TestCase):

    def setUp(self):
        self.model = Model(name='logicals')
        self.a = self.model.binary_var(name='a')
        self.b = self.model.binary_var(name='b')
        self.c = self.model.binary_var(name='c')
        self.z = self.model.binary_var(name='z')

    def tearDown(self):
        self.model.end()
        self.model = None

    @staticmethod
    def set_binary(bv, bb):
        bv._reset_bounds()
        if bb == 1:
            bv.lb = 1
        elif bb == 0:
            bv.ub = 0
        else:
            print('unexpected binary value: {0} for variable {1}'.format(bb, bv.name))

    def check_truth_table4(self, aa, bb, cc, xz):
        m = self.model
        for dv, dvv in zip([self.a, self.b, self.c], [aa, bb, cc]):
            self.set_binary(dv, dvv)
        s = m.solve()
        self.assertIsNotNone(s)
        zv = self.z
        self.assertEqual(xz, zv.solution_value, "wrong value for z")
        # now set z to opposite
        self.set_binary(zv, 1 - xz)
        self.assertIsNone(m.solve())
        zv._reset_bounds()

    def check_truth_table3(self, aa, bb, xz):
        m = self.model
        for dv, dvv in zip([self.a, self.b], [aa, bb]):
            self.set_binary(dv, dvv)
        s = m.solve()
        self.assertIsNotNone(s)
        zv = self.z
        self.assertEqual(xz, zv.solution_value, "wrong value for z")
        # now set z to opposite
        self.set_binary(zv, 1 - xz)
        self.assertIsNone(m.solve())
        zv._reset_bounds()

    def check_truth_table_impossible(self, aa, bb, cc, zz):
        m = self.model
        for dv, dvv in zip([self.a, self.b, self.c, self.z], [aa, bb, cc, zz]):
            dv._reset_bounds()
            if dvv is None:
                pass
            elif dvv == 1:
                dv.lb = 1
            elif dvv == 0:
                dv.ub = 0
            else:
                print('unexpected dvv: {0}'.format(dvv))
        s = m.solve()
        self.assertIsNone(s)

    def test_and_of_or_str(self):
        m = self.model
        a, b, c, z = self.a, self.b, self.c, self.z
        and_of_or = m.logical_and(a, m.logical_or(b, c))
        self.assertEqual('and(a,or(b,c))', str(and_of_or))

    def test_and_of_or(self):
        m = self.model
        a, b, c, z = self.a, self.b, self.c, self.z
        m.add(self.z == m.logical_and(a, m.logical_or(b, c)))
        self.check_truth_table4(0, 0, 0, 0)
        self.check_truth_table4(0, 0, 1, 0)
        self.check_truth_table4(0, 1, 0, 0)
        self.check_truth_table4(0, 1, 1, 0)
        self.check_truth_table4(1, 0, 0, 0)
        self.check_truth_table4(1, 0, 1, 1)
        self.check_truth_table4(1, 1, 0, 1)
        self.check_truth_table4(1, 1, 1, 1)

    def test_or_of_or(self):
        m = self.model
        a, b, c, z = self.a, self.b, self.c, self.z
        m.add(self.z == m.logical_or(a, m.logical_or(b, c)))
        self.check_truth_table4(0, 0, 0, 0)
        self.check_truth_table4(0, 0, 1, 1)
        self.check_truth_table4(0, 1, 0, 1)
        self.check_truth_table4(0, 1, 1, 1)
        self.check_truth_table4(1, 0, 0, 1)
        self.check_truth_table4(1, 0, 1, 1)
        self.check_truth_table4(1, 1, 0, 1)
        self.check_truth_table4(1, 1, 1, 1)

    def test_logand_var_constr(self):
        m = self.model
        a = self.a
        z = self.z
        ij = m.integer_var(name='ij')
        c5 = (ij == 5)
        c5.name = 'cx5'
        m.add(self.z == m.logical_and(a, c5))
        #(m.lp_string)
        m.add(ij >= 6)
        m.maximize(a)
        self.assertIsNotNone(m.solve())
        self.assertEqual(1, a.solution_value)
        self.assertEqual(0, z.solution_value)

    def test_logand_var_constr_non_discrete_ko(self):
        m = self.model
        a = self.a
        ij = m.integer_var(name='ij')
        c_pi = (ij == 3.14)
        six.assertRaisesRegex(self, DOcplexException,
                              "Model.logical_and, arg#1: Not a logical operand",
                              lambda m_: m_.add(self.z == m_.logical_and(a, c_pi)), m)

    def test_logical_not_num_ko1(self):
        six.assertRaisesRegex(self, DOcplexException,
                              "Model.logical.not: Not a logical operand: 3.14",
                              lambda m_: m_.logical_not(3.14), self.model)

    def test_logical_not_contvar_ko2(self):
        m = self.model
        zz = m.continuous_var(name='zz')
        six.assertRaisesRegex(self, DOcplexException,
                              "Model.logical.not: Not a logical operand: zz",
                              lambda m_: m_.logical_not(zz), self.model)

    def test_logical_not_nondiscrete_ct_ko3(self):
        m = self.model
        zz = m.continuous_var(name='zz')
        six.assertRaisesRegex(self, DOcplexException,
                              "Model.logical.not: Not a logical operand: zz",
                              lambda m_: m_.logical_not(zz == 3), self.model)

    def test_logical_not_bvar(self):
        m = self.model
        b1 = m.binary_var(name='b1')
        b2 = m.binary_var(name='b2')
        m.add(b2 == m.logical_not(b1))
        m.minimize(b1)
        self.assertIsNotNone(m.solve())
        self.assertEqual(0, b1.solution_value)
        self.assertEqual(1, b2.solution_value)
        lps = m.lp_string
        self.assertIn('b1 + _not6 = 1', lps)

    def test_logical_not_logand(self):
        # not all bs can be equal to 1, we maximize, all but one set to 1
        m = self.model
        size = 7
        bs = m.binary_var_list(size, name='bb')
        # and is false
        m.add(m.logical_not(m.logical_and(*bs)) == 1)
        m.maximize(m.sum(bs))
        s1 = m.solve()
        self.assertEqual(size - 1, m.objective_value)
        # s1.display()



    def test_logical_not_clone(self):
        m = self.model
        b1 = m.binary_var(name='b1')
        not_b1 = m.logical_not(b1)
        not_b2 = not_b1.clone()
        self.assertIsInstance(not_b2, LogicalNotExpr)
        self.assertIsNot(not_b1, not_b2)  # different
        self.assertEqual(str(not_b1), str(not_b2))

    def test_logical_not_str(self):
        m = self.model
        b1 = m.binary_var(name='b1')
        b2 = m.binary_var(name='b2')
        not_and = m.logical_not(m.logical_and(b1, b2))
        self.assertEqual("not(and(b1,b2))", str(not_and))

    def test_logical_not_repr(self):
        m = self.model
        b1 = m.binary_var(name='b1')
        not_b1 = m.logical_not(b1)
        self.assertEqual("docplex.mp.NotExpr(b1)", repr(not_b1))

    def test_logical_not_name(self):
        with Model('not1') as m:
            b1 = m.binary_var(name='b1')
            not_b1 = m.logical_not(b1)
            self.assertEqual("_not1", not_b1._get_resolved_f_var().name)

    def test_logxor_truth_tables(self):
        # an implementation of XOR and XOR(b1,b2) = OR(b1,b2) && NOT(b1 && b2)
        def logxor(m, b1, b2):
            return m.logical_and(m.logical_or(b1, b2), m.logical_not(m.logical_and(b1, b2)))

        m = self.model
        a, b, z = self.a, self.b, self.z
        m.add(z == logxor(m, a, b))
        self.check_truth_table3(0, 0, 0)
        self.check_truth_table3(0, 1, 1)
        self.check_truth_table3(1, 0, 1)
        self.check_truth_table3(1, 1, 0)
        self.check_truth_table_impossible(0, 0, None, 1)
        self.check_truth_table_impossible(0, 1, None, 0)
        self.check_truth_table_impossible(1, 0, None, 0)
        self.check_truth_table_impossible(1, 1, None, 1)

    def test_logical_not_nondiscrete_ct_ko4(self):
        m = self.model
        zz = m.continuous_var(name='zz')
        six.assertRaisesRegex(self, DOcplexException,
                              "Model.logical.not: Not a logical operand: zz",
                              lambda m_: m_.logical_not(zz == 3), self.model)

    def test_logical_not_bvar2(self):
        m = self.model
        b1 = m.binary_var(name='b1')
        b2 = m.binary_var(name='b2')
        m.add(b2 == m.logical_not(b1))
        m.minimize(b1)
        self.assertIsNotNone(m.solve())
        self.assertEqual(0, b1.solution_value)
        self.assertEqual(1, b2.solution_value)
        lps = m.lp_string
        self.assertIn('b1 + _not6 = 1', lps)

    def test_logical_not_logand2(self):
        # not all bs can be equal to 1, we maximize, all but one set to 1
        m = self.model
        size = 7
        bs = m.binary_var_list(size, name='bb')
        # and is false
        m.add(m.logical_not(m.logical_and(*bs)) == 1)
        m.maximize(m.sum(bs))
        s1 = m.solve()
        self.assertEqual(size - 1, m.objective_value)
        # s1.display()

    def test_logical_not_logor(self):
        # not even one bs can be equal to 1, we maximize, all are zero
        m = self.model
        size = 7
        bs = m.binary_var_list(size, name='bb')
        # and is false
        m.add(m.logical_not(m.logical_or(*bs)) == 1)
        m.maximize(m.sum(bs))
        self.assertIsNotNone(m.solve())
        self.assertEqual(0, m.objective_value)


if __name__ == "__main__":
    unittest.main()
