import unittest

from docplex.mp.model import Model
from docplex.mp.environment import Environment
from docplex.mp.constr import AbstractConstraint

from testutils import ForceProceduralCplexContext

from docplex.mp.model_reader import ModelReader

from testutils import get_test_temp_dir

TEMP_PATH = get_test_temp_dir()

the_env = Environment()

try:
    m = Model()
    name_pwl_vars = m._name_functional_vars
except AttributeError:
    # old default
    name_pwl_vars = True


@unittest.skipUnless(the_env.has_cplex and the_env.cplex_version_as_tuple >= (12, 7, 0),
                     "Piecewise linear constraints require Cplex 12.7 or higher")
class PwlModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Model("pwl", full_obj=True)
        m = self.model
        self.x = m.continuous_var(name='x')
        self.y = m.continuous_var(name='y')
        self.z = m.continuous_var(name='z')

    def tearDown(self):
        self.model.end()

    def test_simple_pwl_step(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(-1, 0), (-1, 1)], 0, "step")
        mdl.add_constraint(self.y == pwl(self.x), ctname="c1")
        mdl.add_constraint(self.y >= 0.5)
        lps = mdl.lp_string
        # print(lps)
        s = mdl.solve()
        self.assertIsNotNone(s)
        self.assertEqual(self.x.solution_value, 0)
        self.assertEqual(self.y.solution_value, 1)
        expected_pwl_name = "_pwl_step#3" if name_pwl_vars else "x4"
        self.assertIn(f'c1: y - {expected_pwl_name} = 0', lps)
        self.assertIn('Pwl', lps)
        self.assertIn(f'pwl1: {expected_pwl_name} = x 0 (-1, 0) (-1, 1) 0', lps)
        # Check that a simple export-import round-trip does not raise exceptions

        xlp = mdl.export_as_lp(basename='wm1')
        if xlp:
            with ModelReader.read(xlp) as m:
                self.assertEqual(3, m.number_of_constraints)
                self.assertEqual(1, m.number_of_pwl_constraints)

    def test_simple_pwl_duplicate_name(self):
        # two pwls with same name???
        mdl = self.model
        pwl1 = mdl.piecewise(0, [(0, 0), (10, 10)], 0, "pwf")
        pwl2 = mdl.piecewise(0, [(0, 10), (10, 0)], 0, "pwf")
        mdl.add_constraint(self.y == pwl1(self.x))
        mdl.add_constraint(self.y == pwl2(self.x))
        lps = mdl.export_as_lp_string()
        s = mdl.solve()
        self.assertIsNotNone(s)
        expected_pw1_name = "_pwl_pwf#3" if name_pwl_vars else "x4"
        expected_pw2_name = "_pwl_pwf#4" if name_pwl_vars else "x5"
        self.assertIn(f'c1: y - {expected_pw1_name} = 0', lps)
        self.assertIn(f'c2: y - {expected_pw2_name} = 0', lps)
        self.assertIn(f'pwl1: {expected_pw1_name} = x 0 (0, 0) (10, 10) 0', lps)
        self.assertIn(f'pwl2: {expected_pw2_name} = x 0 (0, 10) (10, 0) 0', lps)

    def test_simple_pwl_crossing_slopes(self):
        mdl = self.model
        x = self.x
        y = self.y
        # f1 is increasing from 0 to 10
        pwl1 = mdl.piecewise(0, [(0, 0), (10, 10)], 0, "slope1")
        # f2 is decreasing from 10 to 0
        pwl2 = mdl.piecewise(0, [(0, 10), (10, 0)], 0, "slope2")
        mdl.add_constraint(y == pwl1(x))
        mdl.add_constraint(y == pwl2(x))
        lps = mdl.export_as_lp_string()
        s = mdl.solve()
        self.assertIsNotNone(s)
        self.assertAlmostEqual(5.0, s[x])
        self.assertAlmostEqual(5.0, s[y])
        expected_pw1_name = "_pwl_slope1#3" if name_pwl_vars else "x4"
        expected_pw2_name = "_pwl_slope2#4" if name_pwl_vars else "x5"
        self.assertIn(f'c1: y - {expected_pw1_name} = 0', lps)
        self.assertIn(f'c2: y - {expected_pw2_name} = 0', lps)
        self.assertIn(f'pwl1: {expected_pw1_name} = x 0 (0, 0) (10, 10) 0', lps)
        self.assertIn(f'pwl2: {expected_pw2_name} = x 0 (0, 10) (10, 0) 0', lps)

    def test_simple_pwl_expr_arg(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(0, -1), (10, 9)], 0, "slope_0_10")
        mdl.add_constraint(self.x == pwl(2 * self.x - 5))

        mdl.minimize(self.x)
        s = mdl.solve()
        self.assertIsNotNone(s)
        self.assertEqual(6, s[self.x])
        lps = mdl.export_as_lp_string()
        self.assertIn("c1: x4 - 2 x = -5", lps)
        expected_pwl_name = "_pwl_slope_0_10#4" if name_pwl_vars else "x5"
        self.assertIn(f"c2: x - {expected_pwl_name} = 0", lps)
        self.assertIn(f"pwl1: {expected_pwl_name} = x4 0 (0, -1) (10, 9) 0", lps)

    def test_lp_export_long_pwl(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(0, 10), (0, 9), (1, 9), (1, 8),
                                (2, 8), (2, 7), (3, 7), (3, 6),
                                (4, 6), (4, 5), (5, 5), (5, 4),
                                (6, 4), (6, 3), (7, 3), (7, 2),
                                (8, 2), (8, 1), (9, 1), (9, 0)], 0, "steps")

        mdl.add_constraint(self.y == pwl(self.x))
        mdl.add_constraint(self.y == self.x)
        mdl.solve()
        self.assertEqual(self.x.solution_value, 5)
        lps = mdl.lp_string
        mdl.export_as_lp(path=TEMP_PATH, basename="longpwl")
        lps_nobreaks = lps.replace("\n", "")
        lps_nobreaks = lps_nobreaks.replace("       ", " ")
        expected_pwl_varname = "_pwl_steps#3" if name_pwl_vars else "x4"
        self.assertIn(f'{expected_pwl_varname} = x 0 (0, 10) (0, 9) (1, 9) (1, 8) (2, 8) (2, 7) (3, 7)', lps_nobreaks)
        self.assertTrue('(3, 6) (4, 6) (4, 5) (5, 5) (5, 4) (6, 4) (6, 3) (7, 3) (7, 2) (8, 2)' in lps_nobreaks)
        self.assertTrue('(8, 1) (9, 1) (9, 0) 0' in lps_nobreaks)

    def test_pwl_call_num(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0, "slope")
        a = 7
        expr_a = pwl(a)

        mdl.add_constraint(self.y == expr_a)
        s = mdl.solve()
        self.assertIsNotNone(s)
        self.assertEqual(expr_a.solution_value, 17)
        self.assertEqual(self.y.solution_value, 17)

    def test_pwl_with_expr(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0, "slope")
        expr = pwl(self.y)
        mdl.add_constraint(self.y == self.x)
        mdl.add_constraint(self.x >= 4)
        mdl.minimize(expr)
        mdl.solve()
        self.assertEqual(expr._raw_solution_value(), 14)
        self.assertEqual(mdl.objective_value, 14)
        self.assertEqual(self.x.solution_value, 4)

    def test_pwl_get_expr_value(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0, "slope")
        expr = pwl(self.y)
        mdl.add_constraint(self.z == expr)
        mdl.add_constraint(self.y == self.x)
        mdl.add_constraint(self.x >= 4)
        mdl.minimize(self.z)
        mdl.solve()
        self.assertEqual(expr._raw_solution_value(), 14)
        self.assertEqual(mdl.objective_value, 14)
        self.assertEqual(self.x.solution_value, 4)

    def test_pwl_unnamed(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(-1, 0), (-1, 1)], 0)
        mdl.add_constraint(self.y == pwl(self.x))
        mdl.add_constraint(self.y >= 0.5)
        res = mdl.solve()
        self.assertTrue(res)

    def test_pwl_with_model_copy(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0, "slope")
        expr = pwl(self.y)
        mdl.add_constraint(self.z == expr)
        mdl.add_constraint(self.y == self.x)
        mdl.add_constraint(self.x >= 4)
        mdl.minimize(self.z)
        lp_as_str = mdl.export_as_lp_string()
        mdl_copy = mdl.clone(new_name='copy of mdl')
        mdl.end()
        lp_copy_as_str = mdl_copy.export_as_lp_string()
        res = mdl_copy.solve()
        x_copy = mdl_copy.get_var_by_name("x")
        z_copy = mdl_copy.get_var_by_name("z")
        self.assertEqual(lp_as_str.split("Minimize")[1], lp_copy_as_str.split("Minimize")[1])
        self.assertTrue(res)
        self.assertEqual(z_copy._raw_solution_value(), 14)
        self.assertEqual(mdl_copy.objective_value, 14)
        self.assertEqual(x_copy.solution_value, 4)

    def test_pwl_with_model_copy2(self):
        mdl = self.model
        pwl = mdl.piecewise_as_slopes([(0, 0), (1, 10)], 0, (0, 10), "slope")
        expr = pwl(self.y)
        mdl.add_constraint(self.z == expr)
        mdl.add_constraint(self.y == self.x)
        mdl.add_constraint(self.x >= 4)
        mdl.minimize(self.z)
        lp_as_str = mdl.export_as_lp_string()
        mdl_copy = mdl.clone(new_name='copy of mdl')
        mdl.end()
        lp_copy_as_str = mdl_copy.export_as_lp_string()
        res = mdl_copy.solve()
        x_copy = mdl_copy.get_var_by_name("x")
        z_copy = mdl_copy.get_var_by_name("z")
        self.assertEqual(lp_as_str.split("Minimize")[1], lp_copy_as_str.split("Minimize")[1])
        self.assertTrue(res)
        self.assertEqual(z_copy._raw_solution_value(), 14)
        self.assertEqual(mdl_copy.objective_value, 14)
        self.assertEqual(x_copy.solution_value, 4)

    def test_pwl_remove(self):
        mdl = self.model
        pwl1 = mdl.piecewise(0, [(0, 0), (10, 10)], 0, "slope1")
        pwl2 = mdl.piecewise(0, [(0, 10), (10, 0)], 0, "slope2")
        pwl_lin_ct_1 = mdl.add_constraint(self.y == pwl1(self.x))
        lin_ct_1 = mdl.add_constraint(self.x == self.z)
        pwl_lin_ct_2 = mdl.add_constraint(self.z == pwl2(self.x))

        self.assertEqual(pwl_lin_ct_1.index, 0)
        self.assertEqual(lin_ct_1.index, 1)
        self.assertEqual(pwl_lin_ct_2.index, 2)

        self.assertEqual(mdl.number_of_pwl_constraints, 2)
        self.assertEqual(mdl.number_of_linear_constraints, 3)
        self.assertEqual(mdl.number_of_constraints, 5)
        pwl_constraints = [ct for ct in mdl.iter_pwl_constraints()]

        pwl_ct_0 = pwl_constraints[0]
        expected_pwl_varname = "_pwl_slope1#3" if name_pwl_vars else "x4"
        self.assertEqual(
                         f"docplex.mp.PwlConstraint({expected_pwl_varname},(0, [(0, 0), (10, 10)], 0),x)", repr(pwl_ct_0))
        self.assertEqual(f"{expected_pwl_varname} == slope1(x)", str(pwl_ct_0), )

        # Remove first PwlConstraint (Note that the associated linear constraint: y == pwl1(x)._f_var is not removed)
        mdl.remove_constraint(pwl_ct_0)

        self.assertEqual(mdl.number_of_pwl_constraints, 1)
        self.assertEqual(mdl.number_of_linear_constraints, 3)
        self.assertEqual(mdl.number_of_constraints, 4)

        mdl.solve()
        self.assertEqual(self.x.solution_value, 5)
        self.assertEqual(self.z.solution_value, 5)

    def test_pwl_clear_all(self):
        mdl = self.model
        pwl1 = mdl.piecewise(0, [(0, 0), (10, 10)], 0, "slope1")
        pwl2 = mdl.piecewise(0, [(0, 10), (10, 0)], 0, "slope2")
        mdl.add_constraint(self.y == pwl1(self.x))
        mdl.add_constraint(self.x == self.z)
        mdl.add_constraint(self.z == pwl2(self.x))
        self.assertEqual(2, mdl.number_of_pwl_constraints)
        mdl.clear_constraints()
        self.assertEqual(0, mdl.number_of_pwl_constraints)
        cpx = mdl.get_cplex(do_raise=False)
        if cpx:
            self.assertEqual(0, cpx.pwl_constraints.get_num())

    def test_model_pwl_ct_with_number_arg(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(-1, 0), (-1, 1)], 0, name="step")
        mdl.add_constraint(self.x == pwl(2))
        res = mdl.solve()
        self.assertTrue(res)
        self.assertEqual(self.x.solution_value, 1)

    def test_model_create_pwl_ct_expr(self):
        mdl = self.model
        self.xy = self.y
        pwl = mdl.piecewise(0, [(-1, 0), (-1, 1)], 0, name="step")
        pwl_expr = pwl(self.x + self.y)
        mdl.add_constraint(self.z == pwl_expr)
        mdl.add_constraint(self.y >= 0.5)

        self.assertEqual(repr(pwl_expr), "docplex.mp.PwlExpr(pwl_step#, x+y)")

        pwl_ct = mdl.get_pwl_constraint_by_index(idx=0)
        self.assertIsInstance(pwl_ct, AbstractConstraint)

        pwl_ct_vars = set(pwl_ct.iter_variables())
        self.assertEqual({pwl_expr._x_var, pwl_expr.functional_var}, pwl_ct_vars)

        pwl_expr_vars = set(pwl_expr.iter_variables())
        self.assertEqual({self.x, self.y, pwl_expr.functional_var}, pwl_expr_vars)

        s = mdl.solve()
        self.assertIsNotNone(s)

    def check_piecewise_constraint(self):
        mdl = self.model
        x = self.x
        y = self.y
        pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0)
        mdl.add_piecewise_constraint(self.y, pwl, self.x, name='pwl1020')
        self.assertEqual(3, mdl.number_of_variables)
        self.assertEqual(0, mdl.number_of_generated_variables)
        lps = mdl.export_as_lp_string()
        self.assertIn('pwl1020: y = x 0 (0, 10) (10, 20) 0', lps)
        mdl.add(x == 7)
        mdl.maximize(x - y)
        s = mdl.solve()
        self.assertIsNotNone(s)
        self.assertEqual(s[x], 7)
        self.assertEqual(s[self.y], 17)

    def test_piecewise_constraint_procedural_true(self):
        with ForceProceduralCplexContext(True):
            self.check_piecewise_constraint()

    def test_piecewise_constraint_procedural_false(self):
        with ForceProceduralCplexContext(False):
            self.check_piecewise_constraint()


@unittest.skipUnless(the_env.has_cplex and the_env.cplex_version_as_tuple >= (12, 7, 0),
                     "Piecewise linear constraints require Cplex 12.7 or higher")
class PwlModelSamples(unittest.TestCase):
    def setUp(self):
        self.model = Model("pwl")

    def tearDown(self):
        self.model.end()

    def test_pwl_sample1(self):
        # OPL model ---------------------------------
        #
        # int n=2;
        # float objectiveforxequals0=300;
        # float breakpoint[1..n]=[100,200];
        # float slope[1..n+1]=[1,2,-3];
        # dvar int x;
        #
        # maximize piecewise(i in 1..n)
        # {slope[i] -> breakpoint[i]; slope[n+1]}(0,objectiveforxequals0) x;
        #
        # subject to
        # {
        #  true;
        # }
        #
        # OUTPUT:
        #   // solution (optimal) with objective 600
        #   x = 200;
        # -------------------------------------------
        mdl = self.model

        n = 2
        objectiveforxequals0 = 300
        pw_break = [100, 200]
        slope = [1, 2, -3]

        x = mdl.integer_var(lb=-mdl.infinity, name="x")

        pwl = mdl.piecewise_as_slopes([(slope[i], pw_break[i]) for i in range(0, n)], slope[n],
                                      (0, objectiveforxequals0))
        mdl.maximize(pwl(x))
        res = mdl.solve()
        self.assertTrue(res)
        self.assertEqual(x.solution_value, 200)
        self.assertEqual(mdl.objective_value, 600)

    def test_pwl_sample2(self):
        # OPL model ---------------------------------
        #
        # dvar float x;
        # dvar float signx;
        #
        # dvar float y;
        # dvar float signy;
        #
        # maximize signx-signy;
        # subject to {
        #    x == y;
        #    signx == piecewise{0->0; 2->0; 0}(1,1) x;
        #    signy == piecewise{0->0; 2->0; 0}(1,1) y;
        # }
        #
        # OUTPUT:
        #     // solution (optimal) with objective 2
        #     signx = 1;
        #     signy = -1;
        #     x = 0;
        #     y = 0;
        # -------------------------------------------
        mdl = self.model
        x = mdl.continuous_var(lb=-mdl.infinity, name="x")
        signx = mdl.continuous_var(lb=-mdl.infinity, name="signx")
        y = mdl.continuous_var(lb=-mdl.infinity, name="y")
        signy = mdl.continuous_var(lb=-mdl.infinity, name="signy")

        mdl.maximize(signx - signy)
        # Subject to:
        mdl.add_constraint(x == y)
        mdl.add_constraint(signx == mdl.piecewise_as_slopes([(0, 0), (2, 0)], 0, (1, 1), name="sign1")(x))
        mdl.add_constraint(signy == mdl.piecewise_as_slopes([(0, 0), (2, 0)], 0, (1, 1), name="sign2")(y))

        res = mdl.solve()

        # cplex = mdl._Model__engine
        # cplex._CplexEngine__cplex.write("sample2.lp", "lp")
        #
        # print(mdl.export_as_lp_string())
        # mdl.print_solution(print_zeros=True)
        self.assertTrue(res)
        self.assertEqual(x.solution_value, 0)
        self.assertEqual(signx.solution_value, 1)
        self.assertEqual(y.solution_value, 0)
        self.assertEqual(signy.solution_value, -1)
        self.assertEqual(mdl.objective_value, 2)

    def test_pwl_sample3(self):
        # OPL model ---------------------------------
        #
        # dvar float x;
        # dvar float signx;
        #
        # dvar float y;
        # dvar float signy;
        #
        # maximize x;
        # subject to {
        #    x == 2;
        #    signx == piecewise{0->0; 2->0; 0}(1,1) x;
        #    y == -2;
        #    signy == piecewise{0->0; 2->0; 0}(1,1) y;
        # }
        #
        # OUTPUT:
        #     // solution (optimal) with objective 2
        #     x = 2;
        #     signx = 1;
        #     y = -2;
        #     signy = -1;
        # -------------------------------------------
        mdl = self.model
        x = mdl.continuous_var(name="x")
        signx = mdl.continuous_var(name="signx")
        y = mdl.continuous_var(lb=-mdl.infinity, name="y")
        signy = mdl.continuous_var(lb=-mdl.infinity, name="signy")

        mdl.maximize(x)
        # Subject to:
        mdl.add_constraint(x == 2)
        mdl.add_constraint(signx == mdl.piecewise_as_slopes([(0, 0), (2, 0)], 0, (1, 1), name="sign1")(x))
        mdl.add_constraint(y == -2)
        mdl.add_constraint(signy == mdl.piecewise_as_slopes([(0, 0), (2, 0)], 0, (1, 1), name="sign2")(y))

        # print(mdl.export_as_lp_string())
        res = mdl.solve()

        # mdl.print_solution(print_zeros=True)
        self.assertTrue(res)
        self.assertEqual(x.solution_value, 2)
        self.assertEqual(signx.solution_value, 1)
        self.assertEqual(y.solution_value, -2)
        self.assertEqual(signy.solution_value, -1)
        self.assertEqual(mdl.objective_value, 2)

    def test_pwl_sample4(self):
        # OPL model ---------------------------------
        #
        # dvar float x;
        #
        # subject to {
        #     0.3 == piecewise{0->0; 2->0; 0}(1,1) x;			// ==> Solve OK -> x = 0
        # }
        #
        # OUTPUT:
        #     // solution (optimal) with objective 0
        #     x = 0;
        # -------------------------------------------
        mdl = self.model
        x = mdl.continuous_var(lb=-mdl.infinity, name="x")
        mdl.add_constraint(0.3 == mdl.piecewise_as_slopes([(0, 0), (2, 0)], 0, (1, 1), name="sign1")(x))
        res = mdl.solve()
        self.assertTrue(res)
        self.assertEqual(x.solution_value, 0)
        self.assertEqual(mdl.objective_value, 0)

    def test_pwl_sample5(self):
        # OPL model ---------------------------------
        #
        # dvar float x;
        # dvar float sign;
        #
        # maximize(sign);
        #
        # subject to {
        #     sign == piecewise{0->0; 2->0; 0}(1,1) x;
        #     sign <= 0.3;
        # }
        #
        # OUTPUT:
        #     // solution (optimal) with objective 0
        #     sign = 0;
        #     x = 0;
        # -------------------------------------------
        mdl = self.model
        x = mdl.continuous_var(lb=-mdl.infinity, name="x")
        sign = mdl.continuous_var(lb=-mdl.infinity, name="sign")
        mdl.add_constraint(sign == mdl.piecewise_as_slopes([(0, 0), (2, 0)], 0, (1, 1), name="sign1")(x))
        mdl.add_constraint(sign <= 0.3)
        mdl.maximize(sign)
        res = mdl.solve()
        self.assertTrue(res)
        self.assertEqual(x.solution_value, 0)
        self.assertEqual(sign.solution_value, 0)
        self.assertEqual(mdl.objective_value, 0)

    # Skip this test as no goal is defined and behaviour is "undefined" in that case (for this case, the result
    #  is different than the one with OPL version)
    # @unittest.skip(True)
    def test_pwl_sample6(self):
        # OPL model ---------------------------------
        #
        # dvar float x;
        # dvar float sign;
        #
        # subject to {
        #     sign == piecewise{0->0; 2->0; 0}(1,1) x;
        #     sign <= -0.3;
        # }
        #
        # OUTPUT:
        #     // solution (optimal) with objective 0
        #     sign = -1;
        #     x = 0;
        # -------------------------------------------
        mdl = self.model
        x = mdl.continuous_var(lb=-mdl.infinity, name="x")
        sign = mdl.continuous_var(lb=-mdl.infinity, name="sign")
        mdl.add_constraint(sign == mdl.piecewise_as_slopes([(0, 0), (2, 0)], 0, (1, 1), name="sign1")(x))
        mdl.add_constraint(sign <= -0.3)
        # NO GOAL
        res = mdl.solve()

        # mdl.print_solution(print_zeros=True)
        # cplex = mdl._Model__engine
        # cplex._CplexEngine__cplex.write("sample6.lp", "lp")

        # print(mdl.export_as_lp_string())
        self.assertTrue(res)
        self.assertEqual(x.solution_value, 0)
        self.assertEqual(sign.solution_value, -0.3)  # In OPL, the returned value is: -1.0 for "sign"
        self.assertEqual(mdl.objective_value, 0)

    def test_pwl_sample7(self):
        # OPL model ---------------------------------
        #
        # dvar float x;
        # dvar float sign;
        #
        # maximize(sign);
        #
        # subject to {
        #     sign == piecewise{0->0; 2->0; 0}(1,1) x;
        #     sign <= -0.3;
        # }
        #
        # OUTPUT:
        #     // solution (optimal) with objective -0.3
        #     sign = -0.3;
        #     x = 0;
        # -------------------------------------------
        mdl = self.model
        x = mdl.continuous_var(lb=-mdl.infinity, name="x")
        sign = mdl.continuous_var(lb=-mdl.infinity, name="sign")
        mdl.add_constraint(sign == mdl.piecewise_as_slopes([(0, 0), (2, 0)], 0, (1, 1), name="sign1")(x))
        mdl.add_constraint(sign <= -0.3)
        mdl.maximize(sign)
        res = mdl.solve()
        # mdl.print_solution(print_zeros=True)
        self.assertTrue(res)
        self.assertEqual(x.solution_value, 0)
        self.assertEqual(sign.solution_value, -0.3)
        self.assertEqual(mdl.objective_value, -0.3)

    def test_pwl_sailcopw(self):
        # OPL model ---------------------------------
        #
        # int NbPeriods = ...;
        # range Periods = 1..NbPeriods;
        #
        # float Demand[Periods] = ...;
        # float RegularCost = ...;
        # float ExtraCost = ...;
        # float Capacity = ...;
        # float Inventory = ...;
        # float InventoryCost = ...;
        #
        # dvar float+ Boat[Periods];
        # dvar float+ Inv[0..NbPeriods];
        #
        #
        # minimize
        #    sum(t in Periods)
        #        piecewise{ RegularCost -> Capacity ; ExtraCost } Boat[t] +
        #                   InventoryCost  * (sum(t in Periods) Inv[t]);
        #
        # subject to  {
        #   ctInventory:
        #     Inv[0] == Inventory;
        #   forall(t in Periods)
        #     ctDemand:
        #       Boat[t] + Inv[t-1] == Inv[t] + Demand[t];
        # }
        # OUTPUT:
        #     // solution (optimal) with objective 72200
        #     // Quality Incumbent solution:
        #     // MILP objective                                7.2200000000e+004
        #     // MILP solution norm |x| (Total, Max)           7.07850e+004 3.70000e+004
        #     // MILP solution error (Ax=b) (Total, Max)       0.00000e+000 0.00000e+000
        #     // MILP x bound error (Total, Max)               0.00000e+000 0.00000e+000
        #     // MILP x integrality error (Total, Max)         0.00000e+000 0.00000e+000
        #     // MILP slack bound error (Total, Max)           0.00000e+000 0.00000e+000
        #     //
        #
        #     Boat = [90 0 100 0];
        #     Inv = [10 60 0 25 0];
        # -------------------------------------------

        # Input Data
        NbPeriods = 4
        Demand = {1: 40, 2: 60, 3: 75, 4: 25}
        RegularCost = 400
        ExtraCost = 350
        Capacity = 40
        Inventory = 10
        InventoryCost = 20

        # Model
        mdl = self.model
        Periods = [i for i in range(1, NbPeriods + 1)]

        Boat = mdl.continuous_var_dict(Periods, name=lambda ix: 'Boat#%d' % ix)
        Inv = mdl.continuous_var_dict([i for i in range(NbPeriods + 1)], name=lambda ix: 'Inv#%d' % ix)

        mdl.minimize(
            mdl.sum(mdl.piecewise_as_slopes([(RegularCost, Capacity), ], ExtraCost)(Boat[t]) for t in Periods) +
            InventoryCost * mdl.sum([Inv[t] for t in Periods]))

        # Subject to:
        mdl.add_constraint(Inv[0] == Inventory)
        mdl.add_constraints([Boat[t] + Inv[t - 1] == Inv[t] + Demand[t] for t in Periods])

        res = mdl.solve()

        self.assertTrue(res)
        self.assertEqual(Boat[1].solution_value, 90)
        self.assertEqual(Boat[2].solution_value, 0)
        self.assertEqual(Boat[3].solution_value, 100)
        self.assertEqual(Boat[4].solution_value, 0)
        self.assertEqual(Inv[0].solution_value, 10)
        self.assertEqual(Inv[1].solution_value, 60)
        self.assertEqual(Inv[2].solution_value, 0)
        self.assertEqual(Inv[3].solution_value, 25)
        self.assertEqual(Inv[4].solution_value, 0)
        self.assertEqual(mdl.objective_value, 72200)

    def test_pwl_sailcopwg1(self):
        # OPL model ---------------------------------
        #
        # int NbPeriods = ...;
        # range Periods = 1..NbPeriods;
        # int NbPieces = ...;
        #
        # float Cost[1..NbPieces] = ...;
        # float Breakpoint[1..NbPieces-1] = ...;
        # float Demand[Periods] = ...;
        # float Inventory = ...;
        # float InventoryCost = ...;
        #
        # dvar float+ Boat[Periods];
        # dvar float+ Inv[0..NbPeriods];
        #
        #
        # minimize
        #   sum( t in Periods )
        #     piecewise(i in 1..NbPieces-1) {
        #       Cost[i] -> Breakpoint[i];
        #       Cost[NbPieces]
        #     } Boat[t] +
        #   InventoryCost  * ( sum( t in Periods ) Inv[t] );
        #
        # subject to {
        #   ctInit:
        #     Inv[0] == Inventory;
        #   forall( t in Periods )
        #     ctBoat:
        #       Boat[t] + Inv[t-1] == Inv[t] + Demand[t];
        #
        # }
        # OUTPUT:
        #     // solution (optimal) with objective 78450
        #     // Quality Incumbent solution:
        #     // MILP objective                                7.8450000000e+004
        #     // MILP solution norm |x| (Total, Max)           7.84600e+004 3.17500e+004
        #     // MILP solution error (Ax=b) (Total, Max)       0.00000e+000 0.00000e+000
        #     // MILP x bound error (Total, Max)               0.00000e+000 0.00000e+000
        #     // MILP x integrality error (Total, Max)         0.00000e+000 0.00000e+000
        #     // MILP slack bound error (Total, Max)           0.00000e+000 0.00000e+000
        #     //
        #
        #     Boat = [40
        #              50 75 25];
        #     Inv = [10 10 0 0 0];
        # -------------------------------------------

        # Input Data
        NbPeriods = 4
        Demand = {1: 40, 2: 60, 3: 75, 4: 25}
        NbPieces = 2
        Cost = {1: 400, 2: 450}
        Breakpoint = {1: 40}
        Inventory = 10
        InventoryCost = 20

        # Model
        mdl = self.model
        Periods = [i for i in range(1, NbPeriods + 1)]

        Boat = mdl.continuous_var_dict(Periods, name=lambda ix: 'Boat#%d' % ix)
        Inv = mdl.continuous_var_dict([i for i in range(NbPeriods + 1)], name=lambda ix: 'Inv#%d' % ix)

        mdl.minimize(
            mdl.sum(mdl.piecewise_as_slopes(
                [(Cost[i], Breakpoint[i]) for i in range(1, NbPieces)], Cost[NbPieces])(Boat[t]) for t in Periods) +
            InventoryCost * mdl.sum([Inv[t] for t in Periods]))

        # Subject to:
        mdl.add_constraint(Inv[0] == Inventory)
        mdl.add_constraints([Boat[t] + Inv[t - 1] == Inv[t] + Demand[t] for t in Periods])

        res = mdl.solve()

        self.assertTrue(res)
        self.assertEqual(Boat[1].solution_value, 40)
        self.assertEqual(Boat[2].solution_value, 50)
        self.assertEqual(Boat[3].solution_value, 75)
        self.assertEqual(Boat[4].solution_value, 25)
        self.assertEqual(Inv[0].solution_value, 10)
        self.assertEqual(Inv[1].solution_value, 10)
        self.assertEqual(Inv[2].solution_value, 0)
        self.assertEqual(Inv[3].solution_value, 0)
        self.assertEqual(Inv[4].solution_value, 0)
        self.assertEqual(mdl.objective_value, 78450)

    def test_pwl_sailcopwg2(self):
        # OPL model ---------------------------------
        #
        # OUTPUT:
        #     / solution (optimal) with objective 66950
        #     // Quality Incumbent solution:
        #     // MILP objective                                6.6950000000e+004
        #     // MILP solution norm |x| (Total, Max)           6.69600e+004 2.87500e+004
        #     // MILP solution error (Ax=b) (Total, Max)       0.00000e+000 0.00000e+000
        #     // MILP x bound error (Total, Max)               0.00000e+000 0.00000e+000
        #     // MILP x integrality error (Total, Max)         0.00000e+000 0.00000e+000
        #     // MILP slack bound error (Total, Max)           0.00000e+000 0.00000e+000
        #     //
        #
        #     Boat = [40
        #              50 75 25];
        #     Inv = [10 10 0 0 0];
        # -------------------------------------------

        # Input Data
        NbPeriods = 4
        Demand = {1: 40, 2: 60, 3: 75, 4: 25}
        NbPieces = 3
        Cost = {1: 300, 2: 400, 3: 450}
        Breakpoint = {1: 30, 2: 40}
        Inventory = 10
        InventoryCost = 20

        # Model
        mdl = self.model
        Periods = [i for i in range(1, NbPeriods + 1)]

        Boat = mdl.continuous_var_dict(Periods, name=lambda ix: 'Boat#%d' % ix)
        Inv = mdl.continuous_var_dict([i for i in range(NbPeriods + 1)], name=lambda ix: 'Inv#%d' % ix)

        mdl.minimize(
            mdl.sum(mdl.piecewise_as_slopes(
                [(Cost[i], Breakpoint[i]) for i in range(1, NbPieces)], Cost[NbPieces])(Boat[t]) for t in Periods) +
            InventoryCost * mdl.sum([Inv[t] for t in Periods]))

        # Subject to:
        mdl.add_constraint(Inv[0] == Inventory)
        mdl.add_constraints([Boat[t] + Inv[t - 1] == Inv[t] + Demand[t] for t in Periods])

        res = mdl.solve()

        self.assertTrue(res)
        self.assertEqual(Boat[1].solution_value, 40)
        self.assertEqual(Boat[2].solution_value, 50)
        self.assertEqual(Boat[3].solution_value, 75)
        self.assertEqual(Boat[4].solution_value, 25)
        self.assertEqual(Inv[0].solution_value, 10)
        self.assertEqual(Inv[1].solution_value, 10)
        self.assertEqual(Inv[2].solution_value, 0)
        self.assertEqual(Inv[3].solution_value, 0)
        self.assertEqual(Inv[4].solution_value, 0)
        self.assertEqual(mdl.objective_value, 66950)

    def test_pwl_sailcopwg3(self):
        # OPL model ---------------------------------
        #
        # OUTPUT:
        #     // solution (optimal) with objective 1560600
        #     // Quality Incumbent solution:
        #     // MILP objective                                1.5606000000e+006
        #     // MILP solution norm |x| (Total, Max)           1.56023e+006 1.51750e+006
        #     // MILP solution error (Ax=b) (Total, Max)       0.00000e+000 0.00000e+000
        #     // MILP x bound error (Total, Max)               0.00000e+000 0.00000e+000
        #     // MILP x integrality error (Total, Max)         0.00000e+000 0.00000e+000
        #     // MILP slack bound error (Total, Max)           0.00000e+000 0.00000e+000
        #     //
        #
        #     Boat = [50
        #              50 65 25];
        #     Inv = [10 20 10 0 0];
        # -------------------------------------------

        # Input Data
        NbPeriods = 4
        Demand = {1: 40, 2: 60, 3: 75, 4: 25}
        NbPieces = 4
        Cost = {1: 300, 2: 400, 3: 450, 4: 100000}
        Breakpoint = {1: 30, 2: 40, 3: 50}
        Inventory = 10
        InventoryCost = 20

        # Model
        mdl = self.model
        Periods = [i for i in range(1, NbPeriods + 1)]

        Boat = mdl.continuous_var_dict(Periods, name=lambda ix: 'Boat#%d' % ix)
        Inv = mdl.continuous_var_dict([i for i in range(NbPeriods + 1)], name=lambda ix: 'Inv#%d' % ix)

        mdl.minimize(
            mdl.sum(mdl.piecewise_as_slopes(
                [(Cost[i], Breakpoint[i]) for i in range(1, NbPieces)], Cost[NbPieces])(Boat[t]) for t in Periods) +
            InventoryCost * mdl.sum([Inv[t] for t in Periods]))

        # Subject to:
        mdl.add_constraint(Inv[0] == Inventory)
        mdl.add_constraints([Boat[t] + Inv[t - 1] == Inv[t] + Demand[t] for t in Periods])

        res = mdl.solve()

        self.assertTrue(res)
        self.assertEqual(Boat[1].solution_value, 50)
        self.assertEqual(Boat[2].solution_value, 50)
        self.assertEqual(Boat[3].solution_value, 65)
        self.assertEqual(Boat[4].solution_value, 25)
        self.assertEqual(Inv[0].solution_value, 10)
        self.assertEqual(Inv[1].solution_value, 20)
        self.assertEqual(Inv[2].solution_value, 10)
        self.assertEqual(Inv[3].solution_value, 0)
        self.assertEqual(Inv[4].solution_value, 0)
        self.assertEqual(mdl.objective_value, 1560600)


@unittest.skipUnless(the_env.has_cplex and the_env.cplex_version_as_tuple >= (12, 7, 0),
                     "Piecewise linear constraints require Cplex 12.7 or higher")
class PwlWrapperIssueModelSamples(unittest.TestCase):

    def test_pwl_sailcopw_resolve(self):
        # Input Data
        NbPeriods = 4
        Demand = {1: 40, 2: 60, 3: 75, 4: 25}
        RegularCost = 400
        ExtraCost = 550
        Capacity = 40
        Inventory = 10
        InventoryCost = 20

        Periods = [i for i in range(1, NbPeriods + 1)]
        Periods0 = [i for i in range(NbPeriods + 1)]

        mdl = Model("Sailco_PWL")
        # Definition of production cost as a piecewise linear function
        production_cost_pwl = mdl.piecewise_as_slopes([(RegularCost, Capacity), ], ExtraCost)

        # A piecewise linear model
        Boat = mdl.continuous_var_dict(Periods, name='Boat')
        Inv = mdl.continuous_var_dict(Periods0, name='Inv')

        mdl.minimize(
            mdl.sum(production_cost_pwl(Boat[t]) for t in Periods) +
            InventoryCost * mdl.sum([Inv[t] for t in Periods]))

        # Subject to:
        mdl.add_constraint(Inv[0] == Inventory)
        mdl.add_constraints([Boat[t] + Inv[t - 1] == Inv[t] + Demand[t] for t in Periods])

        mdl.solve(url=None, key=None)
        self.assertEqual(mdl.objective_value, 82950)

        # Create a new PWL function and update objective
        NbPieces = 4
        Cost = {1: 300, 2: 400, 3: 450, 4: 100000}
        Breakpoint = {1: 30, 2: 40, 3: 50}

        production_cost_pwl = mdl.piecewise_as_slopes(
            [(Cost[i], Breakpoint[i]) for i in range(1, NbPieces)], Cost[NbPieces])

        mdl.minimize(
            mdl.sum(production_cost_pwl(Boat[t]) for t in Periods) + InventoryCost * mdl.sum([Inv[t] for t in Periods]))

        mdl.solve(url=None, key=None)
        self.assertEqual(mdl.objective_value, 1560600)

    def test_two_piecewise_models(self):
        # RTC-31149
        # static interface prevented two models from handling different picewises simultaneously
        m1 = Model(name='pw1')
        m2 = Model(name='pw2')
        # allocate two vars per model
        x1 = m1.continuous_var(name='x', ub=4)
        y1 = m1.continuous_var(name='y', ub=4)
        x2 = m2.continuous_var(name='x', ub=4)
        y2 = m2.continuous_var(name='y', ub=4)
        # one pw per model (different)
        pw1 = m1.piecewise(preslope=0, breaksxy=[(1, 1), (2, 2), (3, 3), (4, 4)], postslope=0)
        pw2 = m2.piecewise(preslope=0, breaksxy=[(1, 4), (2, 3), (3, 2), (4, 1)], postslope=0)
        # cts
        m1.add(y1 == pw1(x1))
        m2.add(y2 == pw2(x2))
        #
        m1.maximize(y1)
        m2.maximize(y2)

        s1 = m1.solve()
        s2 = m2.solve()

        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)

        self.assertEqual(x1.solution_value, 4)
        self.assertEqual(1, x2.solution_value)
        m1.end()
        m2.end()


if __name__ == "__main__":
    unittest.main(verbosity=1)
