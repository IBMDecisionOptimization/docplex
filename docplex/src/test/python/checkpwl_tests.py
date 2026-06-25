import os
import unittest

from docplex.mp.model import Model
from testutils import get_test_temp_dir

TEMP_PATH = get_test_temp_dir()

def find(file, search="12\.5", path=TEMP_PATH):
    import re
    import sys

    if path is not None:
        file = os.path.join(path, file)

    with open(file, "r") as file:
        for line in file:
            if re.search(search, line):
                return True
    return False




class TestPwlTest(unittest.TestCase):
    def test1(self):
        with Model(name_functional_vars=True) as mdl:
            pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0, "slope")
            x = mdl.continuous_var(name="xxx")
            p = pwl(x)

            y = mdl.continuous_var(name="yyy")
            z = mdl.continuous_var(name="zzz")
            vars = set([x, y, z])

            mdl.max(p)
            ct = (p == y + 2 * z)
            #mdl.add_constraint(ct)
            pwl_expr_vars = [v for v in p.iter_variables() if v not in vars]
            pwl_ct_vars = [v for v in ct.iter_variables() if v not in vars]
            self.assertIs(pwl_expr_vars[-1], pwl_ct_vars[-1], "pwl problem: did not find the correct pwl exp variable.")
            self.assertEqual("pwl" in pwl_expr_vars[-1].lp_name, True, "pwl problem: did not find the correct pwl exp variable.")
            v = pwl_expr_vars[-1]
            v.lb = 1.5
            v.ub = 12.5
            mdl.maximize(p)
            s0 = mdl.solve()
            mdl.export_as_lp(path=TEMP_PATH, basename="tata1")
            mdl.get_cplex().write(os.path.join(TEMP_PATH, "tata.lp"))
            s0.export_as_sol(path=TEMP_PATH, basename="tata")
            #print(s0)
            self.assertEqual(len(s0.iter_variables()), 2)
            self.assertAlmostEqual([v for x, v in s0.iter_var_values()][0], 2.5)
            self.assertEqual(find("tata.lp"), True)
            self.assertEqual(find("tata1.lp"), True)

    def test2(self):
        with Model() as mdl:
            pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0, "slope")
            x = mdl.continuous_var(name="xxx")
            p = pwl(x)
            mdl.maximize(p)
            mdl.add_range(1.5, p, 12.5)
            s = mdl.solve()
            mdl.export_as_lp(path=TEMP_PATH, basename="toto1")
            mdl.get_cplex().write(os.path.join(TEMP_PATH, "toto.lp"))
            s.export_as_sol(path=TEMP_PATH, basename="toto")
            #print(s)

            self.assertEqual(find("toto.lp"), True)
            self.assertEqual(find("toto.sol"), True)
            self.assertEqual(find("toto1.lp"), True)
            self.assertEqual(len(s.iter_variables()),2)
            self.assertAlmostEqual([v for x, v in s.iter_var_values()][0], 2.5)

    def test2bis(self):
        with Model() as mdl:
            pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0, "slope")
            x = mdl.continuous_var(name="xxx")
            p = pwl(x)
            mdl.maximize(p)
            mdl.add_constraint(1.5 <=p)
            mdl.add_constraint(p <= 12.5)
            s = mdl.solve()
            mdl.export_as_lp(path=TEMP_PATH, basename="tutu1")
            mdl.get_cplex().write(os.path.join(TEMP_PATH, "tutu.lp"))
            s.export_as_sol(path=TEMP_PATH, basename="tutu")
            #print(s)

            self.assertEqual(find("tutu.lp"), True)
            self.assertEqual(find("tutu.sol"), True)
            self.assertEqual(find("tutu1.lp"), True)
            self.assertEqual(len(s.iter_variables()),2)
            self.assertAlmostEqual([v for x, v in s.iter_var_values()][0], 2.5)

    def test3(self):
        with Model() as mdl:
            pwl = mdl.piecewise(0, [(0, 10), (10, 20)], 0, "slope")
            x = mdl.continuous_var(name="xxx")
            p = pwl(x)
            y = p.as_var
            y.lb = 1.5
            y.ub = 12.5
            mdl.maximize(p)
            s1 = mdl.solve()
            mdl.export_as_lp(path=TEMP_PATH, basename="titi1")
            mdl.get_cplex().write(os.path.join(TEMP_PATH, "titi.lp"))
            s1.export_as_sol(path=TEMP_PATH, basename="titi")
            #print(s1)
            self.assertEqual(find("titi.lp"), True)
            self.assertEqual(find("titi.sol"), True)
            self.assertEqual(find("titi1.lp"), True)
            self.assertEqual(len(s1.iter_variables()),2)
            self.assertAlmostEqual([v for x, v in s1.iter_var_values()][0], 2.5)
