# Tests to ensure bug fixes don't resurface

import unittest
from docplex.mp.model import Model
from docplex.mp.environment import *


class NoRegressionTests(unittest.TestCase):
    def setUp(self):
        self.model = Model("no regression")

    def tearDown(self):
        self.model.end()

    def test_RTC_289198_1(self):
        mdl = Model(keep_ordering=True)
        my_keys = range(15000)
        x = mdl.integer_var_dict(keys=my_keys, lb=0, ub=5, name="x_")
        mdl.add_constraint(mdl.sum(x) <= 500)
        lps = mdl.export_as_lp_string()
        self.assertIn("x__14998 + x__14999 <= 500", lps)

    def test_RTC_289198_2(self):
        mdl = Model(keep_ordering=True)
        my_keys = range(15000)
        x = mdl.integer_var_list(keys=my_keys, lb=0, ub=5, name="x_")
        mdl.add_constraint(mdl.sum(x) <= 500)
        lps = mdl.export_as_lp_string()
        self.assertIn("x__14998 + x__14999 <= 500", lps)

    def test_RTC_289198_3(self):
        mdl = Model(keep_ordering=True)
        my_keys = range(15)
        x = mdl.integer_var_list(keys=my_keys, lb=0, ub=5, name="x_")
        mdl.add_constraint(sum(x) <= 500)
        lps = mdl.export_as_lp_string()
        self.assertIn("x__13 + x__14 <= 500", lps)

    def test_RTC_29590(self):
        mdl = Model("scale", )
        mdl.parameters.read.datacheck = 0
        n = 100
        qty = mdl.continuous_var_list(n, lb=0, ub=n)
        mdl.add_constraints(qty[i] - qty[i - 1] == 1 for i in range(n))
        nb_eq_csts = mdl.statistics.number_of_eq_constraints
        self.assertEqual(nb_eq_csts, 100)

    def test_RTC_30715(self):
        mdl = Model("float precision")
        x = mdl.continuous_var(name='x')
        c = 1.0 / 3
        mdl.add_constraint(x <= c)
        lps = mdl.export_as_lp_string()
        expected = "<= 0.333333333333" if env_is_64_bit() else "<= 0.333333333"
        self.assertIn(expected, lps)

    def test_RTC_31890(self):
        mdl = Model("31890")
        x = mdl.integer_var(lb=3, ub=8)
        mdl.maximize(4 + x)
        lp = mdl.export_as_lp_string()
        self.assertIn('obj: x1 + 4', lp)

    def test_RTC_31976(self):
        mdl = Model("31976")
        mdl.semicontinuous_var(lb=2, ub=10, name='sc1')
        mdl.semicontinuous_var_list(keys=range(3), lb=3, ub=100, name='sc')
        mdl.continuous_var(name='x')
        mdl.integer_var_cube(keys1=2, keys2=range(2), keys3=range(3))
        mdl.integer_var(name='y')
        mdl.binary_var_dict(keys=5)
        stats = str(mdl.statistics)
        self.assertIn("number of variables: 23", stats)
        self.assertIn("binary=5, integer=13, continuous=1", stats)
        self.assertIn("binary=5, integer=13, continuous=1, semi-continuous=4", stats)


    def test_RTC_30650(self):
        with Model(name='rtc30650', checker='off') as m:
            m.parameters.read.datacheck = 0
            m.apply_parameters()
            v1 = m.continuous_var(ub=10, name="v1")
            v2 = m.continuous_var(ub=10, name="v2")
            m.add_constraint(3 * v1 + 2 * v2 <= float("inf"))
            m.maximize(5 * v1 + 4 * v2)
            s = m.solve()

    def test_RTC_37585(self):
        # should break around 900-1000 iterations if bug is still is there
        with Model() as model:
            x = model.continuous_var(0, 1e20, "x")
            factor = 1.0
            while factor < 5000:
                factor += 1.0
                model.minimize(factor * x)
                model.solve()


if __name__ == "__main__":
    unittest.main()
