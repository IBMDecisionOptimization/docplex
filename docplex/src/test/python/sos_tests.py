import unittest

import six

from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException

from docplex.mp.constants import SOSType
from docplex.mp.model_reader import ModelReader
from testutils import silent_remove, DocplexAbstractTest


class SOSTests(DocplexAbstractTest):

    def test_sos3(self):
        m = self.model
        x1 = m.continuous_var(lb=1, ub=10, name='x1')
        x2 = m.continuous_var(lb=1, ub=10, name='x2')
        six.assertRaisesRegex(self, DOcplexException, "Cannot convert to SOS type: 3 - expecting 1|2|'sos1'|'sos2'",
                              lambda m_: m_.add_sos([x1, x2], 3), m)

    def test_sos1_str(self):
        m = self.model
        xs = m.integer_var_list(keys=2, name='zz', key_format='%s')
        sos1 = m.add_sos1(dvars=xs, name='sos1_xs')
        self.assertEqual('SOS1(\'sos1_xs\')[zz0, zz1]', str(sos1))

    def test_sos2_str(self):
        m = self.model
        xs = m.integer_var_list(keys=3, name='zz', key_format='%s')
        sos2 = m.add_sos2(dvars=xs, name='sos2_xs')
        self.assertEqual('SOS2(\'sos2_xs\')[zz0, zz1, zz2]', str(sos2))

    def test_sos2_as_constraint(self):
        m = self.model
        xs = m.integer_var_list(keys=['a', 'b', 'c'], name=str)
        sos2 = m.add_sos2(dvars=xs, name='sos2_abc')
        sos2_ct = sos2.as_constraint()
        self.assertFalse(sos2_ct.is_added())
        self.assertEqual('sos2_abc: a+b+c <= 2', str(sos2_ct))

    def test_sos1(self):
        m = self.model
        ub = 10
        xs = m.integer_var_list(keys=11, ub=ub)
        sos1 = m.add_sos1(dvars=xs, name='sos1_xs')
        self.assertEqual(11, len(sos1))
        self.assertIsNotNone(sos1)
        self.assertEqual(1, m.number_of_sos)
        self.assertEqual(1, m.number_of_sos1)
        self.assertEqual(0, m.number_of_sos2)
        self.assertEqual(str(sos1.sos_type), 'SOSType.SOS1')
        self.assertEqual(repr(sos1.sos_type), 'docplex.mp.SOSType.SOS1')
        m.maximize(m.sum(xs))
        if m._can_solve():
            s = m.solve()
            self.assertIsNotNone(s)
            self.assertEqual(ub, m.objective_value)
            m.clear_sos()
            self.assertEqual(0, m.number_of_sos)
            s2 = m.solve()
            self.assertIsNotNone(s2)
            self.assertEqual(110, s2.objective_value)

    def test_sos2(self):
        m = self.model
        ub = 10
        xs = m.integer_var_list(keys=7, ub=ub)
        sos = m.add_sos2(dvars=xs, name='sos1_xs')
        self.assertEqual(7, len(sos))
        self.assertIsNotNone(sos)
        self.assertEqual(1, m.number_of_sos)
        self.assertEqual(0, m.number_of_sos1)
        self.assertEqual(1, m.number_of_sos2)
        self.assertEqual(str(sos.sos_type), 'SOSType.SOS2')
        self.assertEqual(repr(sos.sos_type), 'docplex.mp.SOSType.SOS2')
        m.maximize(m.sum(xs))
        if m._can_solve():
            s = m.solve()
            self.assertIsNotNone(s)
            self.assertEqual(2 * ub, m.objective_value)
            # now clear all sos
            m.clear_sos()
            s2 = m.solve()
            self.assertIsNotNone(s2)
            # without sos, all variables are set to their ub.
            self.assertEqual(70, m.objective_value)

    def test_sos1_list0_ko(self):
        self.assertRaises(DOcplexException, lambda m: m.add_sos1(dvars=[]), self.model)

    def test_sos1_list1_ok(self):
        mdl = self.model
        x1 = mdl.continuous_var(name='x1')
        mdl.add_sos1(dvars=[x1])
        self.assertEqual(1, mdl.number_of_sos1)

    def test_sos2_list0_ko(self):
        mdl = self.model
        self.assertRaises(DOcplexException, lambda m: m.add_sos2(dvars=[]), mdl)

    def test_sos2_list1_ko(self):
        mdl = self.model
        x1 = mdl.continuous_var(name='x1')
        six.assertRaisesRegex(self, DOcplexException, 'SOS2 variable set must contain at least 2 variables',
                              lambda m: m.add_sos2(dvars=[x1]), mdl)

    def test_sos2_list2_ok(self):
        mdl = self.model
        x1 = mdl.continuous_var(name='x1')
        x2 = mdl.continuous_var(name='x2')
        mdl.add_sos2(dvars=[x1, x2])
        self.assertTrue(1, mdl.number_of_sos2)

    def test_RTC_31591(self):  # add_sos will crash on OSX because name was None
        mdl = self.model
        x = mdl.continuous_var_list(keys=5, name='x', ub=[40, 1, mdl.infinity, mdl.infinity, 1])
        mdl.minimize(-x[0] - x[1] - 3 * x[2] - 3 * x[3] - 2 * x[4])
        mdl.add_constraint(-x[0] - x[1] + x[2] + x[3] <= 30, ctname='c1')
        mdl.add_constraint(+x[0] + x[2] - 3 * x[3] <= 30, ctname='c2')
        mdl.add_sos2(x)
        mdl.solve()
        self.assertEqual(mdl.objective_value, -92)

    def test_create_sos_from_set(self):
        mdl = self.model
        xl = mdl.continuous_var_list(keys=5, name='x', ub=[40, 1, mdl.infinity, mdl.infinity, 1])
        xset = set(xl)
        six.assertRaisesRegex(self, DOcplexException, 'ordered sequence', lambda m: m.add_sos1(xset), mdl)

    def test_create_sos_from_dict(self):
        mdl = self.model
        xd = mdl.continuous_var_dict(keys=5, name='x', ub=[40, 1, mdl.infinity, mdl.infinity, 1])
        six.assertRaisesRegex(self, DOcplexException, 'ordered sequence', lambda m: m.add_sos1(xd), mdl)

    def test_sos_getitem(self):
        mdl = self.model
        x1 = mdl.continuous_var(name='x1')
        x2 = mdl.continuous_var(name='x2')
        sos1 = mdl.add_sos1([x1, x2])
        self.assertIs(x1, sos1[0])
        self.assertIs(x2, sos1[1])
        self.assertRaises(IndexError, lambda asos: asos[2], sos1)

    def test_sostype_parse_ko3(self):
        six.assertRaisesRegex(self, DOcplexException, "Cannot convert to SOS type: 3 - expecting 1|2|'sos1'|'sos2'",
                              lambda: SOSType.parse(3))

    def test_sostype_parse_ko_sos3(self):
        six.assertRaisesRegex(self, DOcplexException, "Cannot convert to SOS type: 3 - expecting 1|2|'sos1'|'sos2'",
                              lambda: SOSType.parse('sos3'))

    def test_sos2_custom_weights(self):
        with Model(name='wsos') as mdl:
            xs = mdl.continuous_var_list(keys=['x', 'y', 'z'])
            s1 = mdl.add_sos(xs, 1, weights=[3, 11, 17])
            self.assertEqual(s1.weights, [3, 11, 17])
            lps = mdl.export_as_lp_string()
            self.assertIn('S1 :: x : 3 y : 11 z : 17', lps)

    def test_sos2_weights_bad_len(self):
        with Model(name='wsos') as mdl:
            xs = mdl.continuous_var_list(keys=['x', 'y', 'z'])
            six.assertRaisesRegex(self, DOcplexException, "Expecting a sequence of numbers of size 3",
                                  lambda m_: m_.add_sos(xs, 1, weights=[3, 11]), mdl)

    def test_read_model_sos(self):
        with Model("test_sos") as mdl:
            x = mdl.continuous_var_list(keys=5, name='x', ub=[40, 1, mdl.infinity, mdl.infinity, 1])
            mdl.minimize(-x[0] - x[1] - 3 * x[2] - 3 * x[3] - 2 * x[4])
            mdl.add_constraint(-x[0] - x[1] + x[2] + x[3] <= 30, ctname='c1')
            mdl.add_constraint(+x[0] + x[2] - 3 * x[3] <= 30, ctname='c2')
            mdl.add_sos2(x, name='my_sos_2')
            mdl.add_sos1([x[0], x[1], x[3]], 'my_sos_1')
            self.assertEqual(2, mdl.number_of_sos)
            self.assertEqual(1, mdl.number_of_sos1)
            self.assertEqual(1, mdl.number_of_sos1)
            mdl_env = mdl.environment
            if mdl_env.has_cplex:  # and mdl_env.cplex_version < '12.7':
                lp_exported_path = mdl.export_as_lp(basename="read_sos_test")
                if lp_exported_path:
                    try:
                        mdl2 = ModelReader.read(lp_exported_path)
                        self.assertEqual(mdl2.number_of_sos, 2)
                        for s in mdl2.iter_sos():
                            self.assertIn(s.name, ['my_sos_1', 'my_sos_2'])
                    finally:
                        pass
                        silent_remove(lp_exported_path)

    # def test_sos_pplan(self):


if __name__ == "__main__":
    unittest.main()
