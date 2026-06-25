import unittest
import six

from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException
from docplex.mp.vartype import BinaryVarType, IntegerVarType, ContinuousVarType, SemiContinuousVarType, \
    SemiIntegerVarType


class VartypeTests(unittest.TestCase):

    expected_default_lb = 0

    def setUp(self):
        self.model = Model()

    def tearDown(self):
        self.model.end()
        self.model = None
    integer_bound_types = {int} if six.PY3 else {int, long}

    def check_bound_type(self, vartype, actual_bound, symbol):
        if not vartype.is_discrete():
            self.assertIsInstance(actual_bound, float,
                                  "computed %s must return a float, got %g" % (symbol, actual_bound))
        else:
            bound_type = type(actual_bound)
            self.assertTrue(bound_type in self.integer_bound_types
                            or isinstance(actual_bound, float), "bound value is: {}".format(actual_bound))

    def _check_computed_bounds(self, vartype, initial_lb, initial_ub, expected_lb, expected_ub):
        cpx_infinity = 1e+20
        m = self.model
        actual_lb, actual_ub = vartype._compute_lb(initial_lb, m), vartype._compute_ub(initial_ub, m)
        self.check_bound_type(vartype, actual_bound=actual_lb, symbol="LB")
        self.check_bound_type(vartype, actual_bound=actual_ub, symbol="UB")
        self.assertLessEqual(actual_ub, cpx_infinity)
        self.assertGreaterEqual(actual_lb, -cpx_infinity)
        self.assertEqual(actual_lb, expected_lb, "Unexpected lb: %g, expecting: %g" % (actual_lb, expected_lb))
        self.assertEqual(actual_ub, expected_ub, "Unexpected ub: %g, expecting: %g" % (actual_ub, expected_ub))

    def test_compute_binary_var_bounds(self):
        binary = self.model.binary_vartype
        self._check_computed_bounds(binary, 0, 4, 0, 4)
        self._check_computed_bounds(binary, 0, 0.5, 0, 0.5)

    def test_compute_infeasible_binary_var_bounds(self):
        binary = self.model.binary_vartype
        six.assertRaisesRegex(self, DOcplexException,
                                "Lower bound for binary variable",
                              lambda c: c._check_computed_bounds(binary, 5,10, 5, 10), self)

    def test_intvar_bounds(self):
        int_type = self.model.integer_vartype
        lb_specs = [(-2e+20, -1e+20), (-1e+20, -1e+20), (-1, -1), (0, 0)]
        ub_specs = [(0, 0), (1e+19, 1e+19), (2e+20, 1e+20)]
        for user_lb, expected_lb in lb_specs:
            for user_ub, expected_ub in ub_specs:
                # print("lb=%g, ub=%g"%(user_lb, user_ub))
                self._check_computed_bounds(int_type,
                                            user_lb, user_ub,
                                            expected_lb, expected_ub)

    def test_vartype_binary(self):
        binary_type = self.model.binary_vartype
        self.assertTrue(binary_type.is_discrete())
        self.assertEqual(binary_type.default_lb, 0)
        self.assertEqual(binary_type.default_ub, 1)
        self.assertFalse(binary_type.accept_value(0.5))
        self.assertEqual(str(binary_type), "VarType_binary")

    def test_vartype_integer(self):
        integer_type = self.model.integer_vartype
        self.assertTrue(integer_type.is_discrete())
        self.assertEqual(integer_type.default_lb, 0)
        self.assertEqual(integer_type.default_ub, 1e+20)
        self.assertTrue(integer_type.accept_value(1))
        self.assertFalse(integer_type.accept_value(1.3))

    def test_vartype_continuous(self):
        continuous_type = self.model.continuous_vartype
        self.assertFalse(continuous_type.is_discrete())
        self.assertEqual(continuous_type.default_lb, 0)
        self.assertEqual(continuous_type.default_ub, 1e+20)
        self.assertTrue(continuous_type.accept_value(3.14))

    def test_vartype_semicontinuous(self):
        continuous_type = self.model.semicontinuous_vartype
        self.assertFalse(continuous_type.is_discrete())
        # self.assertEqual(continuous_type.default_lb, 0)
        self.assertEqual(continuous_type.default_ub, 1e+20)
        self.assertTrue(continuous_type.accept_value(3.14))

    def test_vartype_semiinteger(self):
        semiint_type = self.model.semiinteger_vartype
        self.assertTrue(semiint_type.is_discrete())
        # self.assertEqual(continuous_type.default_lb, 0)
        self.assertEqual(semiint_type.default_ub, 1e+20)
        self.assertTrue(semiint_type.accept_value(3))
        self.assertFalse(semiint_type.accept_value(-3))

    def test_vartype_parse(self):
        m = self.model
        bt1 = m._parse_vartype('B')
        self.assertIs(bt1, m.binary_vartype)
        bt2 = m._parse_vartype('b')
        self.assertIs(bt2, m.binary_vartype)
        ct1 = m._parse_vartype('C')
        self.assertIs(ct1, m.continuous_vartype)
        ct2 = m._parse_vartype('c')
        self.assertIs(ct2, m.continuous_vartype)

    def test_vartype_equality(self):
        # check any two var types are different...
        all_vartypes = [BinaryVarType, IntegerVarType, ContinuousVarType, SemiContinuousVarType, SemiIntegerVarType]
        for vt1 in all_vartypes:
            for vt2 in all_vartypes:
                ivt1 = vt1()
                ivt2 = vt2()
                if vt1 == vt2:
                    self.assertTrue(ivt1 == ivt2, 'types must be equal: {0}, {1}'.format(ivt1, ivt2))
                else:
                    self.assertTrue(ivt1 != ivt2)

    def test_binary_varlist_with_type_equality(self):
        m = self.model
        # create variables with a different type instance.
        zs = m.var_list(keys=3, vartype=BinaryVarType(), name='zs')
        x = m.binary_var(name='bidon')
        self.assertEqual(zs[0].vartype, x.vartype)

    def test_continuous_varlist_with_type_equality(self):
        m = self.model
        # create variables with a different type instance.
        zs = m.var_list(keys=3, vartype=ContinuousVarType(), name='frees')
        x = m.continuous_var(name='bidon')
        self.assertEqual(zs[0].vartype, x.vartype)

    def test_integer_varlist_with_type_equality(self):
        m = self.model
        # create variables with a different type instance.
        zs = m.var_list(keys=3, vartype=IntegerVarType(), name='ixx')
        x = m.integer_var(name='bidon')
        self.assertEqual(zs[0].vartype, x.vartype)


if __name__ == "__main__":
    unittest.main()



