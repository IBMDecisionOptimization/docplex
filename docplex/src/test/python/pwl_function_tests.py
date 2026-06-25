import unittest

import six
from docplex.mp.pwl import PwlFunction
from docplex.mp.model import Model
from docplex.mp.environment import Environment
from docplex.mp.utils import DOcplexException

the_env = Environment()


@unittest.skipUnless(the_env.has_cplex and the_env.cplex_version_as_tuple >= (12, 7, 0),
                     "Piecewise linear constraints require Cplex 12.7 or higher")
class PwlPiecewiseTests(unittest.TestCase):
    def setUp(self):
        self.model = Model("piecewise")

    def tearDown(self):
        self.model.end()

    def check_pwlfn(self, pwlf, expected_str, continuous, convex, size=-1):
        if not expected_str.startswith('piecewise'):
            # noinspection PyAugmentAssignment
            expected_str = 'piecewise' + expected_str
        self.assertEqual(expected_str, str(pwlf))
        self.assertEqual(continuous, pwlf.is_continuous())
        self.assertEqual(convex, pwlf.is_convex())
        if size >= 0:
            self.assertEqual(pwlf.number_of_points, size)

    def test_pwl_invalid_none(self):
        six.assertRaisesRegex(self,DOcplexException, "Invalid definition for Piecewise Linear Function: None.",
                                lambda m: PwlFunction(m, name='invalid', pwl_def=None),
                                self.model)

    def test_pwl_invalid_breaksxy_type(self):
        six.assertRaisesRegex(self,DOcplexException, "argument 'breaksxy' must be defined",
                                lambda m: m.piecewise(0, None, 0, name='invalid'), self.model)

    def test_pwl_invalid_breaksxy_type2(self):
        six.assertRaisesRegex(self,DOcplexException, "argument 'breaksxy' expects iterable, 0 was passed",
                                lambda m: m.piecewise(0, 0, 0, name='invalid'), self.model)

    def test_pwl_invalid_breaksxy_type3(self):
        six.assertRaisesRegex(self,DOcplexException, "argument 'breaksxy' must be a non-empty list of \(x, y\) tuples.",
                                lambda m: m.piecewise(0, [], 0, name='invalid'), self.model)

    def test_pwl_invalid_breaksxy_type4(self):
        six.assertRaisesRegex(self,DOcplexException, "invalid tuple in 'breaksxy': \(0,\). Each tuple must have 2 items.",
                                lambda m: m.piecewise(0, [(0,)], 0, name='invalid'), self.model)

    def test_pwl_invalid_breaksxy_type5(self):
        six.assertRaisesRegex(self,DOcplexException,
                                "X coordinate in: \(0, 1\) cannot be smaller than previous break abscisse: \(1, 0\).",
                                lambda m: m.piecewise(0, [(1, 0), (0, 1)], 0, name='invalid'), self.model)

    def test_pwl_invalid_breaksxy_type6(self):
        six.assertRaisesRegex(self,DOcplexException, "Model.piecewise.preslope: Expecting number, None was passed",
                                lambda m: m.piecewise(None, [], 0, name='invalid'), self.model)

    def test_pwl_invalid_breaksxy_type7(self):
        six.assertRaisesRegex(self,
            DOcplexException,
            "invalid break: \(1, 3\). There cannot be more than 2 consecutive breaks with same abscisse.",
            lambda m: m.piecewise(0, [(0, 0), (1, 1), (1, 2), (1, 3)], 0, name='invalid'), self.model)

    def test_pwl_invalid_breaksxy_type8(self):
        six.assertRaisesRegex(self,DOcplexException, "invalid item in 'breaksxy': a. Each item must be a \(x, y\) tuple.",
                                lambda m: m.piecewise(0, ['a'], 0, name='invalid'), self.model)

    def test_pwl_accept_tuple_breakxy1(self):
        pwl = PwlFunction(self.model, name='invalid', pwl_def=PwlFunction._PwlAsBreaks(10, (0, 0), 1))
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def.to_string(), "(10, [(0, 0)], 1)")
        self.assertEqual(pwl_def.get_nb_intervals(), 0)

    def test_pwl_accept_tuple_breakxy2(self):
        six.assertRaisesRegex(self,
            DOcplexException, "invalid tuple in 'breaksxy': \(\). Each tuple must have 2 items.",
            lambda m: m.piecewise(1, (), 1, name='invalid'), self.model)

    def test_pwl_undefined_postslope(self):
        six.assertRaisesRegex(self,
            DOcplexException, "Model.piecewise.postslope: Expecting number, None was passed",
            lambda m: m.piecewise(0, [(0, 0)], None, name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type(self):
        six.assertRaisesRegex(self,
            DOcplexException, "argument 'slopebreaksx' must be defined",
            lambda m: m.piecewise_as_slopes(None, 0, name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type2(self):
        six.assertRaisesRegex(self,
            DOcplexException, "Expecting number, 'a' was passed",
            lambda m: m.piecewise_as_slopes([], 'a', name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type3(self):
        six.assertRaisesRegex(self,
            DOcplexException, "not an iterable: 0",
            lambda m: m.piecewise_as_slopes(0, 1, name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type4(self):
        six.assertRaisesRegex(self,
            DOcplexException, "invalid tuple in 'slopebreaksx': \(0,\). Each tuple must have 2 items.",
            lambda m: m.piecewise_as_slopes([(0,)], 1, name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type5(self):
        six.assertRaisesRegex(self,
            DOcplexException, "NaN value was passed",
            lambda m: m.piecewise_as_slopes([(0, float('nan'))], 1, name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type6(self):
        six.assertRaisesRegex(self,
            DOcplexException, "invalid item in 'slopebreaksx': 0. Each item must be a \(x, y\) tuple.",
            lambda m: m.piecewise_as_slopes([0], 1, name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type7(self):
        six.assertRaisesRegex(self,
            DOcplexException, "X coordinate in: \(5, -1\) cannot be smaller than previous break abscisse: \(10, 0\).",
            lambda m: m.piecewise_as_slopes([(10, 0), (5, -1)], 10, name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type8(self):
        six.assertRaisesRegex(self,
            DOcplexException,
            "invalid break: \(2, 0\). There cannot be more than 2 consecutive breaks with same abscisse.",
            lambda m: m.piecewise_as_slopes([(10, 0), (5, 0), (2, 0)], 10, (1, 0), name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type9(self):
        six.assertRaisesRegex(self,
            DOcplexException, "Model.piecewise_as_slopes.anchor: expecting 2-tuple of floats, invalid tuple \(0, 1, 2\) was passed",
            lambda m: m.piecewise_as_slopes([(10, 0), (5, 0)], 10, (0, 1, 2), name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type10(self):
        six.assertRaisesRegex(self,
            DOcplexException, "anchor \(0, 0\) cannot be defined at discontinuity point: \(5, 0\)",
            lambda m: m.piecewise_as_slopes([(10, 0), (5, 0)], 10, (0, 0), name='invalid'), self.model)

    def test_pwl_invalid_slopebreakx_type11(self):
        six.assertRaisesRegex(self,
            DOcplexException, "Infinite value was passed",
            lambda m: m.piecewise_as_slopes([(0, float('inf'))], 1, name='invalid'), self.model)

    def test_pwl_accept_tuple_slopebreakx1(self):
        pwl = PwlFunction(self.model, name='valid', pwl_def=PwlFunction._PwlAsSlopes((10, 0), 1))
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{10 -> 0;1}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(10, [(0, 0)], 1)")
        self.assertEqual(pwl_def_as_breaks.get_nb_intervals(), 0)

    def test_pwl_accept_tuple_slopebreakx2(self):
        pwl = PwlFunction(self.model, name='valid', pwl_def=PwlFunction._PwlAsSlopes((), 1))
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(1, [(0, 0)], 1)")

    def test_pwl_accept_tuple_slopebreakx3(self):
        pwl = self.model.piecewise_as_slopes((), 1)
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(1, [(0, 0)], 1)")

    def test_pwl_accept_tuple_slopebreakx4(self):
        pwl = self.model.piecewise_as_slopes((10, 0), 1)
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{10 -> 0;1}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(10, [(0, 0)], 1)")

    def test_pwl_invalid_anchor(self):
        six.assertRaisesRegex(self,
            DOcplexException, "Model.piecewise_as_slopes.anchor: expecting 2-tuple of floats, invalid tuple \(0,\) was passed",
            lambda m: m.piecewise_as_slopes([], 0, (0,), name='invalid'), self.model)

    def test_pwl_invalid_anchor2(self):
        six.assertRaisesRegex(self,
            DOcplexException, "Model.piecewise_as_slopes.anchor: expecting 2-tuple of floats, None was passed",
            lambda m: m.piecewise_as_slopes([], 0, None, name='invalid'), self.model)

    def test_pwl_invalid_anchor3(self):
        six.assertRaisesRegex(self,
            DOcplexException, "Model.piecewise_as_slopes.anchor: expecting 2-tuple of floats, 0 was passed",
            lambda m: m.piecewise_as_slopes([], 0, 0, name='invalid'), self.model)

    def test_pwl_slope_to_breaksxy(self):
        pwl = PwlFunction(self.model, name='trivial', pwl_def=PwlFunction._PwlAsSlopes([], 10))
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{10}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(10, [(0, 0)], 10)")

    def test_pwl_slope_to_breaksxy2(self):
        pwl = PwlFunction(self.model, name='step', pwl_def=PwlFunction._PwlAsSlopes([(0, 0), (10, 0)], 0, (-1, 0)))
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{0 -> 0;10 -> 0;0}(-1, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(0, [(0, 0), (0, 10)], 0)")
        self.assertEqual(pwl_def_as_breaks.get_nb_intervals(), 0)

    def test_pwl_slope_to_breaksxy3(self):
        pwl = PwlFunction(self.model, name='slope', pwl_def=PwlFunction._PwlAsSlopes([(0, 0), (10, 10)], 0, (0, 0)))
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(str(pwl), "{0 -> 0;10 -> 10;0}(0, 0)")
        self.assertEqual(pwl_def.to_string(), "{0 -> 0;10 -> 10;0}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(0, [(0, 0), (10, 100)], 0)")
        self.assertEqual(pwl_def_as_breaks.get_nb_intervals(), 1)

    def test_pwl_slope_to_breaksxy4(self):
        pwl = PwlFunction(self.model, name='slope', pwl_def=PwlFunction._PwlAsSlopes([(0, 0), (1, 10)], 0, (1, 0)))
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(str(pwl_def), "{0 -> 0;1 -> 10;0}(1, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(0, [(0, -1), (10, 9)], 0)")

    def test_pwl_slope_to_breaksxy5(self):
        pwl = PwlFunction(self.model, name='slope+steps', pwl_def=PwlFunction._PwlAsSlopes(
            [(5, 0), (1, 0), (0, 1), (2, 1)], -3, (0.5, -2)))
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(str(pwl_def), "{5 -> 0;1 -> 0;0 -> 1;2 -> 1;-3}(0.5, -2)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(5, [(0, -3.0), (0, -2.0), (1, -2.0), (1, 0.0)], -3)")

    def test_pwl_slope_to_breaksxy6(self):
        pwl = PwlFunction(self.model, name='trivial', pwl_def=PwlFunction._PwlAsSlopes([(10, 0)], 10))
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(str(pwl_def), "{10 -> 0;10}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(10, [(0, 0)], 10)")
        self.assertEqual(pwl_def_as_breaks.get_nb_intervals(), 0)


@unittest.skipUnless(the_env.has_cplex and the_env.cplex_version_as_tuple >= (12, 7, 0),
                     "Piecewise linear constraints require Cplex 12.7 or higher")
class PwlPiecewiseArithmeticTests(unittest.TestCase):
    cannot_modify_pwl_function_msg = 'Cannot modify a PWL function'

    def setUp(self):
        self.model = Model("piecewise")

    def tearDown(self):
        self.model.end()

    def get_pwl_sample_breaks_1(self, x0, name=None):
        return self.model.piecewise(1, [(x0, 0), (x0 + 2, 0), (x0 + 2, 3), (x0 + 5, 0)], 1, name=name)
        # return PwlFunction(self.model, name=name, pwl_def=PwlFunction._PwlAsBreaks(
        #     1, [(x0, 0), (x0 + 2, 0), (x0 + 2, 3), (x0 + 5, 0)], 1))

    def get_pwl_sample_slopes_1(self, x0, y0, name):
        return PwlFunction(self.model, name=name, pwl_def=PwlFunction._PwlAsSlopes(
            [(1, x0), (0, x0 + 2), (3, x0 + 2), (-1, x0 + 5)], 1, (x0, y0)))

    def test_pwl_arithmetic_one_pwl_eval(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        self.assertEqual(pwl_1.evaluate(-5), -5)
        self.assertEqual(pwl_1.evaluate(0), 0)
        self.assertEqual(pwl_1.evaluate(1.5), 0)
        six.assertRaisesRegex(self, DOcplexException, "Cannot evaluate PWL at a discontinuity",
                                lambda: pwl_1.evaluate(2))
        self.assertEqual(pwl_1.evaluate(3), 2)
        self.assertEqual(pwl_1.evaluate(5), 0)
        self.assertEqual(pwl_1.evaluate(7), 2)

    def test_pwl_arithmetic_one_pwl_1(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = pwl_1 + 42
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(1, [(0, 42), (2, 42), (2, 45), (5, 42)], 1)")
        self.assertEqual(pwl._pwl_def_as_breaks.get_nb_intervals(), 2)

    def test_pwl_arithmetic_one_pwl_1err(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self, DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1 + x)

    def test_pwl_arithmetic_one_pwl_1err2(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self, DOcplexException,
                                "Unsupported operation: x \+ \(1, \[\(0, 0\), \(2, 0\), \(2, 3\), \(5, 0\)\], 1\)",
                                lambda: x + pwl_1)

    def test_pwl_arithmetic_one_pwl_1err3(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self, DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1.pwl_def_as_breaks + x)

    def test_pwl_arithmetic_one_pwl_1radd(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = 42 + pwl_1
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(1, [(0, 42), (2, 42), (2, 45), (5, 42)], 1)")

    def test_pwl_arithmetic_one_pwl_1iadd(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        try:
            pwl_1 += 42
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_2(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = pwl_1 * 7
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(7, [(0, 0), (2, 0), (2, 21), (5, 0)], 7)")

    def test_pwl_arithmetic_one_pwl_2err(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self, DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1 * x)

    def test_pwl_arithmetic_one_pwl_2err2(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self,
            DOcplexException,
            "Multiply expects variable, expr or number, docplex.mp.pwl.PwlFunction\(preslope=1,breaksxy=\[",
            lambda: x * pwl_1)

    def test_pwl_arithmetic_one_pwl_2err3(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self,DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1.pwl_def_as_breaks * x)

    def test_pwl_arithmetic_one_pwl_2rmul(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = 7 * pwl_1
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(7, [(0, 0), (2, 0), (2, 21), (5, 0)], 7)")

    def test_pwl_arithmetic_one_pwl_2imul(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        try:
            pwl_1 *= 7
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_2div(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = pwl_1 / 2
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(0.5, [(0, 0.0), (2, 0.0), (2, 1.5), (5, 0.0)], 0.5)")

    def test_pwl_arithmetic_one_pwl_2div2(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        six.assertRaisesRegex(self,
            DOcplexException,
            "PWL function \(1, \[\(0, 0\), \(2, 0\), \(2, 3\), \(5, 0\)\], 1\) cannot be used as denominator of 2",
            lambda: 2 / pwl_1)

    def test_pwl_arithmetic_one_pwl_idiv(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        try:
            pwl_1 /= 2
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_3(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = (pwl_1 + 3) * 7
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(7, [(0, 21), (2, 21), (2, 42), (5, 21)], 7)")

    def test_pwl_arithmetic_one_pwl_4(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = pwl_1 * 0
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(0, [(0, 0)], 0)")

    def test_pwl_arithmetic_one_pwl_4imul(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        try:
            pwl_1 *= 0
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_5(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = pwl_1.translate(42)
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(1, [(42, 0), (44, 0), (44, 3), (47, 0)], 1)")

    def test_pwl_arithmetic_one_pwl_5err(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        six.assertRaisesRegex(self, DOcplexException,
                                "Invalid type for argument: \[42\].",
                                lambda: pwl_1.translate([42]))

    def test_pwl_arithmetic_one_pwl_5err2(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        six.assertRaisesRegex(self, DOcplexException,
                                "Invalid type for argument: \[42\].",
                                lambda: pwl_1.pwl_def.translate([42]))

    def test_pwl_arithmetic_one_pwl_6(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl = pwl_1 - 42
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.assertEqual(pwl_def.to_string(), "(1, [(0, -42), (2, -42), (2, -39), (5, -42)], 1)")

    def test_pwl_arithmetic_one_pwl_6err(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self,DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1 - x)

    def test_pwl_arithmetic_one_pwl_6err2(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self,DOcplexException,
                                "Unsupported operation: x \- \(1, \[\(0, 0\), \(2, 0\), \(2, 3\), \(5, 0\)\], 1\)",
                                lambda: x - pwl_1)

    def test_pwl_arithmetic_one_pwl_6err3(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self,DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1.pwl_def_as_breaks - x)

    def test_pwl_arithmetic_one_pwl_6isub(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        try:
            pwl_1 -= 42
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_slopes_1(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl = pwl_1 + 42
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1 -> 0;0 -> 2;3 -> 2;-1 -> 5;1}(0, 42)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(1, [(0, 42), (2, 42), (2, 45), (5, 42)], 1)")

    def test_pwl_arithmetic_one_pwl_slopes_1radd(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl = 42 + pwl_1
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1 -> 0;0 -> 2;3 -> 2;-1 -> 5;1}(0, 42)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(1, [(0, 42), (2, 42), (2, 45), (5, 42)], 1)")

    def test_pwl_arithmetic_one_pwl_slopes_1iadd(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        try:
            pwl_1 += 42
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_slopes_1err(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self,DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1.pwl_def + x)

    def test_pwl_arithmetic_one_pwl_slopes_2(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl = pwl_1 * 7
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{7 -> 0;0 -> 2;21 -> 2;-7 -> 5;7}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(7, [(0, 0), (2, 0), (2, 21), (5, 0)], 7)")

    def test_pwl_arithmetic_one_pwl_slopes_2imul(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        try:
            pwl_1 *= 7
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_slopes_2err(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self,DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1.pwl_def - x)

    def test_pwl_arithmetic_one_pwl_slopes_3(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl = (pwl_1 + 3) * 7
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{7 -> 0;0 -> 2;21 -> 2;-7 -> 5;7}(0, 21)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(7, [(0, 21), (2, 21), (2, 42), (5, 21)], 7)")

    def test_pwl_arithmetic_one_pwl_slopes_4(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl = pwl_1 * 0
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{0}(0, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(0, [(0, 0)], 0)")

    def test_pwl_arithmetic_one_pwl_slopes_4err(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        x = self.model.continuous_var(name="x")
        six.assertRaisesRegex(self,DOcplexException, "Invalid type for right hand side operand: x.",
                                lambda: pwl_1.pwl_def * x)

    def test_pwl_arithmetic_one_pwl_slopes_4imul(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        try:
            pwl_1 *= 0
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_slopes_5(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl = pwl_1.translate(42)
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1 -> 42;0 -> 44;3 -> 44;-1 -> 47;1}(42, 0)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(1, [(42, 0), (44, 0), (44, 3), (47, 0)], 1)")

    def test_pwl_arithmetic_one_pwl_slopes_5err(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        six.assertRaisesRegex(self,DOcplexException,
                                "Invalid type for argument: \[42\].",
                                lambda: pwl_1.translate([42]))

    def test_pwl_arithmetic_one_pwl_slopes_5err2(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        six.assertRaisesRegex(self,DOcplexException,
                                "Invalid type for argument: \[42\].",
                                lambda: pwl_1.pwl_def.translate([42]))

    def test_pwl_arithmetic_one_pwl_slopes_6(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl = pwl_1 - 42
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1 -> 0;0 -> 2;3 -> 2;-1 -> 5;1}(0, -42)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(1, [(0, -42), (2, -42), (2, -39), (5, -42)], 1)")

    def test_pwl_arithmetic_one_pwl_slopes_6isub(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        try:
            pwl_1 -= 42
            self.fail('expecting exception, not raised.')
        except DOcplexException as e:
            self.assertIn(self.cannot_modify_pwl_function_msg, str(e))

    def test_pwl_arithmetic_one_pwl_slopes_6rsub(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl = 42 - pwl_1
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{-1 -> 0;0 -> 2;-3 -> 2;1 -> 5;-1}(0, 42)")
        self.assertEqual(pwl_def_as_breaks.to_string(), "(-1, [(0, 42), (2, 42), (2, 39), (5, 42)], -1)")

    def is_equal(self, pwl_def, pwl_def2):
        if pwl_def.preslope != pwl_def2.preslope:
            return False
        if pwl_def.postslope != pwl_def2.postslope:
            return False
        if len(pwl_def.breaksxy) != len(pwl_def2.breaksxy):
            return False
        for b_l, b_r in zip(pwl_def.breaksxy, pwl_def2.breaksxy):
            if b_l != b_r:
                return False
        return True

    def check_equal(self, pwl_def, pwl_def2):
        self.assertTrue(self.is_equal(pwl_def, pwl_def2), "PWLs are different:" +
                        pwl_def.to_string() + " != " + pwl_def2.to_string())

    def test_pwl_arithmetic_two_pwl_1(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsBreaks(0, (0, 0), 0))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.assertEqual(pwl_def, pwl.pwl_def_as_breaks)
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(1, [(0, 0), (2, 0), (2, 3), (5, 0)], 1))

    def test_pwl_arithmetic_two_pwl_1err(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsBreaks(0, (0, 0), 0))
        six.assertRaisesRegex(self,DOcplexException, "Invalid type for right hand side operand: \(0, \[\(0, 0\)\], 0\).",
                                lambda: pwl_1 * pwl_2)

    def test_pwl_arithmetic_two_pwl_2(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsBreaks(1, (-10, -10), 1))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def, PwlFunction._PwlAsBreaks(2, [(0, 0), (2, 2), (2, 5), (5, 5)], 2))

    def test_pwl_arithmetic_two_pwl_3(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsBreaks(-1, (10, -10), -1))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def, PwlFunction._PwlAsBreaks(0, [(0, 0), (2, -2), (2, 1), (5, -5)], 0))

    def test_pwl_arithmetic_two_pwl_4(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_breaks_1(0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(2, [(0, 0), (2, 0), (2, 6), (5, 0)], 2))

    def test_pwl_arithmetic_two_pwl_5(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_breaks_1(-10, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(2, [(-10, -10), (-8, -8), (-8, -5), (-5, -5), (0, 5), (2, 7), (2, 10),
                                                      (5, 10)], 2))

    def test_pwl_arithmetic_two_pwl_6(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_breaks_1(1, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(2, [(0, -1), (1, 0), (2, 0), (2, 3), (3, 2), (3, 5), (5, 1), (6, 1)],
                                                  2))

    def test_pwl_arithmetic_two_pwl_7(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_breaks_1(2, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(2, [(0, -2), (2, 0), (2, 3), (4, 1), (4, 4), (5, 2), (7, 2)], 2))

    def test_pwl_arithmetic_two_pwl_8(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_breaks_1(3, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(2, [(0, -3), (2, -1), (2, 2), (3, 2), (5, 0), (5, 3), (8, 3)], 2))

    def test_pwl_arithmetic_two_pwl_9(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_breaks_1(4, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(2, [(0, -4), (2, -2), (2, 1), (4, 1), (5, 0), (6, 1), (6, 4), (9, 4)],
                                                  2))

    def test_pwl_arithmetic_two_pwl_10(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_breaks_1(5, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(2, [(0, -5), (2, -3), (2, 0), (5, 0), (7, 2), (7, 5), (10, 5)], 2))

    def test_pwl_arithmetic_two_pwl_11(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_breaks_1(6, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def,
                         PwlFunction._PwlAsBreaks(2,
                                                  [(0, -6), (2, -4), (2, -1), (5, -1), (6, 1), (8, 3), (8, 6), (11, 6)],
                                                  2))

    def test_pwl_arithmetic_two_pwl_12(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsBreaks(1, (-10, -10), 1))
        pwl = pwl_1 - pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def, PwlFunction._PwlAsBreaks(0, [(0, 0), (2, -2), (2, 1), (5, -5)], 0))

    def test_pwl_arithmetic_two_pwl_slopes_1(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([], 0))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1 -> 0;0 -> 2;3 -> 2;-1 -> 5;1}(0, 0)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(1, [(0, 0), (2, 0), (2, 3), (5, 0)], 1))

    def test_pwl_arithmetic_two_pwl_slopes_1b(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([], 0))
        pwl = pwl_2 + pwl_1
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1 -> 0;0 -> 2;3 -> 2;-1 -> 5;1}(0, 0)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(1, [(0, 0), (2, 0), (2, 3), (5, 0)], 1))

    def test_pwl_arithmetic_two_pwl_slopes_2(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([], 1, (-10, -10)))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{2 -> 0;1 -> 2;3 -> 2;0 -> 5;2}(5, 5)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(2, [(0, 0), (2, 2), (2, 5), (5, 5)], 2))

    def test_pwl_arithmetic_two_pwl_slopes_3(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([], -1, (10, -10)))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{0 -> 0;-1 -> 2;3 -> 2;-2 -> 5;0}(10, -5)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(0, [(0, 0), (2, -2), (2, 1), (5, -5)], 0))

    def test_pwl_arithmetic_two_pwl_slopes_3b(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([], -1, (10, -10)))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), pwl_def_as_breaks.to_string())
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(0, [(0, 0), (2, -2), (2, 1), (5, -5)], 0))

    def test_pwl_arithmetic_two_pwl_slopes_4(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{2 -> 0;0 -> 2;6 -> 2;-2 -> 5;2}(0, 0)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(2, [(0, 0), (2, 0), (2, 6), (5, 0)], 2))

    def test_pwl_arithmetic_two_pwl_slopes_5(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_slopes_1(-10, 0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(),
                         "{2 -> -10;1 -> -8;3 -> -8;0 -> -5;2 -> 0;1 -> 2;3 -> 2;0 -> 5;2}(5, 10)")
        self.check_equal(pwl_def_as_breaks,
                         PwlFunction._PwlAsBreaks(2, [(-10, -10), (-8, -8), (-8, -5), (-5, -5), (0, 5), (2, 7), (2, 10),
                                                      (5, 10)], 2))

    def test_pwl_arithmetic_two_pwl_slopes_6(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_slopes_1(1, 0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(),
                         "{2 -> 0;1 -> 1;0 -> 2;3 -> 2;-1 -> 3;3 -> 3;-2 -> 5;0 -> 6;2}(6, 1)")
        self.check_equal(pwl_def_as_breaks,
                         PwlFunction._PwlAsBreaks(2, [(0, -1), (1, 0), (2, 0), (2, 3), (3, 2), (3, 5), (5, 1), (6, 1)],
                                                  2))

    def test_pwl_arithmetic_two_pwl_slopes_7(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_slopes_1(2, 0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(),
                         "{2 -> 0;1 -> 2;3 -> 2;-1 -> 4;3 -> 4;-2 -> 5;0 -> 7;2}(7, 2)")
        self.check_equal(pwl_def_as_breaks,
                         PwlFunction._PwlAsBreaks(2, [(0, -2), (2, 0), (2, 3), (4, 1), (4, 4), (5, 2), (7, 2)], 2))

    def test_pwl_arithmetic_two_pwl_slopes_8(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_slopes_1(3, 0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(),
                         "{2 -> 0;1 -> 2;3 -> 2;0 -> 3;-1 -> 5;3 -> 5;0 -> 8;2}(8, 3)")
        self.check_equal(pwl_def_as_breaks,
                         PwlFunction._PwlAsBreaks(2, [(0, -3), (2, -1), (2, 2), (3, 2), (5, 0), (5, 3), (8, 3)], 2))

    def test_pwl_arithmetic_two_pwl_slopes_9(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_slopes_1(4, 0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(),
                         "{2 -> 0;1 -> 2;3 -> 2;0 -> 4;-1 -> 5;1 -> 6;3 -> 6;0 -> 9;2}(9, 4)")
        self.check_equal(pwl_def_as_breaks,
                         PwlFunction._PwlAsBreaks(2, [(0, -4), (2, -2), (2, 1), (4, 1), (5, 0), (6, 1), (6, 4), (9, 4)],
                                                  2))

    def test_pwl_arithmetic_two_pwl_slopes_10(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_slopes_1(5, 0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(),
                         "{2 -> 0;1 -> 2;3 -> 2;0 -> 5;1 -> 7;3 -> 7;0 -> 10;2}(10, 5)")
        self.check_equal(pwl_def_as_breaks,
                         PwlFunction._PwlAsBreaks(2, [(0, -5), (2, -3), (2, 0), (5, 0), (7, 2), (7, 5), (10, 5)], 2))

    def test_pwl_arithmetic_two_pwl_slopes_11(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = self.get_pwl_sample_slopes_1(6, 0, 'pwl_2')
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(),
                         "{2 -> 0;1 -> 2;3 -> 2;0 -> 5;2 -> 6;1 -> 8;3 -> 8;0 -> 11;2}(11, 6)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(
            2, [(0, -6), (2, -4), (2, -1), (5, -1), (6, 1), (8, 3), (8, 6), (11, 6)], 2))

    def test_pwl_arithmetic_two_pwl_slopes_12(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([], 1, (-10, -10)))
        pwl = pwl_1 - pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{0 -> 0;-1 -> 2;3 -> 2;-2 -> 5;0}(5, -5)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(0, [(0, 0), (2, -2), (2, 1), (5, -5)], 0))

    def test_pwl_arithmetic_two_pwl_slopes_12b(self):
        pwl_1 = self.get_pwl_sample_breaks_1(0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([], 1, (-10, -10)))
        pwl = pwl_1 - pwl_2
        pwl_def = pwl.pwl_def
        self.check_equal(pwl_def, PwlFunction._PwlAsBreaks(0, [(0, 0), (2, -2), (2, 1), (5, -5)], 0))

    def test_pwl_arithmetic_two_pwl_slopes_13(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([(0, 0), (2, 0)], 0, (-10, 0)))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1 -> 0;2 -> 0;0 -> 2;3 -> 2;-1 -> 5;1}(5, 2)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(1, [(0, 0), (0, 2), (2, 2), (2, 5), (5, 2)], 1))

    def test_pwl_arithmetic_two_pwl_slopes_14(self):
        pwl_1 = self.get_pwl_sample_slopes_1(0, 0, 'pwl_1')
        pwl_2 = PwlFunction(self.model, name='pwl_2', pwl_def=PwlFunction._PwlAsSlopes([(0, 10), (2, 10)], 0, (-10, 0)))
        pwl = pwl_1 + pwl_2
        pwl_def = pwl.pwl_def
        pwl_def_as_breaks = pwl.pwl_def_as_breaks
        self.assertEqual(pwl_def.to_string(), "{1 -> 0;0 -> 2;3 -> 2;-1 -> 5;1 -> 10;2 -> 10;1}(11, 8)")
        self.check_equal(pwl_def_as_breaks, PwlFunction._PwlAsBreaks(
            1, [(0, 0), (2, 0), (2, 3), (5, 0), (10, 5), (10, 7)], 1))

    #@unittest.skip('waiting for rtc32887 from cplex...')
    def test_rtc32887(self):
        with Model(name='rtc32887') as m:
            y = m.continuous_var(name='y')
            x = m.continuous_var(name='x', lb=9614.41701242187, ub=15108.3695909487)
            pwl = PwlFunction(m, PwlFunction._PwlAsBreaks(preslope=-0.0439939614235899,
                                                          breaksxy=[(13646.2693079536, 1070.63191855911),
                                                                    (14101.1449515521, 1053.94774633369),
                                                                    (14587.3913291918, 1036.11294154099)
                                                                    ],
                                                          postslope=-0.0279465562863657))
            #m.add(y == m._add_pwl(pwl_func=pwl, arg=x))
            m.add_piecewise_constraint(y, pwl, x)
            m.minimize(y)
            if m._can_solve():
                s = m.solve()
                self.assertIsNotNone(s)
                # we get y=14601.950875, x= 1021.55
                print('y=f(x)={0}, x={1}'.format(y.solution_value, x.solution_value))
                arg1_value = x.solution_value
                # but when we evaluate f(x) without cplex, we get 1 adifferent value!!!
                eval1 = pwl.evaluate(x_val=arg1_value)
                self.assertAlmostEqual(y.solution_value, eval1, 1)

    def test_pwl_repr(self):
        pw1 = self.get_pwl_sample_breaks_1(x0=1)
        self.assertEqual('docplex.mp.pwl.PwlFunction(preslope=1,breaksxy=[(1, 0), (3, 0), (3, 3), (6, 0)],postslope=1)', repr(pw1))

    def test_pwl_alone_has_no_pwl(self):
        mdl = self.model
        pwl = mdl.piecewise(0, [(-1, 0), (-1, 1)], 0, "step")
        self.assertFalse(mdl._has_piecewise())

        x, y = mdl.continuous_var_list(keys=['x', 'y'], name=str)
        pwl = mdl.piecewise(0, [(-1, 0), (-1, 1)], 0, "step")
        mdl.add_constraint(y == pwl(x), ctname="c1")
        self.assertTrue(mdl._has_piecewise())


if __name__ == "__main__":
    unittest.main()
