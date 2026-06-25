import unittest

from six import iteritems, assertRaisesRegex

from docplex.mp.utils import is_iterable
from docplex.mp.model import Model
from docplex.mp.basic import Expr
from docplex.mp.linear import LinearExpr, MonomialExpr
from docplex.mp.quad import QuadExpr
from docplex.mp.operand import Operand, LinearOperand
from docplex.mp.error_handler import DOcplexException
from docplex.mp.linear import ZeroExpr

class ExpressionBaseTests(unittest.TestCase):
    def setUp(self):
        m = Model()

        self.x = m.continuous_var(name='x')
        self.y = m.continuous_var(name='y')
        self.z = m.continuous_var(name='z')
        self.zero = m.linear_expr(arg=0)
        self.xexpr =  m.linear_expr(arg=self.x)
        # a constant linear expression with 3
        self.expr3 = m.linear_expr(arg=3)
        # a constant expression
        self.kexpr7 = m._lfactory._new_constant_expr(7, safe_number=True)
        self.ijk = m.integer_var(name='ijk')
        self.model = m

    def tearDown(self):
        self.model.end()
        self.model = None

    def check_expression(self, expr, expected_terms=None, expected_cst=0,
                         expected_str=None):
        self.assertEqual(expr.constant, expected_cst)
        if expected_terms is not None:
            #self.assertEqual(len(expected_terms), expr.number_of_variables())
            for v, k in iteritems(expected_terms):
                self.assertEqual(k, expr[v])
        if expected_str is not None:
            self.assertEqual(expected_str, str(expr))

    def check_not_equal(self, expr1, expr2):
        # principle of symmetry
        self.assertFalse(expr1.equals(expr2))
        self.assertFalse(expr2.equals(expr1))

    def check_equal(self, expr1, expr2):
        # principle of symmetry
        self.assertTrue(expr1.equals(expr2))
        self.assertTrue(expr2.equals(expr1))


class OperandTests(ExpressionBaseTests):
    # check that various expr tests are operands

    def test_var_operand(self):
        # check iter_variables
        x = self.x
        self.assertIs(x, x.as_variable())
        # iterators
        self.assertEqual([x], list(x.iter_variables()))
        self.assertEqual([], list(x.iter_quads()))
        self.assertEqual([(x, 1)], list(x.iter_terms()))
        self.assertEqual(0, x.get_constant())
        self.assertFalse(x.is_constant())
        self.assertTrue(x in x)
        self.assertIsNone(x.as_constant())

    def test_mn_operand(self):
        # check iter_variables
        mn = 7 * self.x
        x = self.x
        self.assertIsNone(mn.as_variable())
        # iterators
        self.assertEqual([x], list(mn.iter_variables()))
        self.assertEqual([], list(x.iter_quads()))
        self.assertEqual([(x, 7)], list(mn.iter_terms()))
        self.assertEqual(0, mn.get_constant())
        self.assertFalse(mn.is_constant())
        self.assertIsNone(x.as_constant())
        self.assertTrue(x in mn)

    def test_lin_operand(self):
        m = self.model
        # check iter_variables
        lin = 17 * self.y + 7 * self.x + 33
        x = self.x
        y = self.y
        self.assertIsNone(lin.as_variable())

        # iterators
        listvars = list(lin.iter_variables())
        self.assertEqual(2, len(listvars))

        if m.keep_ordering:
            self.assertIs(y, listvars[0])
            self.assertIs(x, listvars[1])
        else:
            # nothing is guaranteed: in python 3.6, ordering is OK
            # just check both x and y are in the list
            ordered = (x is listvars[1]) and (y is listvars[0])
            not_ordered = (x is listvars[0]) and (y is listvars[1])
            self.assertTrue( ordered or not_ordered)
        self.assertEqual([], list(x.iter_quads()))

        self.assertEqual(33, lin.get_constant())
        self.assertFalse(lin.is_constant())
        self.assertIsNone(x.as_constant())
        self.assertTrue(x in lin)
        self.assertTrue(y in lin)

    def test_link_operand(self):
        lk = self.model.lfactory.linear_expr(arg=None, constant=3.14)
        self.assertTrue(lk.is_constant())
        self.assertEqual(3.14, lk.as_constant())

    def test_abs_operand(self):
        m = self.model
        absx = m.abs(self.x)
        # constant protocol
        self.assertEqual(0, absx.get_constant())
        self.assertFalse(absx.is_constant())
        # iter terms protocol
        abs_terms = list(absx.iter_terms())
        self.assertEqual(1, len(abs_terms))
        self.assertEqual(1, abs_terms[0][1])
        abs_vars = list(absx.iter_variables())
        self.assertEqual(1, len(abs_vars))

    def test_max_operand(self):
        m = self.model
        max_x = m.max(self.x, self.y)
        # constant protocol
        self.assertEqual(0, max_x.get_constant())
        self.assertFalse(max_x.is_constant())
        # iter terms protocol
        max_terms = list(max_x.iter_terms())
        self.assertEqual(1, len(max_terms))
        self.assertEqual(1, max_terms[0][1])

    def test_iterable(self):
        self.assertFalse(is_iterable(Operand()))
        self.assertFalse(is_iterable(LinearOperand()))

    def test_zero_op(self):
        zz = ZeroExpr(self.model)
        self.assertEqual(0, zz.as_constant())
        self.assertTrue(zz.is_constant())

    def test_constant_expr_op(self):
        kk = self.model.lfactory.constant_expr(3.14)
        self.assertTrue(kk.is_constant())
        self.assertEqual(3.14, kk.as_constant())


class LinearExprInitTests(ExpressionBaseTests):
    # all ways to build a linexpr

    def test_empty_linear_expr(self):
        mdl = self.model
        e = mdl.linear_expr()
        self.assertIsInstance(e, LinearExpr)
        self.assertEqual(0, e.constant)
        self.assertEqual(0, e.number_of_variables())
        self.assertFalse(e._transient)
        self.assertTrue(e.is_zero())
        self.assertTrue(e.is_constant())

    def test_one_expr(self):
        m = self.model
        oneExpr = m.linear_expr(arg=1)
        self.assertEqual(1, oneExpr.constant, "unexpected constant, expecting 0")
        self.assertTrue(oneExpr.is_constant())
        self.assertEqual(1, oneExpr.constant)

    def test_constant_int_expr(self):
        m = self.model
        expr = m.linear_expr(arg=13)
        #self.assertFalse(expr.has_name())
        self.assertEqual(13, expr.constant, "unexpected constant, expecting 0")
        self.assertFalse(expr.is_zero())
        self.assertTrue(expr.is_constant())

    def test_constant_float_expr(self):
        m = self.model
        expr = m.linear_expr(arg=-3.14)
        self.assertIsNone(expr.name)
        self.assertEqual(-3.14, expr.constant, "unexpected constant, expecting -pi")
        self.assertFalse(expr.is_zero())
        self.assertTrue(expr.is_constant())

    def test_expr_from_constant(self):
        mdl = self.model
        e = mdl.linear_expr(arg=7)
        self.assertTrue(e.is_constant())
        self.assertEqual(7, e.constant)
        e.constant = 11
        self.assertEqual(11, e.constant)

    def test_expr_from_var(self):
        y = self.y
        expr = y.to_linear_expr()
        self.assertEqual([y], list(expr.iter_variables()))
        self.assertEqual('y', str(expr))
        self.assertIs(y, expr.as_variable())

    def test_linexpr_as_var(self):
        y = self.y
        vd1 = {y : 1.0}
        liny1 = self.model.linear_expr(arg=vd1)
        self.assertIs(y, liny1.as_variable())

    def test_expr_from_monomial(self):
        mdl = self.model
        y = self.y
        z = 2 * y
        mnm = MonomialExpr(self.model, y, 3.14)
        expr = mdl.linear_expr(mnm)
        self.assertIsInstance(expr, LinearExpr)
        self.assertEqual(expr[y], 3.14)

    def test_monomial_as_var(self):
        mn1 = MonomialExpr(self.model, self.y, 1.0)
        self.assertIs(self.y, mn1.as_variable())

    def test_linear_expr_from_tuple(self):
        m = self.model
        e = m.linear_expr((self.x, 7))
        self.assertEqual('7x', str(e))
        self.assertEqual(0, e.get_constant())

    def test_linear_expr_from_expr(self):
        m = self.model
        # one expr
        expr1 = m.linear_expr((self.x, 7))
        # build an expr from an expr : gives a copy
        expr2 = m.linear_expr(expr1)
        self.assertIsInstance(expr2, Expr)
        self.assertTrue(expr1.equals(expr2))

    def test_linear_expr_from_string(self):
        assertRaisesRegex(self, DOcplexException, "Cannot convert 'foo' to docplex.mp.LinearExpr",\
                                lambda m: m.linear_expr(arg='foo'), self.model)


    def test_linearexpr_set(self):
        expr1 = 7 * self.x + 3
        expr2 = 15 * self.y + 7
        xset = {expr1, expr2}
        self.assertEqual(2, len(xset))
        self.assertIn(expr1, xset)
        self.assertIn(expr2, xset)


class LinearExprEqualityTests(ExpressionBaseTests):

    def test_var_equals_itself(self):
        xx = self.x
        self.assertTrue(xx, xx)

    def test_var_not_equals_num(self):
        self.assertFalse(self.x.equals(3.14))

    def test_var_equals_monomial1(self):
        xx = self.x
        mn = xx * 2
        mn /= 2
        self.assertTrue(xx.equals(mn))

    def test_var_not_equals_mn_coef(self):
        xx = self.x
        self.assertFalse(xx.equals(5*xx))

    def test_var_not_equals_mn_var(self):
        xx = self.x
        yy = self.y
        mny = 2 * yy
        mny /= 2
        self.assertFalse(xx.equals(mny))

    def test_var_equals_linexpr(self):
        xx = self.x
        lexpr = self.model.linear_expr(xx)
        self.assertTrue(xx.equals(lexpr))

    def test_var_not_equals_linexpr_var(self):
        xx = self.x
        yy = self.y
        lexpr = self.model.linear_expr(yy)
        self.assertFalse(xx.equals(lexpr))

    def test_var_not_equals_linexpr_vars(self):
        xx = self.x
        yy = self.y
        self.assertFalse(xx.equals(xx+yy))

    def test_var_not_equals_linexpr_cst(self):
        xx = self.x
        lexpr = xx + 7
        self.assertFalse(xx.equals(lexpr))

    def test_expr_not_equal_str(self):
        expr1 = self.x + 1
        self.assertFalse(expr1.equals("foo"))

    def test_expr_equals(self):
        xx = self.x
        yy = self.y
        e1 = 2 * xx + 3 * yy + 5
        e2 = 2 * xx + 3 * yy + 5
        self.assertTrue(e1.equals(e2))

    def test_expr_not_equals_cst(self):
        expr1 = self.x + 1
        expr2 = self.x + 3
        self.check_not_equal(expr1, expr2)

    def test_numexpr_equals_num(self):
        expr3 = self.model.linear_expr(3)
        self.assertTrue(expr3.equals(3))
        self.assertFalse(expr3.equals(7))

    def test_kexpr_equals_num(self):
        kexpr7 = self.kexpr7
        self.assertTrue(kexpr7.equals(7))
        self.assertFalse(kexpr7.equals(8))

    def test_kexpr_equals_kexpr(self):
        kexpr7 = self.kexpr7
        self.assertTrue(kexpr7, self.model._lfactory._new_constant_expr(7))
        self.assertTrue(kexpr7, self.model._lfactory._new_constant_expr(11))
        self.assertTrue(kexpr7, self.model.linear_expr(arg=7))
        self.assertTrue(kexpr7, self.model.linear_expr(arg=111))

    def test_expr_not_equals_coeff(self):
        expr1 = self.x + 1
        expr2 = 2 * self.x + 1
        self.check_not_equal(expr1, expr2)

    def test_expr_not_equals_nb_vars(self):
        expr1 = self.x + 1
        expr2 = self.x + self.y + 1
        self.check_not_equal(expr1, expr2)

    def test_expr_not_equals_cst_expr(self):
        expr1 = self.model.linear_expr(arg=7)
        expr2 = self.x + 1
        self.check_not_equal(expr1, expr2)

    def test_expr_not_equals_var2(self):
        expr = self.x + 3
        self.assertFalse(expr.equals(self.y))

    # expressions that are equal but have different types
    def test_var_equal_linepr_var(self):
        lexpr = self.model.linear_expr(self.x)
        self.assertTrue(lexpr.equals(self.x))

    def test_var_equal_mn(self):
        lexpr = self.model.linear_expr(self.x)
        lexpr *= 3
        self.assertTrue(lexpr.equals(3*self.x))

    def test_var_not_equals_quad(self):
        xx = self.x
        q = (xx ** 2) + 3 * xx + 7
        l3 = 3 * xx + 7
        self.check_not_equal(l3, q)


class LinearExpressionCloneTests(ExpressionBaseTests):
    def check_clone(self, original):
        cloned = original.clone()
        self.assertFalse(original is cloned)
        self.assertEqual(str(original), str(cloned))
        self.assertTrue(original.equals(cloned))
        self.assertTrue(cloned.equals(original))

    def test_linxpr_clone_empty(self):
        expr = self.model.linear_expr()
        self.check_clone(expr)

    def test_linxpr_clone_cst(self):
        expr = self.model.linear_expr(arg=7)
        self.check_clone(expr)

    def test_linxpr_clone_var(self):
        expr = self.model.linear_expr(arg=self.x)
        self.check_clone(expr)

    def test_linxpr_clone_full(self):
        expr = 3 * self.x - 5 * self.y + 7 * self.z + 11
        self.check_clone(expr)


class LinearExpressionNegateTests(ExpressionBaseTests):

    def check_opposite(self, e1, e2):
        self.assertEqual(e1.constant, -e2.constant)
        self.assertEqual(e1.number_of_variables(), e2.number_of_variables())
        for v in e1.iter_variables():
            self.assertEqual(e1[v], -e2[v])

    def test_opposite_empty(self):
        e = self.model.linear_expr()
        minus_e = e.opposite()
        self.assertTrue(minus_e.equals(e))

    def test_opposite_cst(self):
        e = self.model.linear_expr(arg=-7)
        minus_e = e.opposite()
        self.check_opposite(e, minus_e)

    def test_opposite_var(self):
        e = self.model.linear_expr(arg=self.x)
        minus_e = e.opposite()
        self.check_opposite(e, minus_e)

    def test_opposite_full(self):
        expr = 3 * self.x - 5 * self.y + 7 * self.z + 11
        copy_of_expr = expr.clone()
        minus_expr = expr.opposite()
        self.check_opposite(copy_of_expr, minus_expr)

    def test_negate_empty(self):
        e = self.model.linear_expr()
        e_before = e.clone()
        e.negate()
        self.assertTrue(e.equals(e_before))

    def test_negate_cst(self):
        e = self.model.linear_expr(arg=-77)
        e_before = e.clone()
        e.negate()
        self.check_opposite(e_before, e)

    def test_negate_var(self):
        e = self.model.linear_expr(arg=self.x)
        e_before = e.clone()
        e.negate()
        self.check_opposite(e_before, e)

    def test_negate_full(self):
        e = 3 * self.x - 5 * self.y + 7 * self.z + 11
        e_before = e.clone()
        e.negate()
        self.check_opposite(e_before, e)

    def test_uminus_cst(self):
        e = self.model.linear_expr(arg=-7)
        minus_e = -e
        self.check_opposite(e, minus_e)

    def test_uminus_var(self):
        e = self.model.linear_expr(arg=self.x)
        minus_e = -e
        self.check_opposite(e, minus_e)

    def test_uminus_linexpr(self):
        e = 3 * self.x - 5 * self.y + 7 * self.z + 11
        minus_e = -e
        self.check_expression(minus_e, expected_terms={self.x: -3,
                                                       self.y: 5,
                                                       self.z: -7}, expected_cst=-11)


class LinearExpressionAddTests(ExpressionBaseTests):
    def check_add_result(self, initial, added, expected_cst=0, expected_terms=None, expected_str=None):
        copy_of_initial = initial.clone()
        ls = added + initial
        rs = copy_of_initial + added
        # check that left and right are equal
        self.check_equal(ls, rs)
        # chek result
        self.check_expression(ls, expected_terms=expected_terms,
                              expected_cst=expected_cst,
                              expected_str=expected_str)

    def test_add_zero_cst_expr(self):
        self.check_add_result(initial=self.expr3, added=0, expected_cst=3, expected_str='3')

    def test_add_zero_kexpr(self):
        self.check_add_result(initial=self.kexpr7, added=0, expected_cst=7, expected_str='7')

    def test_add_zero_var(self):
        self.check_add_result(initial=self.xexpr, added=0, expected_terms={self.x: 1})

    def test_add_zero_linexp(self):
        xexpr = 3 * self.x - 5 * self.y + 7 * self.z + 11
        self.check_add_result(initial=xexpr, added=0, expected_cst=11, expected_str='3x-5y+7z+11')

    def test_add_num_cst(self):
        self.check_add_result(initial=self.expr3, added=1, expected_cst=4, expected_terms={}, expected_str='4')

    def test_add_num_kexpr(self):
        self.check_add_result(initial=self.kexpr7, added=1, expected_cst=8, expected_terms={}, expected_str='8')


    def test_add_num_var(self):
        self.check_add_result(initial=self.xexpr, added=1, expected_cst=1, expected_terms={self.x: 1}, expected_str='x+1')

    def test_add_num_linexp(self):
        e = 3 * self.x - 5 * self.y + 7 * self.z + 11
        self.check_add_result(initial=e, added=1, expected_cst=12, expected_terms={self.x: 3,
                                                                                   self.y: -5,
                                                                                   self.z: 7})

    def test_add_kexpr_num(self):
        self.check_add_result(initial=self.kexpr7, added=1, expected_cst=8, expected_terms={})

    def test_add_kexpr_kexpr(self):
        self.check_add_result(initial=self.kexpr7, added=self.kexpr7, expected_cst=14, expected_terms={})

    def test_add_kexpr_var(self):
        self.check_add_result(initial=self.kexpr7, added=self.x, expected_cst=7, expected_terms={self.x: 1})

    def test_add_kexpr_mn(self):
        mn3 = 3 * self.y
        self.check_add_result(initial=self.kexpr7, added=mn3, expected_cst=7, expected_terms={self.y: 3})

    def test_add_kexpr_lin(self):
        e = self.model.linear_expr(3 * self.x - 5 * self.y + 11)
        self.check_add_result(initial=self.kexpr7, added=e, expected_cst=18, expected_terms={self.x: 3, self.y: -5})

    def test_add_var_var(self):
        xexpr = self.model.linear_expr(arg=self.x)
        self.check_add_result(initial=xexpr, added=self.x, expected_terms={self.x: 2})

    def test_add_var_var_collapse(self):
        xexpr = self.model.linear_expr(arg=(self.x, -1))
        xexpr.constant = 77
        #
        self.check_add_result(initial=xexpr, added=self.x, expected_terms={}, expected_cst=77)

    def test_add_var_lin(self):
        e = self.model.linear_expr(3 * self.x - 5 * self.y + 7 * self.z + 11)  # force non transient
        self.check_add_result(initial=self.xexpr, added=e,
                              expected_terms={self.x: 4,
                                              self.y: -5,
                                              self.z: 7}, expected_cst=11)

    def test_add_lin_lin(self):
        e1 = self.model.linear_expr(3 * self.x - 5 * self.y + 7 * self.z + 11)
        e2 = self.model.linear_expr(3 * self.x + 5 * self.y + 7 * self.z + 11)
        self.check_add_result(initial=e1, added=e2,
                              expected_terms={self.x: 6,
                                              self.z: 14}, expected_cst=22)

    # --- iadd
    def test_linexpr_iadd_num(self):
        e = 2 * self.x + 3 * self.y + 55
        e += 44
        self.assertEqual(e.constant, 99)
        self.assertEqual('2x+3y+99', str(e))

    def test_linexpr_iadd_var(self):
        x = self.x
        y = self.y
        z = self.z
        e = 2 * x + 3 * y + 55
        # a l origine z pas dans e
        self.assertFalse(z in e)
        self.assertEqual(2, e.number_of_variables())
        e += z
        # now z is in e
        self.assertEqual(e[z], 1)
        self.assertEqual(3, e.number_of_variables())
        self.assertEqual(str(e), "2x+3y+z+55")

    def test_linexpr_iadd_var_collapse(self):
        e = self.y - self.x
        # if we add x again, we get only y, x is gone
        e += self.x
        self.check_expression(e, expected_str='y', expected_terms={self.y: 1})

    def test_linexpr_iadd_mn(self):
        x = self.x
        y = self.y
        z = self.z
        e = 2 * x + 3 * y + 55
        e += 137 * z

        # now z is in e
        self.assertTrue(z in e)
        self.check_expression(e, expected_terms={x: 2, y: 3, z: 137}, expected_cst=55)

    def test_linexpr_iadd_linexpr(self):
        x = self.x
        y = self.y
        z = self.z
        e = 2 * x + 3 * y + 55
        # a l origine z pas dans e
        self.assertFalse(z in e)
        self.assertEqual(2, e.number_of_variables())
        e += z + 7 * x + 44
        # now z is in e
        self.assertTrue(z in e)
        self.check_expression(e, expected_terms={x: 9, y: 3, z: 1}, expected_cst=99)

    def test_linexpr_iadd_quad(self):
        x = self.x
        y = self.y
        e = x + y
        # iadd x**2
        e += x**2
        self.assertEqual('x^2+x+y', str(e))
        self.assertTrue(e.has_quadratic_term())

    def test_add_expr_no_side_effect(self):
        m = self.model
        x1 = self.model.binary_var('x1')
        x2 = self.model.binary_var('x2')
        x3 = self.model.binary_var('x3')
        e1 = x1 + x2 + 1
        # e1bis is clone of e1
        e1bis = e1.clone()
        self.assertTrue(e1.equals(e1bis))
        # do some arithmetic
        e2 = x1 + 2 * x2 + x3 + 4
        e2 -= x2
        e2 *= 4
        # now check e1 is unchanged
        self.assertEqual(2, e1.number_of_variables())
        self.assertTrue(e1.equals(e1bis))

class LinearExpressionSubtractTests(ExpressionBaseTests):

    def check_sub(self, base, subtracted, expected_terms={}, expected_cst=0, expected_str=None):
        res = base - subtracted
        self.check_expression(res, expected_cst=expected_cst, expected_terms=expected_terms,
                              expected_str=expected_str)

    # base: zero expr.
    def test_zeroexpr_sub_zero(self):
        self.check_sub(base=self.zero, subtracted=0, expected_str='0')

    def test_zeroexpr_sub_var(self):
        self.check_sub(base=self.zero, subtracted=self.z, expected_terms={self.z:-1})

    def test_zeroexpr_sub_num(self):
        self.check_sub(base=self.zero, subtracted=1, expected_cst=-1)

    def test_zeroexpr_sub_linexpr(self):
        e = self.x + 5*self.y -7 * self.z - 11
        self.check_sub(base=self.zero, subtracted=e,
                        expected_terms={self.x: -1, self.y: -5, self.z: 7},
                       expected_cst=11)

    def test_zeroexpr_to_linear(self):
        z1 = self.zero
        z2 = z1.to_linear_expr()
        self.assertIs(z1, z2)

    # base numexpr
    def test_numexpr_sub_zero(self):
        self.check_sub(base=self.expr3, subtracted=0, expected_str='3', expected_cst=3)

    def test_kexpr_sub_zero(self):
        self.check_sub(base=self.kexpr7, subtracted=0, expected_str='7', expected_cst=7)

    def test_numexpr_sub_num(self):
        self.check_sub(base=self.expr3, subtracted=5, expected_str='-2', expected_cst=-2)
    def test_kexpr_sub_num(self):
        self.check_sub(base=self.kexpr7, subtracted=9, expected_str='-2', expected_cst=-2)

    def test_numexpr_sub_var(self):
        self.check_sub(base=self.expr3, subtracted=self.z,
                       expected_cst=3, expected_terms={self.z:-1})
    def test_kexpr_sub_var(self):
        self.check_sub(base=self.kexpr7, subtracted=self.z,
                       expected_cst=7, expected_terms={self.z:-1})

    def test_numexpr_sub_mn(self):
        mn = 7*self.z
        self.check_sub(base=self.expr3, subtracted=mn,
                       expected_cst=3, expected_terms={self.z:-7})

    def test_kexpr_sub_mn(self):
        mn = 7*self.z
        self.check_sub(base=self.kexpr7, subtracted=mn,
                       expected_cst=7, expected_terms={self.z:-7})

    def test_numexpr_sub_lin(self):
        lin = self.x + 5*self.y -7 * self.z - 11
        self.check_sub(base=self.expr3, subtracted=lin,
                       expected_cst=14,
                       expected_terms={self.x: -1, self.y:-5, self.z:7})
    def test_kexpr_sub_lin(self):
        lin = self.x + 5*self.y -7 * self.z - 11
        self.check_sub(base=self.kexpr7, subtracted=lin,
                       expected_cst=18,
                       expected_terms={self.x: -1, self.y:-5, self.z:7})

    # base: linexpr

    def test_linexpr_sub_zero(self):
        e = self.x + 5 * self.y - 7 * self.z - 11
        em0 = e - 0
        self.check_equal(e, em0)

    def test_linexpr_sub_num(self):
        e = self.x + self.y - 11
        self.check_sub(e, subtracted=7, expected_cst=-18, expected_terms={self.x:1, self.y: 1})

    def test_linexpr_sub_var(self):
        e = self.x + 2*self.y - 11
        self.check_sub(e, subtracted=self.y, expected_cst=-11, expected_terms={self.x:1, self.y: 1})

    def test_linexpr_sub_var_collapse(self):
        e = self.x + 2*self.y - 11
        self.check_sub(e, subtracted=self.x, expected_cst=-11, expected_terms={self.y: 2})

    def test_linexpr_sub_mn(self):
        e = self.x + 2*self.y - 11
        mn = 4 * self.x
        self.check_sub(e, subtracted=mn, expected_cst=-11, expected_terms={self.x:-3, self.y: 2})

    def test_linexpr_sub_mn_collapse(self):
        e = self.x + 2*self.y - 11
        mn = 2* self.y
        self.check_sub(e, subtracted=mn, expected_cst=-11, expected_terms={self.x: 1})

    # specia; case
    def test_linexpr_sub_clone(self):
        e = self.x + 5*self.y -7 * self.z - 11
        ebis = e.clone()
        self.check_sub(base=e, subtracted=ebis)

    # incremental sub

    def test_expr_isub_var(self):
        mdl = self.model
        x = self.x
        y = self.y
        z = self.z
        e = 2 * x + 3 * y + 55
        # a l origine z pas dans e
        self.assertFalse(z in e)
        self.assertEqual(2, e.number_of_variables())
        e -= z
        # now z is in e
        self.assertEqual(-1, e[z])
        self.assertEqual(3, e.number_of_variables())
        self.assertEqual(str(e), "2x+3y-z+55")

    def test_expr_isub_expr(self):
        x = self.x
        y = self.y
        z = self.z
        e = 2 * x + 3 * y + 55
        # a l origine z pas dans e
        self.assertFalse(z in e)
        self.assertEqual(2, e.number_of_variables())
        e -= (3 * z + 7 * x + 44)
        # now z is in e
        self.assertEqual(-3, e[z])
        self.assertEqual(3, e.number_of_variables())
        self.assertEqual(str(e), "-5x+3y-3z+11")

    def test_linexpr_isub_quad(self):
        x = self.x
        y = self.y
        e = x + y
        # iadd x**2
        e -= x**2
        self.assertEqual('-x^2+x+y', str(e))
        self.assertTrue(e.has_quadratic_term())


class LinearExpressionMultiplyTests(ExpressionBaseTests):
    def check_mul(self, initial, factor, expected_cst=0, expected_terms=None, expected_str=None):
        copy_of_initial = initial.clone()
        ls = factor * initial
        rs = copy_of_initial * factor
        # check that left and right are equal
        self.check_equal(ls, rs)
        # chek result
        self.check_expression(ls, expected_terms=expected_terms,
                              expected_cst=expected_cst,
                              expected_str=expected_str)

    # base = zero expr (empty expr)
    def test_zeroexpr_multiply_num(self):
        e = self.model.linear_expr(arg=0)
        self.check_mul(initial=e, factor=7, expected_cst=0, expected_terms={})

    def test_zeroexpr_multiply_varexpr(self):
        zero = self.model.linear_expr(arg=0)
        self.check_mul(initial=zero, factor=self.xexpr, expected_cst=0, expected_terms={})

    # ---  base: cst expr
    def test_cstexpr_multiply_zero(self):
        e = self.model.linear_expr(arg=5)
        self.check_mul(initial=e, factor=0, expected_cst=0, expected_terms={})

    def test_cstexpr_multiply_num_cst(self):
        e = self.model.linear_expr(arg=5)
        self.check_mul(initial=e, factor=7, expected_cst=35, expected_terms={})

    def test_cstexpr_multiply_cstexpr(self):
        e1 = self.model.linear_expr(arg=5)
        e2 = self.model.linear_expr(arg=-7)
        self.check_mul(initial=e1, factor=e2, expected_cst=-35, expected_terms={})

    def test_kexpr_multiply_zero(self):
        self.check_mul(initial=self.kexpr7, factor=0, expected_cst=0, expected_terms={})

    def test_kexpr_multiply_num_cst(self):
        self.check_mul(initial=self.kexpr7, factor=7, expected_cst=49, expected_terms={})

    # def test_cstexpr_multiply_cstexpr(self):
    #     e1 = self.model.linear_expr(arg=5)
    #     e2 = self.model.linear_expr(arg=-7)
    #     self.check_mul(initial=e1, factor=e2, expected_cst=-35, expected_terms={})

    # base: varexpr
    def test_varexpr_multiply_zero(self):
        x = self.x
        e = self.model.linear_expr(arg=x)
        self.check_mul(initial=e, factor=0, expected_cst=0, expected_terms={})

    def test_varexpr_multiply_num(self):
        e = self.model.linear_expr(arg=self.x)
        self.check_mul(initial=e, factor=7, expected_cst=0, expected_terms={self.x: 7})

    def test_varexpr_multiply_cstexpr(self):
        e = self.model.linear_expr(arg=self.x)
        self.check_mul(initial=e, factor=self.expr3, expected_cst=0, expected_terms={self.x: 3})

    def test_varexpr_multiply_kexpr(self):
        e = self.model.linear_expr(arg=self.x)
        self.check_mul(initial=e, factor=self.kexpr7, expected_cst=0, expected_terms={self.x: 7})

    # base: linexpr
    def test_linexp_multiply_num(self):
        e = 3 * self.x - 5 * self.y + 7 * self.z + 11
        self.check_mul(initial=e, factor=7, expected_cst=77,
                       expected_terms={self.x: 21, self.y: -35, self.z: 49})

    def test_linexp_multiply_zero(self):
        e = 3 * self.x - 5 * self.y + 7 * self.z + 11
        self.check_mul(initial=e, factor=0, expected_cst=0, expected_terms={})

    def test_linexpr_multiply_zeroexpr(self):
        e = 3 * self.x - 5 * self.y + 7 * self.z + 11
        zx = ZeroExpr(self.model)
        mz = e * zx
        self.assertEqual('0', str(mz))


    def test_linexp_multiply_minus_one(self):
        e = self.model.linear_expr(3 * self.x - 5 * self.y + 7 * self.z + 11)
        opp = e.opposite()
        rm = e * (-1)
        self.assertTrue(opp.equals(rm))
        lm = (-1) * e
        self.assertTrue(opp.equals(lm))
        self.check_mul(initial=e, factor=-1, expected_terms={self.x: -3, self.y: 5, self.z: -7},
                       expected_cst=-11)

    # --- multiply by a constant linear expr (3)

    def test_linexpr_multiply_empty_cstexpr(self):
        e = self.model.linear_expr()
        self.check_mul(initial=e, factor=self.expr3, expected_terms={})
    def test_linexpr_multiply_empty_kexpr(self):
        e = self.model.linear_expr()
        self.check_mul(initial=e, factor=self.kexpr7, expected_terms={})

    def test_linexpr_multiply_cstexpr(self):
        e = 3 * self.x - 5 * self.y + 7 * self.z + 11
        self.check_mul(initial=e, factor=self.expr3,
                       expected_terms={self.x: 9, self.y: -15, self.z: 21},
                       expected_cst=33)
    def test_linexpr_multiply_kexpr(self):
        e = 3 * self.x - 5 * self.y + 7 * self.z + 11
        self.check_mul(initial=e, factor=self.kexpr7,
                       expected_terms={self.x: 21, self.y: -35, self.z: 49},
                       expected_cst=77)

    # incremental mul
    def test_cstexpr_imul_zero(self):
        e = self.model.linear_expr(arg=7)
        e *= 0
        self.check_expression(e, expected_str='0')

    def test_cstexpr_imul_num(self):
        e = self.model.linear_expr(arg=7)
        e *= 9
        self.check_expression(e, expected_cst=63, expected_terms={})

    def test_cstexpr_imul_varexpr(self):
        e = self.model.linear_expr(arg=7)
        e *= self.xexpr
        self.check_expression(e, expected_cst=0, expected_terms={self.x: 7})

    def test_cstexpr_imul_linexpr(self):
        e = self.model.linear_expr(arg=7)
        l = self.x + 2 * self.y + 3
        e *= l
        self.check_expression(e, expected_cst=21, expected_terms={self.x: 7, self.y: 14})

    def test_varexpr_imul_zero(self):
        e = self.model.linear_expr(arg=self.y)
        e *= 0
        self.assertTrue(e.is_zero())

    def test_varexpr_imul_one(self):
        e = self.model.linear_expr(arg=self.y)
        e *= 1
        self.check_expression(e, expected_terms={self.y:1})

    def test_varexpr_imul_minus_one(self):
        e = self.model.linear_expr(arg=self.y)
        e *= -1
        self.check_expression(e, expected_terms={self.y: -1})

    def test_varexpr_imul_var(self):
        e = self.model.linear_expr(arg=self.y)
        e *= self.z
        self.assertTrue(e.has_quadratic_term())
        self.assertEqual('y*z', str(e))

    def test_varexpr_imul_linexpr(self):
        e1 = self.model.linear_expr(arg=self.y)
        e2 = (self.x + self.y)
        e1*= e2
        self.assertTrue(e1.has_quadratic_term())
        self.assertEqual('x*y+y^2', str(e1))

    def test_linexpr_imul_zero(self):
        e = 2 * self.x + 3 * self.y + 5
        e *= 0
        self.assertTrue(e.is_zero())

    def test_linexpr_imul_one(self):
        e = 2 * self.x + 3 * self.y + 5
        ebis = e.clone()
        e *= 1
        self.assertTrue(ebis.equals(e))
        self.assertEqual(str(e), str(ebis))

    def test_linexpr_imul_minus_one(self):
        e = 2 * self.x + 3 * self.y + 5
        ebis = e.clone().negate()
        e *= -1
        self.assertTrue(ebis.equals(e))
        self.assertEqual(str(e), str(ebis))

    def test_linexpr_imul_num(self):
        e = 2 * self.x + 3 * self.y + 5
        e *= 100
        self.assertEqual(str(e), '200x+300y+500')

    def test_linexpr_imul_var(self):
        e = 2 * self.x + 3 * self.y + 5
        e *= self.x
        self.assertEqual(2, e.number_of_quadratic_terms)
        self.assertEqual('5x', str(e.linear_part))
        self.assertEqual(2, e.get_quadratic_coefficient(self.x))
        self.assertEqual(3, e.get_quadratic_coefficient(self.x, self.y))

    def test_linexpr_imul_mn(self):
        e = 2 * self.x + 3 * self.y + 5
        m = 7*self.x
        e *= m
        self.assertEqual(2, e.number_of_quadratic_terms)
        self.assertEqual('35x', str(e.linear_part))
        self.assertEqual(14, e.get_quadratic_coefficient(self.x))
        self.assertEqual(21, e.get_quadratic_coefficient(self.x, self.y))

    def test_linexpr_imul_linexpr(self):
        e = 2 * self.x + 3 * self.y + 5
        e2 = 7*self.y + 100
        e *= e2
        self.assertEqual(2, e.number_of_quadratic_terms)
        self.assertEqual('200x+335y+500', str(e.linear_part))
        self.assertEqual(21, e.get_quadratic_coefficient(self.y))
        self.assertEqual(14, e.get_quadratic_coefficient(self.x, self.y))

    def test_linexpr_imul_quad_num(self):
        e = 2 * self.x + 3 * self.y + 5
        q3 = self.model._qfactory.new_quad(quads=None, linexpr=3)  # constant 3 as quad
        e *= q3
        self.assertFalse(e.is_quad_expr())
        self.assertEqual('6x+9y+15', str(e))

    def test_linexpr_imul_quad_lin(self):
        e = 2 * self.x + 3 * self.y + 5
        e2 = self.y + 3
        q3 = self.model._qfactory.new_quad(quads=None, linexpr=e2)  # constant 3 as quad
        e *= q3
        # expecting (2x+3y+5)*(y+3) -> 2xy+3y^2+(9+5)y+15)
        self.assertEqual(2, e.number_of_quadratic_terms)
        self.assertEqual('6x+14y+15', str(e.linear_part))

    def test_linexpr_imul_quad_quad(self):
        e = 2 * self.x + 3 * self.y + 5
        e2 = self.y + 3
        xy = self.x * self.y
        try:
            e *= xy
            was_raised = False
        except DOcplexException:
            was_raised = True
        self.assertTrue(was_raised)
    # -- ordering


    def test_mul_expr_ordering(self):
        # build a long expr with ordering, scale it, check order is ok
        with Model('alpha') as m:
            names = "abcdefghijklmnopqrstuvwxyz"
            xs = m.continuous_var_list(keys=names)
            e = m.linear_expr()
            for x in xs:
                e += x
            expected = "+".join(l for l in names)
            self.assertEqual(expected, str(e))
            e2 = e * 2
            expected2 = "+".join("2%s" % l for l in names)
            self.assertEqual(expected2, str(e2))
            # print(str(e2))

class LinearExpressionDivideTests(ExpressionBaseTests):

    def test_var_divide_cst(self):
        x = self.x
        expr = x / 2
        self.assertEqual(0.5, expr[x])

    def test_mn_divide_cst(self):
        m = 7 * self.x
        m2 = m / 2
        self.check_expression(m2, expected_terms={self.x: 3.5})

    def test_expr_divide_cst(self):
        x = self.x
        y = self.y
        expr = (x + y + 3) / 2
        self.check_expression(expr, expected_terms={self.x:0.5, self.y: 0.5}, expected_cst=1.5)


class LinearExpressionStrTests(ExpressionBaseTests):
    def _check_expression_string(self, expr, expected_str,compact=True):
        actual_str = expr.to_string(use_space=not compact)
        self.assertEqual(expected_str, actual_str)

    def test_expr_str_zero(self):
        expr = self.model.linear_expr(0)
        self._check_expression_string(expr, '0')

    def test_expr_str_one(self):
        expr = self.model.linear_expr(1)
        self._check_expression_string(expr, '1')

    def test_expr_str_minus_one(self):
        expr = self.model.linear_expr(-1)
        self._check_expression_string(expr, '-1')

    def test_expr_str_positive_digits(self):
        expr = self.model.linear_expr(1234.567)
        self._check_expression_string(expr, '1234.567')

    def test_expr_str_negative_digits(self):
        expr = self.model.linear_expr(-1234.567)
        self._check_expression_string(expr, '-1234.567')

    def test_expr_str_var(self):
        m = self.model
        xx = m.continuous_var(name='xx')
        expr0 = m.linear_expr(xx)
        self._check_expression_string(expr0, 'xx')

    def test_exp_str_anonymous_var(self):
        with Model() as am:
            xx = am.continuous_var()
            expr0 = am.linear_expr(xx)
            self._check_expression_string(expr0, 'x1')

    def test_exp_str_anonymous_mn(self):
        with Model() as am:
            xx = am.continuous_var()
            expr0 = 7 * xx
            self._check_expression_string(expr0, '7x1')

    def test_expr_str_sum_anonymous_vars(self):
        with Model() as am:
            xs = am.continuous_var_list(3)  # anonymous
            sum_xs = am.sum(xs)
            self.assertEqual('x1+x2+x3', str(sum_xs))

    def test_expr_str_negate_x(self):
        m = self.model
        x = m.continuous_var(name='xx')
        expr0 = - m.linear_expr(x)
        self._check_expression_string(expr0, '-xx')

    def test_expr_str_two_x(self):
        m = self.model
        x = m.continuous_var(name='xx')
        expr0 = m.linear_expr().add_term(x, 2)
        self._check_expression_string(expr0, '2xx')

    def test_expr_str_affine_pp(self):
        m = self.model
        x = m.continuous_var(name='xx')
        expr = 2 * x + 3
        self._check_expression_string(expr, '2xx+3')

    def test_expr_str_affine_pn(self):
        m = self.model
        x = m.continuous_var(name='xx')
        expr = 2 * x - 3
        self._check_expression_string(expr, '2xx-3')

    def test_expr_str_affine_np(self):
        m = self.model
        x = m.continuous_var(name='xx')
        expr = -2 * x + 3
        self._check_expression_string(expr, '-2xx+3')

    def test_expr_str_affine_nn(self):
        m = self.model
        x = m.continuous_var(name='xx')
        expr = -2 * x - 3
        self._check_expression_string(expr, '-2xx-3')

    # ---

    def test_expr_str_affine_space_ppp(self):
        x = self.x
        y = self.y
        expr = 2 * x + 3 * y + 5
        self._check_expression_string(expr, "2 x + 3 y + 5", compact=False)

    def test_expr_str_affine_space_ppn(self):
        x = self.x
        y = self.y
        expr = 2 * x + 3 * y - 5
        self._check_expression_string(expr, "2 x + 3 y - 5", compact=False)

    def test_expr_str_affine_space_p1p(self):
        x = self.x
        y = self.y
        expr = 2 * x + y + 5
        self._check_expression_string(expr, "2 x + y + 5", compact=False)

    def test_expr_str_affine_space_p1n(self):
        x = self.x
        y = self.y
        expr = 2 * x + y - 5
        self._check_expression_string(expr, "2 x + y - 5", compact=False)

    def test_expr_str_affine_space_pnp(self):
        x = self.x
        y = self.y
        expr = 2 * x - 3 * y + 5
        self._check_expression_string(expr, "2 x - 3 y + 5", compact=False)

    def test_expr_str_affine_space_pnn(self):
        x = self.x
        y = self.y
        expr = 2 * x - 3 * y - 5
        self._check_expression_string(expr, "2 x - 3 y - 5", compact=False)

    def test_expr_str_affine_space_1pp(self):
        x = self.x
        y = self.y
        expr = x + 3 * y + 5
        self._check_expression_string(expr, "x + 3 y + 5", compact=False)

    def test_expr_str_affine_space_1pn(self):
        x = self.x
        y = self.y
        expr = x + 3 * y - 5
        self._check_expression_string(expr, "x + 3 y - 5", compact=False)

    def test_expr_str_affine_space_11p(self):
        x = self.x
        y = self.y
        expr = x + y + 5
        self._check_expression_string(expr, "x + y + 5", compact=False)

    def test_expr_str_affine_space_11n(self):
        x = self.x
        y = self.y
        expr = x + y - 5
        self._check_expression_string(expr, "x + y - 5", compact=False)

    def test_expr_str_affine_space_1np(self):
        x = self.x
        y = self.y
        expr = x - 3 *y + 5
        self._check_expression_string(expr, "x - 3 y + 5", compact=False)

    def test_expr_str_affine_space_1nn(self):
        x = self.x
        y = self.y
        expr = x -3 * y - 5
        self._check_expression_string(expr, "x - 3 y - 5", compact=False)

    def test_expr_str_affine_space_npp(self):
        x = self.x
        y = self.y
        expr = -3 * x + 2 *y + 5
        self._check_expression_string(expr, "-3 x + 2 y + 5", compact=False)

    def test_expr_str_affine_space_npn(self):
        x = self.x
        y = self.y
        expr = -3 * x + 2 * y - 5
        self._check_expression_string(expr, "-3 x + 2 y - 5", compact=False)

    def test_expr_str_affine_space_n1p(self):
        x = self.x
        y = self.y
        expr = -3 * x + y + 5
        self._check_expression_string(expr, "-3 x + y + 5", compact=False)

    def test_expr_str_affine_space_n1n(self):
        x = self.x
        y = self.y
        expr = -3 * x + y - 5
        self._check_expression_string(expr, "-3 x + y - 5", compact=False)

    def test_expr_str_affine_space_nnp(self):
        expr = -2 * self.x - 3 * self.y + 5
        self._check_expression_string(expr, "-2 x - 3 y + 5", compact=False)

    def test_expr_str_affine_space_nnn(self):
        expr = -2 * self.x - 3 * self.y - 5
        self._check_expression_string(expr, "-2 x - 3 y - 5", compact=False)



class ExpressionIsDiscreteTests(ExpressionBaseTests):
    def test_discrete_zero_linexpr(self):
        ze = self.model.linear_expr()
        self.assertTrue(ze.is_discrete())

    def test_discrete_expr_ok1(self):
        expr = 2 * self.ijk + 3
        self.assertTrue(expr.is_discrete())

    def test_discrete_expr_ok2(self):
        expr = 2 * self.ijk + 3.0  # an integer value with a float type is ok
        self.assertTrue(expr.is_discrete())

    def test_discrete_expr_ok3(self):
        k3 = self.model.linear_expr(3)
        self.assertTrue(k3.is_discrete())

    def test_discrete_kexpr_ok7(self):
        k7 = self.model._lfactory._new_constant_expr(7)
        self.assertTrue(k7.is_discrete())

    def test_discrete_kexpr_ko777(self):
        k777 = self.model._lfactory._new_constant_expr(7.77)
        self.assertFalse(k777.is_discrete())

    def test_discrete_expr_ok4(self):
        k7 = self.model.linear_expr(7.0)
        self.assertTrue(k7.is_discrete())

    def test_discrete_expr_ko1(self):
        x = self.x
        expr = 2.1 * x + 3
        self.assertFalse(expr.is_discrete())

    def test_discrete_expr_ko3(self):
        expr = self.x + 3
        self.assertFalse(expr.is_discrete())

    def test_discrete_expr_ko4(self):
        expr = 1.1 * self.x + 3
        self.assertFalse(expr.is_discrete())

    def test_discrete_expr_ko5(self):
        expr = 1.1 * self.x + 3.14
        self.assertFalse(expr.is_discrete())

    def test_discrete_expr_ko2(self):
        m = self.model
        ijk = self.ijk
        expr = 2 * ijk + 3.14
        self.assertFalse(expr.is_discrete())

    def test_discrete_var_ko3_continuous(self):
        m = self.model
        ijk = self.x
        expr = m.linear_expr(ijk)
        self.assertFalse(expr.is_discrete())

    def test_discrete_var_ok_binary(self):
        m = self.model
        x = m.binary_var(name='b')
        expr = m.linear_expr(x)
        self.assertTrue(expr.is_discrete())

    def test_discrete_var_ok_integer(self):
        m = self.model
        x = m.integer_var(name='b')
        expr = m.linear_expr(x)
        self.assertTrue(expr.is_discrete())

    def test_discrete_mnm_ok(self):
        m = self.model
        mn = m.integer_var(name='b') * 7
        self.assertTrue(mn.is_discrete())

    def test_discrete_mnm_ko1(self):
        m = self.model
        mn = m.integer_var(name='b') * 7.7
        self.assertFalse(mn.is_discrete())

    def test_discrete_mnm_ko2(self):
        m = self.model
        mn = m.continuous_var(name='b') * 2
        self.assertFalse(mn.is_discrete())

    def test_discrete_mnm_ko3(self):
        m = self.model
        mn = m.continuous_var(name='b') * 2.2
        self.assertFalse(mn.is_discrete())


class ExpressionCloneTests(ExpressionBaseTests):

    def test_mn_clone(self):
        mn = 7 * self.x
        mn2 = mn.clone()
        self.assertIsNot(mn, mn2)
        self.assertEqual(mn2.coef, mn.coef)
        self.assertIs(mn.var, mn2.var)



class DocplexZeroExprTests(ExpressionBaseTests):


    def setUp(self):
        ExpressionBaseTests.setUp(self)
        self.zero = ZeroExpr(self.model)

    def test_zero_str(self):
        self.assertEqual("0", str(self.zero))

    def test_zero_repr(self):
        self.assertEqual("docplex.mp.ZeroExpr()", repr(self.zero))

    def test_zero_negate(self):
        self.assertIs(self.zero, self.zero.negate())

    def test_zero_add_cst(self):
        e = self.zero + 3
        self.assertEqual(3, e)

    def test_zero_plus_var(self):
        self.assertIs(self.x, self.zero.plus(self.x))

    def test_zero_add_mnm(self):
        mnm = 3 * self.x
        e = self.zero + mnm
        self.assertIs(e, mnm)

    def test_zero_radd_mnm(self):
        mnm = 3 * self.x
        e = mnm + self.zero
        self.assertEqual("3x", str(e))  # this is a different instance...

    def test_zero_add_var(self):
        v = self.model.binary_var(name='b')
        le = v + self.zero
        self.assertEqual("b", str(le))
        re = self.zero + v
        self.assertIs(re, v)

    def test_zero_add_expr(self):
        v = self.model.binary_var(name='b')
        expr = 3 * v + 7
        le = self.zero + expr
        re = expr + self.zero  # expr is cloned here. pity
        self.assertIs(le, expr)
        self.assertTrue(re.equals(expr))

    def test_zero_sub_var(self):
        v = self.model.binary_var(name='b')
        e = self.zero - v
        self.assertIsInstance(e, Expr)
        self.assertEqual("-b", str(e))

    def test_zero_rsub_num(self):
        e = 3 - self.zero
        self.assertEqual(3, e)

    def test_zero_sub_rexpr(self):
        arg = 3 * self.x - 5 * self.y + 17
        e = self.zero - arg
        self.assertEqual("-3x+5y-17", str(e))

    def test_zero_rsub_rexpr(self):
        arg = 3 * self.x - 5 * self.y + 17
        e = arg - self.zero
        self.assertEqual("3x-5y+17", str(e))

    def test_zero_mul_any(self):
        self.assertIs(self.zero, self.zero * 3)
        self.assertIs(self.zero, self.zero * 0)
        self.assertIs(self.zero, self.zero * self.x)
        mnm = 3 * self.x
        self.assertIs(self.zero, self.zero * mnm)
        linexpr = 3 * self.x + 47 * self.y + 66
        self.assertIs(self.zero, self.zero * linexpr)

    def test_zero_times_any(self):
        self.assertIs(self.zero, self.zero.times(3))
        self.assertIs(self.zero, self.zero.times(0))
        self.assertIs(self.zero, self.zero.times(self.x))
        mnm = 3 * self.x
        self.assertIs(self.zero, self.zero.times(mnm))
        linexpr = 3 * self.x + 47 * self.y + 66
        self.assertIs(self.zero, self.zero.times(linexpr))

    def test_zero_rmul_cst(self):
        self.assertIs(self.zero, 3 * self.zero)
        self.assertIs(self.zero, 0 * self.zero)

    def test_zero_rmul_var(self):
        zmul = self.x * self.zero
        self.assertTrue(zmul.is_zero())

    def test_zero_rmul_mnm(self):
        mnm = 3 * self.x
        e = mnm * self.zero
        self.assertTrue(e.is_zero())

    def test_zero_rmul_linexpr(self):
        linexpr = 3 * self.x + 47 * self.y + 66
        e = linexpr * self.zero
        self.assertTrue(e.is_zero())

    def test_zero_iadd(self):
        zz = self.zero
        zz += self.x
        self.assertIs(self.x, zz)

    def test_zero_isub(self):
        zz = self.zero
        zz -= self.x
        # zz is a linear expression with -1 * x and zero constant
        self.assertEqual(0, zz.get_constant())
        self.assertEqual(1, zz.number_of_terms())
        self.assertEqual('-x', str(zz))

    def test_zero_clone(self):
        self.assertIs(self.zero.clone(), self.zero)

    def test_zero_nb_vars(self):
        self.assertEqual(0, self.zero.number_of_variables())

    def test_zero_is_discrete(self):
        self.assertTrue(self.zero.is_discrete())

    def test_zero_equals(self):
        zz = self.zero
        self.assertTrue(zz.equals(self.zero))
        self.assertTrue(zz.equals(ZeroExpr(self.model)))
        self.assertTrue(zz.equals(self.model.linear_expr()))
        self.assertTrue(zz.equals(0))
        self.assertFalse(zz.equals(1))
        self.assertFalse(zz.equals(self.x))
        self.assertFalse(zz.equals(self.x+1))
        self.assertFalse(zz.equals(7*self.x))

    def test_zero_coeff(self):
        self.assertEqual(0, self.zero.unchecked_get_coef(self.x))

    def test_zero_vars(self):
        self.assertEqual([], list(self.zero.iter_terms()))
        self.assertFalse(self.zero.contains_var(self.x))
        self.assertFalse(self.x in self.zero)

    def test_zero_solution(self):
        m = self.model
        if m._can_solve():
            s = m.solve()
            self.assertEqual(0, self.zero.solution_value)

    def test_zero_size(self):
        self.assertEqual(0, self.zero.size)



class DocplexMonomialTest(ExpressionBaseTests):
    def initMonomials(self):
        self.coef1 = 7
        self.coef2 = 13
        self.monomial1 = 7 * self.x
        self.monomial2 = 13 * self.y

    def setUp(self):
        ExpressionBaseTests.setUp(self)
        self.initMonomials()

    def test_mnm_name(self):
        self.assertIsNone(self.monomial1.name)

    def test_monomial_str(self):
        self.assertEqual("7x", str(self.monomial1))
        self.assertEqual("13y", str(self.monomial2))

    def test_monomial_accessors(self):
        self.assertIs(self.x, self.monomial1.var)
        ok = self.monomial1.contains_var(self.x)
        self.assertTrue(ok)
        self.assertEqual(self.monomial1.coef, self.coef1)

    def test_monomial_clone(self):
        mnm2 = self.monomial1.clone()
        self.assertTrue(mnm2.equals(self.monomial1))

    def test_monomial_iter_variables(self):
        mnm_iter = self.monomial1.iter_variables()
        mnm_vars = list(mnm_iter)
        self.assertEqual(1, len(mnm_vars))
        self.assertIs(mnm_vars[0], self.x)

    def test_monomial_isin(self):
        self.assertTrue(self.x in self.monomial1)
        zzz = self.model.continuous_var(name="zorglub")
        self.assertFalse(zzz in self.monomial1)

    def test_monomial_getitem(self):
        self.assertEqual(7, self.monomial1[self.x])
        # check a variable npot in monomial
        zzz = self.model.continuous_var(name='zzz')
        self.assertEqual(0, self.monomial1[zzz])

    def test_monomial_set(self):
        m1 = self.monomial1
        m2 = self.monomial2
        mset = {m1, m2}
        self.assertEqual(2, len(mset))

    def test_monomial_add_zero(self):
        e = self.monomial1 + 0
        self.assertEqual(str(e), "7x")
        self.assertEqual('7x', str(e))

    def test_monomial_add_cst(self):
        e = self.monomial1 + 3
        self.assertEqual(str(e), "7x+3")

    def test_monomial_radd_cst(self):
        e = 4 + self.monomial1
        self.assertEqual(str(e), "7x+4")

    def test_monomial_subtract_cst(self):
        e = self.monomial1 - 3
        self.assertEqual(str(e), "7x-3")

    def test_monomial_rsub_cst(self):
        e = 11 - self.monomial1
        self.assertEqual(str(e), "-7x+11")

    def test_monomial_uminus(self):
        nm = - self.monomial1
        self.assertEqual(str(nm), "-7x")
        self.assertIsNot(nm, self.monomial1)

    def test_monomial_negate(self):
        old_coef = self.monomial1.coef
        nm1 = self.monomial1.negate()
        self.assertIs(nm1, self.monomial1)
        self.assertEqual(nm1.coef, -old_coef)
        self.assertIs(nm1.var, self.monomial1.var)

    def test_monomial_times_one(self):
        e = self.monomial1 * 1
        # yields the same: is it good or not?
        self.assertTrue(self.monomial1.equals(e))

    def test_monomial_times_cst(self):
        e = self.monomial1 * 2
        self.assertEqual(14, e.coef)
        self.assertIsNot(e, self.monomial1)

    def test_monomial_times_zero(self):
        z = self.monomial1 * 0
        self.assertEqual("0", str(z))

    def test_monomial_multiply3(self):
        self.monomial1.multiply(3)
        self.assertEqual('21x', str(self.monomial1))

    def test_monomial_rtimes_one(self):
        e = 1.0 * self.monomial1
        self.assertTrue(self.monomial1.equals(e))

    def test_monomial_rtimes_cst(self):
        e = 2 * self.monomial1
        self.assertEqual(14, e.coef)
        self.assertIsNot(e, self.monomial1)

    def test_monomial_rtimes_zero(self):
        z = 0 * self.monomial1
        self.assertEqual("0", str(z))

    def test_monomial_times_linear_expr_cst(self):
        expr2 = self.model.linear_expr(2)
        z = self.monomial1 * expr2
        self.assertTrue(z, 14 * self.x)
        # monomail1 is untouched
        self.assertEqual(str(self.monomial1), "7x")

    def test_monomial_solution_value(self):
        mdl = self.model
        ij = mdl.integer_var(name='ij', lb=3)
        im = 33.3 * ij
        self.x.lb = 1
        mdl.minimize(self.x + im)
        self.assertTrue(mdl.solve())
        self.assertAlmostEqual(7, self.monomial1.solution_value, delta=1e-6)
        self.assertAlmostEqual(99.9, im.solution_value, delta=1e-5)


    def test_monomial_str_rich(self):
        self.assertEqual("7 x", self.monomial1.to_string(use_space=True))

    def test_monomial_repr(self):
        self.assertEqual("docplex.mp.MonomialExpr(7x)", repr(self.monomial1))

    def test_monomial_equal_ko(self):
        mnm2 = 8 * self.x
        self.assertFalse(mnm2.equals(self.monomial1))
        self.assertFalse(mnm2.equals(3.14))
        mdl = self.model
        zs = mdl.integer_var_list(4, lb=1, name='z')
        self.assertFalse(mnm2.equals(mdl.sum(zs)))

    def test_monomial_divide_float(self):
        e = self.monomial1 / 0.5
        self.assertIsInstance(e, LinearOperand)
        self.assertEqual("14x", str(e))
        self.assertIsNot(e, self.monomial1)

    def test_monomial_divide_int(self):
        m1 = self.monomial1
        m2 = m1 * 2
        m22 = m2 / 2  # *2 /2 = 1 !
        self.assertTrue(m22.equals(self.monomial1))  # equals
        self.assertIsNot(m22, m1)  # not is

    @staticmethod
    def _try_divide(num, denom):
        z = num / denom

    def test_monomial_divide_other(self):
        m2 = 1.7 * self.x
        self.assertRaises(DOcplexException, self._try_divide, m2, self.monomial1)

    def test_monomial_zero_divide(self):
        self.assertRaises(DOcplexException, self._try_divide, self.monomial1, 0)

    def test_monomial_inverse(self):
        self.assertRaises(DOcplexException, self._try_divide, 1, self.monomial1)

    def test_sum_divide(self):
        mdl = self.model
        x = mdl.binary_var_list(3, name='b')
        e = mdl.sum([x[i] * i / (1.0 + i) for i in range(len(x))])
        self.assertEqual("0.500b_1+0.667b_2", str(e))

    # aithmetic to self
    def test_monomial_iadd_num(self):
        m1 = 77 * self.x
        m1 += 3
        self.assertEqual('77x+3', str(m1))

    def test_monomial_iadd_var(self):
        m1 = 77 * self.x
        m1 += self.x
        self.assertEqual('78x', str(m1))

    def test_monomial_iadd_mn(self):
        z = 77 * self.x
        m2 = 3 * self.y
        z += m2
        self.assertEqual('77x+3y', str(z))

    def test_monomial_iadd_square(self):
        z = 77 * self.x
        y2 = self.y ** 2
        z += y2
        self.assertEqual('y^2+77x', str(z))
        # y2 unchanged
        self.assertEqual('y^2', str(y2))

    def test_monomial_iadd_full_quad(self):
        z = 77 * self.x
        q = 3 * self.y ** 2 + self.y + 3
        z += q
        self.assertEqual(1, z.number_of_quadratic_terms)
        self.assertEqual(3, z.get_quadratic_coefficient(self.y))
        self.assertEqual('77x+y+3', str(z.linear_part))

    # sub to self
    def test_mn_isub_num(self):
        m1 = 77 * self.x
        m1 -= 3
        self.assertEqual('77x-3', str(m1))

    def test_mn_isub_var(self):
        m1 = 77 * self.x
        m1 -= self.y
        self.assertEqual('77x-y', str(m1))

    def test_mn_isub_square(self):
        m1 = 77 * self.x
        y2 = self.y**2
        m1 -= y2
        self.assertEqual('-y^2+77x', str(m1))

    def test_mn_imul_num(self):
        m1 = 77 * self.x
        m1 *= 2
        self.assertEqual('154x', str(m1))

    def test_mn_imul_zero(self):
        y = self.y
        z = 2 * y
        z *= 0
        self.assertIsInstance(z, ZeroExpr)

    def test_mn_imul_one(self):
        y = self.y
        z = 2 * y
        z *= 1
        self.assertIsInstance(z, MonomialExpr)
        self.assertEqual(z.coef, 2)

    def test_monomial_imul_var(self):
        y = self.y
        z = 2 * y
        z *= y
        self.assertIsInstance(z, QuadExpr)
        self.assertEqual(z.get_quadratic_coefficient(y), 2)
        self.assertEqual('2y^2', str(z))

    def test_monomial_imul_mn(self):
        y = self.y
        z = 2 * y
        mn= 3 * y
        z *= mn
        self.assertIsInstance(z, QuadExpr)
        self.assertEqual(z.get_quadratic_coefficient(y), 6)

    def test_monomial_imul_linexpr(self):
        x = self.x
        y = self.y
        z = 2 * y
        zz = x + 3 *y
        self.assertIsInstance(z, MonomialExpr)
        self.assertIsInstance(zz, LinearExpr)
        self.assertEqual(z.coef, 2)
        z *= zz
        self.assertIsInstance(z, QuadExpr)
        self.assertEqual(z.get_quadratic_coefficient(y), 6)

    def test_monomial_imul_quad_num(self):
        y = self.y
        z = 2 * y
        q3 = self.model._qfactory.new_quad(quads=None, linexpr=3)  # constant 3 as quad
        z *= q3

    def test_monomial_imul_quad_lin(self):
        y = self.y
        z = 2 * y
        lin = 5 * self.x + 7 * y
        ql = self.model._qfactory.new_quad(quads=None, linexpr=lin)  # constant 3 as quad
        z *= ql
        self.assertEqual(2, z.number_of_quadratic_terms)
        self.assertEqual('0', str(z.linear_part))
        self.assertEqual(14, z.get_quadratic_coefficient(self.y))

    def test_monomial_imul_quad_quad(self):
        y = self.y
        y2 = self.y ** 2
        z = 2 * y
        try:
            z *= y2
            raised = False
        except DOcplexException as dox:
            raised = True
        self.assertTrue(raised)

    def test_mn_idiv_num(self):
        m1 = 77 * self.x
        m1 /= 7
        self.assertEqual('11x', str(m1))

class DocplexTransientExprTests(ExpressionBaseTests):
    def setUp(self):
        ExpressionBaseTests.setUp(self)
        self.nc_model = Model(keep_all_exprs=False)

    def tearDown(self):
        ExpressionBaseTests.tearDown(self)
        self.nc_model.end()
        self.nc_model = None

    def test_linear_expr_side_effect(self):
        m = self.nc_model
        a = m.continuous_var(name='a')
        b = m.continuous_var(name='b')
        c = m.continuous_var(name='c')
        e = a + b + c
        e2 = e.clone()
        ref_str_e2 = str(e2)
        self.assertEqual(str(e), ref_str_e2)
        # modify e -> e2 is not modified....
        z = e + b  # this modifies e !!!
        self.assertEqual("a+2b+c", str(e))
        self.assertEqual("a+2b+c", str(z))
        self.assertEqual(str(e2), ref_str_e2)

    def test_transient_from_var(self):
        mdl = self.nc_model
        x = mdl.integer_var(name='x')
        y = mdl.integer_var(name='y')
        e = x + y
        self.assertFalse(e.is_kept())

    def test_transient_from_mnm(self):
        mdl = self.nc_model
        x = mdl.integer_var(name='x')
        y = mdl.integer_var(name='y')
        e = 4 * x + y
        self.assertFalse(e.is_kept())

    def test_keep_transient(self):
        mdl = self.nc_model
        x = mdl.integer_var(name='x')
        y = mdl.integer_var(name='y')
        e = x + y
        self.assertFalse(e.is_kept())
        e.keep()
        self.assertTrue(e.is_kept())

    def test_triangular_cts(self):
        m = Model()
        xs = m.integer_var_list(keys=['x', 'y', 'z'])
        e = 0
        for i, v in enumerate(xs):
            e += v
            m.add(e.clone() <= 3 + i)
        lps = m.export_as_lp_string()
        self.assertIn("x <= 3", lps)
        self.assertIn("x + y <= 4", lps)
        self.assertIn("x + y + z <= 5", lps)

class DOcplexNormalizeExprTests(ExpressionBaseTests):

    def test_iadd_normalized(self):
        e = 2 * self.x + 3 * self.y + 1
        self.assertTrue(e.is_normalized())
        e += (-2 * self.x)
        self.assertTrue(e.is_normalized())

    def test_normalize(self):
        e = 2 * self.x + 3 * self.y + 1
        self.assertTrue(e.is_normalized())
        e += (-2 * self.x)
        self.assertTrue(e.is_normalized())
        self.assertEqual('3y+1', str(e))
        self.assertNotIn(self.x, e)

    def test_normalized(self):
        e = 2 * self.x + 3 * self.y + 1
        self.assertTrue(e.is_normalized())
        e += (-2 * self.x)
        en = e.clone()
        self.assertTrue(en.is_normalized())
        self.assertEqual('3y+1', str(en))
        self.assertNotIn(self.x, en)
        self.assertIsNot(e, en)

    def test_normalized_no1(self):
        e = 2 * self.x + 3 * self.y + 1
        self.assertTrue(e.is_normalized())
        ee2 = e.clone()
        ee2.set_coefficient(self.y, 0)
        self.assertTrue(ee2.is_normalized())

    def test_normalized_var_cst(self):
        ct = self.x <= 77
        self.assertTrue(ct.lhs.is_normalized())
        self.assertTrue(ct.rhs.is_normalized())

    def test_normalized_mn(self):
        ct = (2 * self.x <= self.y + 77)
        self.assertTrue(ct.lhs.is_normalized())

    def test_normalized_linexpr(self):
        ct = (2 * self.x + 3 * self.y <= 0)
        self.assertTrue(ct.lhs.is_normalized())
        self.assertTrue(ct.rhs.is_normalized())





if __name__ == "__main__":
    unittest.main()
