import unittest
import six

from docplex.mp.utils import DOcplexException
from docplex.mp.advmodel import AdvModel
from docplex.mp.utils import is_iterable

from docplex.mp.linear import ZeroExpr
from testutils import skipIfCplexCE, are_functional_vars_named


try:
    import pandas as pd
    from pandas import DataFrame
except ImportError:
    pd = None
    DataFrame = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

import os
docplex_debug = os.environ.get("DOCPLEX_DEBUG")

class AdvModelTestBase(unittest.TestCase):
    def setUp(self):
        self.model = AdvModel(name="advmodel")
        m = self.model
        self.zero = ZeroExpr(m)

    def tearDown(self):
        self.model.end()

    def check_quad(self, q, expected_nb_quadratics, expected_str):
        self.assertEqual(expected_nb_quadratics, q.number_of_quadratic_terms)
        self.assertTrue(q.is_normalized())
        actual_str = str(q)
        if is_iterable(expected_str, accept_string=False):
            self.assertTrue(any((text == actual_str) for text in expected_str))
        else:
            self.assertEqual(actual_str, expected_str)


# noinspection PyUnresolvedReferences
class ScalProdVarsAlldifferentTests(AdvModelTestBase):
    def test_scalprodvars_alldiff_empty(self):
        spv = self.model.scal_prod_vars_all_different(terms=[], coefs=[])
        self.assertTrue(spv.is_zero())

    def test_scalprodvars_alldiff_one(self):
        m = self.model
        a = m.continuous_var(name='a')
        spv = self.model.scal_prod_vars_all_different(terms=[a], coefs=[7])
        self.assertEqual('7a', str(spv))

    def test_scalprodvars_alldiff_two(self):
        m = self.model
        a = m.continuous_var(name='a')
        b = m.continuous_var(name='b')
        spv = self.model.scal_prod_vars_all_different(terms=[a, b], coefs=[7, 11])
        self.assertEqual('7a+11b', str(spv))

    def test_scalprodvars_alldiff_accumulate(self):
        with AdvModel(name='nochecks', checker='off') as m:
            a = m.continuous_var(name='a')
            spv = m.scal_prod_vars_all_different(terms=[a, a], coefs=[7, 11])
            self.assertEqual('11a', str(spv))

    def test_scalprodvars_alldiff_collapse(self):
        with AdvModel(name='nochecks', checker='off') as m:
            a = m.continuous_var(name='a')
            spv = m.scal_prod_vars_all_different(terms=[a, a], coefs=[7, -7])
            # last one wins
            self.assertEqual('-7a', str(spv))

    def test_scalprodvars_alldiff_too_many_vars(self):
        m = self.model
        a = m.continuous_var(name='a')
        b = m.continuous_var(name='b')
        spv = self.model.scal_prod_vars_all_different(terms=[a, b], coefs=[7])
        self.assertEqual('7a', str(spv))

    def test_scalprodvars_alldiff_too_many_coefs(self):
        m = self.model
        a = m.continuous_var(name='a')
        b = m.continuous_var(name='b')
        spv = self.model.scal_prod_vars_all_different(terms=[a, b], coefs=[7, 11, 13])
        self.assertEqual('7a+11b', str(spv))

    def test_scalprodvars_alldiff_eq_scalprod(self):
        m = self.model
        size = 11
        dvars = m.continuous_var_list(size)
        r1 = range(1, size + 1)
        e1 = m.scal_prod(dvars, r1)
        e2 = m.scal_prod_vars_all_different(dvars, r1)
        self.assertTrue(e1.equals(e2))

    def test_scalprodvars_alldiff_num_coefs(self):
        # coefs is a plain number
        m = self.model
        size = 3
        koef = 7
        dvars = m.continuous_var_list(size, name='a')
        e1 = koef * m.sum(dvars)
        e2 = m.scal_prod_vars_all_different(dvars, coefs=koef)
        self.assertTrue(e1.equals(e2))

    def test_scalprodvars_alldiff_zero_coefs(self):
        # coefs is a plain number
        m = self.model
        dvars = m.continuous_var_list(3, name='a')
        e = m.scal_prod_vars_all_different(dvars, coefs=0)
        self.assertEqual('0', str(e))

    def test_scal_prod_vars_all_different_ko_checked(self):
        with AdvModel(checker='default') as m:
            marker_name = 'zorglub'
            a = m.continuous_var(name='a')
            z = m.continuous_var(name=marker_name)
            six.assertRaisesRegex(self, DOcplexException, '%s appears twice' % marker_name,
                                  lambda md: md.scal_prod_vars_all_different([z, a, z], [1, 3, 5]), m)

    def test_scal_prod_vars_all_different_no_var(self):
        with AdvModel(checker='default') as m:
            a = m.continuous_var(name='a')
            z = m.continuous_var(name='z')
            six.assertRaisesRegex(self, DOcplexException, 'Expecting decision variable, got: 10',
                                  lambda md: md.scal_prod_vars_all_different([z, a, 10], [1, 3, 5]), m)





class ScalProdTriplTests(AdvModelTestBase):

    def test_scal_prod_triple_strings_ko(self):
        with AdvModel(checker='on') as advm:
            x = advm.continuous_var(name='x')
            six.assertRaisesRegex(self, DOcplexException, 'Expecting variable or linear expression,',
                                  lambda m: m.scal_prod_triple([x], ['foo'], [3]), advm)

    def test_scal_prod_triple(self):
        m = self.model
        m.set_checker('off')
        size = range(10)
        vars1 = m.continuous_var_list(size, name="v1")
        vars2 = m.continuous_var_list(size, name="v2")
        coefs = [i + 1 for i in size]
        e1 = m.scal_prod_triple(vars1, vars2, coefs)
        e11 = m.scal_prod_triple(vars1, vars2, 10)
        e111 = m.scal_prod_triple(vars1[0], vars2, 10)
        e1111 = m.scal_prod_triple(vars1, vars2[0], 10)
        e11111 = m.scal_prod_triple(vars1[0], vars2[0], 10)
        e111111 = m.scal_prod_triple(vars1[0], vars2[0], coefs)
        e1111111 = m.scal_prod_triple(vars1[0], vars2, coefs)
        #  e11111111 = m.scal_prod_triple(vars1, vars2[0], coefs)

        e2 = m.quad_expr()
        e22 = m.quad_expr()
        e222 = m.quad_expr()
        e2222 = m.quad_expr()
        e22222 = m.quad_expr()
        e222222 = m.quad_expr()
        e2222222 = m.quad_expr()
        e22222222 = m.quad_expr()

        e22222 += vars1[0] * vars2[0] * 10
        for i in size:
            e2 += vars1[i] * vars2[i] * coefs[i]
            e22 += vars1[i] * vars2[i] * 10
            e222 += vars1[0] * vars2[i] * 10
            e2222 += vars1[i] * vars2[0] * 10
            e222222 += vars1[0] * vars2[0] * coefs[i]
            e2222222 += vars1[0] * vars2[i] * coefs[i]
            e22222222 += vars1[i] * vars2[0] * coefs[i]

        self.assertTrue(e1.equals(e2))
        self.assertTrue(e11.equals(e22))
        self.assertTrue(e111.equals(e222))
        self.assertTrue(e1111.equals(e2222))
        self.assertTrue(e11111.equals(e22222))
        self.assertTrue(e111111.equals(e222222))
        self.assertTrue(e1111111.equals(e2222222))

    def test_scal_prod_vars_triple(self):
        m = self.model
        size = 10
        vars1 = m.continuous_var_list(size, name="v1")
        vars2 = m.continuous_var_list(size, name="v2")
        coefs = range(1, size + 1)
        e1 = m.scal_prod_triple_vars(vars1, vars2, coefs)
        e3 = m.scal_prod_triple(vars1, vars2, coefs)

        e2 = m.quad_expr()

        for i in range(size):
            e2 += vars1[i] * vars2[i] * coefs[i]

        # print e2
        # print e1
        self.assertTrue(e1.equals(e2))
        self.assertTrue(e1.equals(e3))

    def test_scal_prod_triple_num_nonzero(self):
        vars1 = self.model.continuous_var_list(keys=['x', 'y', 'z'])
        vars2 = reversed(vars1)
        #  [x, y, z] * [z, y, x] -> xz + y^2 + zx
        e = self.model.scal_prod_triple(vars1, vars2, 3)
        self.check_quad(e, expected_nb_quadratics=2, expected_str=['3y^2+6x*z', '6x*z+3y^2'])

    def test_scal_prod_triple_num_zero(self):
        vars1 = self.model.continuous_var_list(keys=['x', 'y', 'z'])
        vars2 = reversed(vars1)
        #  [x, y, z] * [z, y, x] -> xz + y^2 + zx
        e = self.model.scal_prod_triple(vars1, vars2, 0)
        self.assertEqual('0', str(e))

    # @unittest.skip('no typechecking')
    def test_scal_prod_triple_string_coef(self):
        vars1 = self.model.continuous_var_list(keys=['x', 'y', 'z'])
        vars2 = reversed(vars1)
        six.assertRaisesRegex(self, DOcplexException, 'scal_prod_triple expects iterable or number as coefficients',
                              lambda m: m.scal_prod_triple(vars1, vars2, 'a'), self.model)

    def test_scal_prod_triple_vars(self):
        m = self.model
        vars1 = m.continuous_var_list(keys=['x', 'y', 'z'])
        vars2 = list(reversed(vars1))
        coefs = [3, 5, 7]
        q = m.scal_prod_triple_vars(left_terms=vars1, right_terms=vars2, coefs=coefs)
        self.check_quad(q, expected_nb_quadratics=2, expected_str=['5y^2+10x*z', '10x*z+5y^2'])

    def test_scal_prod_triple_vars_num_coef(self):
        m = self.model
        vars1 = m.continuous_var_list(keys=['x', 'y', 'z'])
        vars2 = reversed(vars1)
        q = m.scal_prod_triple_vars(left_terms=vars1, right_terms=vars2, coefs=3)

    def test_scal_prod_triple_vars_zero_coef(self):
        m = self.model
        vars1 = m.continuous_var_list(keys=['x', 'y', 'z'])
        vars2 = reversed(vars1)
        q = m.scal_prod_triple_vars(left_terms=vars1, right_terms=vars2, coefs=0)
        self.assertEqual('0', str(q))

    def test_scal_prod_triple_vars_right_var(self):
        with AdvModel(keep_ordering=True) as om:
            vars1 = om.continuous_var_list(keys=['x', 'y', 'z'])
            vars2 = vars1[0]
            q = om.scal_prod_triple_vars(left_terms=vars1, right_terms=vars2, coefs=[1, 2, 3])
            self.check_quad(q, expected_nb_quadratics=3, expected_str=['x^2+3x*z+2x*y', 'x^2+2x*y+3x*z'])

    def test_scal_prod_triple_vars_left_right_var(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        q = m.scal_prod_triple_vars(left_terms=x, right_terms=y, coefs=7)
        self.assertEqual('7x*y', str(q))


class QuadMatrixSumTests(AdvModelTestBase):
    def test_quad_matrix_sum0(self):
        q0 = self.model.quad_matrix_sum(matrix=[], dvars=[])
        self.assertEqual('0', str(q0))

    def test_quad_matrix_sum1(self):
        m = [[7]]
        x = self.model.continuous_var(name='x')
        q = self.model.quad_matrix_sum(matrix=m, dvars=[x])
        self.assertEqual('7x^2', str(q))

    def test_quad_matrix_sum2_asymmetric(self):
        with AdvModel(keep_ordering=True) as om:
            m2 = [[1, 2], [3, 4]]
            xs = om.continuous_var_list(keys=['x', 'y'])
            q = om.quad_matrix_sum(matrix=m2, dvars=xs)
            self.assertEqual('x^2+5x*y+4y^2', str(q))

    def test_quad_matrix_sum2_symmetric_flag(self):
        with AdvModel(keep_ordering=True) as om:
            m2 = [[1, 3], [3, 7]]
            xs = om.continuous_var_list(keys=['x', 'y'])
            q = om.quad_matrix_sum(matrix=m2, dvars=xs, symmetric=True)
            self.assertEqual('x^2+6x*y+7y^2', str(q))

    def test_quad_matrix_sum2_symmetric_noflag(self):
        with AdvModel(keep_ordering=True) as om:
            m2 = [[1, 3], [3, 7]]
            xs = om.continuous_var_list(keys=['x', 'y'])
            # noinspection PyArgumentEqualDefault
            q = om.quad_matrix_sum(matrix=m2, dvars=xs, symmetric=False)
            self.assertEqual('x^2+6x*y+7y^2', str(q))

    def test_quad_matrix_sum_symmetric(self):
        with AdvModel(keep_ordering=False) as om:
            m3 = [[1, 7, 3], [7, 4, -5], [3, -5, 6]]
            xs = om.integer_var_list(keys=3, name='x')
            q = om.quad_matrix_sum(m3, xs, symmetric=True)
            self.check_quad(q, expected_nb_quadratics=6,
                            expected_str='x_0^2+14x_0*x_1+6x_0*x_2+4x_1^2-10x_1*x_2+6x_2^2')

    @unittest.skipUnless(sp, 'scipy is required')
    def test_quad_matrix_sum_sp2_symmetric(self):
        mat = sp.coo_matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]], shape=(3, 3))
        xs = self.model.continuous_var_list(3, name=['x', 'y', 'z'])
        q = self.model.quad_matrix_sum(mat, xs, symmetric=True)
        self.assertEqual('2x*z+y^2', str(q))

    @unittest.skipUnless(sp, 'scipy is required')
    def test_quad_matrix_sum_sp2_asymmetric(self):
        mat = sp.coo_matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]], shape=(3, 3))
        xs = self.model.continuous_var_list(3, name=['x', 'y', 'z'])
        q = self.model.quad_matrix_sum(mat, xs, symmetric=False)
        self.assertEqual('2x*z+y^2', str(q))

class SumSquareVarsTests(AdvModelTestBase):
    def test_sumsq_empty(self):
        q = self.model.sumsq_vars_all_different([])
        self.assertTrue(q.is_zero())

    def test_sumsq_one(self):
        x = self.model.continuous_var(name='xx')
        q = self.model.sumsq_vars_all_different([x])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='xx^2')

    def test_sumsq_two(self):
        x = self.model.continuous_var(name='xx')
        y = self.model.continuous_var(name='yy')
        q = self.model.sumsq_vars_all_different([x, y])
        self.check_quad(q, expected_nb_quadratics=2, expected_str='xx^2+yy^2')

    def test_sumsq_add_ko(self):
        with AdvModel(name='nochecks', checker='off') as m:
            x = m.continuous_var(name='xx')
            y = m.continuous_var(name='yy')
            q = m.sumsq_vars_all_different([x, y, x, x])
            # each variable counts once: weird but it's in the name!
            self.check_quad(q, expected_nb_quadratics=2, expected_str='xx^2+yy^2')

    def test_sumsq_vars(self):
        with AdvModel('sumsq_vars_test', keep_ordering=True) as m:
            x = m.continuous_var_list(keys=5, name='x', ub=200)
            q = m.sumsq_vars(x)
            self.check_quad(q, expected_nb_quadratics=5, expected_str='x_0^2+x_1^2+x_2^2+x_3^2+x_4^2')

    def test_sumsq_vars_ordered(self):
        # check that keep_ordering=True keeps the ordering of variables
        # when performing a sumsq_vars
        with AdvModel("varsumsq_ordered", keep_ordering=True) as m:
            alphabet = 'zorglub'
            vd = m.binary_var_list(keys=alphabet)
            e = m.sumsq_vars_all_different(vd)
            # expected is concatenation of keys with '^2' , separated by '+'
            expected = '+'.join(c + '^2' for c in alphabet)
            self.assertEqual(expected, str(e))

    def test_sumsq_vars_all_different_ko_checked(self):
        with AdvModel(checker='default') as m:
            marker_name = 'zorglub'
            a = m.continuous_var(name='a')
            z = m.continuous_var(name=marker_name)
            six.assertRaisesRegex(self, DOcplexException, '%s appears twice' % marker_name,
                                  lambda md: md.sumsq_vars_all_different([z, a, z]), m)

    def test_sumsq_vars_all_different_no_var_checked(self):
        with AdvModel(checker='default') as m:
            a = m.continuous_var(name='a')
            z = m.continuous_var(name='z')
            six.assertRaisesRegex(self, DOcplexException, 'Expecting decision variable, got: 4',
                                  lambda md: md.sumsq_vars_all_different([z, a, 4]), m)


class MatrixConstraintTests(unittest.TestCase):
    def setUp(self):
        self.model = AdvModel()
        self.xl = self.model.continuous_var_list(3, name=['x', 'y', 'z'])

    def tearDown(self):
        self.model.end()
        self.model = None


class MatrixConstraintErrorTests(MatrixConstraintTests):
    def test_matrix_varset_ko(self):
        mat = [[1, 2, 3]]  # 2 rows 2 cols
        six.assertRaisesRegex(self, DOcplexException, 'ordered sequence',
                              lambda m: m.matrix_constraints(mat, set(self.xl), [33]), self.model)

    def test_matrix_rhs_set_ko(self):
        mat = [[1, 2, 3], [4, 5, 6]]  # 2 rows 2 cols
        six.assertRaisesRegex(self, DOcplexException, 'ordered sequence',
                              lambda m: m.matrix_constraints(mat, self.xl, {33, 44}),
                              self.model)

    def test_matrix_list2_rhs_comp_ko(self):
        # use a comprehension for
        mat = [[1, 2, 3], [4, 5, 6]]  # 2 rows 3 cols
        rhs = (i ** 2 for i in range(7, 9))
        self.assertRaises(DOcplexException, lambda m: m.matrix_constraints(mat, self.xl, rhs), self.model)


class ListMatrixConstraintTests(MatrixConstraintTests):
    def test_matrix_list_empty(self):
        mat = []  # 2 rows 2 cols
        xs = []
        rhs = []
        cts = self.model.matrix_constraints(mat, xs, rhs)
        self.assertEqual(0, len(cts))

    # python lists
    def test_matrix_list1(self):
        mat = [[1, 2, 3]]  # 2 rows 2 cols
        rhs = [33]
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(1, len(cts))
        self.assertEqual('x+2y+3z <= 33', str(cts[0]))

    def test_matrix_list2(self):
        mat = [[1, 2, 3], [4, 5, 6]]  # 2 rows 3 cols
        rhs = [33, 77]
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(2, len(cts))
        self.assertEqual('x+2y+3z <= 33', str(cts[0]))
        self.assertEqual('4x+5y+6z <= 77', str(cts[1]))

    def test_matrix_list2_ge(self):
        mat = [[1, 2, 3], [4, 5, 6]]  # 2 rows 3 cols
        rhs = [33, 77]
        cts = self.model.matrix_constraints(mat, self.xl, rhs, sense='GE')
        self.assertEqual(2, len(cts))
        self.assertEqual('x+2y+3z >= 33', str(cts[0]))
        self.assertEqual('4x+5y+6z >= 77', str(cts[1]))

    def test_matrix_list_bad_column_sizes(self):
        mat = [[1, 2, 3], [4, 5, 6, 7]]  # 2 rows 3 cols
        rhs = [1] * 3
        six.assertRaisesRegex(self, DOcplexException, 'All columns should have same length',
                              lambda m: m.matrix_constraints(mat, self.xl, rhs), self.model)


@unittest.skipUnless(np, 'numpy is not present')
class NumpyMatrixConstraintTests(MatrixConstraintTests):
    def test_np_matrix1(self):
        mat = np.array([[1, 2, 3]])
        rhs = [77]
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(1, len(cts))
        self.assertEqual('x+2y+3z <= 77', str(cts[0]))

    def test_np_matrix2(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows 3 cols
        rhs = [33, 77]
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(2, len(cts))
        self.assertEqual('x+2y+3z <= 33', str(cts[0]))
        self.assertEqual('4x+5y+6z <= 77', str(cts[1]))

    def test_np_matrix2_np_rhs(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows 3 cols
        rhs = np.array([33, 77])
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(2, len(cts))
        self.assertEqual('x+2y+3z <= 33', str(cts[0]))
        self.assertEqual('4x+5y+6z <= 77', str(cts[1]))


@unittest.skipUnless(pd, 'pandas is not present')
class PandasMatrixConstraintTests(MatrixConstraintTests):
    def test_df_matrix1(self):
        mat = DataFrame(data={'a': 1, 'b': 2, 'c': 3}, index=[1])
        rhs = [77]
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(1, len(cts))
        self.assertEqual('x+2y+3z <= 77', str(cts[0]))

    def test_df_matrix2(self):
        mat = DataFrame.from_records([(1, 2, 3), (4, 5, 6)])
        rhs = [33, 77]
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(2, len(cts))
        self.assertEqual('x+2y+3z <= 33', str(cts[0]))
        self.assertEqual('4x+5y+6z <= 77', str(cts[1]))

    def test_df_matrix2_rhs_series(self):
        mat = DataFrame.from_records([(1, 2, 3), (4, 5, 6)])
        rhs = pd.Series([33, 77])
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(2, len(cts))
        self.assertEqual('x+2y+3z <= 33', str(cts[0]))
        self.assertEqual('4x+5y+6z <= 77', str(cts[1]))

    @skipIfCplexCE
    def test_matrix_santas(self):
        from examples.modeling.quadratic.santa.santasbag import build_test_model
        s = build_test_model(bagfile_name=None)
        self.assertIsNotNone(s)
        self.assertTrue(abs(s.objective_value - 35659.6) <= 1)


@unittest.skipUnless(sp, 'scipy is not present')
class ScipySparseConstraintTests(MatrixConstraintTests):

    def test_sp_matrix_diag_coo(self):
        mat = sp.coo_matrix(([1, 1, 1], ([0, 1, 2], [2, 1, 0])), shape=(3, 3))
        rhs = [4, 9, 16]
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(3, len(cts))
        self.assertEqual('z <= 4', str(cts[0]))
        self.assertEqual('y <= 9', str(cts[1]))
        self.assertEqual('x <= 16', str(cts[2]))

    def test_sp_matrix_diag_csr(self):
        mat = sp.csr_matrix(([1, 1, 1], ([0, 1, 2], [2, 1, 0])), shape=(3, 3))
        rhs = [4, 9, 16]
        cts = self.model.matrix_constraints(mat, self.xl, rhs)
        self.assertEqual(3, len(cts))
        self.assertEqual('z <= 4', str(cts[0]))
        self.assertEqual('y <= 9', str(cts[1]))
        self.assertEqual('x <= 16', str(cts[2]))

    def test_sp_quad_matrix_sum1(self):
        # 3 nonzeros: (0,2) + (1,1) + (2,0) -> xz + y^2 + zx
        mat = sp.coo_matrix(([3], ([0], [2])), shape=(3, 3))
        q = self.model.quad_matrix_sum(mat, self.xl)
        self.assertEqual(str(q), '3x*z')

    def test_sp_quad_matrix_sum2(self):
        # 3 nonzeros: (0,2) + (1,1) + (2,0) -> xz + y^2 + zx
        mat = sp.coo_matrix(([3, 7], ([0, 1], [2, 1])), shape=(3, 3))
        q = self.model.quad_matrix_sum(mat, self.xl)
        self.assertEqual(str(q), '3x*z+7y^2')


class VectorCompareTests(AdvModelTestBase):
    def setUp(self):
        AdvModelTestBase.setUp(self)
        self.a = self.model.continuous_var(name='a')
        self.b = self.model.continuous_var(name='b')
        self.size = 3
        self.xl = self.model.continuous_var_list(keys=self.size, name='x')

    def test_compare_lists(self):
        l1 = self.xl
        l2 = list(range(1, self.size + 1))
        cts = self.model.vector_compare(l1, l2, 'le')
        self.assertEqual(self.size, len(cts))
        self.assertEqual(['x_0 <= 1', 'x_1 <= 2', 'x_2 <= 3'], [str(c) for c in cts])

    def test_compare_comps(self):
        l1 = self.xl
        l2 = range(1, self.size + 1)
        cts = self.model.vector_compare(l1, l2, 'le')
        self.assertEqual(self.size, len(cts))
        self.assertEqual(['x_0 <= 1', 'x_1 <= 2', 'x_2 <= 3'], [str(c) for c in cts])

    def test_compare_bad_lists(self):
        l1 = self.xl
        six.assertRaisesRegex(self, DOcplexException,
                              "same length", lambda m_: m_.vector_compare(l1, [1], 'le'), self.model)

    def test_compare_list_empty1(self):
        l1 = []
        l2 = []
        cts = self.model.vector_compare(l1, l2, 'le')
        self.assertEqual([], cts)

    @unittest.skipUnless(np, 'numpy is not present')
    def test_compare_np_arrays(self):
        s1 = np.array(self.xl)
        s2 = np.array(list(range(1, self.size + 1)))
        cts = self.model.vector_compare(s1, s2, 'le')
        self.assertEqual(self.size, len(cts))
        self.assertEqual(['x_0 <= 1', 'x_1 <= 2', 'x_2 <= 3'], [str(c) for c in cts])

    @unittest.skipUnless(pd, 'pandas is not present')
    def test_compare_series(self):
        s1 = pd.Series(self.xl)
        s2 = pd.Series(list(range(1, self.size + 1)))
        cts = self.model.vector_compare(s1, s2, 'le')
        self.assertEqual(self.size, len(cts))
        self.assertEqual(['x_0 <= 1', 'x_1 <= 2', 'x_2 <= 3'], [str(c) for c in cts])


class ListMatrixRangeTests(MatrixConstraintTests):

    def test_matrix_list_empty(self):
        rrs = self.model.matrix_ranges(coef_mat=[], dvars=[], lbs=[], ubs=[])
        self.assertEqual(0, len(rrs))

    def test_matrix_list1(self):
        mat = [[1, 2, 3]]  # 2 rows 2 cols
        lbs = [33]
        ubs = [77]
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(1, len(rrs))
        self.assertEqual('33 <= x+2y+3z <= 77', str(rrs[0]))

    def test_matrix_list2(self):
        mat = [[1, 2, 3], [4, 5, 6]]  # 2 rows 3 cols
        lbs = [33, 34]
        ubs = [77, 78]
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(2, len(rrs))
        self.assertEqual(['33 <= x+2y+3z <= 77', '34 <= 4x+5y+6z <= 78'], list(map(str, rrs)))

    def test_matrix_list_bad_lb_size(self):
        mat = [[1, 2, 3], [4, 5, 6]]  # 2 rows 3 cols
        lbs = [1] * 4
        ubs = [2] * 3
        six.assertRaisesRegex(self, DOcplexException, 'Incorrect size for range lower bounds',
                              lambda m: m.matrix_ranges(mat, self.xl, lbs, ubs), self.model)

    def test_matrix_list_bad_ub_size(self):
        mat = [[1, 2, 3], [4, 5, 6]]  # 2 rows 3 cols
        lbs = [1] * 2
        ubs = [2] * 7
        six.assertRaisesRegex(self, DOcplexException, 'Incorrect size for range upper bounds',
                              lambda m: m.matrix_ranges(mat, self.xl, lbs, ubs), self.model)


@unittest.skipUnless(np, 'numpy is not present')
class NumpyMatrixRangeTests(MatrixConstraintTests):

    def test_np_matrix1(self):
        mat = np.array([[1, 2, 3]])
        lbs = [11]
        ubs = [77]
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(1, len(rrs))
        self.assertEqual('11 <= x+2y+3z <= 77', str(rrs[0]))

    def test_np_matrix2(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows 3 cols
        lbs = [100, 101]
        ubs = [7000, 7001]
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(2, len(rrs))
        self.assertEqual(['100 <= x+2y+3z <= 7000', '101 <= 4x+5y+6z <= 7001'], list(map(str, rrs)))

    def test_np_matrix2_np_bounds(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows 3 cols
        lbs = np.array([100, 101])
        ubs = np.array([7000, 7001])
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(2, len(rrs))
        self.assertEqual(['100 <= x+2y+3z <= 7000', '101 <= 4x+5y+6z <= 7001'], list(map(str, rrs)))


@unittest.skipUnless(pd, 'pandas is not present')
class PandasMatrixRangeTests(MatrixConstraintTests):
    def test_df_matrix1(self):
        mat = DataFrame(data={'a': 1, 'b': 2, 'c': 3}, index=[1])
        lbs = [100]
        ubs = [7000]
        cts = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(1, len(cts))
        self.assertEqual('100 <= x+2y+3z <= 7000', str(cts[0]))

    def test_df_matrix2(self):
        mat = DataFrame.from_records([(1, 2, 3), (4, 5, 6)])
        lbs = [100, 101]
        ubs = [7000, 7001]
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(2, len(rrs))
        self.assertEqual(['100 <= x+2y+3z <= 7000', '101 <= 4x+5y+6z <= 7001'], list(map(str, rrs)))

    def test_df_matrix2_bound_series(self):
        mat = DataFrame.from_records([(1, 2, 3), (4, 5, 6)])
        lbs = pd.Series([100, 101])
        ubs = pd.Series([7000, 7001])
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(2, len(rrs))
        self.assertEqual(['100 <= x+2y+3z <= 7000', '101 <= 4x+5y+6z <= 7001'], list(map(str, rrs)))


@unittest.skipUnless(sp, 'scipy is not present')
class ScipySparseRangeTests(MatrixConstraintTests):

    def test_sp_matrix_diag_coo(self):
        mat = sp.coo_matrix(([1, 1, 1], ([0, 1, 2], [2, 1, 0])), shape=(3, 3))
        ubs = [4, 9, 16]
        lbs = [1, 2, 3]
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(3, len(rrs))
        self.assertEqual('1 <= z <= 4', str(rrs[0]))
        self.assertEqual('2 <= y <= 9', str(rrs[1]))
        self.assertEqual('3 <= x <= 16', str(rrs[2]))

    def test_sp_matrix_diag_csr(self):
        mat = sp.csr_matrix(([1, 1, 1], ([0, 1, 2], [2, 1, 0])), shape=(3, 3))
        ubs = [4, 9, 16]
        lbs = [1, 2, 3]
        rrs = self.model.matrix_ranges(mat, self.xl, lbs, ubs)
        self.assertEqual(3, len(rrs))
        self.assertEqual('1 <= z <= 4', str(rrs[0]))
        self.assertEqual('2 <= y <= 9', str(rrs[1]))
        self.assertEqual('3 <= x <= 16', str(rrs[2]))


class ScalProdTripleTests(AdvModelTestBase):

    def setUp(self):
        self.model = AdvModel(name="quads", keep_ordering=True)
        m = self.model
        qf = m._qfactory
        self.zero = ZeroExpr(m)
        self.x = m.continuous_var(name='x')
        self.y = m.continuous_var(name='y')
        self.z = m.continuous_var(name='z')
        self.x2 = self.x ** 2
        self.y2 = self.y ** 2
        self.xy = self.x * self.y
        self.q3 = qf.new_quad(quads=None, linexpr=3)  # constant 3 as a quad
        self.qz = qf.new_quad(quads=None, linexpr=0)  # constant zero as a quad
        self.q1 = qf.new_quad(quads=None, linexpr=1)  # constant 1 as quad

    def tearDown(self):
        self.model.end()
        self.model = None
    # scalprod_triple
    def test_scalprod_triple_var_var(self):
        q = self.model.scal_prod_triple(coefs=[3], left_terms=[self.x], right_terms=[self.y])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='3x*y')

    def test_scalprod_triple_var_mn(self):
        q = self.model.scal_prod_triple(coefs=[3], left_terms=[self.x], right_terms=[7 * self.y])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='21x*y')

    def test_scalprod_triple_var_lin1(self):
        q = self.model.scal_prod_triple(coefs=[3], left_terms=[self.x], right_terms=[self.y + 5])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='3x*y+15x')

    def test_scalprod_triple_mn_var(self):
        q = self.model.scal_prod_triple(coefs=[3], left_terms=[3 * self.x], right_terms=[self.y])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='9x*y')

    def test_scalprod_triple_mn_mn(self):
        q = self.model.scal_prod_triple(coefs=[3], left_terms=[3 * self.x], right_terms=[-7 * self.y])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='-63x*y')

    def test_scalprod_triple_mn_lin(self):
        q = self.model.scal_prod_triple(coefs=[3], left_terms=[3 * self.x], right_terms=[-7 * self.y + 1])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='-63x*y+9x')

    def test_scalprod_triple_lin_var(self):
        q = self.model.scal_prod_triple(coefs=[2], left_terms=[5 * self.x + 3], right_terms=[self.y])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='10x*y+6y')

    def test_scalprod_triple_lin_mn(self):
        q = self.model.scal_prod_triple(coefs=[2], left_terms=[5 * self.x + 3], right_terms=[2 * self.y])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='20x*y+12y')

    def test_scalprod_triple_lin_lin(self):
        q = self.model.scal_prod_triple(coefs=[2], left_terms=[self.x + 3], right_terms=[5 * self.y + 7])
        self.check_quad(q, expected_nb_quadratics=1, expected_str='10x*y+14x+30y+42')

    def test_scalprod_triple_zero_expr(self):
        zz = ZeroExpr(self.model)
        q = self.model.scal_prod_triple(left_terms=[self.x], right_terms=[zz], coefs=[2])
        self.assertEqual('0', str(q))

    def test_scalprod_triple_abs(self):
        m = self.model
        q = self.model.scal_prod_triple(left_terms=[m.abs(self.x)], right_terms=[self.y], coefs=[2])
        expected = '2y*_abs3' if are_functional_vars_named() else '2y*x4'
        self.assertEqual(expected, str(q))

    def test_scalprod_triple_31985(self):
        from docplex.mp.advmodel import AdvModel
        with AdvModel(name='31985') as m:
            q = m.scal_prod_triple(left_terms=self.x, right_terms=self.y, coefs=5)
            self.assertEqual('5x*y', str(q))


if __name__ == "__main__":
    unittest.main()
