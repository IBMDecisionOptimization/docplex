# Tests with the Python ecosystem

from unittest import TestCase, skipUnless, main, skip

from docplex.mp.model import Model
from docplex.mp.advmodel import AdvModel
from docplex.mp.linear import Expr, LinearExpr
from testutils import RedirectedOutputToStringContext


try:
    import numpy as np

except ImportError:

    np = None

try:
    import pandas as pd

except ImportError:
    pd = None

@skipUnless(np, "numpy is not present")
class NumpyTests(TestCase):
    def setUp(self):
        self.model = Model("numpy", output_level="warning")
        self.x = self.model.continuous_var(name='x')
        self.y = self.model.continuous_var(name='y')

    def tearDown(self):
        self.model.end()
        self.model = None

    def has_numpy(self):
        return self.model.environment.has_numpy

    def _check_solve(self, expect_ok=True):
        if self.model._can_solve():
            sol = self.model.solve()
            if expect_ok:
                self.assertIsNotNone(sol)
            else:
                self.assertIsNone(sol)

    def test_numpy_int_timelimit(self):
        m = self.model
        raw = 119
        m.parameters.timelimit = np.int64(raw)
        m.apply_parameters()
        self.assertEqual(raw, m.parameters.timelimit())

    def test_numpy_int_writelevel(self):
        m = self.model
        raw = 3
        m.parameters.output.writelevel = np.int64(raw)
        m.apply_parameters()
        self.assertEqual(raw, m.parameters.output.writelevel())


    @skipUnless(np, "numpy is not present")
    def test_numpy_var_int_bounds(self):
        numpy_lb = np.int64(3)
        numpy_ub = np.int64(11)
        ix = self.model.integer_var(name="var_with_numpy_bounds", lb=numpy_lb, ub=numpy_ub)
        self.assertEqual(3, ix.lb)
        self.assertEqual(11, ix.ub)

    def test_numpy_batch_var_lower_bound_num(self):
        m = self.model
        numpy_lb = np.int64(3)
        ixs = self.model.integer_var_list(2, name="var_with_numpy_bounds")
        m.change_var_lower_bounds(ixs, numpy_lb)
        lb0 = ixs[0].lb
        self.assertEqual(3, ixs[0].lb)
        self.assertTrue(isinstance(ixs[0].lb, float))

    def test_numpy_batch_var_lower_bound_seq(self):
        m = self.model
        numpy_lb3 = np.int64(3)
        numpy_lb7 = np.int64(7)
        ixs = self.model.integer_var_list(2, name="var_with_numpy_bounds")
        m.change_var_lower_bounds(ixs, (numpy_lb3, numpy_lb7) )
        self.assertTrue(all(isinstance(x_.lb, float) for x_ in ixs))
        self.assertEqual([3,7], [x.lb for x in ixs])

    def test_numpy_batch_var_upper_bound_num(self):
        m = self.model
        numpy_ub = np.int64(333)
        ixs = self.model.integer_var_list(2, name="var_with_numpy_bounds")
        m.change_var_upper_bounds(ixs, numpy_ub)
        self.assertEqual(333, ixs[0].ub)
        self.assertTrue(isinstance(ixs[0].ub, float))

    def test_numpy_batch_var_upper_bound_seq(self):
        m = self.model
        numpy_ub3 = np.int64(333)
        numpy_ub7 = np.int64(777)
        ixs = self.model.integer_var_list(2, name="var_with_numpy_bounds")
        m.change_var_upper_bounds(ixs, (numpy_ub3, numpy_ub7) )
        self.assertTrue(all(isinstance(x_.ub, float) for x_ in ixs))
        self.assertEqual([333,777], [x.ub for x in ixs])


    @skipUnless(np, "numpy is not present")
    def test_numpy_contvar_large_ub(self):
        np_large_ub = np.int64(79800000000)
        m = self.model
        ix = m.continuous_var(name="var_with_numpy_bounds", ub=np_large_ub)

    @skipUnless(np, "numpy is not present")
    def test_numpy_intvar_large_ub(self):
        np_large_ub = np.int64(7980000000)
        ix = self.model.integer_var(name="var_with_numpy_bounds", lb=0, ub=np_large_ub)

    @skipUnless(np, "numpy is not present")
    def test_numpy_intvarlist_large_ub(self):
        np_large_ub = np.int64(7980000000)
        ixl = self.model.integer_var_list(2, name="var_with_numpy_bounds", lb=0, ub=np_large_ub)


    @skipUnless(np, "numpy is not present")
    def test_numpy_floatvar_large_ub(self):
        np_large_ub = np.float64(7980000000.5)
        ix = self.model.integer_var(name="var_with_numpy_bounds", lb=0, ub=np_large_ub)

    @skipUnless(np, "numpy is not present")
    def test_numpy_floatvarlist_large_ub(self):
        np_large_ub = np.int64(7980000000.5)
        ixl = self.model.integer_var_list(2, name="var_with_numpy_bounds", lb=0, ub=np_large_ub)

    @skipUnless(np, "numpy is not present")
    def test_numpy_change_var_lb_int(self):
        v = self.model.continuous_var(name="v")
        numpy_lb = np.int64(3000)
        v.lb = numpy_lb
        self.assertEqual(3000, v.lb)
        self.assertIsNotNone(self.model.solve())

    @skipUnless(np, "numpy is not present")
    def test_numpy_change_var_ub_int(self):
        v = self.model.continuous_var(name="v")
        numpy_ub = np.int64(999999)
        v.ub = numpy_ub
        self.assertEqual(999999, v.ub)
        self.assertIsNotNone(self.model.solve())

    @skipUnless(np, "numpy is not present")
    def test_numpy_var_float_bounds(self):
        numpy_lb = np.float64(3.14)
        numpy_ub = np.float64(12.34)
        ix = self.model.continuous_var(name="var_with_numy_bounds", lb=numpy_lb, ub=numpy_ub)
        self.assertEqual(3.14, ix.lb)
        self.assertEqual(12.34, ix.ub)

    @skipUnless(np, "numpy is not present")
    def test_numpy_change_var_lb_float(self):
        v = self.model.continuous_var(name="v")
        numpy_lb = np.float64(3.14)
        v.lb = numpy_lb
        self.assertEqual(3.14, v.lb)

    @skipUnless(np, "numpy is not present")
    def test_numpy_change_var_ub_float(self):
        v = self.model.continuous_var(name="v")
        numpy_fub = np.float64(3.14)
        v.ub = numpy_fub
        self.assertEqual(3.14, v.ub)


    def test_numpy_custom_lbs(self):
        nbs = np.array([100, 1000, 10000])
        xs = self.model.continuous_var_list(len(nbs), lb=nbs)

    def test_numpy_custom_ubs(self):
        nbs = np.array([100, 1000, 10000])
        xs = self.model.continuous_var_list(len(nbs), ub=nbs)

    @skipUnless(np, "numpy is not present")
    def test_numpy_ile(self):
        numpy_irhs = np.int64(33)
        numpy_k = np.int64(7)
        self.model.add_constraint(numpy_k * self.x <= numpy_irhs)
        self._check_solve()

    @skipUnless(np, "numpy is not present")
    def test_numpy_ieq(self):
        numpy_irhs = np.int64(33)
        numpy_k = np.int64(7)
        self.model.add_constraint(numpy_k * self.x == numpy_irhs)

    @skipUnless(np, "numpy is not present")
    def test_numpy_ige(self):
        numpy_irhs = np.int64(33)
        numpy_k = np.int64(7)
        self.model.add_constraint(numpy_k * self.x >= numpy_irhs)
        self._check_solve()

    @skipUnless(np, "numpy is not present")
    def test_numpy_fle(self):
        numpy_rhs = np.float64(33.3)
        numpy_k = np.float64(7.7)
        self.model.add_constraint(numpy_k * self.x <= numpy_rhs)
        self._check_solve()

    @skipUnless(np, "numpy is not present")
    def test_numpy_feq(self):
        numpy_rhs = np.float64(33.3)
        numpy_k = np.float64(7.7)
        self.model.add_constraint(numpy_k * self.x == numpy_rhs)
        self._check_solve()

    @skipUnless(np, "numpy is not present")
    def test_numpy_fge(self):
        numpy_rhs = np.float64(33.3)
        numpy_k = np.float64(7.77)
        self.model.add_constraint(numpy_k * self.x >= numpy_rhs)
        self._check_solve()

    # see RTC-28146
    @skipUnless(np, "numpy is not present")
    def test_numpy_ile_numpy_first(self):
        numpy_lhs = np.int64(33)
        numpy_k = np.int64(7)
        self.model.add_constraint(numpy_lhs <= numpy_k * self.x)

    @skipUnless(np, "numpy is not present")
    def test_numpy_ige_numpy_first(self):
        numpy_lhs = np.int64(33)
        numpy_k = np.int64(7)
        self.model.add_constraint(numpy_lhs >= numpy_k * self.x)

    @skipUnless(np, "numpy is not present")
    def test_numpy_ieq_numpy_first(self):
        numpy_lhs = np.int64(33)
        numpy_k = np.int64(7)
        self.model.add_constraint(numpy_lhs == numpy_k * self.x)

    def test_numpy_ctblock_rhs(self):
        numpy_rhss = [np.int64(33), np.int64(333)]
        self.model.add_constraints([self.x <= numpy_rhss[0], self.y <= numpy_rhss[1]])

    def test_numpy_indicator_rhs(self):
        numpy_rhs = np.int64(33)
        b = self.model.binary_var()
        self.model.add_indicator(b, self.x <= numpy_rhs)

    def test_numpy_range_bounds(self):
        numpy_lhs = np.int64(34)
        numpy_rhs = np.int64(66)
        rrct = self.model.add_range(lb=numpy_lhs, expr=self.x, ub=numpy_rhs)
        self.assertEqual(34, rrct.lb)
        self.assertEqual(66, rrct.ub)
        # rhs is ub (minus constant)
        self.assertEqual(66, rrct.cplex_num_rhs())
        # rangeval is - (ub-lb)
        self.assertEqual(-32, rrct.cplex_range_value())

    @skipUnless(np, "numpy is not present")
    def test_numpy_add_numpy_first(self):
        numpy_k = np.int64(33)
        expr = numpy_k + self.x
        self.assertIsInstance(expr, Expr)
        self.assertEqual("x+33", str(expr))

    @skipUnless(np, "numpy is not present")
    def test_numpy_mul_numpy_first(self):
        numpy_k = np.int64(77)
        expr = numpy_k * self.x
        self.assertIsInstance(expr, Expr)
        self.assertEqual("77x", str(expr))

    @skipUnless(np, "numpy is not present")
    def test_numpy_sub_numpy_first(self):
        numpy_k = np.int64(123)
        expr = numpy_k - self.x
        self.assertIsInstance(expr, Expr)
        self.assertEqual("-x+123", str(expr))

    @skipUnless(np, "numpy is not present")
    def test_numpy_mul_zero_int_numpy(self):
        numpy_k = np.int64(0)
        expr = numpy_k * self.x
        self.assertTrue(expr.is_zero())

    @skipUnless(np, "numpy is not present")
    def test_numpy_mul_zero_float_numpy(self):
        numpy_k = np.float64(0)
        expr = numpy_k * self.x
        self.assertTrue(expr.is_zero())

    @skipUnless(np, "numpy is not present")
    def test_numpy_sum(self):
        m = self.model
        size = 10
        the_range = range(0, size)
        np_coefs = np.array([k + 1 for k in the_range])
        alldvars = self.model.continuous_var_list(size, lb=1, name='x')
        expr = m.sum(alldvars[i] * np_coefs[i] for i in the_range)
        m.minimize(expr)
        if m._can_solve():
            m.solve()
            expected = size * (size + 1) / 2
            self.assertEqual(expected, expr.solution_value)

    @skipUnless(np, "numpy is not present")
    def test_scalprod_numpy_vars(self):
        m = self.model
        size = 10
        the_range = range(0, size)
        np_coefs = np.array([k + 1 for k in the_range])
        xs = self.model.continuous_var_list(size, lb=1, name='x')
        npxs = np.array(xs)
        expr = m.scal_prod(npxs, np_coefs)
        m.minimize(expr)
        if m._can_solve():
            m.solve()
            expected = size * (size + 1) / 2
            self.assertEqual(expected, expr.solution_value)

    @skipUnless(np, "numpy is not present")
    def test_scalprod_numpy_coefs(self):
        m = self.model
        size = 10
        the_range = range(0, size)
        np_coefs = np.array([k + 1 for k in the_range])
        alldvars = self.model.continuous_var_list(size, lb=1, name='x')
        expr = m.scal_prod(alldvars, np_coefs)
        m.minimize(expr)
        if m._can_solve():
            m.solve()
            expected = size * (size + 1) / 2
            self.assertEqual(expected, expr.solution_value)

    @skipUnless(np, "numpy is not present")
    def test_numpy_scalprod_empty(self):
        m = self.model
        size = 10
        np_empty_coefs = np.array([])
        alldvars = self.model.continuous_var_list(size, lb=1, name='x')
        expr = m.scal_prod(alldvars, np_empty_coefs)
        self.assertTrue(expr.is_constant())
        self.assertEqual(0, expr.constant)

    @skipUnless(np, "numpy is not present")
    def setup_numpy_scalprod(self, size):
        m = self.model
        dvars1 = m.continuous_var_list(size, lb=1, name='x1')
        dvars2 = m.continuous_var_list(size, lb=1, name='x2')
        # 1 build anumpy vector of dvars
        npdvars = np.array([dvars1, dvars2])
        # 2 create a numpy vector of coefs
        npcoefs = np.array([range(1, size + 1), range(size, 2 * size)])
        # 3 multiply
        npscalprod = npdvars * npcoefs
        # print(npscalprod)
        return npscalprod

    @skipUnless(np, "numpy is not present")
    def test_numpy_scalprod_numpy_int_zero(self):
        m = self.model
        size = 10
        alldvars = self.model.continuous_var_list(size, lb=1, name='x')
        np_izero = np.int64(0)
        npz_dot = m.dot(alldvars, np_izero)
        self.assertEqual("0", str(npz_dot))

    @skipUnless(np, "numpy is not present")
    def test_numpy_scalprod_numpy_float_zero(self):
        m = self.model
        size = 10
        alldvars = self.model.continuous_var_list(size, lb=1, name='x')
        np_fzero = np.float64(0)
        npz_dot = m.dot(alldvars, np_fzero)
        self.assertEqual("0", str(npz_dot))

    @skipUnless(np, "numpy is not present")
    def test_numpy_scalprod_flatten(self):
        with Model(keep_ordering=True) as m:
            size = 3
            npscalprod = self.setup_numpy_scalprod(size)
            # need to take the flatten iterator here
            e = m.sum(npscalprod.flat)
            self.assertIsInstance(e, LinearExpr)
            self.assertEqual(2 * size, e.number_of_variables())
            self.assertEqual(str(e), "x1_0+2x1_1+3x1_2+3x2_0+4x2_1+5x2_2")

    @skipUnless(np, "numpy is not present")
    def test_numpy_scalprod_raw(self):
        m = self.model
        size = 3
        npscalprod = self.setup_numpy_scalprod(size)
        # need to take the flatten iterator here
        e = m.sum(npscalprod)
        self.assertIsInstance(e, LinearExpr)
        self.assertEqual(2 * size, e.number_of_variables())
        # self.assertEqual(str(e), "x1_0+2x1_1+2x2_0+3x2_1")

    @skipUnless(np, "numpy is not present")
    def test_numpy_print_mnm_array(self):
        # build a ndarray of dvars and just check that print will not crash
        m = self.model
        size = 3
        dvars = m.continuous_var_list(size, lb=1, name='x1')
        mnms = [dv * 7 for dv in dvars]
        # 1 build anumpy vector of dvars
        np_mnms = np.array(mnms)
        with RedirectedOutputToStringContext() as out:
            print(np_mnms)
        self.assertEqual(out.get_str(), "[7x1_0 7x1_1 7x1_2]\n")

    @skipUnless(np, "numpy is not present")
    def test_numpy_print_linexpr_array(self):
        # build a ndarray of dvars and just check that print will not crash
        m = self.model
        size = 3
        z = m.continuous_var(name='zz')
        dvars = m.continuous_var_list(size, lb=1, name='x1')
        exprs = [z + dv * 7 + 13 for dv in dvars]
        # 1 build anumpy vector of dvars
        np_exprs = np.array(exprs)
        with RedirectedOutputToStringContext() as out:
            print(np_exprs)
        self.assertTrue(out.get_str().startswith('[zz+7x1_0+13 zz+7x1_1+13 zz+7x1_2+13]'))

    @skipUnless(np, "numpy is not present")
    def test_numpy_create_vector_cts(self):
        m = self.model
        size = 3
        dvars = m.continuous_var_list(size, lb=1, name='x1')
        # np_dvars = np.array(dvars)
        # np_rhs = np.array([i for i in range(1, size+1)])
        # this is crashing
        # np_cts = np_dvars <= np_rhs
        #
        # m.add_constraints(np_cts)
        # self.assertEqual(size, m.number_of_constraints)

    @skipUnless(np, "numpy is not present")
    def test_numpy_block_cts_rhs(self):
        m = self.model
        x = self.x
        cts = [ x <= np.int64(77), self.y <= np.float64(3.14)]
        m.add_constraints(cts)
        self.assertEqual(2, m.number_of_constraints)


    @skipUnless(np, "numpy is not present")
    def test_numpy_lp(self):
        # define constraitns with numpy int and numpy float coefs, send to lp
        m = self.model
        x = self.x
        y = self.y
        numpy_i33 = np.int64(33)
        numpy_i7 = np.int64(7)

        numpy_f75 = np.float64(7.5)
        numpy_f325 = np.float64(3.1415)
        m += ( numpy_i7 * x <= numpy_i33)
        m += (numpy_f75 * y <= numpy_f325)
        m.minimize(numpy_f75*x + numpy_i7*y)
        lps = m.export_as_lp_string()
        self.assertIn("7.5", lps)
        self.assertIn("3.1415", lps)


    @skipUnless(np, "numpy is not present")
    def test_numpy_array_as_varlist_keys(self):
        npk = np.array([3, 5, 7, 9])
        npvs = self.model.continuous_var_list(keys=npk, name='np')
        self.assertEqual(len(npk), len(npvs))

    @skipUnless(np, "numpy is not present")
    def test_numpy_pwl(self):
        from numbers import Number
        m = self.model
        nph = np.float64(0.5)
        np1 = np.float64(1.0)
        np2 = np.float64(2.0)
        np3 = np.float64(3.0)
        nppwl = m.piecewise(preslope=-nph, breaksxy=[(np1, np3), (2, np2)], postslope=nph)


    def test_numpy_int_obj_offset(self):
        npoff = np.int64(999)
        m = self.model
        m.minimize(self.x+npoff)

    def test_solution_values_from_numpy(self):
        m = self.model
        size = 17
        xl = self.model.continuous_var_list(size, name='pdr')
        nxs = np.array(xl)
        sol = m.new_solution(var_value_dict={xl[i]: i for i in range(size)})
        svals = sol.get_values(nxs)
        self.assertEqual(size,len(svals))
        for i in range(size):
            self.assertEqual(i, svals[i])

    def test_numpy_quad_obj_coef(self):
        m = self.model
        np1 = np.int64(3)
        xx = m.continuous_var(name='xx')
        m.maximize(np1 * xx ** 2)
        lps = m.export_as_lp_string()
        self.assertIn('[ 6 xx^2 ]/2', lps)

    def test_quad_matrix_sum_numpy_array(self):
        with AdvModel() as am:
            # coefs as a square matrix 3-3
            mat = np.array([[1,2], [3,4]])
            xl = am.continuous_var_list(2, name=['x', 'y'])
            q = am._aggregator.quad_matrix_sum(mat, xl, symmetric=False)
            self.assertEqual('x^2+5x*y+4y^2', str(q))


@skipUnless(pd, "pandas is not present")
class PandasTests(TestCase):
    def setUp(self):
        mdl = Model("pandas_test")
        self.model = mdl
        self.xyz = mdl.integer_var_list(3, name=["x", "y", "z"], lb=0, ub=30)
        if pd:
            pdf = pd.DataFrame({"dvar": pd.Series(self.xyz),
                                "icoefs": pd.Series([11, 22, 33]),
                                "costs": pd.Series([5, 7, 11]),
                                "fcoefs": pd.Series([5.5, 6.6, 7.7]),
                                "pdnames": pd.Series(["foo", "bar", "gee"])})
            # new column: use pandas * operator.
            pdf["var_icoef"] = pdf["dvar"] * pdf["icoefs"]
            self.pdf = pdf
        else:
            self.pdf = None

    def test_pandas_dataframe_as_vardict_keys(self):
        pdf = pd.DataFrame({'c1': [1, 2],
                            'c2': [3, 4]},
                           index=['r1', 'r2']
                           )
        #print(pdf)
        pdvs = self.model.continuous_var_dict(keys=pdf, name='pandas')
        self.assertEqual(2, len(pdvs))
        # expecting 'c1' -> Var(name='pandas_c1', ...)
        self.assertEqual('pandas_r1', pdvs['r1'].name)
        self.assertEqual('pandas_r2', pdvs['r2'].name)

    def test_pandas_dataframe_as_var_dict_index(self):
        with Model("RTC-32344") as mdl:
            df_data = [(-99,1,990,510,0,16.000,19.500,20.500,24.000,7.00,714.00000000),
                       (-99,1,990,510,0,16.000,16.000,16.000,20.000,4.00,408.00000000),
                       (-99,1,990,510,0,16.000,16.000,16.000,20.500,4.50,459.00000000),
                       (-99,1,990,510,0,16.000,16.000,16.000,21.000,5.00,510.00000000)]
            df = pd.DataFrame(df_data,columns=['Assoc','Div_Nbr','Dept','Job','Day','Start','Lunch_Start','Lunch_End','End','Length','Pref_Penalty'])
            # print(df.index.values.tolist())
            my_var_dict = mdl.binary_var_dict(keys=df, name='var_dict_indexed_by_df')
            self.assertEqual(mdl.statistics.number_of_binary_variables,4)

    def test_pandas_series_as_varlist_keys(self):
        items = ['a', 'b', 'c', 'd', 'e']
        pds = pd.Series(items)
        pdvs = self.model.continuous_var_list(keys=pds, name='pandas')
        self.assertEqual(len(items), len(pdvs))
        self.assertEqual(['pandas_%s' % i for i in items], [v.name for v in pdvs])

    def test_pandas_series_as_vardict_keys(self):
        items = ['1', '2', '3', '4']
        pds = pd.Series(items, index=['a', 'b', 'c', 'd'])
        #print(pds)
        pdvs = self.model.continuous_var_dict(keys=pds, name='pandas')
        # TODO: finalize asserts (for now values, why not indices????)
        #print('keys: {0!s}'.format(pdvs.keys()))

    def test_pandas_custom_lbs(self):
        nbs = pd.Series([100, 1000, 10000])
        xs = self.model.continuous_var_list(len(nbs), lb=nbs)

    def test_numpy_custom_ubs(self):
        nbs = pd.Series([100, 1000, 10000])
        xs = self.model.continuous_var_list(len(nbs), ub=nbs)

    def test_pandas_strings_as_varnames(self):
        pdnames = self.pdf["pdnames"]
        xyz = self.model.integer_var_list(3, name=pdnames)
        x = xyz[0]
        self.assertEqual(x.name, pdnames[0])

    def test_export_lp_unicode(self):
        mdl = self.model
        pdnames = self.pdf["pdnames"]
        xyz = mdl.integer_var_list(3, name=pdnames)
        badname_var = mdl.continuous_var(name="foo the bar")
        c = mdl.add_constraint(sum(xyz) <= badname_var + 1, ctname=u"some_unicode_ct")
        lps = mdl.export_as_lp_string()
        self.assertTrue("some_unicode_ct" in lps)
        self.assertTrue("foo_the_bar" in lps)

    def _check_one_expr(self, expr):
        # add a constraint expr <= 1 to the model
        self.model.add_constraint(expr >= 1)

    @skipUnless(pd, "pandas is not present")
    def test_pandas_sum_dtf_dvars(self):
        # should not crash
        e = self.model.sum(self.pdf["dvar"])
        self.assertIsInstance(e, Expr)

    def test_pandas_sum_dtf_exprs(self):
        self.model.sum(self.pdf["var_icoef"])

    def test_pandas_sum_dtf_int_coefs_dotprod(self):
        pdf = self.pdf
        sum_ex = self.model.sum(pdf["dvar"][i] * pdf["icoefs"][i] for i in range(len(pdf)))
        self._check_one_expr(sum_ex)

    def test_pandas_sum_dtf_float_coefs_dotprod(self):
        pdf = self.pdf
        sum_ex = self.model.sum(pdf["dvar"][i] * pdf["fcoefs"][i] for i in range(len(pdf)))
        self._check_one_expr(sum_ex)

    def test_pandas_scalprod_varlist_dtf_int_coefs(self):
        self.model.scal_prod(self.xyz, self.pdf["icoefs"])

    def test_pandas_scalprod_varlist_dtf_float_coefs(self):
        sp = self.model.scal_prod(self.xyz, self.pdf["fcoefs"])
        self._check_one_expr(sp)

    def test_pandas_scalprod_dtf_dvars_dtf_int_coefs(self):
        pdf = self.pdf
        sp = self.model.scal_prod(pdf["dvar"], pdf["icoefs"])
        self._check_one_expr(sp)

    def test_pandas_scalprod_dtf_dvars_dtf_float_coefs(self):
        pdf = self.pdf
        sp = self.model.scal_prod(pdf["dvar"], pdf["fcoefs"])
        self._check_one_expr(sp)

    def test_pandas_scalprod_dtf_exprs(self):
        pdf = self.pdf
        sp = self.model.scal_prod(pdf["var_icoef"], 2)
        self._check_one_expr(sp)

    @skipUnless(pd, "pandas is not present")
    def test_pandas_linearct_lhs(self):
        pdf = self.pdf
        self.model.add_constraint(pdf["dvar"][0] <= pdf["icoefs"][0])

    def test_pandas_linearct_rhs(self):
        if pd:
            pdf = self.pdf
            self.model.add_constraint(pdf["dvar"][0] <= pdf["icoefs"][1])

    def test_pandas_linearct_lhs_rhs(self):
        if pd:
            pdf = self.pdf
            self.model.add_constraint(
                pdf["icoefs"][0] * pdf["dvar"][0] <= pdf["dvar"][1] * pdf["icoefs"][1] + pdf["icoefs"][2])

    @skipUnless(pd, "pandas is not present")
    def test_pandas_set_objective_sum_dvars(self):
        mdl = self.model
        pdf = self.pdf
        mdl.maximize(mdl.sum(pdf["dvar"]))

    @skipUnless(pd, "pandas is not present")
    def test_pandas_set_objective_sum_exprs(self):
        mdl = self.model
        pdf = self.pdf
        mdl.maximize(mdl.sum(pdf["var_icoef"]))

    @skipUnless(pd, "pandas is not present")
    def test_pandas_set_objective_scalprod_dvars(self):
        mdl = self.model
        pdf = self.pdf
        mdl.maximize(mdl.scal_prod(pdf["dvar"], pdf["costs"]))

    @skipUnless(pd, "pandas is not present")
    def test_pandas_irv(self):
        # need a fresh model as the default one as x,y,z
        # do it the irv way
        with Model(name="irv") as mdl:
            xyz = mdl.integer_var_list(3, name=["x", "y", "z"], lb=0, ub=30)
            pdf = pd.DataFrame({"dvar": pd.Series(xyz),
                                "coef": pd.Series([11, 22, 33]),
                                "costs": pd.Series([5, 7, 11])})
            pdf["obj"] = pdf["dvar"] * pdf["coef"]
            obj = mdl.sum(pdf["dvar"])
            for i in range(3):
                mdl.add_constraint(pdf["dvar"][i] <= pdf["coef"][i])

            mdl.add_constraint(mdl.scal_prod(pdf["dvar"], pdf["coef"]) <= 50)
            # the pdf[costs] contains numpy/pandas integers, that have to be converted to floats for cplex
            mdl.add_constraint(pdf["costs"][0] * pdf["dvar"][1] <= 1742)
            mdl.maximize(obj)

    def test_pandas_sum_series(self):
        mdl = self.model
        xl = mdl.continuous_var_list(3, name='x', key_format="%s")
        xs = pd.Series(xl)  # a pandas series of variables...
        e = mdl.sum(xs)
        self.assertEqual("x0+x1+x2", str(e))

    def test_pandas_sum_series_nonstd_index(self):
        mdl = self.model
        dd = {1: 1, 10: 10, 100: 100}
        ss = pd.Series(dd, index=[1, 10, 100])
        r = mdl.sum(ss)
        # r is an int64
        x = mdl.integer_var(name='ij')
        mdl.add(r * x <= 77)
        self.assertIsNotNone(mdl.solve())
        lps = mdl.lp_string
        self.assertIn('111 ij <= 77', lps)
        self.assertEqual("111", str(r))  # the result is an expr.

    def test_pandas_sum_groupby_series(self):
        with Model(name="mining") as mdl:
            nb_mines = 3
            nb_years = 2
            work_vars = mdl.binary_var_matrix(keys1=nb_mines, keys2=nb_years, name='work')
            df = pd.DataFrame({'work': work_vars})
            df.index.names = ['range_mines', 'range_years']
            # sum of worked mines is less than 2
            df.work.groupby(level='range_years').agg(lambda works: mdl.add(sum(works) <= 2))
            self.assertEqual(nb_years, mdl.number_of_constraints)
            cts = [ct for ct in mdl.iter_constraints()]
            self.assertEqual("work_0_0+work_1_0+work_2_0 <= 2", str(cts[0]))


    def test_pandas_sum_join_series(self):
        with Model("mining", keep_ordering=True) as mdl:
            nb_mines = 2
            nb_years = 3

            range_years = range(nb_years)
            #
            work_vars = mdl.binary_var_matrix(keys1=nb_mines, keys2=nb_years, name='work')
            df = pd.DataFrame({'work': work_vars})
            df.index.names = ['range_mines', 'range_years']
            #
            s_discounts = pd.Series((y+1 for y in range_years), index=range_years, name='discounts')
            s_discounts.index.name = 'range_years'
            #
            df_join = df.join(s_discounts)
            e = mdl.sum(df_join.work * df_join.discounts)
            self.assertEqual("work_0_0+2work_0_1+3work_0_2+work_1_0+2work_1_1+3work_1_2", str(e))


    @skipUnless(pd and pd.__version__ >= '0.18.0', 'requires pandas 0.18 or higher')
    def test_pandas_range_as_keys(self):
        keys = pd.RangeIndex(start=0, stop=3, step=1)
        # should not raise any error
        xs = self.model.continuous_var_dict(keys=keys, name='pdr')

    def test_solution_values_from_pandas(self):
        m = self.model
        size = 17
        xl = self.model.continuous_var_list(size, name='pdr')
        pds = pd.Series(xl)
        sol = m.new_solution(var_value_dict={xl[i]: i for i in range(size)})
        svals = sol.get_values(pds)
        self.assertEqual(size,len(svals))
        for i in range(size):
            self.assertEqual(i, svals[i])


    def test_quad_matrix_sum_dataframe(self):
        with AdvModel() as am:
            labels = ['ga', 'bu']
            xl = self.model.continuous_var_list(labels, name=str)
            xs = pd.Series(xl)
            # coefs as a square matrix 3-3
            mat = {"ga": [1,2], "bu": [3,4]}

            dfv = pd.DataFrame(mat, index=labels, columns=labels)
            q = am._aggregator.quad_matrix_sum(dfv, xs, symmetric=False)
            self.assertEqual('ga^2+5ga*bu+4bu^2', str(q))





if __name__ == "__main__":
    if np:
        main()
