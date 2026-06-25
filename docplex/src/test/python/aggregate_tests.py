# pylint: disable=W0614
# pylint: disable=W0212
# pylint: disable=too-many-lines


import unittest
from functools import reduce

import six

from docplex.mp.basic import Expr
from docplex.mp.constr import AbstractConstraint
from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model
from docplex.mp.operand import LinearOperand
from docplex.mp.sumfn import docplex_sum
from docplex.mp.utils import is_iterable, DOcplexException
from testutils import DocplexAbstractTest

import os
docplex_debug = os.environ.get("DOCPLEX_DEBUG")

class DOcplexScalprodTests(DocplexAbstractTest):
    def test_scal_prod_with_zeros(self):
        m = self.model
        SIZE = 100
        (LB, UB) = (1, 3)
        allnames = ['foo%d' % i for i in range(1, SIZE + 1)]
        varlist = m.continuous_var_list(keys=allnames, lb=LB, ub=UB)
        # build a list of integers from 1 to 50 interleaved with zeroes:
        # e.g.e [1, 0, 2, 0, 3, 0 ...
        coefs = reduce(lambda l, k: l + [k, 0], range(1, 1 + int(SIZE / 2)), [])
        dot_expr = m.dot(varlist, coefs)
        self.assertTrue(dot_expr.is_normalized())
        self.assertEqual(dot_expr.number_of_variables(), SIZE / 2)

    def test_scal_prod_null_null(self):
        m = self.model
        sp = m.scal_prod([], [])
        self.assertTrue(sp.is_zero())

    def test_scal_prod_var_null(self):
        m = self.model
        x = m.continuous_var(name='x')
        sp = m.scal_prod([x], [])
        self.assertTrue(sp.is_zero())

    def test_scal_prod_null_coefs(self):
        m = self.model
        sp = m.scal_prod([], [1, 2, 3])
        self.assertTrue(sp.is_zero())

    def test_scal_prod_one_var(self):
        m = self.model
        x = m.continuous_var(name='x')
        scal_prod = m.scal_prod([x], 3)
        exp_prod = 3 * x
        self.assertTrue(exp_prod.equals(scal_prod))

    def test_scal_prod_with_one_cst(self):
        m = self.model
        allvars = m.continuous_var_list('abcdefghij')
        cst = 7
        scal_prod_cst = m.scal_prod(allvars, cst)
        self.assertIsInstance(scal_prod_cst, LinearExpr, 'not an expr')
        for (_, k) in scal_prod_cst.iter_terms():
            self.assertEqual(cst, k)

    def test_scal_prod_all_iter_variables(self):
        m = self.model
        allvars = m.continuous_var_list('abc')
        iota = m.dot(m.iter_variables(), range(1, len(allvars)+1))
        self.assertEqual("a+2b+3c", str(iota))

    def test_scal_prod_coef_starvation(self):
        m = self.model
        allnames = 'abcdefghij'
        xvars = m.continuous_var_dict(allnames, lb=0)
        coefs = [1, 2, 3]
        sorted_vars = [xvars[n] for n in allnames]
        scal_prod_cst = m.scal_prod(sorted_vars, coefs)
        # print(scal_prod_cst)
        self.assertIsInstance(scal_prod_cst, LinearExpr, 'not an expr')
        self.assertEqual(len(coefs), scal_prod_cst.number_of_variables())

    def test_scal_prod_no_vars(self):
        coefs = [3, 5, 7]
        m = self.model
        scal_prod_no_vars = m.scal_prod([], coefs)
        self.assertTrue(scal_prod_no_vars.is_zero())

    def test_scal_prod_var_starvation_coef_list(self):
        coefs = [3, 5, 7]
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        sp = m.scal_prod([x, y], coefs)
        self.assertTrue(sp.equals(3 * x + 5 * y))

    def test_scal_prod_var_starvation_coef_comp(self):
        coefs = (2 * i + 1 for i in range(1, 7))
        m = self.model
        x = m.continuous_var(name='x')
        scal_prod = m.scal_prod([x], coefs)
        self.assertTrue(x in scal_prod)
        self.assertEqual(3, scal_prod[x])
        self.assertEqual(0, scal_prod.constant)

    def test_scalprod_coef_generator(self):
        m = self.model

        def gen_odds():
            i = 1
            while True:
                yield i
                i += 2

        allvars = m.continuous_var_list(range(1, 4), lb=0, name='x')
        sum_exp = m.sum(allvars)
        self.assertTrue(sum_exp.equals(allvars[0] + allvars[1] + allvars[2]))

        sp = m.scal_prod(allvars, gen_odds())
        sps = sp.to_string()
        self.assertEqual(sps, 'x_1+3x_2+5x_3')  # white space is not pretty
        self.assertIsInstance(sp, LinearExpr)
        for v in allvars:
            self.assertTrue(v in sp)

    def test_scal_prod_no_coefs(self):
        m = self.model
        allnames = 'abcdefghij'
        coefs = [0]
        var_seq = m.binary_var_list(allnames)
        scal_prod_seq = m.scal_prod(var_seq, coefs)
        self.assertTrue(scal_prod_seq.is_zero())
        vars_gen = (var_seq[n] for n in range(len(allnames)))
        scal_prod_gen = m.scal_prod(vars_gen, coefs)
        self.assertTrue(scal_prod_gen.is_zero())

    def test_scal_prod_num_coef(self):
        m = self.model
        ze_coef = 5
        allvars = m.continuous_var_list(3, ub=1000, name='xz')
        scal_prod5 = m.scal_prod(allvars, ze_coef)
        self.assertEqual(3, scal_prod5.number_of_variables())
        for v in allvars:
            self.assertTrue(v in scal_prod5)
            self.assertEqual(ze_coef, scal_prod5[v])

    def test_scal_prod_zero_coeff(self):
        m = self.model
        allvars = m.continuous_var_list(3, ub=1000, name='xz')
        zdot = m.dot(allvars, 0.0)
        self.assertEqual("0", str(zdot))

    def test_scal_prod_vars_repeated(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        z = m.continuous_var(name='z')
        spr = m.scal_prod([x, y, z, y], 2)
        self.assertEqual("2x+4y+2z", str(spr))

    def test_scal_prod_collapse(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        z = m.continuous_var(name='z')
        spz = m.scal_prod([x, y, z, x, y, z], [1, 2, 3, -1, -2, -3])
        self.assertEqual("0", str(spz))
        # this is the only assertion checking terms are normalized (no zero coeffs)
        # perf impact so far limited to dot, seldom used anyway
        # self.assertTrue(spz.is_zero())

    def test_scal_prod_gen_vs_seq(self):
        m = self.model
        NBVARS = 3

        allnames = ['x%d' % v for v in range(1, NBVARS + 1)]
        var_seq = m.continuous_var_list(allnames)
        var_gen = (var_seq[v] for v in range(len(var_seq)))
        coefs = [(1 + k) * (1 + k) for k in range(len(var_seq))]
        # 1. scal prod with seq
        scal_prod_seq = m.scal_prod(var_seq, coefs)
        scal_prod_gen = m.scal_prod(var_gen, coefs)
        self.assertTrue(scal_prod_seq.equals(scal_prod_gen))
        rank = 1
        for var in var_seq:
            coef1 = scal_prod_gen[var]
            coef2 = scal_prod_seq[var]
            expectedCoef = rank * rank
            self.assertEqual(expectedCoef, coef1)
            self.assertEqual(expectedCoef, coef2)
            rank += 1

    def test_scal_prod_full(self):
        # mix variables, monomials, linexprs, numbers
        m = self.model
        m.float_precision = 1
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        z = m.continuous_var(name='z')
        terms = [x, 3 * x, x + y + 1, z + x + y + 3, z]
        koefs = [1, 3, 5, 7, 0]
        dot = m.dot(terms, koefs)
        # expected x coef is 1 + 9 +5 + 7 = 22
        # expected y coef is 5 + 7 = 12
        # expected z coef is 7
        # expected constant (from linear expression terms) is 5*1 + 7*3 = 26
        self.assertEqual("22x+12y+7z+26", str(dot))


    def test_scal_prodf_squares(self):
        m = self.model
        xs = m.continuous_var_dict(range(1,4), name='x')
        sqs = m.scal_prod_f(xs, lambda k: k*k)
        self.assertEqual('x_1+4x_2+9x_3', str(sqs))

    def test_scal_prodf_dict_empty(self):
        m = self.model
        sqs = m.scal_prod_f({}, lambda k: k*k)
        self.assertEqual('0', str(sqs))

    def test_scal_prodf_dict_not_alldiff(self):
        m = self.model
        xs = m.continuous_var_dict(range(1,4), name=['x', 'y', 'z'])
        xs[5] = xs[1]
        sqs = m.scal_prod_f(xs, lambda k: k*k, assume_alldifferent=False)
        self.assertEqual('26x+4y+9z', str(sqs))

    def test_scal_prodf_matrix(self):
        m = self.model
        bm = m.binary_var_matrix(keys1=range(1,3), keys2=range(1,4), name='b')
        sqs = m.scal_prod_f(bm, lambda kk: kk[0] + 2*kk[1])
        self.assertEqual('3b_1_1+5b_1_2+7b_1_3+4b_2_1+6b_2_2+8b_2_3', str(sqs))

    def test_scal_prodf_set(self):
        m = self.model
        xl = m.continuous_var_list(3, name=['x', 'y', 'z'])
        xs = set(xl)
        six.assertRaisesRegex(self, DOcplexException,
                              "Model.dotf expects either a dictionary or an ordered sequence of variables",
                              lambda m_: m_.dotf(xs, lambda _: 3.14),
                              m)

    def test_scal_prodf_list_empty(self):
        m = self.model
        zz = m.dotf([], lambda i: (i+1)**2)
        self.assertEqual('0', str(zz))

    def test_scal_prodf_list_squares(self):
        m = self.model
        xl = m.continuous_var_list(3, name=['x', 'y', 'z'])
        sqs = m.dotf(xl, lambda i: (i+1)**2)
        self.assertEqual('x+4y+9z', str(sqs))

    def test_scal_prodf_list_squares_checker_off(self):
        with Model(checker='off') as m1:
            xl = m1.continuous_var_list(3, name=['x', 'y', 'z'])
            sqs = m1.dotf(xl, lambda i: (i+1)**2)
            self.assertEqual('x+4y+9z', str(sqs))

    def test_scal_prodf_list_squares_not_alldiff(self):
        m = self.model
        [x,y,z] = m.continuous_var_list(3, name=['x', 'y', 'z'])
        xl2 = [x,y,z,x]
        sqs = m.dotf(xl2, lambda i: (i+1)**2, assume_alldifferent=False)
        self.assertEqual('17x+4y+9z', str(sqs))

    def test_scal_prodf_list_squares_not_alldiff_checker_off(self):
        with Model(checker='off') as m1:
            [x,y,z] = m1.continuous_var_list(3, name=['x', 'y', 'z'])
            xl2 = [x,y,z,x]
            sqs = m1.dotf(xl2, lambda i: (i+1)**2, assume_alldifferent=False)
            self.assertEqual('17x+4y+9z', str(sqs))


class DOcplexAggregateSumTests(DocplexAbstractTest):
    def test_builtin_sum_vars(self):
        m = self.model
        (LB, UB) = (1, 3)
        allnames = ['x1', 'x2', 'x3']
        allvars = m.continuous_var_list(allnames, LB, UB)
        py_sum = sum(allvars)
        self.assertIsInstance(py_sum, LinearExpr, "sum([vars] should be an expr")
        m_sum = m.sum(allvars)
        self.assertTrue(m_sum.equals(py_sum))
        self.assertEqual("x1+x2+x3", str(py_sum))

    def test_sum_empty(self):
        # sum of an empty sequence is zero
        m = self.model
        expr = m.sum([])
        self.assertEqual('0', str(expr))
        self.assertTrue(expr.is_zero())

    def test_sum_num(self):
        # sum with a plainnumber returns the number
        num = 7
        sum7 = self.model.sum(num)
        self.assertEqual(num, sum7)

    def test_sum_one_var(self):
        m = self.model
        x1 = m.continuous_var(name='x1')
        sum_x1 = m.sum(x1)
        self.assertIsInstance(sum_x1, LinearOperand)
        self.assertEqual("x1", str(sum_x1))

    def test_sum_one_mn(self):
        m = self.model
        x1 = m.continuous_var(name='x1')
        mn1 = 7 * x1
        sum_x1 = m.sum(mn1)
        self.assertIsInstance(sum_x1, LinearOperand)
        self.assertEqual("7x1", str(sum_x1))


    def test_sum_one_linexpr(self):
        m = self.model
        xs = m.continuous_var_list(3, name='x')
        sum_xs = m.sum(xs)
        self.assertIsInstance(sum_xs, LinearExpr)
        # a linear expr is not an iterable, otherwise sum would loop forever
        self.assertFalse(is_iterable(sum_xs))
        sum_sum = m.sum(sum_xs)
        # as a consequence sum(sum(..)) is involutive
        self.assertIs(sum_sum, sum_xs)

    def test_sum_vars(self):
        m = self.model
        size = 36
        allvars = m.continuous_var_list(size, lb=3, ub=11)
        sum_list = m.sum(allvars)
        self.assertIsInstance(sum_list, LinearExpr)
        self.assertEqual(0, sum_list.constant)
        self.assertEqual(sum_list.number_of_variables(), size)
        for var2 in allvars:
            self.assertTrue(var2 in sum_list)

        # using sum with a comprehension
        sum_odds = m.sum(v for i, v in enumerate(allvars) if i % 2 == 1)

        self.assertIsInstance(sum_odds, LinearExpr)
        self.assertEqual(sum_odds.constant, 0)
        # in [0, 1, 2] only 1 is odd
        self.assertEqual(sum_odds.number_of_variables(), int(size / 2))
        iter_allvars = iter(allvars)
        next(iter_allvars)
        var_1 = next(iter_allvars)  # ugly way to get allvars[1]
        self.assertTrue(sum_odds.contains_var(var_1))
        self.assertEqual(1, sum_odds[var_1])

    def test_sum_vars_with_repeated(self):
        m = self.model
        allnames = ['x', 'y', 'z']
        var_list = m.continuous_var_list(allnames)
        term_list = var_list
        term_list.extend(var_list)
        sum2 = m.sum(term_list)
        self.assertEqual("2x+2y+2z", str(sum2))

    def test_sum_vars_exprs_with_repeated(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        z = m.continuous_var(name='z')
        terms = [x, y, z, x, 2 * x + 1, 3 * y + 2, 4 * z + 3, 7 * x + 4, 6 * y + 5, 5 * z + 6]

        sum_terms = m.sum(terms)
        self.assertEqual("11x+10y+10z+21", str(sum_terms))

    def test_sum_var_dict(self):
        m = self.model
        allnames = ['x', 'y', 'z']
        var_dict = m.continuous_var_dict(allnames)
        sum_xyz = m.sum(var_dict)
        xyz = var_dict['x'] + var_dict['y'] + var_dict['z']
        # beware of ordering: dict does not guarantee key ordering
        self.assertTrue(xyz.equals(sum_xyz))

    def test_sum_vars_exprs_with_repeated_collapse(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        z = m.continuous_var(name='z')
        terms = [x, y, z, -x, 2 * x + 1, 3 * y + 2, 4 * z + 3, 7 * x + 4, -4 * y + 5, 5 * z + 6]

        sum_terms = m.sum(terms)
        # self.assertEqual("9x+10z+21", str(sum_terms))
        self.assertEqual(sum_terms[x], 9)
        self.assertEqual(sum_terms[z], 10)
        self.assertEqual(sum_terms[y], 0)
        self.assertEqual(21, sum_terms.constant)

    def test_sum_vars_full_collapse(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        args = [x, -y, -x, y]
        e = m.sum(args)
        self.assertEqual("0", str(e))

    def test_sum_vars_repeated_ordering(self):
        with Model(keep_ordering=True) as m:
            x = m.continuous_var(name='x')
            y = m.continuous_var(name='y')
            z = m.continuous_var(name='z')
            args = [z, x, y, y, ]
            e = m.sum(args)
            self.assertEqual('z+x+2y', str(e))

    def test_sum_vars_repeated_not_ordering(self):
        with Model(keep_ordering=False) as unordered_m:
            x = unordered_m.continuous_var(name='x')
            y = unordered_m.continuous_var(name='y')
            z = unordered_m.continuous_var(name='z')
            e = unordered_m.sum([z, x, y, y])
            self.assertEqual('x+2y+z', str(e))

    def test_sum_exprs_not_vars(self):
        m = self.model
        (LB, UB) = (1, 3)
        allnames = ['x1', 'x2', 'x3']
        vardict = m.continuous_var_dict(allnames, LB, UB)
        allvars = [vardict[n] for n in allnames]
        exprs = [allvars[i] + allvars[(i + 1) % 3] for i in range(3)]

        sumOfExprs = m.sum(exprs)
        # print(sumOfExprs)
        self.assertTrue(isinstance(sumOfExprs, LinearExpr))
        self.assertEqual(0, sumOfExprs.constant)
        for v in allvars:
            self.assertEqual(2, sumOfExprs.get_coef(v))

    def test_sum_non_iterable_num_nonzero(self):
        m = self.model
        sum_3 = m.sum(3)
        self.assertEqual(3, sum_3)

    def test_sum_filter_empty(self):
        m = self.model
        keys = range(1, 3)
        xs = m.integer_var_dict(3, ub=999, name='x')
        emptysum = m.sum(xs[k] for k in keys if k > 10)
        self.assertIsInstance(emptysum, Expr)
        # the comprehension yields nothing -> sum is 0
        ct = m.add_constraint(emptysum <= 1)
        self.assertEqual("0 <= 1", str(ct))
        self.assertTrue(not m._can_solve() or m.solve())

    def test_sum_mixed(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        args = [1, x, 3 * y, -7 * x + 5 * y + 42, 3.14]
        mix = m.sum(args)
        # expecting -6x+8y+46.14
        self.assertEqual("-6x+8y+46.140", str(mix))

    def test_sum_mixed_collapse(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        args = [1, -x, -y, 2 * x + 3 * y - 3, -x, -2 * y, 2]
        mix = m.sum(args)
        # all terms collapse
        # self.assertTrue(mix.is_normalized())
        self.assertEqual("0", str(mix))

    # ----- aggregate tests

    def test_sum_fn_empty(self):
        empty_sum = docplex_sum([])
        self.assertEqual(0, empty_sum)

    def test_sum_fn_vars(self):
        mdl = self.model
        allnames = ['x', 'y', 'z']
        allvars = mdl.continuous_var_list(allnames)
        var_sum = docplex_sum(allvars)
        self.assertEqual(3, var_sum.number_of_variables())
        self.assertEqual("x+y+z", str(var_sum))

    def test_sum_fn_exprs(self):
        mdl = self.model
        allnames = ['x', 'y', 'z']
        allvars = mdl.continuous_var_list(allnames, lb=0, name=str)
        exprs = [v + 1 for v in allvars]
        expr_sum = docplex_sum(exprs)
        self.assertEqual(3, expr_sum.number_of_variables())
        self.assertEqual("x+y+z+3", str(expr_sum))

    def test_sum_fn_var_expr_mixed(self):
        mdl = self.model
        allnames = ['x', 'y', 'z']
        allvars = mdl.continuous_var_list(allnames, lb=0, name=str)
        terms = []
        for v in allvars:
            terms.append(v)
            terms.append(2 * v + 3)
        expr_sum = docplex_sum(terms)
        self.assertEqual(3, expr_sum.number_of_variables())

    def test_sum_fn_dict(self):
        mdl = self.model
        allnames = ['x', 'y', 'z']
        allvars = mdl.continuous_var_dict(allnames, lb=0, name=str)
        expr_sum = docplex_sum(allvars)
        self.assertEqual(3, expr_sum.number_of_variables())
        self.assertEqual("x+y+z", str(expr_sum))

    @unittest.skipUnless(six.PY2, "Use of Linear Expr in a set requires to implement __hash__ method")
    def test_sum_fn_set(self):
        mdl = self.model
        allnames = ['x', 'y', 'z']
        allvars = mdl.continuous_var_dict(allnames, lb=0, name=str)
        expr_sum = docplex_sum({v + 1 for v in allvars.values()})
        self.assertEqual(3, expr_sum.number_of_variables())
        self.assertEqual("x+y+z+3", str(expr_sum))

    def test_sum_fn_one_number(self):
        self.assertEqual(1, docplex_sum(1))

    def test_sum_fn_None(self):
        self.assertEqual(0, docplex_sum(None))

    def test_sum_fn_mix_models(self):
        mdl1 = self.model
        mdl2 = Model(solver_agent=self.GlobalEngineCode)
        x1 = mdl1.continuous_var(lb=0, name="x1")
        x2 = mdl2.continuous_var(lb=0, name="x2")
        six.assertRaisesRegex(self, DOcplexException,
                                "Cannot mix objects belonging to different models",
                                lambda x1, x2: docplex_sum([x1, x2]),
                                x1, x2)

    def test_sum_fn_list_numbers(self):
        l = [i for i in range(1, 6)]
        # expecting 1+2+3+4+5 = 15
        self.assertEqual(15, docplex_sum(l))

    def test_sum_fn_var_expr_num(self):
        mdl = self.model
        x = mdl.continuous_var(name='x')
        terms = [13, x, 3 * x + 7]
        self.assertEqual("4x+20", str(mdl.sum(terms)))
        self.assertEqual("4x+20", str(docplex_sum(terms)))

    def test_null_sum_novars_constraint_feasible(self):
        m = self.model
        trivial_feasible_ct = m.add_constraint(m.sum([]) <= 2, 'feasible')
        self.assertIsInstance(trivial_feasible_ct, AbstractConstraint)
        self.assertTrue(trivial_feasible_ct.is_added())
        self.assertTrue(not m._can_solve() or m.solve())

    def test_null_sum_novars_constraint_infeasible(self):
        m = self.model
        trivial_infeasible_ct = m.add_constraint(m.sum([]) == 2, 'infeasible')
        self.assertIsInstance(trivial_infeasible_ct, AbstractConstraint)
        self.assertTrue(trivial_infeasible_ct.is_added())
        # either engine cannot solve or it fails.
        self.assertFalse(m._can_solve() and m.solve())

    def test_null_sum_novars_constraints_infeasible(self):
        m = self.model
        x = m.continuous_var(name='x')
        trivial_infeasible_ct, ctx5 = m.add([m.sum([]) == 2, (x <= 5)], ['zero_eq_2', 'ctx5'])
        self.assertIsInstance(trivial_infeasible_ct, AbstractConstraint)
        self.assertTrue(trivial_infeasible_ct.is_added())
        self.assertEqual(2, m.number_of_constraints)
        # either engine cannot solve or it fails.
        self.assertFalse(m._can_solve() and m.solve())

class DocplexAggregateSumSqTests(DocplexAbstractTest):
    def test_sumsq_empty(self):
        e = self.model.sumsq([])
        self.assertEqual(str(e), '0')

    def test_sumsq_num_n(self):
        e = self.model.sumsq(-7)
        self.assertEqual(e, 49)

class DocplexAggregateSumVarTests(DocplexAbstractTest):

    def test_sumvars_comp_checker_on(self):
        with Model(checker='on') as m:
            # should not crash: see RTC-37337
            xyz_vars = m.continuous_var_list(keys=['x', 'y', 'z'])
            e = m.sum_vars((z for z in xyz_vars))
            self.assertEqual('x+y+z', str(e))

    def test_sumvars_comp_checker_off(self):
        with Model(checker='off') as m:
            # should not crash: see RTC-37337
            xyz_vars = m.continuous_var_list(keys=['x', 'y', 'z'])
            e = m.sum_vars((z for z in xyz_vars))
            self.assertEqual('x+y+z', str(e))

    def test_sumvars_empty_list(self):
        m = self.model
        e = m.sum_vars([])
        self.assertEqual('0', str(e))

    def test_sumvars_list(self):
        m = self.model
        # should not crash: see RTC-37337
        xyz_vars = m.continuous_var_list(keys=['x', 'y', 'z'])
        e = m.sum_vars(xyz_vars)
        self.assertEqual('x+y+z', str(e))

    def test_sumvars_dict(self):
        m = self.model
        # should not crash: see RTC-37337
        xs = m.continuous_var_dict(keys=6, name='x', key_format='%s')
        e = m.sum_vars(xs)
        self.assertEqual('x0+x1+x2+x3+x4+x5', str(e))

    def test_sumvars_non_iterable(self):
        six.assertRaisesRegex(self, DOcplexException,
                              r"Model.sumvars\(\) expects an iterable returning variables",
                              lambda m_: m_.sum_vars(3.14), self.model)

    def test_sumvars_bad_list(self):
        m = self.model
        xs = m.continuous_var_list(3, name='x')
        l = xs[:]
        l.append(3.14)
        six.assertRaisesRegex(self, DOcplexException,
                              r"Model.sumvars\(\): Expecting an iterable returning variables",
                              lambda m_: m_.sum_vars(l), self.model)


class SumVarsTests(DocplexAbstractTest):
    def test_sumvars_empty(self):
        q = self.model.sum_vars_all_different([])
        self.assertTrue(q.is_zero())

    def test_sumvars_one(self):
        x = self.model.continuous_var(name='xx')
        e = self.model.sum_vars_all_different([x])
        self.assertEqual('xx', str(e))

    def test_sumvars_two(self):
        x = self.model.continuous_var(name='xx')
        y = self.model.continuous_var(name='yy')
        e = self.model.sum_vars_all_different([x, y])
        self.assertEqual('xx+yy', str(e))

    def test_sumvars_all_different_last_one_wins(self):
        # disable check to see what happens actually
        with Model(checker='off') as m:
            x = m.continuous_var(name='xx')
            y = m.continuous_var(name='yy')
            e = m.sum_vars_all_different([x, y, x, x])
            self.assertEqual('xx+yy', str(e))

    def test_sumvars_all_different_dict(self):
        # disable check to see what happens actually
        with Model(checker='off', keep_ordering=True) as m:
            bs = m.binary_var_dict(keys=['a', 'b','c'], name=str)
            e = m.sum_vars_all_different(bs)
            self.assertEqual('a+b+c', str(e))

    def test_sumvars_vs_sumvarsalldiff(self):
        with Model(keep_ordering=True) as om:
            x = om.continuous_var_list(keys=20, name='x')
            e1 = om.sum_vars(x)
            e2 = om.sum_vars_all_different(x)
            self.assertTrue(e1.equals(e2))
            self.assertTrue(e2.equals(e1))

    def test_sumvar_all_different_duplicate(self):
        with Model(checker='default') as m:
            marker_name = 'xx'
            x = m.continuous_var(name=marker_name)
            y = m.continuous_var(name='yy')
            six.assertRaisesRegex(self, DOcplexException, '%s appears twice' % marker_name,
                                  lambda md: md.sum_vars_all_different([x, y, x]), m)

    def test_sumvars_all_different_not_vars(self):
        with Model(checker='default') as m:
            x = m.continuous_var(name='x')
            a = 5
            six.assertRaisesRegex(self, DOcplexException, 'Expecting decision variable, got: 5',
                                  lambda md: md.sum_vars_all_different([x, a]), m)

    @unittest.skipUnless(docplex_debug, 'this kind of time testing is not reliable, especially in jenkins')
    def test_sumvars_faster_sumvars_alldiff(self):
        size = 600000
        m = self.model
        m.set_checker('off')
        x = self.model.continuous_var_list(keys=size, name='x')
        from private.timing import MyTimer
        with MyTimer('sum') as t0:
            self.model.sum(x)
        with MyTimer('sum_var') as t1:
            self.model.sum_vars(x)
        with MyTimer('sum_var_alldifferent') as t2:
            self.model.sum_vars_all_different(x)
        self.assertLess(1.2 * t2.secs, t1.secs,
                        msg='sum_vars_all_different() should be faster: \n'
                            'sum_vars_all_different(50000 vars)/sum_vars(50000 vars) should be greater than 1.5. \n'
                            'Got: {}'.format(t1.secs / t2.secs))

    def test_sumvars_alldiff_comp(self):
        with Model(checker='default') as m:  # force checking
            xyz_vars = m.continuous_var_list(keys=['x', 'y', 'z'])
            e = m.sum_vars_all_different((z for z in xyz_vars))
            self.assertEqual('x+y+z', str(e))

    def test_sumvars_comp_checker_off(self):
        with Model(checker='off') as m:
            # should not crash: see RTC-37337
            xyz_vars = m.continuous_var_list(keys=['x', 'y', 'z'])
            e = m.sum_vars((z for z in xyz_vars))
            self.assertEqual('x+y+z', str(e))

    def test_sumvars_dict_checker_off(self):
        with Model(checker='off') as m:
            xyz_vars = m.continuous_var_dict(keys=['x', 'y', 'z'], name=str)
            e = m.sum_vars(xyz_vars)
            self.assertEqual('x+y+z', str(e))

if __name__ == "__main__":
    unittest.main()
