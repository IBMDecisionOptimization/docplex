import unittest
import six
import operator

from docplex.mp.model import Model
from docplex.mp.linear import Expr, LinearExpr
from docplex.mp.dvar import is_var
from docplex.mp.utils import is_iterable
from docplex.mp.error_handler import DOcplexException

from testutils import RedirectedOutputToStringContext
from collections import namedtuple


class DecisionVarTestsBase(unittest.TestCase):
    expected_default_lb = 0

    def setUp(self):
        self.model = Model()

    def tearDown(self):
        self.model.end()
        self.model = None


class DecisionVarTests(DecisionVarTestsBase):

    def test_model_create_one_binary_variable(self):
        oldNbVars = self.model.number_of_variables
        oldNbBinaryVars = self.model.number_of_binary_variables
        the_var_name = 'a complicated name'
        var = self.model.binary_var(the_var_name)
        newNbVars = self.model.number_of_variables
        newNbBinaryVars = self.model.number_of_binary_variables

        self.assertEqual(newNbVars, 1 + oldNbVars, "bad increment of #variables")
        self.assertEqual(newNbBinaryVars, 1 + oldNbBinaryVars, "bad increment #binaries")
        self.assertEqual(var.name, the_var_name)
        self.assertEqual(0, var.lb)
        self.assertEqual(1, var.ub)
        self.assertTrue(var.is_binary(), "bad variable type, expecting binary")

    def test_model_create_one_integer_variable(self):
        oldNbVars = self.model.number_of_variables
        oldNbTypedVars = self.model.number_of_integer_variables
        the_var_name = 'this is an integer variable'
        expectedLb = 127
        expectedUb = 348
        var = self.model.integer_var(expectedLb, expectedUb, the_var_name)
        newNbVars = self.model.number_of_variables
        newNbTypedVars = self.model.number_of_integer_variables

        self.assertEqual(newNbVars, 1 + oldNbVars, "bad increment of #variables")
        self.assertEqual(newNbTypedVars, 1 + oldNbTypedVars, "bad increment #binaries")
        self.assertEqual(var.name, the_var_name)
        self.assertEqual(expectedLb, var.lb)
        self.assertEqual(expectedUb, var.ub)
        self.assertTrue(var.is_integer(), "bad variable type, expecting integer")

    def test_create_contvar_default_anonymous(self):
        m = self.model
        cvar = self.model.continuous_var()
        self.assertEqual(0, cvar.index)
        # default LB is 0 !!!
        self.assertEqual(cvar.lb, self.expected_default_lb)
        self.assertEqual(cvar.ub, m.infinity)
        self.assertFalse(cvar.has_user_name())
        self.assertEqual('x1', str(cvar))

    def test_model_create_one_continuous_variable(self):
        oldNbVars = self.model.number_of_variables
        oldNbTypedVars = self.model.number_of_continuous_variables
        the_var_name = 'this is an continuous variable'
        expectedLb = 2.17
        expectedUb = 3.14
        var = self.model.continuous_var(expectedLb, expectedUb, the_var_name)
        newNbVars = self.model.number_of_variables
        newNbTypedVars = self.model.number_of_continuous_variables

        self.assertEqual(newNbVars, 1 + oldNbVars, "bad increment of #variables")
        self.assertEqual(newNbTypedVars, 1 + oldNbTypedVars, "bad increment #binaries")
        self.assertEqual(var.name, the_var_name)
        self.assertEqual(expectedLb, var.lb)
        self.assertEqual(expectedUb, var.ub)
        self.assertFalse(var.is_integer(), "bad variable type, expecting continuous")
        self.assertFalse(var.is_binary(), "bad variable type, expecting continuous")
        self.assertTrue(var.is_continuous(), "bad variable type, expecting continuous")

    def test_create_one_default_continuous_var(self):
        mdl = self.model
        INF = mdl.infinity
        x = mdl.continuous_var(name='x')
        self.assertEqual(0, x.lb)
        self.assertEqual(INF, x.ub)

    def test_var_free_continuous(self):
        mdl = self.model
        x = self.model.continuous_var(name='x', lb=-mdl.infinity)
        self.assertTrue(x.is_free())
        self.assertTrue(x.has_free_lb())
        self.assertTrue(x.has_free_ub())

    def test_var_free_ub_continuous_default(self):
        x = self.model.continuous_var(lb=0, name='x')
        self.assertFalse(x.is_free())
        self.assertFalse(x.has_free_lb())
        self.assertTrue(x.has_free_ub())

    def test_var_free_ub_continuous_custom_big(self):
        x = self.model.continuous_var(lb=0, name='x', ub=3e+30)
        self.assertFalse(x.is_free())
        self.assertFalse(x.has_free_lb())
        self.assertTrue(x.has_free_ub())

    def test_var_free_lb_continuous_default(self):
        mdl = self.model
        x = self.model.continuous_var(lb=-mdl.infinity, ub=1, name='x')
        self.assertFalse(x.is_free())
        self.assertTrue(x.has_free_lb())
        self.assertFalse(x.has_free_ub())

    def test_var_free_lb_continuous_custom_deep(self):
        mdl = self.model
        x = self.model.continuous_var(lb=-3e+30, ub=1, name='x')
        self.assertFalse(x.is_free())
        self.assertTrue(x.has_free_lb())
        self.assertFalse(x.has_free_ub())

    def test_var_non_free_continuous(self):
        x = self.model.continuous_var(lb=1, ub=3, name='x')
        self.assertFalse(x.is_free())
        self.assertFalse(x.has_free_lb())
        self.assertFalse(x.has_free_ub())

    def test_create_positive_var(self):
        m = self.model
        NAME = 'positive'
        var = m.continuous_var(lb=0, name=NAME)
        self.assertEqual(var.lb, 0)
        self.assertEqual(var.ub, m.infinity)

    def test_create_negative_var(self):
        m = self.model
        NAME = 'negative'
        var = m.continuous_var(lb=-m.infinity, ub=0, name=NAME)
        self.assertEqual(var.lb, -m.infinity)

    def test_model_access_var_by_name(self):
        m = self.model
        NAME = 'xxx'
        var = m.continuous_var(name=NAME)
        var_query = m.get_var_by_name(NAME)
        self.assertTrue(var is var_query)
        # an anonymous variable has an index anyway
        anonymous_var = m.continuous_var()
        self.assertTrue(anonymous_var.has_valid_index())

    def test_model_find_matching_vars(self):
        m = self.model
        m.continuous_var_list(keys=3, name='prefix_foo')
        m.continuous_var_list(keys=5, name='prefix_bar')
        self.assertEqual(3, len(m.find_matching_vars(pattern='foo')))
        self.assertEqual(5, len(m.find_matching_vars(pattern='bar')))
        self.assertEqual(8, len(m.find_matching_vars(pattern='prefix_')))

    def test_model_find_matching_vars_case(self):
        m = self.model
        m.continuous_var_list(keys=3, name='prefix_foo')
        m.continuous_var_list(keys=5, name='prefix_FOO')
        m.continuous_var_list(keys=7, name='prefix_BAR')
        self.assertEqual(3, len(m.find_matching_vars(pattern='foo', match_case=True)))
        self.assertEqual(5, len(m.find_matching_vars(pattern='FOO', match_case=True)))
        self.assertEqual(8, len(m.find_matching_vars(pattern='foo', match_case=False)))
        self.assertEqual(15, len(m.find_matching_vars(pattern='prefix_', match_case=False)))
        self.assertEqual(0, len(m.find_matching_vars(pattern='PREFIX', match_case=True)))

    def test_model_find_re_matching_vars(self):
        m = self.model
        xs = m.continuous_var_matrix(keys1=3, keys2=5, name=lambda ij: "foo_{{{0}}}_{{{1}}}".format(*ij))
        # red herring
        y = m.continuous_var(name="foo_bar")
        import re
        re_foo = re.compile("foo_\{[0-9]\}_\{[0-9]\}")
        all_matches_foo_i_j = m.find_re_matching_vars(re_foo)
        self.assertEqual(15, len(all_matches_foo_i_j))
        self.assertEqual(16, len(m.find_matching_vars("foo_")))

    def test_model_access_var_by_index(self):
        m = self.model
        x = m.continuous_var(name='x')
        y = m.continuous_var(name='y')
        id_x = x.index
        id_y = y.index
        self.assertIs(x, m.get_var_by_index(id_x))
        self.assertIs(y, m.get_var_by_index(id_y))
        # add another
        z = m.continuous_var(name='z')
        id_z = z.index
        self.assertIs(z, m.get_var_by_index(id_z))

    def _check_vardict(self, vardict, keySeq, expected_vartype, expectedLB, expectedUB):
        varseq = vardict.values()
        self._check_varlist(varseq, keySeq, expected_vartype, expectedLB, expectedUB)

    def _check_varlist(self, varseq, keySeq, expectedVarType, expectedLB, expectedUB):
        expectedLen = len(keySeq) if is_iterable(keySeq) else keySeq
        self.assertEqual(expectedLen, len(varseq))
        for var in varseq:
            self.assertEqual(expectedVarType, var.vartype)
            self.assertEqual(expectedLB, var.lb)
            self.assertEqual(expectedUB, var.ub)

    ##  empty variable lists
    def test_empty_binary_varlist(self):
        m = self.model
        self.assertEqual([], m.binary_var_list([]))

    def test_empty_integer_varlist(self):
        m = self.model
        self.assertEqual([], m.integer_var_list([]))

    def test_empty_continuous_varlist(self):
        m = self.model
        self.assertEqual([], m.continuous_var_list([]))

    # anonymous var lists
    def test_anonymous_binary_var_list(self):
        m = self.model
        bvars = m.binary_var_list(3)  # name is default str
        self.assertEqual(3, len(bvars))
        self.assertFalse(any(v.has_user_name() for v in bvars))

    def test_anonymous_integer_var_list(self):
        m = self.model
        bvars = m.integer_var_list(3)  # name is default str
        self.assertEqual(3, len(bvars))
        self.assertFalse(any(v.has_user_name() for v in bvars))

    def test_anonymous_continuous_var_list(self):
        m = self.model
        bvars = m.continuous_var_list(3)  # name is default str
        self.assertEqual(3, len(bvars))
        self.assertFalse(any(v.has_user_name() for v in bvars))

    # varlist w/ prefix
    def test_prefixed_binary_var_list(self):
        m = self.model
        bvar1 = m.binary_var_list(1, name="zorglub")
        self.assertEqual(1, len(bvar1))
        var = bvar1[0]
        self.assertEqual(m.binary_vartype, var.vartype)
        self.assertEqual("zorglub_0", var.name)

    def test_create_default_binary_vardict(self):
        m = self.model
        KEYS = range(1, 11)
        bvars = m.binary_var_dict(KEYS)
        self._check_vardict(bvars, KEYS, m.binary_vartype, 0, 1)

    def test_create_varlist_from_size(self):
        m = self.model
        SIZE = 11
        dvars = m.binary_var_list(SIZE, name='x', key_format='%s')
        self.assertEqual(SIZE, len(dvars))

    def test_create_varlist_string_key_format(self):
        m = self.model
        SIZE = 3
        dvars = m.binary_var_list(SIZE, name='zzz', key_format='#')
        self.assertEqual(dvars[0].name, 'zzz#0')

    def test_create_varlist_with_nameseq(self):
        m = self.model
        SIZE = 11
        keys = range(SIZE)
        names = ['zzz%d' % i for i in range(SIZE)]
        bvars = m.binary_var_dict(keys, name=names)
        self.assertEqual(SIZE, len(bvars))
        # self.assertEqual(bvars[SIZE].name, '')
        for i in range(SIZE):
            self.assertEqual(bvars[i].name, names[i])

    def test_varlist_with_plain_tuples(self):
        # plain tuples are stringified without ()
        m = self.model
        tkeys = [(1, "a"), (2, "b")]
        tvars = m.continuous_var_list(keys=tkeys, name="v")
        self.assertEqual([v.name for v in tvars], ["v_1_a", "v_2_b"])

    def test_varlist_with_namedtuples(self):
        # plain tuples are stringified without ()
        m = self.model
        tn = namedtuple("TFoo", ["f1", "f2"])
        tkeys = [tn(1, "a"), tn(2, "b")]
        tvars = m.continuous_var_list(keys=tkeys, name="v")
        self.assertEqual([v.name for v in tvars], ["v_1_a", "v_2_b"])

    def test_varlist_with_nametuples_str_override(self):
        # plain tuples are stringified without ()
        m = self.model

        class KT(namedtuple("TFoo", ["a", "b"])):
            def __str__(self):
                return "KT#%d#%s" % (self.a, self.b)

        tkeys = [KT(1, "a"), KT(2, "b")]
        tvars = m.continuous_var_list(keys=tkeys, name="v")
        self.assertEqual([v.name for v in tvars], ["v_KT#1#a", "v_KT#2#b"])

    def test_create_varlist_with_namelist_too_short(self):
        six.assertRaisesRegex(self, DOcplexException, 'An array of names should have same len as keys',
                              lambda m: m.integer_var_list(3, name=['foo']), self.model)

    def test_create_varlist_with_badtype_name(self):
        six.assertRaisesRegex(self, DOcplexException, 'Cannot use this for naming variables: 3.14',
                              lambda m: m.integer_var_list(3, name=3.14), self.model)

    def test_create_varlist_with_namelist_too_long(self):
        m = self.model
        bvars = m.binary_var_list(2, name=['foo', 'bar', 'gee'])
        self.assertEqual(2, len(bvars))
        self.assertEqual(bvars[0].name, 'foo')
        self.assertEqual(bvars[1].name, 'bar')

    @unittest.skipUnless(six.PY3, "test with dict.values on Python 3 only")
    def test_create_var_list_with_dict_values101(self):
        # assume Py3
        m = self.model
        size = 101
        keyd = {i : "foo_%d" %i for i in range(size)}
        lv = m.integer_var_list(keys=keyd.values(), name="x")
        self.assertEqual(size, len(lv))

    @unittest.skipUnless(six.PY3, "test with dict.values on Python 3 only")
    def test_create_var_list_with_dict_keys101(self):
        # assume Py3
        m = self.model
        size = 101
        keyd = {i : "foo_%d" %i for i in range(size)}
        lv = m.integer_var_list(keys=keyd.keys(), name="x")
        self.assertEqual(size, len(lv))

    @unittest.skipUnless(six.PY3, "test with dict.values on Python 3 only")
    def test_create_var_list_with_dict_values10(self):
        # assume Py3
        size = 10
        keyd = {i : "foo_%d" %i for i in range(size)}
        m = self.model
        lv = m.integer_var_list(keys=keyd.values(), name="x")
        self.assertEqual(size, len(lv))

    def test_create_vardict_with_keyset(self):
        m = self.model
        keys = {"foo", "bar", "gee"}
        bvars = m.binary_var_dict(keys, name=keys)
        self.assertEqual(3, len(bvars))

    def test_create_vardict_with_large_keyset(self):
        m = self.model
        keys = ({(i,j) for i in range(12) for j in range(13)})
        bvars = m.binary_var_dict(keys, name='y')
        self.assertEqual(12*13, len(bvars))
        bv00 = bvars[0,0]
        self.assertEqual('y_0_0', bv00.name)

    def test_create_vardict_with_duplicate_keys_nocheck(self):
        m = self.model
        keys = ["foo", "bar", "gee", "foo"]
        bvs = m.binary_var_dict(keys, name=keys)
        self.assertEqual(3, len(bvs))
        # first var "foo" gets kicked out of dict
        self.assertEqual(3, m.get_var_by_name("foo").index)
        self.assertEqual("foo", m.get_var_by_index(0).name)

    def test_create_vardict_with_duplicate_keys_fullcheck_ko(self):
        with Model(checker='full') as fm:
            keys = ["foo", "bar", "gee", "foo"]
            six.assertRaisesRegex(self, DOcplexException,
                                  "Duplicated key", lambda m_: m_.binary_var_dict(keys, name=keys), fm)

    def test_create_vardict_with_key_iter(self):
        keys = [1, 3, 5, 7]
        key_iter = iter(keys)
        bvars = self.model.binary_var_dict(keys=key_iter, name="odds")
        self.assertEqual(4, len(bvars))

    def test_create_vardict_with_key_gen(self):
        # a local generator generates odd integers
        def gen_odds(n):
            i = 1
            while i <= n:
                yield i
                i += 2

        keygen = gen_odds(10)
        # expecting keyset: [1,3,5,7,9] : len is 5
        bvars = self.model.binary_var_dict(keys=keygen, name="odds")
        self.assertEqual(5, len(bvars))

    def test_create_var_dict_from_keyset(self):
        m = self.model
        keys = {1, 2, 3, 4, 3}
        bvars = m.continuous_var_dict(keys, name=str)
        self.assertEqual(4, len(bvars))

    def test_create_default_integer_vars_dict(self):
        m = self.model
        INF = m.infinity
        KEYS = range(1, 11)
        default_integer_var_dict = m.integer_var_dict(KEYS)
        self._check_vardict(default_integer_var_dict, KEYS, expectedLB=0, expectedUB=INF,
                            expected_vartype=m.integer_vartype)

    def test_create_integer_vars1(self):
        m = self.model
        integer_vartype = m.integer_vartype
        INF = m.infinity
        self.assertEqual({}, m.integer_var_dict([], 0, 1))
        keys1 = "abcdef"
        keys2 = "ghijkl"
        keys3 = "mnopqr"
        vars1 = m.integer_var_dict(keys1, lb=3)
        self._check_vardict(vars1, keys1, integer_vartype, 3, INF)
        vars2 = m.integer_var_dict(keys2, ub=7)
        self._check_vardict(vars2, keys2, integer_vartype, self.EXPECTED_DEFAULT_LB, 7)
        vars3 = m.integer_var_dict(keys3, lb=5, ub=11)
        self._check_vardict(vars3, keys3, integer_vartype, 5, 11)

    def test_create_vars_custom_lbs_list(self):
        m = self.model
        size = 3
        custom_lbs = [1, 4, 9]
        varsByKeys = m.continuous_var_dict(size, lb=custom_lbs)
        for i in range(size):
            var = varsByKeys[i]
            self.assertEqual(var.lb, custom_lbs[i])

    def test_create_vars_custom_lbs_dict(self):
        m = self.model
        size = 3
        custom_lbs = {k: (k + 1) ** 3 for k in range(size)}
        varsByKeys = m.continuous_var_dict(size, lb=custom_lbs)
        for i in range(size):
            var = varsByKeys[i]
            self.assertEqual(var.lb, custom_lbs[i])

    def test_create_vars_custom_bounds_tuple(self):
        m = self.model
        size = 3
        custom_lbs = (1, 4, 9)
        custom_ubs = (1, 8, 27)
        vd = m.continuous_var_dict(size, lb=custom_lbs, ub=custom_ubs)
        for k, dv in six.iteritems(vd):
            i = k + 1
            self.assertEqual(dv.lb, i * i)
            self.assertEqual(dv.ub, i * i * i)

    def test_create_vars_custom_lbs_empty_list(self):
        self.assertRaises(DOcplexException, lambda mm: mm.continuous_var_list(3, lb=[]), self.model)

    def test_create_vars_custom_lbs_list_too_short(self):
        self.assertRaises(DOcplexException, lambda mm: mm.continuous_var_list(3, lb=[1]), self.model)

    def test_create_vars_custom_lbs_list_too_big(self):
        m = self.model
        size = 3
        custom_lbs = [1, 4, 9, 17]
        varsByKeys = m.continuous_var_dict(size, lb=custom_lbs)
        for i in range(size):
            var = varsByKeys[i]
            self.assertEqual(var.lb, custom_lbs[i])

    def test_create_vars_custom_lbs_comp(self):
        m = self.model
        size = 3
        custom_lbs = (i ** 3 for i in range(size))
        dvars = m.continuous_var_list(size, lb=custom_lbs)
        for i in range(size):
            var = dvars[i]
            self.assertEqual(var.lb, i ** 3)

    def test_create_vars_custom_lbs_comp_ko(self):
        m = self.model
        badlist = [1, 2, "foo"]
        size = len(badlist)
        custom_lbs = (x for x in badlist)
        six.assertRaisesRegex(self, DOcplexException, "Variable lb expect numbers",
                              lambda m_: m_.continuous_var_list(size, lb=custom_lbs), m)


    def test_create_vars_custom_lbs_functional(self):
        m = self.model
        keys = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
        # custom lb is a functin of the key (here a tuple) returning the second arg
        custom_lb_fn = lambda t: t[1]

        dvars = m.continuous_var_dict(keys, lb=custom_lb_fn)
        for i in range(len(keys)):
            key = keys[i]
            var = dvars[key]
            self.assertEqual(var.lb, custom_lb_fn(key))

    def test_create_vars_custom_ubs(self):
        m = self.model
        keys = 'abc'
        customUbs = [8, 81, 64]
        varsByKeys = m.continuous_var_dict(keys, ub=customUbs)
        for i in range(len(keys)):
            var = varsByKeys[keys[i]]
            self.assertEqual(var.ub, customUbs[i])

    def test_create_vars_custom_ubs_fn(self):
        m = self.model
        xs = m.continuous_var_list(keys=10, ub=lambda k: 1 if 3 <= k <= 7 else None)
        ubs = [x.ub for x in xs]
        for u, ub in enumerate(ubs):
            expected_ub = 1 if 3 <= u <= 7 else 1e+20
            self.assertEqual(ub, expected_ub)

    def test_create_vars_custom_ubs_array(self):
        m = self.model
        xs = m.continuous_var_list(keys=4, ub=[None, 1, 1, None])
        ubs = [x.ub for x in xs]
        for u, ub in enumerate(ubs):
            expected_ub = 1 if 1 <= u <= 2 else 1e+20
            self.assertEqual(ub, expected_ub)

    def test_create_vars_custom_ranges(self):
        m = self.model
        keys = 'abc'
        customLbs = [1, 2, 3]
        customUbs = [8, 81, 64]
        varsByKeys = m.continuous_var_dict(keys, lb=customLbs, ub=customUbs)
        for i in range(len(keys)):
            var = varsByKeys[keys[i]]
            self.assertEqual(var.lb, customLbs[i])
            self.assertEqual(var.ub, customUbs[i])

    def test_create_vars_functional_names(self):
        m = self.model
        name_fn = lambda k: "before_{}_after".format(k)
        dvars = m.integer_var_list(keys=3, name=name_fn)
        for i, v in enumerate(dvars):
            self.assertEqual(v.name, name_fn(i))

    def test_create_vars_functional_ranges(self):
        m = self.model
        SIZE = 7
        keys = range(1, SIZE + 1)
        customLbFn = lambda x: x
        customUbFn = lambda y: y ** 2
        varsByKeys = m.continuous_var_dict(keys, lb=customLbFn, ub=customUbFn)
        for k in keys:
            var = varsByKeys[k]
            self.assertEqual(var.lb, customLbFn(k))
            self.assertEqual(var.ub, customUbFn(k))

    def test_create_vars_name_custom_rule(self):
        m = self.model
        all_keys = [(n, 1) for n in ["foo", "bar", "gee"]]
        allvars = m.binary_var_dict(all_keys, name=lambda x: x[0])
        self.assertEqual(len(allvars), len(all_keys))
        for key in all_keys:
            var = allvars[key]
            self.assertEqual(var.name, key[0])

    def test_create_empty_continuous_vardict(self):
        m = self.model
        self.assertEqual({}, m.continuous_var_dict([]))

    # the expected fefault LB for all variables

    EXPECTED_DEFAULT_LB = 0

    def test_create_continuous_vardict(self):
        m = self.model
        continuous_vartype = m.continuous_vartype
        INF = m.infinity
        keys1 = "abcdef"
        keys2 = "ghijkl"
        keys3 = "mnopqr"
        keys4 = "stu"

        vars1 = m.continuous_var_dict(keys1)
        self._check_vardict(vars1, keys1, continuous_vartype, self.EXPECTED_DEFAULT_LB, INF)
        vars2 = m.continuous_var_dict(keys2, lb=2.5)
        self._check_vardict(vars2, keys2, continuous_vartype, 2.5, INF)
        vars3 = m.continuous_var_dict(keys3, ub=8.5)
        self._check_vardict(vars3, keys3, continuous_vartype, self.EXPECTED_DEFAULT_LB, 8.5)
        vars4 = m.continuous_var_dict(keys4, lb=4.5, ub=16.5)
        self._check_vardict(vars4, keys4, continuous_vartype, 4.5, 16.5)

    def test_create_typed_vardict(self):
        m = self.model
        keys = ['ga', 'bu', 'zo', 'meu']
        vd = m.var_dict(keys, m.integer_vartype, name=str)
        self.assertEqual(len(keys), len(vd))
        for k in keys:
            self.assertIn(k, vd)
            self.assertEqual(vd[k].name, k)

    def test_create_varlist_from_iter(self):
        m = self.model
        # this is an iterator
        SIZE = 100
        iter_keys = (i for i in range(1, SIZE + 1) if i % 2 == 1)
        allvars = m.integer_var_list(iter_keys, name='z')
        self.assertEqual(SIZE / 2, len(allvars))

    # square matrices of vars
    def _check_var_matrix(self, matrix, keys1, keys2, expected_size, expected_lb, expected_ub, namer):
        self.assertEqual(len(matrix), expected_size)
        for k1 in keys1:
            for k2 in keys2:
                self.assertTrue((k1, k2) in matrix)
                dv = matrix[k1, k2]
                xpected_name = namer(k1, k2)
                self.assertTrue(dv.has_valid_index())
                self.assertEqual(xpected_name, dv.name)
                self.assertEqual(expected_lb, dv.lb)
                self.assertEqual(expected_ub, dv.ub)

    def test_create_var_matrix_from_dimensions(self):
        m = self.model
        DIM1, DIM2 = 3, 4
        var_matrix = m.continuous_var_matrix(DIM1, DIM2, lb=3, ub=17, name='m')
        range1 = range(DIM1)
        range2 = range(DIM2)
        guessed_namer = lambda k1, k2: 'm_%d_%d' % (k1, k2)
        self._check_var_matrix(var_matrix, range1, range2, DIM1 * DIM2, 3, 17, guessed_namer)

    def test_create_typed_var_matrix(self):
        m = self.model
        DIM1, DIM2 = 3, 4
        integer_vartype = m.integer_vartype
        var_matrix = m.var_matrix(integer_vartype, DIM1, DIM2, lb=3, ub=17, name='m')
        range1 = range(DIM1)
        range2 = range(DIM2)
        guessed_namer = lambda k1, k2: 'm_%d_%d' % (k1, k2)
        self._check_var_matrix(var_matrix, range1, range2, DIM1 * DIM2, 3, 17, guessed_namer)

    def test_create_var_matrix(self):
        m = self.model
        DIM1, DIM2 = 3, 4
        INF = m.infinity
        keys1 = range(1, DIM1 + 1)
        keys2 = range(1, DIM2 + 1)
        var_matrix = m.continuous_var_matrix(keys1, keys2, lb=0, name='m2')
        guessed_namer = lambda k1, k2: 'm2_%s_%s' % (str(k1), str(k2))
        self._check_var_matrix(var_matrix, keys1, keys2, DIM1 * DIM2, 0, INF, guessed_namer)

    def test_var_cube_namer(self):
        m = self.model
        K1 = [1, 2]
        K2 = ["foo"]
        K3 = [1.1, 2.2]
        allvars = m.continuous_var_cube(K1, K2, K3, lb=0, name="cube")
        self.assertEqual(4, len(allvars))
        for k1 in K1:
            for k2 in K2:
                for k3 in K3:
                    pk = (k1, k2, k3)
                    self.assertTrue(pk in allvars)
                    var = allvars[pk]
                    self.assertTrue(var.is_continuous())
                    expected_name = "cube_%s_%s_%s" % (str(k1), str(k2), str(k3))
                    self.assertEqual(expected_name, var.name)

    def test_rename_var(self):
        with Model("rename_var") as m:
            xvar = m.continuous_var(name='x')
            m.maximize(xvar)
            NEW_NAME = "spirou_et_fantasio_panade_a_champignac"
            # using the proprty here
            xvar.name = NEW_NAME
            self.assertEqual(NEW_NAME, xvar.name)
            # lp_as_string = m.export_as_lp_string()
            # self.assertTrue(lp_as_string.find(NEW_NAME) > 0)

            path = m.dump_as_lp()
            with open(path) as iss:
                lp_file = iss.read()
            self.assertTrue(lp_file.find(NEW_NAME) > 0)

    def test_var_rename_warned(self):
        m = self.model
        xvar = m.continuous_var(name='x')
        xvar.name = " a name with blanks"
        self.assertEqual(1, m.number_of_warnings)

    def test_var_rename_none(self):
        m = self.model
        xvar = m.continuous_var(name='x')

        def anonymize(dv):
            dv.name = None

        six.assertRaisesRegex(self, DOcplexException, "expects a string",
                              lambda dv_: anonymize(dv_), xvar)

    def test_var_from_unnamed_to_named(self):
        m = self.model
        xvar = m.continuous_var()  # unnamed
        self.assertFalse(xvar.has_user_name())
        xvar.name = "foo"
        self.assertTrue(xvar.has_user_name())

    @unittest.skip('does not work')
    def test_var_from_named_to_unnamed(self):
        m = self.model
        xvar = m.continuous_var(name='foo')
        self.assertTrue(xvar.has_user_name())
        xvar.name = None
        self.assertFalse(xvar.has_user_name())

    def test_varname_collision(self):
        # declaring two vars with same name produces a wanring, not a fatal
        # accessing variable by name returns the last one
        # the first var can still be obtained by index.
        m = self.model
        varname = 'foo'
        cvar = m.continuous_var(name=varname)
        self.assertIs(cvar, m.get_var_by_name(varname))
        ivar = m.integer_var(name=varname)
        self.assertIs(ivar, m.get_var_by_name(varname))
        self.assertEqual(2, m.number_of_variables)

    def test_var_lp_names(self):
        # anonymous vars
        m = self.model
        b = m.binary_var()
        ij = m.integer_var()
        x = m.continuous_var()
        sx = m.semicontinuous_var(lb=1)
        sn = m.semiinteger_var(lb=1)
        self.assertEqual(['x1', 'x2', 'x3', 'x4', 'x5'], list(map(operator.attrgetter('lp_name'), m.iter_variables())))
        self.assertEqual(['b1', 'i2', 'x3', 's4', 'n5'], list(map(operator.attrgetter('lpt_name'), m.iter_variables())))


class DecisionVarString(DecisionVarTestsBase):
    def test_var_str_binary_named(self):
        z = self.model.binary_var(name='z')
        self.assertEqual(str(z), 'z')

    def test_var_str_binary_anonymous(self):
        z = self.model.binary_var()
        self.assertEqual(str(z), 'x1')

    def test_var_str_integer_default_bounds_anon(self):
        z = self.model.integer_var()
        self.assertEqual('x1', str(z))

    def test_var_str_integer_default_bounds_named(self):
        z = self.model.integer_var(name='ivar')
        self.assertEqual('ivar', str(z))

    def test_var_str_integer_non_default_bounds_named(self):
        z = self.model.integer_var(lb=3, ub=7, name='jvar')
        self.assertEqual('jvar', str(z))

    def test_var_str_integer_non_default_ub_anon(self):
        z = self.model.integer_var(ub=7)
        self.assertEqual('x1', str(z))

    def test_var_str_continuous_default_bounds(self):
        x = self.model.continuous_var(name='x')
        self.assertEqual('x', str(x))

    def test_var_str_continuous_non_default_bounds(self):
        x = self.model.continuous_var(name='fooo', lb=1)
        self.assertEqual('fooo', str(x))

    def test_var_str_anonymous_default_bounds(self):
        anon = self.model.continuous_var()
        self.assertEqual("x1", str(anon))

    def test_var_str_anonymous_non_default_lb_ub(self):
        anon = self.model.continuous_var(lb=3, ub=11)
        self.assertEqual("x1", str(anon))


class DecisionVarReprTests(DecisionVarTestsBase):
    def test_var_repr_binary_named(self):
        z = self.model.binary_var(name='zzz')
        self.assertEqual(repr(z), "docplex.mp.Var(type=B,name='zzz')")

    def test_var_repr_binary_unnamed(self):
        z = self.model.binary_var()
        self.assertEqual(repr(z), "docplex.mp.Var(type=B)")

    def test_var_repr_integer_named_default_bounds(self):
        z = self.model.integer_var(name='ijk')
        self.assertEqual(repr(z), "docplex.mp.Var(type=I,name='ijk')")

    def test_var_repr_integer_named_non_default_lb(self):
        z = self.model.integer_var(name='ijk', lb=3)
        self.assertEqual(repr(z), "docplex.mp.Var(type=I,name='ijk',lb=3)")

    def test_var_repr_integer_named_non_default_ub(self):
        z = self.model.integer_var(name='ijk', ub=1234)
        self.assertEqual(repr(z), "docplex.mp.Var(type=I,name='ijk',ub=1234)")

    def test_var_repr_cont_named_default_bounds(self):
        z = self.model.continuous_var(name='xyz')
        self.assertEqual(repr(z), "docplex.mp.Var(type=C,name='xyz')")

    def test_var_repr_cont_named_non_default_lb(self):
        z = self.model.continuous_var(name='xyz', lb=3.14)
        self.assertEqual(repr(z), "docplex.mp.Var(type=C,name='xyz',lb=3.14)")

    def test_var_repr_cont_named_non_default_ub(self):
        z = self.model.continuous_var(name='xyz', ub=789.345)
        self.assertEqual(repr(z), "docplex.mp.Var(type=C,name='xyz',ub=789.345)")

    def test_var_repr_semicont_named_default_ub(self):
        z = self.model.semicontinuous_var(name='semi1', lb=3)
        self.assertEqual(repr(z), "docplex.mp.Var(type=S,name='semi1',lb=3)")

    def test_var_repr_semicont_named_custom_ub(self):
        z = self.model.semicontinuous_var(name='semi1', lb=3, ub=9999)
        self.assertEqual(repr(z), "docplex.mp.Var(type=S,name='semi1',lb=3,ub=9999)")

    def test_var_repr_semicont_unnamed(self):
        z = self.model.semicontinuous_var(lb=3)
        self.assertEqual(repr(z), "docplex.mp.Var(type=S,lb=3)")

    def test_var_repr_semiint_named_default_ub(self):
        z = self.model.semiinteger_var(name='semi1', lb=3)
        self.assertEqual(repr(z), "docplex.mp.Var(type=N,name='semi1',lb=3)")

    def test_var_repr_semiint_named_custom_ub(self):
        z = self.model.semiinteger_var(name='semi1', lb=3, ub=9999)
        self.assertEqual(repr(z), "docplex.mp.Var(type=N,name='semi1',lb=3,ub=9999)")

    def test_var_repr_semiint_unnamed(self):
        z = self.model.semiinteger_var(lb=3)
        self.assertEqual(repr(z), "docplex.mp.Var(type=N,lb=3)")


class DecisionVarListTests(DecisionVarTestsBase):
    def test_anonymous_default_list(self):
        m = self.model
        size = 3
        cs = m.continuous_var_list(size)  # no bounds , no name
        self.assertEqual(size, len(cs))
        self.assertTrue(all(v.index >= 0 for v in cs))

    def test_var_list_container_index(self):
        m = self.model
        size = 3
        xs = m.continuous_var_list(keys=3, name='x')
        ys = m.continuous_var_list(keys=3, name='y')
        a = m.binary_var()
        self.assertIsNone(a.container)
        for i in range(size):
            self.assertEqual(0, xs[i].container.index)
            self.assertEqual(1, ys[i].container.index)

    def test_typed_varlist_type_equals(self):
        m = self.model
        xs = m.continuous_var_list(2, name='x')
        self.assertEqual(xs[0].vartype, xs[1].vartype)

    def test_updated_container(self):
        with Model(name='updated_ddicts') as mm:
            keys = ['a', 'b', 'c']
            xd = mm.continuous_var_dict(keys, name='zz')
            xdvs = [dv for dv in xd.values()]

            xd.update(mm.continuous_var_dict(keys=['foo', 'bar'], name='xx'))

            self.assertTrue(all(dv.container is None for dv in xdvs))
            #self.assertEqual(1, len(mm._all_containers))
            mm2 = mm.clone()
            self.assertEqual(1, len(mm2._all_containers))



class DecisionVarMatrixTests(DecisionVarTestsBase):

    def test_model_create_var_matrix(self):
        m = self.model
        DIM1 = 10
        DIM2 = 5
        keys1 = range(1, DIM1 + 1)
        keys2 = range(1, DIM2 + 1)
        var_matrix = m.binary_var_matrix(keys1, keys2, name='my_matrix', key_format='~~{{%s}}')
        self.assertEqual(DIM1 * DIM2, len(var_matrix))
        for k1 in keys1:
            for k2 in keys2:
                dvar = var_matrix[k1, k2]
                self.assertTrue(is_var(dvar), "not a modeling variable")
                # oui je sais ces noms sont baroques mais bon c est le QA non??
                expected_name = 'my_matrix~~{{%d}}~~{{%d}}' % (k1, k2)
                self.assertEqual(expected_name, dvar.name)


class DecisionVarIterTests(DecisionVarTestsBase):
    def test_iter_continuous_vars(self):
        m = self.model
        xs = m.continuous_var_list(3, name='x')
        ys = m.continuous_var_list(5, name='y')
        itl = list(m.iter_continuous_vars())
        self.assertEqual(8, len(itl))

    def test_iter_integer_vars(self):
        m = self.model
        xs = m.integer_var_list(3, name='x')
        ys = m.integer_var_list(5, name='y')
        itl = list(m.iter_integer_vars())
        self.assertEqual(8, len(itl))

    def test_iter_binary_vars(self):
        m = self.model
        xs = m.binary_var_list(3, name='x')
        ys = m.binary_var_list(5, name='y')
        itl = list(m.iter_binary_vars())
        self.assertEqual(8, len(itl))


class DecisionVarArithmeticTests(DecisionVarTestsBase):
    # TODO
    pass

    def setUp(self):
        DecisionVarTestsBase.setUp(self)
        self.x = self.model.continuous_var(name='x')
        self.y = self.model.continuous_var(name='y')

    def test_var_uplus_sign(self):
        # plus sign is syntactic sugar +x "is" exactly the x object
        e = + self.x
        self.assertIs(e, self.x)

    def test_var_uminus(self):
        x = self.x
        # minus sign returns an expression (noneed to know concrete type).
        # with zero constant, one var (x) and coeff is -1
        e = - x
        self.assertEqual('-x', str(e))
        # self.assertIsInstance(e, LinearOperand)
        self.assertEqual(1, e.number_of_variables())
        self.assertEqual(-1, e[x])
        self.assertEqual(0, e.constant)

    def test_var_add_expr_no_side_effect(self):
        x1 = self.model.binary_var('x1')
        x2 = self.model.binary_var('x2')
        x3 = self.model.binary_var('x3')
        e1 = x1 + x2 + 1
        e1bis = e1.clone()
        e2 = x3 + e1
        self.assertTrue(e1.equals(e1bis))

    def test_var_sub_expr_no_side_effect(self):
        x1 = self.model.binary_var('x1')
        x2 = self.model.binary_var('x2')
        x3 = self.model.binary_var('x3')
        e1 = x1 + x2 + 1
        e1bis = e1.clone()
        e2 = x3 - e1
        self.assertTrue(e1.equals(e1bis))

    def test_var_call_add_var(self):
        x1 = self.model.binary_var('x1')
        x2 = self.model.binary_var('x2')
        e = x1.add(x2)
        self.assertEqual('x1+x2', str(e))

    def test_var_add_cst(self):
        m = self.model
        b1 = m.binary_var('b1')
        OPERAND = 3
        sum_right = b1 + OPERAND
        self.assertTrue(isinstance(sum_right, LinearExpr))
        self.assertEqual(OPERAND, sum_right.constant)
        self.assertEqual(1, sum_right.number_of_variables())
        self.assertTrue(b1 in sum_right)
        self.assertEqual(1, sum_right.get_coef(b1))

        sum_left = OPERAND + b1
        self.assertTrue(isinstance(sum_left, LinearExpr))
        self.assertEqual(OPERAND, sum_left.constant)
        self.assertEqual(1, sum_left.number_of_variables())
        self.assertTrue(b1 in sum_left)
        self.assertEqual(1, sum_left.get_coef(b1))

    def test_var_sub_cst(self):
        m = self.model
        x = self.x
        OPERAND = 7
        x_minus_cst = x - OPERAND
        self.assertTrue(isinstance(x_minus_cst, LinearExpr))
        self.assertEqual(-OPERAND, x_minus_cst.constant)
        self.assertEqual(1, x_minus_cst.number_of_variables())
        self.assertTrue(x in x_minus_cst)
        self.assertEqual(1, x_minus_cst.get_coef(x))

        cst_minus_x = OPERAND - x
        self.assertTrue(isinstance(cst_minus_x, LinearExpr))
        self.assertEqual(OPERAND, cst_minus_x.constant)
        self.assertEqual(1, cst_minus_x.number_of_variables())
        self.assertTrue(x in cst_minus_x)
        self.assertEqual(-1, cst_minus_x.get_coef(x))

    def test_var_call_sub_var(self):
        x1 = self.model.binary_var('x1')
        x2 = self.model.binary_var('x2')
        e = x1.subtract(x2)
        self.assertEqual('x1-x2', str(e))
        self.assertEqual(2, e.size)

    def test_var_mul_cst(self):
        x = self.x
        OPERAND = 11
        x_mul_cst = x * OPERAND
        self.assertTrue(isinstance(x_mul_cst, Expr))
        self.assertEqual(0, x_mul_cst.constant)
        self.assertEqual(1, x_mul_cst.number_of_variables())
        self.assertTrue(x_mul_cst.contains_var(x))
        self.assertEqual(OPERAND, x_mul_cst.get_coef(x))

        cst_mul_x = OPERAND * x
        self.assertTrue(isinstance(cst_mul_x, Expr))
        self.assertEqual(0, cst_mul_x.constant)
        self.assertEqual(1, cst_mul_x.number_of_variables())
        self.assertTrue(cst_mul_x.contains_var(x))
        self.assertEqual(OPERAND, cst_mul_x.get_coef(x))

        self.assertTrue(cst_mul_x.equals(x_mul_cst))

    def test_var_negate(self):
        m = self.model
        negate_x = -self.x
        self.assertIsInstance(negate_x, Expr)
        self.assertEqual(str(negate_x), '-x')

    def test_var_add_var(self):
        m = self.model
        e = self.x + self.y
        self.assertEqual('x+y', str(e))

    def test_var_add_self(self):
        m = self.model
        e = self.x + self.x
        self.assertEqual('2x', str(e))

    def test_expr_subtract(self):
        m = self.model
        x1 = self.model.binary_var('x1')
        x2 = self.model.binary_var('x2')
        x3 = self.model.binary_var('x3')
        allvars = [x1, x2, x3]
        d1 = dict(zip(allvars, [1, 2, 3]))
        d2 = dict(zip(allvars, [3, 2, 1]))
        expr1 = m.linear_expr(d1, constant=7)
        expr2 = m.linear_expr(d2, constant=11)
        resultExpr = expr1 - expr2
        self.assertEqual(resultExpr.constant, -4)
        expectedCoefs = [-2, 0, 2]
        for i in range(3):
            self.assertEqual(expectedCoefs[i], resultExpr.get_coef(allvars[i]))
        # expr1 is not modified
        self.assertNotEqual(id(expr1), id(resultExpr))
        self.assertNotEqual(id(expr2), id(resultExpr))

    def test_var_times_zero(self):
        from docplex.mp.linear import ZeroExpr
        zz = ZeroExpr(self.model)
        e = self.x * zz
        self.assertEqual('0', str(e))


class DecisionVarMultitypeVarList(DecisionVarTestsBase):
    def test_multitype_var_list_empty(self):
        ml = self.model._lfactory.new_multitype_var_list(size=0, vartypes=[])
        self.assertEqual([], ml)

    def test_multitype_var_list_explicit(self):
        m = self.model
        vts = [vt for vt in m.iter_vartypes()]
        nb_types = len(vts)
        lbs = [i + 1 for i in range(nb_types)]
        ubs = [100 + i for i in range(nb_types)]
        names = ["x_{0}".format(vt.cplex_typecode) for vt in vts]
        xl = m._lfactory.new_multitype_var_list(size=nb_types, vartypes=vts, lbs=lbs, ubs=ubs, names=names)
        self.assertEqual(nb_types, len(xl))

    def test_multitype_var_list_explicit_no_lbs(self):
        m = self.model
        vts = [vt for vt in m._vartypes()]
        nb_types = len(vts)
        ubs = [100 + i for i in range(nb_types)]

        six.assertRaisesRegex(self, DOcplexException,
                              "you must provide an explicit lower bound for variable of type semi-continuous",
                              lambda m_: m_._lfactory.new_multitype_var_list(size=nb_types, vartypes=vts, ubs=ubs), m)

    def test_multitype_var_list_explicit_ignore_names(self):
        with Model(ignore_names=True) as m:
            vts = [vt for vt in m._vartypes()]
            nb_types = len(vts)
            lbs = [i + 1 for i in range(nb_types)]
            ubs = [100 + i for i in range(nb_types)]
            names = ["x_{0}".format(vt.cplex_typecode) for vt in vts]
            xl = m._lfactory.new_multitype_var_list(size=nb_types, vartypes=vts, lbs=lbs, ubs=ubs, names=names)
            self.assertEqual(nb_types, len(xl))
            for x in xl:
                self.assertIsNone(x.name)
            if m.has_cplex():
                try:
                    cpxnames = m.cplex.variables.get_names()
                except:
                    cpxnames = []
                self.assertEqual([], cpxnames)


class DOcplexSemiConTests(DecisionVarTestsBase):
    def setUp(self):
        DecisionVarTestsBase.setUp(self)

    def test_semicont_default_lb_error(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        six.assertRaisesRegex(self, DOcplexException, "Type semi-continuous has no default lower",
                              lambda v_: v_.vartype.default_lb, xs)

    def test_min_one_semicontvar_zero(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        self.assertEqual(1, m.number_of_semicontinuous_variables)
        self.assertEqual(1, m.number_of_variables)
        self.assertEqual(33, xs.lb)
        m.minimize(xs)
        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(0, xs.solution_value)
        # was failing with autopublish: what can we do?

    def test_semicont_vars_stats(self):
        m = self.model
        xs1 = m.semicontinuous_var(name='sc1', lb=33)
        xs2 = m.semicontinuous_var(name='sc2', lb=33)
        self.assertEqual(2, m.number_of_semicontinuous_variables)
        stats = m.statistics
        self.assertEqual(2, stats.number_of_semicontinuous_variables)
        self.assertEqual(2, stats.number_of_variables)

        with RedirectedOutputToStringContext() as stdout:
            stats.print_information()
        self.assertIn('semi-continuous=2', stdout.get_str())

    def test_min_one_semicontvar_lb(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        m.add(xs >= 1)
        m.minimize(xs)
        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(33, xs.solution_value)

    def test_semicontvar_accept_zero_value(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        self.assertTrue(xs.accepts_value(0))

    def test_semicontvar_accept_negative_ko(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        self.assertFalse(xs.accepts_value(-1))

    def test_semicontvar_accept_sub_lb_ko(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        self.assertFalse(xs.accepts_value(30))

    def test_semicontvar_accept_lb_ok(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        self.assertTrue(xs.accepts_value(33))

    def test_semicontvar_accept_sup_lb_ok(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        self.assertTrue(xs.accepts_value(66))

    def test_semicontvarlist(self):
        m = self.model
        size = 10
        m.semicontinuous_var_list(keys=size, lb=33, name='sc')
        self.assertEqual(size, m.number_of_variables)
        self.assertEqual(size, m.number_of_semicontinuous_variables)

    def test_semicontvar_matrix(self):
        m = self.model
        semicont_vartype = m.semicontinuous_vartype
        dim1 = 3
        dim2 = 5
        size = dim1 * dim2
        scm = m.semicontinuous_var_matrix(keys1=dim1, keys2=dim2, lb=33.2, name='scm')
        self.assertEqual(size, m.number_of_variables)
        self.assertEqual(size, m.number_of_semicontinuous_variables)
        for i in range(dim1):
            for j in range(dim2):
                scv = scm[i, j]
                self.assertEqual(scv.vartype, semicont_vartype)
                self.assertEqual(scv.lb, 33.2)

    def test_semicontinuous_stats_aggregated(self):
        mdl = self.model
        sc = mdl.semicontinuous_var(lb=2, ub=10, name='sc1')
        sc_l = mdl.semicontinuous_var_list(3, lb=3, ub=100, name='sc')
        x = mdl.continuous_var(name='x')
        stats = mdl.statistics
        self.assertEqual(4, stats.number_of_semicontinuous_variables)

    def test_rtc32126(self):
        mdl = self.model
        sc1 = mdl.semicontinuous_var(lb=2, ub=10, name='sc1')
        c2 = mdl.continuous_var(name='x')
        self.assertEqual(c2.vartype.__class__.__name__, 'ContinuousVarType')

    def test_set_negative_lb(self):
        mdl = self.model
        sc1 = mdl.semicontinuous_var(lb=2, ub=10, name='sc1')
        six.assertRaisesRegex(self, DOcplexException, 'semi-continuous variable expects strict positive',
                              lambda sc: sc.set_lb(-1), sc1)


class DecisionVarBoundsChangeTests(DecisionVarTestsBase):
    def check_bound_by_solve(self, mdl, var, bound, up):
        if mdl._can_solve:
            if up:
                mdl.maximize(var)
            else:
                mdl.minimize(var)
            s = mdl.solve()
            self.assertIsNotNone(s)
            self.assertEqual(bound, var.solution_value)

    def test_update_one_var_lb(self):
        m = self.model
        x = m.continuous_var(name='x')
        self.assertEqual(x.lb, 0)
        x.lb = 10
        self.assertEqual(10, x.lb)
        self.check_bound_by_solve(m, var=x, bound=10, up=False)

    def test_update_one_var_lb2_reset(self):
        m = self.model
        x = m.continuous_var(name='x')
        self.assertEqual(x.lb, 0)
        x.lb = 10
        x.lb = None  # reset to default -> 0
        self.assertEqual(0, x.lb)
        self.check_bound_by_solve(m, var=x, bound=0, up=False)

    def test_update_one_var_ub(self):
        m = self.model
        x = m.continuous_var(name='x')
        x.ub = 17
        self.assertEqual(x.ub, 17)
        self.check_bound_by_solve(m, var=x, bound=17, up=True)

    def test_update_one_var_ub2_reset(self):
        m = self.model
        x = m.continuous_var(name='x')
        x.ub = 17
        x.ub = None
        self.assertEqual(x.ub, m.infinity)
        # to check by solve we cheat
        x.lb = 999999
        self.check_bound_by_solve(m, var=x, bound=999999, up=False)

    def test_update_var_ubs(self):
        mdl = self.model
        SIZE = 3
        xs = mdl.continuous_var_list(SIZE, name='x')
        new_ubs = [k * k + 3 for k in range(SIZE)]
        for v, nk in zip(xs, new_ubs):
            v.ub = nk
            self.assertEqual(v.ub, nk)

        if mdl._can_solve():
            mdl.maximize(mdl.sum(xs))
            s = mdl.solve()
            self.assertIsNotNone(s)
            for i, x in enumerate(xs):
                self.assertEqual(3 + i * i, x.solution_value)

    def test_update_var_lbs(self):
        mdl = self.model
        SIZE = 3
        xs = mdl.continuous_var_list(SIZE, name='x')
        new_ubs = [k + 3 for k in range(SIZE)]
        for v, nk in zip(xs, new_ubs):
            v.lb = nk
            self.assertEqual(v.lb, nk)

        if mdl._can_solve():
            mdl.minimize(mdl.sum(xs))
            s = mdl.solve()
            self.assertIsNotNone(s)
            for i, x in enumerate(xs):
                self.assertEqual(3 + i, x.solution_value)

    def test_batch_lower_bounds_seq_seq(self):
        mdl = self.model
        size = 3
        xs = mdl.continuous_var_list(size, name='x')
        nlbs = list(range(1, size + 1))
        mdl.change_var_lower_bounds(xs, nlbs)
        for i, (x, lb) in enumerate(zip(xs, nlbs)):
            self.assertEqual(x.lb, lb)
        # now goto cplex
        mdl.solve()
        cpx_lbs = mdl.cplex.variables.get_lower_bounds()
        for cpxlb, nlb in zip(nlbs, cpx_lbs):
            self.assertAlmostEqual(cpxlb, nlb, delta=1e-6)

    def test_batch_lower_bounds_seq_seq_no_solve(self):
        mdl = self.model
        size = 3
        xs = mdl.continuous_var_list(size, name='x')
        nlbs = list(range(1, size + 1))
        mdl.change_var_lower_bounds(xs, nlbs)
        for i, (x, lb) in enumerate(zip(xs, nlbs)):
            self.assertEqual(x.lb, lb)
        # now goto cplex
        mdl.sync_cplex_engine()
        cpx_lbs = mdl.cplex.variables.get_lower_bounds()
        for cpxlb, nlb in zip(nlbs, cpx_lbs):
            self.assertAlmostEqual(cpxlb, nlb, delta=1e-6)

    def test_batch_lower_bounds_seq_num(self):
        mdl = self.model
        size = 3
        xs = mdl.continuous_var_list(size, name='x')
        nlb = 13
        mdl.change_var_lower_bounds(xs, nlb)
        for x in xs:
            self.assertEqual(x.lb, nlb)
        # now goto cplex
        mdl.solve()
        cpx_lbs = mdl.cplex.variables.get_lower_bounds()
        for cpxlb in cpx_lbs:
            self.assertAlmostEqual(cpxlb, nlb, delta=1e-6)

    def test_batch_lower_bounds_comp_seq_no_checks(self):
        with Model(checker='off') as mdl:
            size = 3
            xs = mdl.continuous_var_list(size, name='x')
            nlbs = list(range(1, size + 1))
            zs = (z for z in xs)
            mdl.change_var_lower_bounds(zs, nlbs)
            for i, (x, lb) in enumerate(zip(xs, nlbs)):
                self.assertEqual(x.lb, lb)
            # now goto cplex
            mdl.solve()
            cpx_lbs = mdl.cplex.variables.get_lower_bounds()
            for cpxlb, nlb in zip(nlbs, cpx_lbs):
                self.assertAlmostEqual(cpxlb, nlb, delta=1e-6)

    def test_batch_lower_bounds_comp_comp_no_checks(self):
        with Model(checker='off') as mdl:
            size = 3
            xs = mdl.continuous_var_list(size, name='x')
            nlbs = range(1, size + 1)
            zs = (z for z in xs)
            mdl.change_var_lower_bounds(zs, nlbs)
            for i, (x, lb) in enumerate(zip(xs, nlbs)):
                self.assertEqual(x.lb, lb)
            # now goto cplex
            mdl.solve()
            cpx_lbs = mdl.cplex.variables.get_lower_bounds()
            for cpxlb, nlb in zip(nlbs, cpx_lbs):
                self.assertAlmostEqual(cpxlb, nlb, delta=1e-6)

    def test_batch_lower_bounds_comp_num_no_checks(self):
        with Model(checker='off') as mdl:
            size = 3
            nlb = 13
            xs = mdl.continuous_var_list(size, name='x')
            zs = (z for z in xs)
            mdl.change_var_lower_bounds(zs, nlb)
            for x in xs:
                self.assertEqual(x.lb, nlb)
            # now goto cplex
            mdl.solve()
            cpx_lbs = mdl.cplex.variables.get_lower_bounds()
            for cpxlb in cpx_lbs:
                self.assertAlmostEqual(cpxlb, nlb, delta=1e-6)

    def test_batch_lower_bounds_defaults(self):
        mdl = self.model
        x = mdl.continuous_var(name='x', lb=7)
        ij = mdl.integer_var(name='ij', lb=7)
        b = mdl.binary_var()
        xs = [x, ij, b]

        mdl.change_var_lower_bounds(xs, None)
        for x in xs:
            self.assertEqual(x.lb, 0)
        # now goto cplex
        mdl.solve()
        cpx_lbs = mdl.cplex.variables.get_lower_bounds()
        for cpxlb in cpx_lbs:
            self.assertAlmostEqual(cpxlb, 0, delta=1e-6)

    def test_batch_upper_bounds_seq_seq(self):
        mdl = self.model
        size = 3
        xs = mdl.continuous_var_list(size, name='x')
        nubs = list(range(100, 100 + size + 1))
        mdl.change_var_upper_bounds(xs, nubs)
        for i, (x, ub) in enumerate(zip(xs, nubs)):
            self.assertEqual(x.ub, ub)
        # now goto cplex
        mdl.solve()
        cpx_ubs = mdl.cplex.variables.get_upper_bounds()
        for cpxub, nub in zip(nubs, cpx_ubs):
            self.assertAlmostEqual(cpxub, nub, delta=1e-6)

    def test_batch_upper_bounds_seq_num(self):
        mdl = self.model
        size = 3
        xs = mdl.continuous_var_list(size, name='x')
        nub = 777
        mdl.change_var_upper_bounds(xs, 777)
        for x in xs:
            self.assertEqual(x.ub, nub)
        # now goto cplex
        mdl.solve()
        cpx_ubs = mdl.cplex.variables.get_upper_bounds()
        for cpxub in cpx_ubs:
            self.assertAlmostEqual(cpxub, nub, delta=1e-6)

    def test_batch_upper_bounds_comp_seq_no_checks(self):
        mdl = self.model
        size = 3
        xs = mdl.continuous_var_list(size, name='x')
        zs = (z for z in xs)
        nubs = list(range(100, 100 + size + 1))
        mdl.change_var_upper_bounds(zs, nubs)
        for i, (x, ub) in enumerate(zip(xs, nubs)):
            self.assertEqual(x.ub, ub)
        # now goto cplex
        mdl.solve()
        cpx_ubs = mdl.cplex.variables.get_upper_bounds()
        for cpxub, nub in zip(nubs, cpx_ubs):
            self.assertAlmostEqual(cpxub, nub, delta=1e-6)

    def test_batch_upper_bounds_comp_comp_no_checks(self):
        with Model(checker='off') as mdl:
            size = 3
            xs = mdl.continuous_var_list(size, name='x')
            zs = (z for z in xs)
            nubs = range(100, 100 + size + 1)
            mdl.change_var_upper_bounds(zs, nubs)
            for i, (x, ub) in enumerate(zip(xs, nubs)):
                self.assertEqual(x.ub, ub)
            # now goto cplex
            mdl.solve()
            cpx_ubs = mdl.cplex.variables.get_upper_bounds()
            for cpxub, nub in zip(nubs, cpx_ubs):
                self.assertAlmostEqual(cpxub, nub, delta=1e-6)

    def test_batch_upper_bounds_comp_num_no_checks(self):
        with Model(checker='off') as mdl:
            size = 3
            xs = mdl.continuous_var_list(size, name='x')
            zs = (x for x in xs)
            nub = 777
            mdl.change_var_upper_bounds(zs, 777)
            for x in xs:
                self.assertEqual(x.ub, nub)
            # now goto cplex
            mdl.solve()
            cpx_ubs = mdl.cplex.variables.get_upper_bounds()
            for cpxub in cpx_ubs:
                self.assertAlmostEqual(cpxub, nub, delta=1e-6)

    def test_batch_upper_bounds_defaults(self):
        mdl = self.model
        x = mdl.continuous_var(name='x', lb=7)
        ij = mdl.integer_var(name='ij', lb=7)
        b = mdl.binary_var()
        xs = [x, ij, b]

        mdl.change_var_upper_bounds(xs, None)
        self.assertAlmostEqual(1, b.ub, delta=1e-6)
        self.assertGreaterEqual(x.ub, 1e+20)
        self.assertGreaterEqual(ij.ub, 1e+20)
        # now goto cplex
        mdl.solve()
        cpx_ubs = mdl.cplex.variables.get_upper_bounds()
        self.assertAlmostEqual(1e+20, cpx_ubs[0], delta=1)
        self.assertAlmostEqual(1e+20, cpx_ubs[1], delta=1)
        self.assertAlmostEqual(1, cpx_ubs[2], delta=1e-6)


class DOcplexSemiIntTests(DecisionVarTestsBase):
    def setUp(self):
        DecisionVarTestsBase.setUp(self)

    def test_semiint_default_lb_error(self):
        m = self.model
        xs = m.semiinteger_var(name='si', lb=33)
        six.assertRaisesRegex(self, DOcplexException, "Type semi-integer has no default lower bound",
                              lambda v_: v_.vartype.default_lb, xs)

    def test_min_one_semiintvar_zero(self):
        m = self.model
        xs = m.semiinteger_var(name='sc', lb=33)
        self.assertEqual(1, m.number_of_semiinteger_variables)
        self.assertEqual(1, m.number_of_variables)
        self.assertEqual(33, xs.lb)
        m.minimize(xs)
        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(0, xs.solution_value)

    def test_semiint_vars_stats(self):
        m = self.model
        xs1 = m.semicontinuous_var(name='sc1', lb=33)
        xs2 = m.semicontinuous_var(name='sc2', lb=33)
        self.assertEqual(2, m.number_of_semicontinuous_variables)
        stats = m.statistics
        self.assertEqual(2, stats.number_of_semicontinuous_variables)
        self.assertEqual(2, stats.number_of_variables)

        with RedirectedOutputToStringContext() as stdout:
            stats.print_information()
        self.assertIn('semi-continuous=2', stdout.get_str())

    def test_min_one_semicontvar_lb(self):
        m = self.model
        xs = m.semiinteger_var(name='sc', lb=33)
        m.add(xs >= 1)
        m.minimize(xs)
        s = m.solve()
        self.assertIsNotNone(s)
        self.assertEqual(33, xs.solution_value)

    def test_semiintvar_accept_zero_value(self):
        m = self.model
        xs = m.semiinteger_var(name='sc', lb=33)
        self.assertTrue(xs.accepts_value(0))

    def test_semiintvar_accept_negative_ko(self):
        m = self.model
        xs = m.semiinteger_var(name='sc', lb=33)
        self.assertFalse(xs.accepts_value(-1))

    def test_semiintvar_accept_sub_lb_ko(self):
        m = self.model
        xs = m.semicontinuous_var(name='sc', lb=33)
        self.assertFalse(xs.accepts_value(30))

    def test_semiintvar_accept_lb_ok(self):
        m = self.model
        xs = m.semiinteger_var(name='sc', lb=33)
        self.assertTrue(xs.accepts_value(33))

    def test_semiintvar_accept_sup_lb_ok(self):
        m = self.model
        xs = m.semiinteger_var(name='sc', lb=33)
        self.assertTrue(xs.accepts_value(66))

    def test_semiintvarlist(self):
        m = self.model
        size = 10
        xs = m.semiinteger_var_list(keys=size, lb=33, name='sc')
        self.assertEqual(size, m.number_of_variables)
        self.assertEqual(size, m.number_of_semiinteger_variables)

    def test_semiintvar_matrix(self):
        m = self.model
        semiint_vartype = m.semiinteger_vartype
        dim1 = 3
        dim2 = 5
        size = dim1 * dim2
        scm = m.semiinteger_var_matrix(keys1=dim1, keys2=dim2, lb=33, name='scm')
        self.assertEqual(size, m.number_of_variables)
        self.assertEqual(size, m.number_of_semiinteger_variables)
        for i in range(dim1):
            for j in range(dim2):
                scv = scm[i, j]
                self.assertEqual(scv.vartype, semiint_vartype)
                self.assertEqual(scv.lb, 33)

    def test_semiintvardict(self):
        m = self.model
        keys = ['ga', 'bu', 'zo']
        size = len(keys)
        xs = m.semiinteger_var_dict(keys=keys, lb=33, name='sc')
        self.assertEqual(size, m.number_of_variables)
        self.assertEqual(size, m.number_of_semiinteger_variables)
        self.assertEqual(xs['ga'].name, 'sc_ga')

    def test_semiinteger_stats_aggregated(self):
        mdl = self.model
        mdl.semiinteger_var(lb=2, ub=10, name='sc1')
        mdl.semiinteger_var_list(3, lb=3, ub=100, name='sc')
        mdl.continuous_var(name='x')
        stats = mdl.statistics
        self.assertEqual(4, stats.number_of_semiinteger_variables)

    def test_set_negative_lb(self):
        mdl = self.model
        sc1 = mdl.semiinteger_var(lb=2, ub=10, name='sc1')
        six.assertRaisesRegex(self, DOcplexException, 'semi-integer variable expects strict positive',
                              lambda sc: sc.set_lb(-1), sc1)


class VarKeyTests(DecisionVarTestsBase):

    def check_container(self, expected_nb_dims, expected_shape, expected_dimstring):
        ctn = self.model.get_var_by_index(0).container
        self.assertEqual(expected_nb_dims, ctn.nb_dimensions)
        self.assertEqual(expected_shape, ctn.shape)
        self.assertIn(expected_dimstring, ctn.dimension_string)

    def test_get_key_from_create_varlist(self):
        keyList = ('P1', 'P2', 'P3')
        prodVars = self.model.integer_var_list(keyList, name='Production')
        self.check_container(1, (3,), "[3]")
        for ivar, var in enumerate(prodVars):
            print('Key of ', var, ' is ', var.get_key())
            self.assertEqual(var.get_key(), keyList[ivar])

    def test_get_key_from_create_vardict(self):
        keyList = ('P1', 'P2', 'P3')
        vardict = self.model.continuous_var_dict(keyList, name='Production')
        self.check_container(1, (3,), "[3]")
        for k, dv in six.iteritems(vardict):
            dvar_key = dv.get_key()
            print('-- Key of ', dv, ' is ', dvar_key)
            self.assertEqual(dvar_key, k)

    def test_get_key_from_var_matrix(self):
        keys1 = (1, 4, 9, 16, 25)
        keys2 = (3, 5, 7)
        vm = self.model.integer_var_matrix(keys1, keys2, name="ijk")
        self.check_container(2, (5, 3), "[5][3]")
        for k1 in keys1:
            for k2 in keys2:
                self.assertEqual((k1, k2), vm[k1, k2].get_key())

    def test_var_cube_keys(self):
        m = self.model

        def tnamer(kt):
            return '_'.join(str(ktf) for ktf in kt)

        cube = m.integer_var_cube(keys1=['ga', 'bu'],
                                  keys2=['foo', 'bar', 'gee'],
                                  keys3=['a', 'b', 'c', 'd'], name=tnamer)

        self.check_container(3, (2, 3, 4), "[2][3][4]")
        for dv in m.iter_variables():
            vname = dv.name
            keys_from_name = vname.split('_')
            keys = list(dv.get_key())
            self.assertEqual(keys_from_name, keys)

    def test_multidict4(self):
        m = self.model
        k1 = [1, 2]
        k2 = [3, 4]
        k3 = [5, 6]
        k4 = [7, 8, 9]
        cube4 = m.var_hypercube('C', [k1, k2, k3, k4], name='X')
        self.assertEqual(24, len(cube4))
        self.assertIn((2, 4, 6, 8), cube4)


class VarContainerTests(unittest.TestCase):

    def test_container_name_string(self):
        with Model() as m:
            zename = 'foo'
            cube = m.integer_var_matrix(keys1=3, keys2=2, name=zename)
            ctns = list(m.iter_var_containers())
            self.assertEqual(1, len(ctns))
            ct0 = ctns[0]
            self.assertEqual(zename, ct0.name)

    def test_container2_name_function(self):
        with Model() as m:
            mx = m.integer_var_matrix(keys1=3, keys2=2, name=lambda ij: 'foo_%d_%d_bar' % ij)
            ctns = list(m.iter_var_containers())
            self.assertEqual(1, len(ctns))
            ct0 = ctns[0]
            self.assertEqual('foo_', ct0.name)

    def test_container1_name_function(self):
        with Model() as m:
            m.integer_var_list(keys=3, name=lambda i: 'foo_%d_bar' % i)
            ctns = list(m.iter_var_containers())
            self.assertEqual(1, len(ctns))
            ct0 = ctns[0]
            self.assertEqual('foo_', ct0.name)


class MiscTests(unittest.TestCase):
    def test_variable_sets(self):
        # relies on variable hashing, assumed to be unique for each variable.
        with Model() as m:
            size = 3
            xl = m.integer_var_list(size, name=["x1", "x2", "x3"])
            zz = m.continuous_var()
            x_set1 = {x for x in xl}
            self.assertEqual(size, len(x_set1))
            # "in" versions
            self.assertIn(xl[0], x_set1)
            self.assertNotIn(zz, x_set1)
            # true/false variant
            self.assertTrue(xl[0] in x_set1)
            self.assertFalse(zz in x_set1)
            xl1 = xl[:]
            xl2 = [x for x in xl]
            xl1.extend(xl2)
            self.assertEqual(2 * size, len(xl1))
            x_set2 = set(xl1)
            self.assertEqual(size, len(x_set2))

    def test_vars_as_dict_keys(self):
        with Model() as m:
            x, y, z = m.continuous_var_list(keys=['x', 'y', 'z'])
            dd = {x: 'foo', y: 'bar', z: 'gee'}
            self.assertEqual('foo', dd[x])
            self.assertEqual('bar', dd[y])


class TupleKeyNameTests(DecisionVarTestsBase):

    @staticmethod
    def check_detuplified(varname):
        return ' ' not in varname and '(' not in varname

    # generated tuples are ok
    def test_var_matrix_prefixed(self):
        m = self.model
        keys1 = ["a", "b", "c"]
        keys2 = [100, 200, 300]
        mx = m.binary_var_matrix(keys1, keys2, name="b")
        allnames = [v.name for v in mx.values()]
        self.assertTrue(all(self.check_detuplified(vn) for vn in allnames))

    def test_var_matrix_explicit(self):
        m = self.model
        keys1 = ["a", "b", "c"]
        keys2 = [100, 200, 300]
        ks = [(k1, k2) for k1 in keys1 for k2 in keys2]
        mx = m.binary_var_dict(ks, name="bbbb")
        allnames = [v.name for v in mx.values()]
        self.assertTrue(all(self.check_detuplified(vn) for vn in allnames))

    def test_var_cube_explicit(self):
        m = self.model
        keys1 = ["a", "b", "c"]
        keys2 = [1, 2, 3]
        keys3 = [4, 5, 6]
        ks = [(k1, k2, k3) for k1 in keys1 for k2 in keys2 for k3 in keys3]
        mx = m.binary_var_dict(ks, name="bbbb")
        allnames = [v.name for v in mx.values()]
        self.assertTrue(all(self.check_detuplified(vn) for vn in allnames))


class ChangeVarTypeTests(DecisionVarTestsBase):

    def test_binary_to_continuous(self):
        m = self.model
        bs = m.binary_var_list(4, name='b')
        sumbs = m.sum(bs)
        m.add(sumbs <= 3.5)
        m.maximize(m.sum(bs))
        s1 = m.solve()
        # s1.display()
        self.assertEqual(3, sumbs.solution_value)
        # now switch
        for b in bs:
            b.set_vartype('C')
        # still a MIP as problem type deos not change!
        self.assertFalse(m._contains_discrete_artefacts())
        # m.print_information()
        s2 = m.solve(log_output=False)
        # m.report()
        self.assertAlmostEqual(3.5, s2.objective_value, delta=1e-3)
        # s2.display()

    def test_batch_binary_to_continuous_1(self):
        m = self.model
        bs = m.binary_var_list(4, name='b')
        sumbs = m.sum(bs)
        m.add(sumbs <= 3.5)
        m.maximize(m.sum(bs))
        s1 = m.solve()
        # s1.display()
        self.assertEqual(3, sumbs.solution_value)
        # now switch
        m.change_var_types(bs, 'C')
        self.assertFalse(m._contains_discrete_artefacts())
        # m.print_information()
        s2 = m.solve(log_output=False)
        # m.report()
        self.assertAlmostEqual(3.5, s2.objective_value, delta=1e-3)
        # s2.display()

    def test_batch_binary_to_type_list(self):
        m = self.model
        bs = m.binary_var_list(4, name='b')
        sumbs = m.sum(bs)
        m.add(sumbs <= 3.5)
        m.maximize(m.sum(bs))
        s1 = m.solve()
        # s1.display()
        self.assertEqual(3, sumbs.solution_value)
        # now switch
        m.change_var_types(bs, ['C', 'I', 'I', 'C'])
        self.assertEqual(2, m.number_of_integer_variables)
        self.assertEqual(2, m.number_of_continuous_variables)
        self.assertEqual(0, m.number_of_binary_variables)
        # m.print_information()
        s2 = m.solve(log_output=False)
        # m.report()
        self.assertAlmostEqual(3.5, s2.objective_value, delta=1e-3)
        lps = m.export_as_lp_string()
        self.assertIn('Generals\n b_1 b_2', lps)

    def test_integer_to_continuous(self):
        m = self.model
        bs = m.integer_var_list(4, name='ij', lb=1, ub=4)
        sumbs = m.sum(bs)
        m.add(sumbs <= 15.5)
        m.maximize(m.sum(bs))
        s1 = m.solve()
        # s1.display()
        self.assertEqual(15, sumbs.solution_value)
        # now switch
        for b in bs:
            b.set_vartype('C')
        # still a MIP as problem type deos not change!
        self.assertFalse(m._contains_discrete_artefacts())
        # m.print_information()
        s2 = m.solve(log_output=False)
        # m.report()
        self.assertAlmostEqual(15.5, s2.objective_value, delta=1e-3)
        # s2.display()

    def test_binary_to_integer(self):
        m = self.model
        bs = m.binary_var_list(4, name='b')
        sumbs = m.sum(bs)
        m.add(sumbs <= 3.14)
        m.maximize(m.dot(bs, [k for k in range(1, len(bs) + 1)]))
        s1 = m.solve()
        # s1.display()
        self.assertEqual(9, s1.objective_value)  # 2 + 3 + 4 = 9
        # now switch
        for b in bs:
            b.set_vartype('I')
            b.ub = 2
        # m.print_information()
        s2 = m.solve(log_output=False)
        # m.report()
        # expected: 0, 0, 1, 2 -> 8 + 3 = 11
        # s2.display()
        self.assertAlmostEqual(11, s2.objective_value, delta=1e-3)

    def test_continuous_to_integer(self):
        m = self.model
        xs = m.continuous_var_list(3, name='x')
        sumxs = m.sum(xs)
        m.add(sumxs <= 3.14)
        m.maximize(sumxs)
        s1 = m.solve()
        # s1.display()
        self.assertEqual(3.14, s1.objective_value)
        # now switch
        for x in xs:
            x.set_vartype('I')

        # m.print_information()
        s2 = m.solve(log_output=False)
        # s2.display()
        self.assertAlmostEqual(3, s2.objective_value, delta=1e-3)

    def test_integer_to_binary_ok(self):
        m = self.model
        ij = m.integer_var(name='ij', lb=0, ub=4)
        m.maximize(ij)
        s1 = m.solve()
        self.assertIsNotNone(s1)
        # s1.display()
        self.assertEqual(4, ij.solution_value)
        ij.set_vartype(m.binary_vartype)
        self.assertEqual(0, ij.lb)
        self.assertEqual(4, ij.ub)
        s2 = m.solve()
        self.assertIsNotNone(s2)
        self.assertEqual(1, ij.solution_value)

    def test_integer_to_binary_lb_ko(self):
        m = self.model
        ij = m.integer_var(name='ij', lb=2, ub=4)
        m.maximize(ij)
        s1 = m.solve()
        self.assertIsNotNone(s1)
        # s1.display()
        self.assertEqual(4, ij.solution_value)
        # should fail...
        six.assertRaisesRegex(self, DOcplexException,
                              "Lower bound for binary variable should be less than 1",
                              lambda v_: v_.set_vartype('B'), ij)

    def test_integer_to_binary_ub_ko(self):
        m = self.model
        ij = m.integer_var(name='ij', lb=-2, ub=-1)
        m.maximize(ij)
        s1 = m.solve()
        self.assertIsNotNone(s1)
        # s1.display()
        self.assertEqual(-1, ij.solution_value)
        # should fail...
        six.assertRaisesRegex(self, DOcplexException,
                              "Upper bound for binary variable should be greater than 0",
                              lambda v_: v_.set_vartype('B'), ij)

    def test_integer_to_binary_large_ok(self):
        m = self.model
        ij = m.integer_var(name='ij', lb=-2, ub=5)
        self.assertEqual(ij.cplex_typecode, 'I')
        m.maximize(ij)
        s1 = m.solve()
        self.assertIsNotNone(s1)
        # s1.display()
        self.assertEqual(5, ij.solution_value)
        ij.set_vartype('B')
        self.assertEqual(ij.cplex_typecode, 'B')
        # bounds are unchanged
        self.assertEqual(ij.lb, -2)
        self.assertEqual(ij.ub, 5)
        s2 = m.solve()
        self.assertAlmostEqual(1, s2[ij], delta=1e-6)

    def test_continuous_to_semicontinuous_ko(self):
        m = self.model
        xx = m.continuous_var(name='xx', lb=0, ub=100.5)
        m.maximize(xx)
        # canot switch to semicon, as lb is 0
        six.assertRaisesRegex(self, DOcplexException, 'semi-continuous variable expects strict positive lower bound',
                              lambda v_: v_.set_vartype('S'), xx)

    def test_continuous_to_semicontinuous_ok(self):
        m = self.model
        xx = m.continuous_var(name='xx', lb=0, ub=100.5)
        m.minimize(xx)
        s1 = m.solve()
        self.assertIsNotNone(s1)
        self.assertAlmostEqual(0, s1.objective_value, delta=1e-4)
        # change lb to 3
        xx.lb = 3
        xx.set_vartype('S')
        m.add( xx >= 1)
        s2 = m.solve(log_output=True, clean_before_solve=True)
        self.assertIsNotNone(s2)
        self.assertAlmostEqual(3, s2.objective_value, delta=1e-4)

    def test_integer_to_semiinteger_ko(self):
        m = self.model
        ij = m.integer_var(name='ij', lb=0, ub=4)
        m.maximize(ij)
        s1 = m.solve()
        # s1.display()
        self.assertEqual(4, ij.solution_value)
        # now switch
        six.assertRaisesRegex(self, DOcplexException, 'semi-integer variable expects strict positive lower bound',
                              lambda v_: v_.set_vartype('N'), ij)

    def test_integer_to_semiinteger_ok(self):
        m = self.model
        ij = m.integer_var(name='ij', lb=3, ub=4)
        m.minimize(ij)
        s1 = m.solve()
        # s1.display()
        self.assertEqual(3, ij.solution_value)
        # now switch
        ij.set_vartype('N')
        self.assertEqual('N', ij.cplex_typecode)
        s2 = m.solve()
        assert s2
        self.assertEqual(0, ij.solution_value)


class DecisionVarLogicalOperatorTests(DecisionVarTestsBase):

    def setUp(self):
        DecisionVarTestsBase.setUp(self)
        m = self.model
        self.b1 = m.binary_var(name='b1')
        self.b2 = m.binary_var(name='b2')
        self.z = m.binary_var(name='z')
        self.x = m.continuous_var(name='x')

    def test_boolvar_and_boolvar(self):
        m = self.model
        b1, b2, z = self.b1, self.b2, self.z
        m.add(z == b1 & b2)
        m.add(z == 1)
        m.minimize(b1 + b2)
        self.assertIsNotNone(m.solve())
        self.assertEqual(1, b1.solution_value)
        self.assertEqual(1, b2.sv)
        self.assertTrue(b1.to_bool())
        self.assertTrue(b2.to_bool())

    def test_boolvar_or_boolvar(self):
        m = self.model
        b1, b2, z = self.b1, self.b2, self.z
        m.add(z == b1 | b2)
        m.add(z == 0)
        m.maximize(b1 + b2)
        self.assertIsNotNone(m.solve())
        self.assertEqual(0, b1.solution_value)
        self.assertEqual(0, b2.solution_value)
        self.assertFalse(b1.to_bool())
        self.assertFalse(b2.to_bool())

    def test_contvar_or_ko(self):
        m = self.model
        six.assertRaisesRegex(self, DOcplexException,
                              "Logical or is available only for binary variables, x has type continuous",
                              lambda m_: m_.add(self.z == self.x | self.b1), m)

    def test_contvar_and_ko(self):
        m = self.model
        six.assertRaisesRegex(self, DOcplexException,
                              "Logical and is available only for binary variables, x has type continuous",
                              lambda m_: m_.add(self.z == self.x & self.b1), m)

    def test_boolvar_not(self):
        m = self.model
        b1, z = self.b1, self.z
        m.add(z == b1.logical_not())
        m.minimize(10 * b1 + z)
        m.solve()
        self.assertEqual(0, b1.solution_value)
        self.assertEqual(1, z.solution_value)
        lps = m.export_as_lp_string()
        self.assertIn("b1 + _not4 = 1", lps)


if __name__ == "__main__":
    unittest.main()
