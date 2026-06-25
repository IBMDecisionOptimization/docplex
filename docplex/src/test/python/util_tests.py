
import re
import unittest
from datetime import datetime

from docplex.mp.mfactory import fix_format_string
from docplex.mp.numutils import *
from docplex.mp.utils import *


from docplex.mp.model import Model
from docplex.mp.numutils import resolve_abs_rel_tolerances


class Dumb(object):
    pass

int_types = set()
float_types = set()

try:
    import numpy as np
    from numpy import int32, float32, int64, float64, int16, uint16, uint32, uint64, float32, int_, bool_
    npv = numpy.__version__
    if  npv < '2.0':
        from numpy import float_

except ImportError:
    np = None

def generate_all_odds(nmax):
    i = 1
    while i < nmax:
        yield i
        i += 2

class UtilsTest(unittest.TestCase):
    def test_is_iterable(self):
        self.assertTrue(is_iterable([]))
        self.assertTrue(is_iterable({}))
        self.assertTrue(is_iterable([1, 2, 3]))
        self.assertTrue(is_iterable({1: 'a', 2: 'b', 3: 'c'}))
        self.assertFalse(is_iterable(None))
        self.assertFalse(is_iterable(1))
        self.assertFalse(is_iterable(3.14))
        #self.assertFalse(is_iterable('foo'))

        def generator_with_one_value():
            yield 1

        gen1 = generator_with_one_value()
        self.assertTrue(is_iterable(gen1))

        gen_odds_1000 = generate_all_odds(1000)
        self.assertTrue(is_iterable(gen_odds_1000))
        odd_comp = (k for k in range(10) if k%2 == 1)
        self.assertTrue(is_iterable(odd_comp))

    def test_has_len(self):
        self.assertTrue(has_len([]))
        self.assertTrue(has_len({}))
        self.assertTrue(has_len([1, 2, 3]))
        self.assertTrue(has_len({1: 'a', 2: 'b', 3: 'c'}))
        self.assertFalse(has_len(None))
        self.assertFalse(has_len(1))
        self.assertTrue(has_len('foo'))

        def generator_with_one_value():
            yield 1

        gen1 = generator_with_one_value()
        self.assertFalse(has_len(gen1))

    def check_candidate_iter(self, candidate_iter):
        self.assertIs(candidate_iter, iter(candidate_iter))
        self.assertTrue(is_iterator(candidate_iter))

    def test_is_iterator(self):
        self.assertFalse(is_iterator(None))
        self.assertFalse(is_iterator([]))
        self.assertTrue(is_iterator(iter([])))
        self.assertFalse(is_iterator(1))

        tpl = (1,2,3,4)
        self.check_candidate_iter(iter(tpl))

        l = [1,2,3]
        iter_l = iter(l)
        self.check_candidate_iter(iter_l)

        ns = {1,2,3}
        iter_s = iter(ns)
        self.check_candidate_iter(iter_s)

        dd = {1:"foo", 2:"bar"}
        iter_dd = iter(dd)
        self.check_candidate_iter(iter_dd)

        if np:
            npa = np.array([[1,2], [3,4]])
            self.assertTrue(is_iterator(npa.flat))


    def test_is_number(self):
        self.assertTrue(is_number(0))
        self.assertTrue(is_number(1))
        self.assertTrue(is_number(-1))
        self.assertTrue(is_number(3.14159))
        self.assertFalse(is_number('3.14'))
        self.assertFalse(is_number(''))
        self.assertFalse(is_number([3.14]))
        self.assertFalse(is_number({1: 3.14}))
        self.assertFalse(is_number(set([3.14])))
        self.assertFalse(is_number(Dumb()))

    # def test_is_zero(self):
    #     self.assertTrue(is_zero(0))
    #     self.assertTrue(is_zero(0.))
    #     self.assertFalse(is_zero(1))
    #     self.assertFalse(is_zero(6.28))
    #     self.assertFalse(is_zero("foo"))
    #     self.assertFalse(is_zero(None))
    #     self.assertFalse(is_zero(Dumb()))
    #     self.assertFalse(is_zero([1,2,3]))
    
    def test_is_number_by_type(self):
        # This test that is_int() and is_number() is true for python integer types (int and long)
        for it in int_types:
            i = it(0)
            self.assertTrue(is_int(i))
            self.assertTrue(is_number(i))
        # This test that is_int() is false for python float types, but is_number is true
        for it in float_types:
            i = it(0.0)
            self.asserFalse(is_int(i))
            self.assertTrue(is_number(i))

    @unittest.skipIf(np is None, "Skipped because numpy wasn't found")
    def test_is_number_by_numpy_type(self):
        # build the list of numpy types
        numpy_int_types = set()
        numpy_float_types = set()
        
        numpy_int_types.add(int64)
        numpy_int_types.add(int32)
        numpy_int_types.add(int16)
        numpy_int_types.add(uint64)
        numpy_int_types.add(uint32)
        numpy_int_types.add(uint16)
        numpy_int_types.add(int_)
        #numpy_int_types.add(bool_)
        
        numpy_float_types.add(float64)
        numpy_float_types.add(float32)
        if  npv < '2.0':
            numpy_float_types.add(float_)
        # This test that is_int() and is_number() is true for numpy integer types
        for it in numpy_int_types:
            i = it(0)
            self.assertTrue(is_int(i), "%s(0) should be recognized as an int" % it)
            self.assertTrue(is_number(i), "%s(0) should be recognized as a number" % it)
        # This test that is_int() is false for numpy float types, but is_number is true
        for it in numpy_float_types:
            i = it(0.0)
            self.assertFalse(is_int(i), "%s(0) should NOT be recognized as an int" % it)
            self.assertTrue(is_number(i),  "%s(0) should be recognized as a number" % it)

    def test_is_string(self):
        self.assertFalse(is_string(None))
        self.assertTrue(is_string(""))
        self.assertTrue(is_string("foo bar"))
        self.assertTrue(is_string(u"this is a unicode string"))
        self.assertFalse(is_string(1234))
        self.assertFalse(is_string(3.14))
        self.assertFalse(is_string([]))


    def test_normalize_basename(self):
        self.assertEqual("foo", normalize_basename("foo"))
        self.assertEqual("foo", normalize_basename("FoO"))
        self.assertEqual("_foo_bar_", normalize_basename(" FOO BAR "))
        self.assertEqual("_foo___bar_", normalize_basename(" FOO   BAR "))
        self.assertEqual("foo-bar", normalize_basename("foo-bar"))

    def test_fix_format_string(self):
        self.assertEqual('_%s_%s_%s_%s', fix_format_string('', 4))
        self.assertEqual('foo_%s', fix_format_string('foo_%s'))
        self.assertEqual('foo_%s', fix_format_string('foo_%s'))
        self.assertEqual('foo_%s', fix_format_string('foo'))
        self.assertEqual('foo_%s_%s', fix_format_string('foo_%s', 2))
        self.assertEqual('foo_%s_%s', fix_format_string('foo_%s_%s', 2))
        self.assertEqual('foo_%s_%s', fix_format_string('foo', 2))
        self.assertEqual('foo_%s_%s_%s', fix_format_string('foo_%s', 3))
        self.assertEqual('foo_%s_%s_%s', fix_format_string('foo_%s_%s', 3))
        self.assertEqual('foo_%s_%s_%s', fix_format_string('foo', 3))

    def test_exception_publish_percent1(self):
        ex = DOcplexException("message with cause: %s", "unknown")
        self.assertEqual("message with cause: unknown", ex.message)

    def test_exception_publish_percent2(self):
        ex = DOcplexException("message with cause: %s, %s", "unknown", "vraiment")
        self.assertEqual("message with cause: unknown, vraiment", ex.message)

    def test_exception_publish_format(self):
        ex = DOcplexException("{0} exception {1} etc {0}", "bla", "bli")
        self.assertEqual("bla exception bli etc bla", ex.message)

    def test_exception_publish_no_slot(self):
        ex = DOcplexException("fixed message", "unknown")
        self.assertEqual("fixed message", ex.message)

    def test_int64_is_number(self):
        if np:
            zebignum = np.int64(9999999999)
            self.assertTrue(is_int(zebignum))

    def test_round_nearest_towards_infinity(self):
        INF = 1e+20
        self.assertEqual(INF, round_nearest_towards_infinity(1e+30))
        self.assertEqual(-INF, round_nearest_towards_infinity(-1e+30))
        self.assertEqual(0, round_nearest_towards_infinity(0))
        self.assertEqual(1, round_nearest_towards_infinity(1))
        self.assertEqual(2, round_nearest_towards_infinity(1.5))
        self.assertEqual(1, round_nearest_towards_infinity(0.5))
        self.assertEqual(0, round_nearest_towards_infinity(-0.5))
        self.assertEqual(-1, round_nearest_towards_infinity(-1.5))
        self.assertEqual(1, round_nearest_towards_infinity(1.1))
        self.assertEqual(1, round_nearest_towards_infinity(0.999))
        self.assertEqual(-3, round_nearest_towards_infinity(-3.001))
        self.assertEqual(-3, round_nearest_towards_infinity(-2.999))
        self.assertTrue(is_int(round_nearest_towards_infinity(3.14)))
        
    def test_round_nearest_halfway_from_zero(self):
        INF = 1e+20
        self.assertEqual(INF, round_nearest_halfway_from_zero(1e+30))
        self.assertEqual(-INF, round_nearest_halfway_from_zero(-1e+30))
        self.assertEqual(0, round_nearest_halfway_from_zero(0))
        self.assertEqual(1, round_nearest_halfway_from_zero(1))
        self.assertEqual(2, round_nearest_halfway_from_zero(1.5))
        self.assertEqual(0, round_nearest_halfway_from_zero(0.5))
        self.assertEqual(0, round_nearest_halfway_from_zero(-0.5))
        self.assertEqual(-2, round_nearest_halfway_from_zero(-1.5))
        self.assertEqual(-2, round_nearest_halfway_from_zero(-2.5))
        self.assertEqual(-4, round_nearest_halfway_from_zero(-3.5))
        self.assertEqual(1, round_nearest_halfway_from_zero(1.1))
        self.assertEqual(2, round_nearest_halfway_from_zero(1.5))
        self.assertEqual(1, round_nearest_halfway_from_zero(0.999))
        self.assertEqual(-3, round_nearest_halfway_from_zero(-3.001))
        self.assertEqual(-3, round_nearest_halfway_from_zero(-2.999))
        self.assertTrue(is_int(round_nearest_halfway_from_zero(3.14)))

class ResolveTolerancesTests(unittest.TestCase):

    def setUp(self):
        self.logger = Model()

    def tearDown(self) -> None:
        self.logger.end()
        self.logger = None

    def test_tolerances_none_defaults(self):
        abs, rels = resolve_abs_rel_tolerances(self.logger, None, None, size=2, accept_none=True)
        self.assertEqual(abs, [1e-6, 1e-6])
        self.assertEqual(rels, [1e-4, 1e-4])

    def test_tolerances_none_custom(self):
        abs, rels = resolve_abs_rel_tolerances(self.logger, None, None, size=2, accept_none=True, default_abstol=3, default_reltol=0.1)
        self.assertEqual(abs, [3, 3])
        self.assertEqual(rels, [0.1, 0.1])

    def test_tolerances_num_num(self):
        abs, rels = resolve_abs_rel_tolerances(self.logger, 3, 0.1, size=2)
        self.assertEqual(abs, [3, 3])
        self.assertEqual(rels, [0.1, 0.1])

    def test_tolerances_seq_num(self):
        abs, rels = resolve_abs_rel_tolerances(self.logger, (2,3), 0.1, size=2)
        self.assertEqual(abs, [2,3])
        self.assertEqual(rels, [0.1, 0.1])

    def test_tolerances_num_seq(self):
        abs, rels = resolve_abs_rel_tolerances(self.logger, 3, (0.1, 0.2), size=2)
        self.assertEqual(abs, [3,3])
        self.assertEqual(rels, [0.1, 0.2])

    def test_tolerances_seq_seq(self):
        abs, rels = resolve_abs_rel_tolerances(self.logger, (3, 7), (0.1, 0.2), size=2)
        self.assertEqual(abs, [3,7])
        self.assertEqual(rels, [0.1, 0.2])

    def test_tolerances_seq_None(self):
        abs, rels = resolve_abs_rel_tolerances(self.logger, abstols=(3, 7), reltols=None, size=2)
        self.assertEqual(abs, [3,7])
        self.assertEqual(rels, [1e-4, 1e-4])


if __name__ == "__main__":
    unittest.main()


