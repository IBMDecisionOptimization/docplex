# -*- coding: latin-1 -*-
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Tests the API with unicode strings.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase
from cplex import SparsePair


class ApiUnicodeTests(CplexTestCase):

    def testApi(self):
        cpx = self._newCplex()
        cpx.read(self._getResource("examples/data/noswot.mps"))

        # Note that we use latin-1 because of the specially-formatted
        # comment at the top of the file. That is, all strings in this
        # file are encoded as latin-1 when they are declared.

        v = ["motörhead", "Ørsted", "Gauß", "Nuñoz"]

        # install a bunch of names using latin1 or unicode objects

        cpx.variables.add(names = v)
        cpx.linear_constraints.add(names = v)
        for n in v:
            cpx.quadratic_constraints.add(name = n)
            cpx.SOS.add(name = n)
            cpx.indicator_constraints.add(name = n)
            cpx.solution.pool.filter.add_diversity_filter(0.0, 1.0,
                                                         [[0],[0.0]], [1.0],
                                                         name = n + "div")
            cpx.solution.pool.filter.add_range_filter(0.0, 1.0, [[0], [0.0]],
                                                     name = n + "rng")
            cpx.MIP_starts.add([v, [0.0] * len(v)],
                              cpx.MIP_starts.effort_level.repair, n)

        cpx.set_problem_name(v[0])
        self.assertEqual(v[0], cpx.get_problem_name())

        v_div = [a + "div" for a in v]
        v_rng = [a + "rng" for a in v]

        for var in v:
            result = cpx.variables.get_cols(var)
            self.assertTrue(isinstance(result, SparsePair))
        for var in cpx.variables.get_cols(v):
            self.assertTrue(isinstance(var, SparsePair))

        for var in v:
            result = cpx.linear_constraints.get_rows(var)
            self.assertTrue(isinstance(result, SparsePair))
        for var in cpx.linear_constraints.get_rows(v):
            self.assertTrue(isinstance(var, SparsePair))

        for var in v:
            result = cpx.quadratic_constraints.get_rhs(var)
            self.assertTrue(isinstance(result, float))
        for var in cpx.quadratic_constraints.get_rhs(v):
            self.assertTrue(isinstance(var, float))

        for var in v:
            self.assertTrue(isinstance(cpx.SOS.get_sets(var), SparsePair))
        for var in cpx.SOS.get_sets(v):
            self.assertTrue(isinstance(var, SparsePair))

        for var in v:
            result = cpx.indicator_constraints.get_rhs(var)
            self.assertTrue(isinstance(result, float))
        for var in cpx.indicator_constraints.get_rhs(v):
            self.assertTrue(isinstance(var, float))

        for var in v_div:
            result = cpx.solution.pool.filter.get_diversity_filters(var)
            self.assertTrue(isinstance(result[0], SparsePair))
            self.assertTrue(isinstance(result[1][0], float))
        for var in cpx.solution.pool.filter.get_diversity_filters(v_div):
            self.assertTrue(isinstance(var[0], SparsePair))
            self.assertTrue(isinstance(var[1][0], float))

        for var in v_rng:
            result = cpx.solution.pool.filter.get_range_filters(var)
            self.assertTrue(isinstance(result, SparsePair))
        for var in cpx.solution.pool.filter.get_range_filters(v_rng):
            self.assertTrue(isinstance(var, SparsePair))

        for var in v:
            result = cpx.MIP_starts.get_starts(var)
            self.assertTrue(isinstance(result[0], SparsePair))
            self.assertTrue(isinstance(result[1], int))
        for var in cpx.MIP_starts.get_starts(v):
            self.assertTrue(isinstance(var[0], SparsePair))
            self.assertTrue(isinstance(var[1], int))

        self.assertEqual(v[0], cpx.get_problem_name())

        # Extract the names for these objects in the new encoding, make sure
        # they match.

        num = cpx.variables.get_num()
        names = cpx.variables.get_names(num - len(v), num - 1)
        for i, n in enumerate(names):
            self.assertEqual(n, v[i])

        num = cpx.linear_constraints.get_num()
        names = cpx.linear_constraints.get_names(num - len(v), num - 1)
        for i, n in enumerate(names):
            self.assertEqual(n, v[i])

        num = cpx.quadratic_constraints.get_num()
        names = cpx.quadratic_constraints.get_names(num - len(v), num - 1)
        for i, n in enumerate(names):
            self.assertEqual(n, v[i])

        num = cpx.SOS.get_num()
        names = cpx.SOS.get_names(num - len(v), num - 1)
        for i, n in enumerate(names):
            self.assertEqual(n, v[i])

        num = cpx.indicator_constraints.get_num()
        names = cpx.indicator_constraints.get_names(num - len(v), num - 1)
        for i, n in enumerate(names):
            self.assertEqual(n, v[i])

        cpxfilter = cpx.solution.pool.filter
        num = cpxfilter.get_num()
        all_names = cpxfilter.get_names(num - 2 * len(v), num - 1)
        div_names = all_names[0:8:2]
        rng_names = all_names[1:8:2]
        for i, n in enumerate(div_names):
            self.assertEqual(n, v_div[i])
        for i, n in enumerate(rng_names):
            self.assertEqual(n, v_rng[i])

        num = cpx.MIP_starts.get_num()
        names = cpx.MIP_starts.get_names(num - len(v), num - 1)
        for i, n in enumerate(names):
            self.assertEqual(n, v[i])


def main():
    unittest.main()


if __name__ == '__main__':
    main()
