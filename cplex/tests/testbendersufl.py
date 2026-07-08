# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Tests benders on a model that we generate from scratch.

No command line arguments are required.
"""
import unittest
try:
    import collections.abc as collections_abc # For Python >= 3.3
except ImportError:
    import collections as collections_abc
from random import randint, shuffle, sample, seed
from cplextestcase import CplexTestCase
from cplex import SparsePair


def flatten(lst):
    for l in lst:
        if isinstance(l, collections_abc.Iterable):
            yield from flatten(l)
        else:
            yield l


class UFL:
    def __init__(self, n=0, m=0, f=None, c=None):
        self.n = n
        self.m = m
        self.f = f
        self.c = c
        self.x = None
        self.y = None

    def model(self, c):
        '''Build Cplex model for a UFL'''
        cols = c.variables
        cap = [self.m]*self.n
        y = list(cols.add(lb=[0]*self.n,
                          ub=[1]*self.n,
                          types=['B']*self.n,
                          names=['y_{}'.format(i) for i in range(self.n)]))

        x = [list(cols.add(lb=[0]*self.n,
                           types=['C']*self.n,
                           names=['x_{}_{}'.format(i, j)
                                  for j in range(self.n)]))
             for i in range(self.m)]

        # Assignment constraint
        rows = c.linear_constraints
        d = [1.]*self.m
        rows.add(lin_expr=[SparsePair(ind=tmp, val=[1.]*self.n)
                           for tmp in x],
                 rhs=[1.] * self.m,
                 senses='E' * self.m,
                 names=['assign_{}'.format(i) for i in range(self.m)])

        # fixed cost constraint
        x_cols = [list(tmp) for tmp in zip(*x)]
        rows.add(lin_expr=[SparsePair(ind=x_cols[i] + [y[i]],
                                      val=d + [-cap[i]])
                           for i in range(self.n)],
                 rhs=[0.0]*self.n, senses='L'*self.n,
                 names=['fixed_cost{}'.format(i) for i in range(self.n)])

        c.objective.set_linear([(i, v)
                                for i, v in zip(flatten(x), flatten(self.c))])
        c.objective.set_linear([(i, v)
                                for i, v in zip(flatten(y), flatten(self.f))])

        self.x = x
        self.y = y
        return


def random_ufl(n, m, fixedpen=0.2):
    ''' Build a random UFL problem. '''
    max_travel = 1000
    c = [[randint(0, max_travel) for i in range(n)]
         for j in range(m)]
    f = [sum(x) * fixedpen for x in zip(*c)]
    return UFL(n=n, m=m, f=f, c=c)


class BendersUFLTests(CplexTestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the random number generator.
        seed(0)

    def testRtc39220(self):
        ufl = random_ufl(50,100)
        c = self._newCplex()
        ufl.model(c)
        c.parameters.benders.strategy.set(3)
        # For RTC-39220, if we write the model before calling solve(),
        # then we don't get a segfault.
        #c.write("mod.lp")
        c.solve()
        # If we don't segfault, then we're good.


def main():
    unittest.main()


if __name__ == '__main__':
    main()
