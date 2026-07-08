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
Tests callback with unicode strings.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase
from cplex.callbacks import UserCutCallback


class MyCB(UserCutCallback):

    def __init__(self, env):
        super().__init__(env)
        self.var_names = None
        self.ran_once = False

    def __call__(self):
        if self.has_incumbent():
            v = self.var_names
            for var in v:
                assert self.get_incumbent_values(var) == 0.0
            for var in self.get_incumbent_values(v):
                assert var == 0.0
            for var in v:
                assert self.get_incumbent_linear_slacks(var) == 0.0
            for var in self.get_incumbent_linear_slacks(v):
                assert var == 0.0
            for var in v:
                assert self.get_incumbent_quadratic_slacks(var) == 0.0
            for var in self.get_incumbent_quadratic_slacks(v):
                assert var == 0.0
            for var in v:
                assert self.get_SOS_feasibilities(var) == 1
            for var in self.get_SOS_feasibilities(v):
                assert var == 1
            self.ran_once = True
        if self.ran_once:
            self.abort()


class CallbackUnicodeTests(CplexTestCase):

    def testCallback(self):
        cpx = self._newCplex()
        cpx.read(self._getResource("examples/data/noswot.mps"))
        # We use latin-1 here because of the specially-formatted comment at
        # the top of the file.  That is, all strings in this file are encoded
        # as latin-1 when they are declared.
        v = ["motörhead", "Ørsted", "Gauß", "Nuñoz"]

        # install a bunch of names using latin1 or unicode objects
        cpx.variables.add(names = v)
        cpx.linear_constraints.add(names = v)
        for n in v:
            cpx.quadratic_constraints.add(name = n)
            cpx.SOS.add(name = n)
            cpx.indicator_constraints.add(name = n)

        cpx.parameters.mip.limits.nodes.set(1000)
        cb = cpx.register_callback(MyCB)
        self.assertFalse(cb.ran_once)
        cb.var_names = v

        cpx.solve()
        self.assertTrue(cb.ran_once)


def main():
    unittest.main()


if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------
# This extra filler is to workaround the following Python bug:
# http://bugs.python.org/issue20844
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
