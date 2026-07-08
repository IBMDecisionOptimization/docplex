# -*- coding: utf-8 -*-
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
Tests deterministic time limits.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase
from cplex.callbacks import IncumbentCallback
import os

SOLLIM = 18
PREFIX = "testpyintsolfileprefix"
PROBLEMFILE = "../../data/caso8.mps"


class MyIncumbentCallback(IncumbentCallback):

    def __init__(self, env):
        super().__init__(env)
        self.incumbents = []

    def __call__(self):
        self.incumbents.append(self.get_values())


class IntSolfilePrefixTests(CplexTestCase):

    def testOne(self):
        with cplex.Cplex() as cpx:
            self._setAllStreams(cpx, None)
            cpx.read(PROBLEMFILE)
            cpx.parameters.threads.set(1)
            # This parameter is hidden in the Python API, but it is set to zero,
            # under the hood.
            #cpx.parameters.mip.mipcbredlp.set(0)
            cpx.parameters.output.intsolfileprefix.set(PREFIX)
            cpx.parameters.mip.limits.solutions.set(SOLLIM)
            cb = cpx.register_callback(MyIncumbentCallback)
            cpx.solve()
            # The incumbents that were found in the incumbent callback.
            incumbents = cb.incumbents

        # Make sure we saw a reasonable amount of incumbents
        inccount = len(incumbents)
        self.assertTrue(
            inccount >= 10,
            "Not enough incumbents (found only {0})".format(inccount))

        # Check that each incumbent produced a solution file and that the
        # values in the files are the same as the values we saw in the
        # incumbent callback.
        for i, incumbent in enumerate(incumbents):
            # Test for existence of file
            solfilename = self.getsolfilename(i)
            self.assertTrue(os.path.exists(solfilename),
                            "didn't find " + solfilename)
            # Now read the incumbent file as MIP start and make sure
            # the values are as expected.
            with cplex.Cplex() as cpx:
                self._setAllStreams(cpx, None)
                cpx.read(PROBLEMFILE)
                cpx.MIP_starts.read(solfilename)
                mipstart = cpx.MIP_starts.get_starts(0)[0].val
                self.assertEqual(
                    len(incumbent), len(mipstart),
                    "Inconsistent lengths: {0} vs. {1}".format(
                        len(incumbent), len(mipstart)))
                for (start, inc) in zip(mipstart, incumbent):
                    self.assertFalse(
                        abs(start - inc) > 1e-6,
                        "Inconsistent values: {0} vs. {1}".format(
                            inc, start))

    # NOTE: if we didn't support Python 2.6, we could use the tearDownClass
    #       @classmethod here (which would only run once after all tests).
    def tearDown(self):
        """Called after every test method."""
        # clean up intsolfiles
        for i in range(SOLLIM):
            self._failSafeDelete(os.path.join('.', self.getsolfilename(i)))

    @staticmethod
    def getsolfilename(index):
        return "%s-%05d.sol" % (PREFIX, index + 1)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
