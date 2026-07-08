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
Tests network parameters (RTC-9769).

No command line arguments are required.
"""
import unittest
import re

from cplextestcase import CplexTestCase
from testutil import OutputProcessor


TESTMODEL = "../../data/netgen40.mps.gz"


class NetworkParameterTests(CplexTestCase):

    def testAllWithSolve(self):
        # In the solve output, we'll look for lines like:
        # * "Extracted network with 2996 nodes and 22996 arcs."
        # * "Network - Optimal:  Objective =    1.1068572024e+10"
        # * "Network time = 0.08 sec. (4.76 ticks)  Iterations = 18447 (3239)"
        outproc = OutputProcessor(
            ["Extracted network with",
             "Network -",
             "Network time"])
        cpx = self._newCplex()
        self._setAllStreams(cpx, outproc)
        cpx.read(TESTMODEL)
        # Use the network solver.
        cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.network)
        # Use non-default values for all of the network parameters.
        cpx.parameters.network.display.set(
            cpx.parameters.network.display.values.true_objective_values)
        cpx.parameters.network.iterations.set(1000000000)
        cpx.parameters.network.netfind.set(
            cpx.parameters.network.netfind.values.general_scaling)
        cpx.parameters.network.pricing.set(
            cpx.parameters.network.pricing.values.partial)
        cpx.parameters.network.tolerances.feasibility.set(1e-05)
        cpx.parameters.network.tolerances.optimality.set(1e-05)
        # Solve
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.optimal)
        # Check for expected output from network solver.
        for item in outproc.regex_list:
            self.assertEqual(item.num_matches, 1)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
