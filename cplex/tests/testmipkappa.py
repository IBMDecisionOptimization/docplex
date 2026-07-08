# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Test for the mipkappa feature.

No command line arguments are required.
"""
import unittest
import sys
import cplex
from cplex.callbacks import MIPInfoCallback
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase

THRESHOLD_ILLPOSED = 1e+14
THRESHOLD_UNSTABLE = 1e+10
THRESHOLD_SUSPICIOUS = 1e+07
EPSZERO = 1e-9


class KappaInfo():
    pct_stable = None
    pct_suspicious = None
    pct_unstable = None
    pct_illposed = None
    max_ = None
    attention = None

    def check_fractions(self):
        # Each fraction is between 0 and 1
        assert 0 <= self.pct_stable and self.pct_stable <= 1
        assert 0 <= self.pct_suspicious and self.pct_suspicious <= 1
        assert 0 <= self.pct_unstable and self.pct_unstable <= 1
        assert 0 <= self.pct_illposed and self.pct_illposed <= 1

    def check(self):
        self.check_max()
        self.check_sum()
        self.check_consistent()
        self.check_attention()
        self.check_fractions()

    def check_max(self):
        assert self.max_ >= 0.0

    def check_sum(self):
        # Fractions sum up to 1
        pct_total = (self.pct_stable +
                     self.pct_suspicious +
                     self.pct_unstable +
                     self.pct_illposed)
        assert pct_total <= 1.0 + EPSZERO and \
            pct_total >= 1.0 - EPSZERO, \
            "WARNING: percentages don't add up!"

    def check_consistent(self):
        # Percentages and max are consistent
        condition1 = (self.max_ >= THRESHOLD_ILLPOSED and
                      self.pct_illposed > EPSZERO)
        condition2 = (self.max_ >= THRESHOLD_UNSTABLE and
                      self.pct_illposed + self.pct_unstable > EPSZERO)
        condition3 = (self.max_ >= THRESHOLD_SUSPICIOUS and
                      self.pct_illposed + self.pct_unstable +
                      self.pct_suspicious > EPSZERO)
        condition4 = (self.max_ < THRESHOLD_SUSPICIOUS and
                      self.pct_stable > 1.0 - EPSZERO)
        assert condition1 or condition2 or condition3 or condition4
        assert self.max_ >= THRESHOLD_SUSPICIOUS and self.attention > 0 or \
            self.max_ < THRESHOLD_SUSPICIOUS and self.attention == 0

    def check_attention(self):
        assert type(self.attention) == type(1.0), \
            "kappa_attention is not a float!"


class KappaCallback(MIPInfoCallback):

    def __init__(self, env):
        super().__init__(env)
        self.called_once = False
        self.no_kappastats = True

    def __call__(self):
        self.called_once = True
        qm = self.quality_metric
        if self.no_kappastats:
            for which in [qm.kappa_stable,
                          qm.kappa_suspicious,
                          qm.kappa_unstable,
                          qm.kappa_illposed,
                          qm.kappa_max,
                          qm.kappa_attention]:
                try:
                    self.get_float_quality(which)
                    raise AssertionError("Expected a failure!")
                except CplexSolverError as cse:
                    # Notice that we check cse.args[1] here and not
                    # cse.args[2].  This is due to the way exceptions
                    # are built in SWIG_callback.c:fast_getcallbackinfo.
                    assert cse.args[1] == error_codes.CPXERR_NO_KAPPASTATS
        else:
            ki = KappaInfo()
            ki.pct_stable = self.get_float_quality(qm.kappa_stable)
            ki.pct_suspicious = self.get_float_quality(qm.kappa_suspicious)
            ki.pct_unstable = self.get_float_quality(qm.kappa_unstable)
            ki.pct_illposed = self.get_float_quality(qm.kappa_illposed)
            ki.max_ = self.get_float_quality(qm.kappa_max)
            ki.attention = self.get_float_quality(qm.kappa_attention)
            ki.check()


class MipKappaTests(CplexTestCase):

    def setUp(self):
        self.c = self._newCplex()
        self.kappastats = self.c.parameters.mip.strategy.kappastats.values

    def testRosagood(self):
        for which in self.kappastats:
            self.runWithKappaStats("../../data/rosagood.mps", which)

    def testCaso8(self):
        for which in self.kappastats:
            self.runWithKappaStats("../../data/caso8.mps", which)

    def runWithKappaStats(self, filename, kappastat):
        no_kappastats = kappastat in [self.kappastats.none,
                                      self.kappastats.auto]

        self.c.read(filename)

        # Set overall node limit in case callback conditions are not met
        self.c.parameters.mip.limits.nodes.set(1000)

        # Set advanced start indicator to 0
        self.c.parameters.advance.set(0)

        # Install info callback only in deterministic parallel.
        # Info callback queries mipkappa statistics using different
        # queries. In opportunistic parallel, between two consecutive
        # queries another thread might update the statistics.
        # Therefore there is no guarantee that the mipkappa statistics
        # as queried from the callback are in a consistent state.
        # See RTC-22707.
        info_callback = None
        if (self.c.parameters.parallel.get() !=
            self.c.parameters.parallel.values.opportunistic):
            info_callback = self.c.register_callback(KappaCallback)
            info_callback.no_kappastats = no_kappastats

        # set kappastats parameter
        self.c.parameters.mip.strategy.kappastats.set(kappastat)

        # optimize
        self.c.solve()
        qm = self.c.solution.quality_metric

        if no_kappastats:
            for which in [qm.kappa_stable,
                          qm.kappa_suspicious,
                          qm.kappa_unstable,
                          qm.kappa_illposed,
                          qm.kappa_max,
                          qm.kappa_attention]:
                with self.assertRaises(CplexSolverError) as cm:
                    self.c.solution.get_float_quality(which)
                self.assertEqual(cm.exception.args[2],
                                 error_codes.CPXERR_NO_KAPPASTATS)
        else:
            ki = KappaInfo()
            get_float_quality = self.c.solution.get_float_quality
            ki.pct_stable = get_float_quality(qm.kappa_stable)
            ki.pct_suspicious = get_float_quality(qm.kappa_suspicious)
            ki.pct_unstable = get_float_quality(qm.kappa_unstable)
            ki.pct_illposed = get_float_quality(qm.kappa_illposed)
            ki.max_ = get_float_quality(qm.kappa_max)
            ki.attention = get_float_quality(qm.kappa_attention)
            ki.check()

        if info_callback is not None:
            self.assertTrue(info_callback.called_once,
                            "kappastat: {0}".format(kappastat))


def main():
    unittest.main()

if __name__ == "__main__":
    main()
