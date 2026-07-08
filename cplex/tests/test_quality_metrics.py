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
Tests the QualityMetrics object (see also the test*solution.py tests).

No command line arguments are required.
"""
import unittest
from contextlib import closing
from io import StringIO
from cplextestcase import CplexTestCase


class QualityMetricsTests(CplexTestCase):

    def testFileLikeObjectStream(self):
        with self._newCplex() as cpx, \
             self._getTempFileName(ext='.txt', delete=False) as tmp, \
             open(tmp, 'w') as ftmp:
            stream = cpx.set_results_stream(ftmp)
            cpx.solve()
            qm = cpx.solution.get_quality_metrics()
            stream.write("foo")
        with open(tmp, "r") as ftmp:
            self.assertIn("foo", ftmp.readlines())

    def testStringStream(self):
        with self._newCplex() as cpx, \
             closing(StringIO()) as strio:
            stream = cpx.set_results_stream(strio)
            cpx.solve()
            qm = cpx.solution.get_quality_metrics()
            stream.write("foo")
            output = strio.getvalue()
        self.assertIn("foo", output)

    def testNoneStream(self):
        with self._newCplex() as cpx:
            stream = cpx.set_results_stream(None)
            cpx.solve()
            qm = cpx.solution.get_quality_metrics()
            stream.write("foo")


def main():
    unittest.main()


if __name__ == '__main__':
    main()
