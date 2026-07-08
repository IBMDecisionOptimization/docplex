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
Tests PWL API.

No command line arguments are required.
"""
import unittest
import cplex
from contextlib import contextmanager
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase, getTempLPFile
from interfacetestcase import InterfaceTestCase, override

# CPX_PARAM_LPREADER is hidden (see RTC-31881).
CPX_PARAM_LPREADER = 1152
CPX_LPREADER_LEGACY = 0
CPX_LPREADER_NEW = 1

def set_lpreader_param(cpx, value):
    # CPX_PARAM_LPREADER is hidden (see RTC-31881).
    cplex._internal._procedural.setintparam(cpx._env._e,
                                            CPX_PARAM_LPREADER,
                                            value)

def use_new_lpreader(cpx):
    """Use the new LP reader explicitly."""
    # CPX_PARAM_LPREADER is hidden (see RTC-31881).
    # cpx.parameters.read.lpreader.set(
    #     cpx.parameters.read.lpreader.values.new)
    set_lpreader_param(cpx, CPX_LPREADER_NEW)

def use_legacy_lpreader(cpx):
    """Use the legacy LP reader explicitly."""
    # CPX_PARAM_LPREADER is hidden (see RTC-31881).
    # cpx.parameters.read.lpreader.set(
    #     cpx.parameters.read.lpreader.values.legacy)
    set_lpreader_param(cpx, CPX_LPREADER_LEGACY)


class PWLTests(InterfaceTestCase, CplexTestCase):

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        return cpx.pwl_constraints

    def get_expected_definition(self):
        """Returns the PWL definition used for testing.

        Returns a list containing the following:
        [vary, varx, preslope, postslope, breakx, breaky]
        """
        return [0, 1, 0.5, 2., [0., 1., 2.], [0., 1., 4.]]

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        names = self.getTestNames(cpx)
        indices = []
        pwl = self.get_interface(cpx)
        var = cpx.variables
        self.assertEqual(pwl.get_num(), 0)
        self.assertEqual(var.get_num(), 0)
        var.add(names=['y', 'x'])
        (vary, varx, preslope, postslope,
         breakx, breaky) = self.get_expected_definition()
        self.assertEqual(var.get_indices(['y', 'x']), [vary, varx])
        for name in names:
            idx = pwl.add(vary='y', varx='x',
                          preslope=preslope,
                          postslope=postslope,
                          breakx=breakx,
                          breaky=breaky,
                          name=name)
            indices.append(idx)
        self.assertEqual(len(indices), len(names))
        self.assertEqual(pwl.get_num(), len(indices))
        return names, indices

    @override(InterfaceTestCase)
    def addOne(self, cpx, name):
        pwl = self.get_interface(cpx)
        cpx.variables.add(lb=[0.0, 0.0])
        (vary, varx, preslope, postslope,
         breakx, breaky) = self.get_expected_definition()
        idx = pwl.add(vary=vary, varx=varx,
                      preslope=preslope,
                      postslope=postslope,
                      breakx=breakx,
                      breaky=breaky,
                      name=name)
        return idx

    @override(InterfaceTestCase)
    def defaultName(self):
        return "p"

    def testAddWithNoNames(self):
        with self._newCplex() as cpx:
            pwl = self.get_interface(cpx)
            cpx.variables.add(lb=[0.0, 0.0])
            expected = self.get_expected_definition()
            (vary, varx, preslope, postslope,
             breakx, breaky) = expected
            idx = pwl.add(vary=vary, varx=varx,
                          preslope=preslope,
                          postslope=postslope,
                          breakx=breakx,
                          breaky=breaky)
            self.assertEqual(pwl.get_num(), 1)
            actual = pwl.get_definitions(idx)
            self.assertEqual(actual, expected)

    @unittest.skip("FIXME: We get a CPXERR_NOT_MIP instead")
    @override(InterfaceTestCase)
    def testGetIndicesNoNames(self):
        """RTC-31974"""
        # Rather than removing the @unittest.skip decorator above, this
        # test should just be deleted, so we use the inherited one in
        # interfacetestcase.py
        InterfaceTestCase.testGetIndicesNoNames(self)

    def testGetDefByIndex(self):
        with self._newCplex() as cpx:
            _, indices = self.doSetUp(cpx)
            pwl = self.get_interface(cpx)
            expected = self.get_expected_definition()
            for idx in indices:
                actual = pwl.get_definitions(idx)
                self.assertEqual(actual, expected)

    def testGetDefByName(self):
        with self._newCplex() as cpx:
            names, _ = self.doSetUp(cpx)
            pwl = self.get_interface(cpx)
            expected = self.get_expected_definition()
            for name in names:
                actual = pwl.get_definitions(name)
                self.assertEqual(actual, expected)

    def testGetDefAll(self):
        with self._newCplex() as cpx:
            self.doSetUp(cpx)
            pwl = self.get_interface(cpx)
            expected = self.get_expected_definition()
            deflst = pwl.get_definitions()
            for actual in deflst:
                self.assertEqual(actual, expected)

    def testGetDefRangeByIndex(self):
        with self._newCplex() as cpx:
            _, indices = self.doSetUp(cpx)
            pwl = self.get_interface(cpx)
            expected = self.get_expected_definition()
            deflst = pwl.get_definitions(indices[0], indices[-1])
            self.assertEqual(len(deflst), len(indices))
            for actual in deflst:
                self.assertEqual(actual, expected)

    def testGetDefRangeByName(self):
        with self._newCplex() as cpx:
            names, _ = self.doSetUp(cpx)
            pwl = self.get_interface(cpx)
            expected = self.get_expected_definition()
            deflst = pwl.get_definitions(names[0], names[-1])
            self.assertEqual(len(deflst), len(names))
            for actual in deflst:
                self.assertEqual(actual, expected)

    def testGetDefBadIndex(self):
        try:
            with self._newCplex() as cpx:
                pwl = self.get_interface(cpx)
                _, indices = self.doSetUp(cpx)
                defn = pwl.get_definitions(max(indices) + 1)
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_INDEX_RANGE)

    def testGetDefWhenNone(self):
        """RTC-33154"""
        try:
            with self._newCplex() as cpx:
                pwl = self.get_interface(cpx)
                pwl.get_definitions(0)
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_INDEX_RANGE)

    def testWriteReadMPS(self):
        self.checkWriteRead(file_ext='.mps', delete_names=False)

    def testWriteReadSAV(self):
        self.checkWriteRead(file_ext='.sav', delete_names=False)

    def testWriteReadLP(self):
        self.checkWriteRead(file_ext='.lp', delete_names=False)

    def testWriteReadMPSNoNames(self):
        self.checkWriteRead(file_ext='.mps', delete_names=True)

    def testWriteReadSAVNoNames(self):
        self.checkWriteRead(file_ext='.sav', delete_names=True)

    def testWriteReadLPNoNames(self):
        self.checkWriteRead(file_ext='.lp', delete_names=True)

    def checkWriteRead(self, file_ext, delete_names):
        with self._getTempFileName(ext=file_ext, delete=True) as tmp:
            with self._newCplex() as cpx:
                names, indices = self.doSetUp(cpx)
                if delete_names:
                    cpx.advanced.delete_names()
                cpx.write(tmp)
            with self._newCplex() as cpx:
                pwl = self.get_interface(cpx)
                self.assertEqual(pwl.get_num(), 0)
                cpx.read(tmp)
                self.assertEqual(pwl.get_num(), len(indices))
                expected = self.get_expected_definition()
                for idx in indices:
                    actual = pwl.get_definitions(idx)
                    self.assertEqual(actual, expected)

    def testSimplePwlFromLegacyLPReader(self):
        self.checkSimplePWLFromLP(uselegacyreader=True)

    @unittest.skip("FIXME: Auto-generates 'p1' instead of 'p2'")
    def testSimplePwlFromNewLPReader(self):
        self.checkSimplePWLFromLP(uselegacyreader=False)

    def checkSimplePWLFromLP(self, uselegacyreader=True):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x 0.5 (0, 0) (1, 1) (2, 4) 2.0
y = x 1.0 (0, 0) (0, 2) (1, 1) (2, 2) 2.0
abs: y = x -1.0 (0, 0) 1.0
end
"""
        with self._newCplex() as cpx:
            if uselegacyreader:
                use_legacy_lpreader(cpx)
            else:
                use_new_lpreader(cpx)
            self.assertTrue(uselegacyreader)
            with getTempLPFile(model) as tmp:
                cpx.read(tmp)
                self.assertEqual(cpx.variables.get_indices(['y', 'x']),
                                 [0, 1])
                self.assertEqual(cpx.pwl_constraints.get_num(), 3)
                self.assertEqual(cpx.pwl_constraints.get_names(0), "pwl1")
                # "p2" is auto-generated.
                self.assertEqual(cpx.pwl_constraints.get_names(1), "p2")
                self.assertEqual(cpx.pwl_constraints.get_names(2), "abs")
                self.assertEqual(
                    cpx.pwl_constraints.get_definitions(0),
                    [0, 1, 0.5, 2.0, [0., 1., 2.], [0., 1., 4.]])
                self.assertEqual(
                    cpx.pwl_constraints.get_definitions(1),
                    [0, 1, 1.0, 2.0, [0., 0., 1., 2.], [0., 2., 1., 2.]])
                self.assertEqual(
                    cpx.pwl_constraints.get_definitions(2),
                    [0, 1, -1.0, 1.0, [0.], [0.]])

    def testPWLFromLPErrNoID(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: * = x 0.5 (0, 0) (1, 1) (2, 4) 2.0
end
"""
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_NO_ID)

    def testPWLFromLPNewlineAfterName(self):
        model = """\
min
0 y + 0 x
pwl
pwl1:
y = x 0.5 (0, 0) (1, 1) (2, 4) 2.0
end
"""
        self.checkPWLFromLPNewlines(model)

    def testPWLFromLPNewlineAfterVars(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x
0.5 (0, 0) (1, 1) (2, 4) 2.0
end
"""
        self.checkPWLFromLPNewlines(model)

    def testPWLFromLPNewlineAfterPreslope(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x 0.5
(0, 0) (1, 1) (2, 4) 2.0
end
"""
        self.checkPWLFromLPNewlines(model)

    def testPWLFromLPNewlineInBreaks(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x 0.5 (0, 0)
(1, 1) (2, 4) 2.0
end
"""
        self.checkPWLFromLPNewlines(model)

    def testPWLFromLPNewlineAfterBreaks(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x 0.5 (0, 0) (1, 1) (2, 4)
2.0
end
"""
        self.checkPWLFromLPNewlines(model)

    def checkPWLFromLPNewlines(self, model):
        with self._newCplex() as cpx:
            use_legacy_lpreader(cpx)
            with getTempLPFile(model) as tmp:
                cpx.read(tmp)
                self.assertEqual(cpx.variables.get_indices(['y', 'x']),
                                 [0, 1])
                self.assertEqual(cpx.pwl_constraints.get_num(), 1)
                self.assertEqual(cpx.pwl_constraints.get_names(0), "pwl1")
                self.assertEqual(
                    cpx.pwl_constraints.get_definitions(0),
                    [0, 1, 0.5, 2.0, [0., 1., 2.], [0., 1., 4.]])

    def testPWLFromLPErrBadIDY(self):
        """Bad y variable."""
        model = """\
min
0 x
pwl
pwl1: y = x 0.5 (0, 0) (1, 1) (2, 4) 2.0
end
"""
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_NAME_NOT_FOUND)

    def testPWLFromLPErrBadIDX(self):
        """Bad x variable."""
        model = """\
min
0 y
pwl
pwl1: y = x 0.5 (0, 0) (1, 1) (2, 4) 2.0
end
"""
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_NAME_NOT_FOUND)

    def testPWLFromLPErrNoOpOrSense(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y < x 0.5 (0, 0) (1, 1) (2, 4) 2.0
end
"""
        self.checkForIllDefinedPwl(model)

    def testPWLFromLPErrBadNumPreslope(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x p (0, 0) (1, 1) (2, 4) 2.0
end
"""
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_BAD_NUMBER)

    def testPWLFromLPErrBadNumX(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x 0.5 (x, 0) (1, 1) (2, 4) 2.0
end
"""
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_BAD_NUMBER)

    def testPWLFromLPErrBadNumY(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x 0.5 (0, y) (1, 1) (2, 4) 2.0
end
"""
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_BAD_NUMBER)

    def testPWLNoBreaks(self):
        try:
            with self._newCplex() as cpx:
                cpx.variables.add(names=["y", "x"])
                cpx.pwl_constraints.add(vary="y", varx="x",
                                        preslope=0, postslope=0,
                                        breakx=[], breaky=[])
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_ILL_DEFINED_PWL)

    def testPwlNotLastSection(self):
        model = """\
min
0 y + 0 x
pwl
pwl1: y = x 0.5 (1, 1) 2.0
bounds
0 <= x <= 1000
end
"""
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_NAME_NOT_FOUND)

    def testPwlWithNameOnly(self):
        model = """\
min
0 y + 0 x
pwl
pwl1:
pwl2: y = x 1.0 (2, 1) 2.0
end
"""
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_NAME_NOT_FOUND)

    def testLastPwlWithNameOnly(self):
        """Handle special case where last PWL constraint is empty."""
        model = """\
min
0 y + 0 x
pwl
pwl1:
end
"""
        self.checkForIllDefinedPwl(model)

    def testBadBreak(self):
        model="""\
Minimize
 obj: x + y
PWL
 mypwl: y = x 1 (2, 3 4
End
"""
        self.checkForIllDefinedPwl(model)

    def checkForIllDefinedPwl(self, model):
        try:
            with self._newCplex() as cpx:
                use_legacy_lpreader(cpx)
                with getTempLPFile(model) as tmp:
                    cpx.read(tmp)
                    self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_ILL_DEFINED_PWL)

    def testBreakForName(self):
        model="""\
Minimize
 obj: x(0,0) + y(0,0)
PWL
 pwl(0): y(0,0) = x(0,0) 0 (1,1) 1
End
"""
        with self._newCplex() as cpx:
            use_legacy_lpreader(cpx)
            with getTempLPFile(model) as tmp:
                cpx.read(tmp)
                self.assertEqual(cpx.variables.get_indices(
                    ['x(0,0)', 'y(0,0)']), [0, 1])
                self.assertEqual(cpx.pwl_constraints.get_num(), 1)
                self.assertEqual(cpx.pwl_constraints.get_names(0), "pwl(0)")
                self.assertEqual(
                    cpx.pwl_constraints.get_definitions(0),
                    [1, 0, 0.0, 1.0, [1.0], [1.0]])

    def testCommaInName(self):
        model="""\
Minimize
 obj: x, + y
PWL
 y = x, 0 (1,1) 1
End
"""
        with self._newCplex() as cpx:
            use_legacy_lpreader(cpx)
            with getTempLPFile(model) as tmp:
                cpx.read(tmp)
                self.assertEqual(cpx.variables.get_indices(
                    ['x,', 'y']), [0, 1])
                self.assertEqual(cpx.pwl_constraints.get_num(), 1)
                self.assertEqual(cpx.pwl_constraints.get_names(0), "p1")
                self.assertEqual(
                    cpx.pwl_constraints.get_definitions(0),
                    [1, 0, 0.0, 1.0, [1.0], [1.0]])

    @unittest.skip("FIXME: Fails on some platforms.  Why?")
    def testUTF8Chars(self):
        model="""\
Minimize
 obj: ö + Ø
PWL
 ö = Ø 0 (1,1) 1
End
"""
        with self._newCplex() as cpx:
            use_legacy_lpreader(cpx)
            cpx.parameters.read.fileencoding.set("utf-8")
            with getTempLPFile(model) as tmp:
                cpx.read(tmp)
                self.assertEqual(cpx.variables.get_indices(
                    ['ö', 'Ø']), [0, 1])
                self.assertEqual(cpx.pwl_constraints.get_num(), 1)
                self.assertEqual(
                    cpx.pwl_constraints.get_definitions(0),
                    [0, 1, 0.0, 1.0, [1.0], [1.0]])

class PWLEncodingTests(PWLTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


def main():
    unittest.main()

if __name__ == '__main__':
    main()
