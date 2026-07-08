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
Tests Annotation API.

No command line arguments are required.
"""
import os
import unittest
from cplex.exceptions import CplexError, CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
from interfacetestcase import InterfaceTestCase, override


class LongAnnotationTests(InterfaceTestCase, CplexTestCase):
    """Mostly generic tests for annotations.

    These tests can be shared with DoubleAnnotationTests below because
    we override methods like get_interface, cast, and get_contents.
    """

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        """Long annotation specific."""
        return cpx.long_annotations

    def cast(self, x):
        """Long annotations specific.

        This makes some of the tests interesting depending on whether we
        run them with DoubleAnnotationTests or not.
        """
        return int(x)

    def get_contents(self):
        """Long annotation specific."""

        return """\
<?xml version='1.0' encoding='utf-8'?>
<CPLEXAnnotations>
  <CPLEXAnnotation name='anno1' type='long' default='1'>
    <object type='0'>
      <anno index='0' value='2'/>
    </object>
  </CPLEXAnnotation>
</CPLEXAnnotations>
"""

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        """Generic test setup for annotations.

        Returns list of names and a list of indices for newly created
        annotations.
        """
        names = self.getTestNames(cpx)
        indices = []
        anno = self.get_interface(cpx)
        objtype = anno.object_type.objective
        self.assertEqual(anno.get_num(), 0)
        for val, name in enumerate(names):
            idx = anno.add(name, self.cast(val + 0.1))
            indices.append(idx)
            anno.set_values(idx, objtype, 0, val * self.cast(2))
        self.assertEqual(len(indices), len(names))
        self.assertEqual(anno.get_num(), len(indices))
        return names, indices

    @override(InterfaceTestCase)
    def addOne(self, cpx, name):
        anno = self.get_interface(cpx)
        idx = anno.add(name, 1.1)
        return idx

    @override(InterfaceTestCase)
    def testAddOneNoNames(self):
        # You cannot create annotations without names, so we override
        # InterfaceTestCase.testGetNameRangeNoNames to be a no-op.
        pass

    @override(InterfaceTestCase)
    def testAddOneNone(self):
        # You cannot create annotations without names, so we override
        # InterfaceTestCase.testGetNameRangeNoNames to be a no-op.
        pass

    @override(InterfaceTestCase)
    def testAddOneEmptyString(self):
        # You cannot create annotations without names, so we override
        # InterfaceTestCase.testGetNameRangeNoNames to be a no-op.
        pass

    @override(InterfaceTestCase)
    def testGetNameNoNames(self):
        # You cannot create annotations without names, so we override
        # InterfaceTestCase.testGetNameRangeNoNames to be a no-op.
        pass

    @override(InterfaceTestCase)
    def testGetNameAllNoNames(self):
        # You cannot create annotations without names, so we override
        # InterfaceTestCase.testGetNameRangeNoNames to be a no-op.
        pass

    @override(InterfaceTestCase)
    def testGetNameRangeNoNames(self):
        # You cannot create annotations without names, so we override
        # InterfaceTestCase.testGetNameRangeNoNames to be a no-op.
        pass

    def checkAdd(self, defval):
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            annoname = "anno1"
            idx = anno.add(annoname, defval)
            self.assertEqual(anno.get_num(), 1)
            self.assertEqual(idx, 0)
            self.assertEqual(anno.get_indices(annoname), 0)

    def testAddWithInt(self):
        self.checkAdd(1)

    def testAddWithFloat(self):
        self.checkAdd(1.1)

    def getDefValCheck(self, vals):
        self.assertTrue(len(vals) >= 2)
        for idx, val in enumerate(vals):
            self.assertEqual(val, self.cast(idx + 0.1))

    def testGetDefValByIndex(self):
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            _, indices = self.doSetUp(cpx)
            defvals = [anno.get_default_values(idx)
                       for idx in indices]
            self.getDefValCheck(defvals)

    def testGetDefValByName(self):
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            annonames, _ = self.doSetUp(cpx)
            # by name
            defvals = [anno.get_default_values(name)
                       for name in annonames]
            self.getDefValCheck(defvals)

    def testGetDefValAll(self):
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            annonames, indices = self.doSetUp(cpx)
            defvals = anno.get_default_values()
            self.getDefValCheck(defvals)

    def testGetDefValRangeByIndex(self):
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            _, indices = self.doSetUp(cpx)
            defvals = anno.get_default_values(indices[0], indices[-1])
            self.getDefValCheck(defvals)

    def testGetDefValRangeByName(self):
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            annonames, _ = self.doSetUp(cpx)
            defvals = anno.get_default_values(annonames[0], annonames[-1])
            self.getDefValCheck(defvals)

    def testGetDefValBad(self):
        try:
            with self._newCplex() as cpx:
                anno = self.get_interface(cpx)
                annonames, indices = self.doSetUp(cpx)
                val = anno.get_default_values(max(indices) + 1)
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_INDEX_RANGE)

    def testSetGetObjByName(self):
        annoname = "anno1"
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.objective
            anno.add(annoname, defval)
            anno.set_values(annoname, objtype, 0, setval)
            self.assertEqual(anno.get_values(annoname, objtype),
                             [setval])

    def testSetGetObj(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.objective
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            anno.set_values(idx, objtype, 0, setval)
            val = anno.get_values(idx, objtype)
            self.assertEqual(val, [setval])

    def testSetGetVar(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.variable
            varidx = list(cpx.variables.add(names=["var1"]))[0]
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            anno.set_values(idx, objtype, varidx, setval)
            val = anno.get_values(idx, objtype)
            self.assertEqual(val, [setval])

    def testSetGetVarDefVal(self):
        defval = self.cast(1.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.variable
            cpx.variables.add(names=["x1", "x2", "x3"])
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            # If we don't set any annotations explicitly, then we expect
            # to get back default values.
            vals = anno.get_values(idx, objtype)
            self.assertEqual(vals, [defval, defval, defval])

    def testSetGetVarDefValNew(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.variable
            varidx = list(cpx.variables.add(names=["var1"]))[0]
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            anno.set_values(idx, objtype, varidx, setval)
            val = anno.get_values(idx, objtype)
            self.assertEqual(val, [setval])
            # Now add another variable, and make sure it has the default
            # value for this annotation.
            cpx.variables.add(names=["var2"])
            vals = anno.get_values(idx, objtype)
            self.assertEqual(vals, [setval, defval])

    def testSetGetVarMult(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.variable
            varindices = list(cpx.variables.add(
                names=["x1", "x2", "x3"]))
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            anno.set_values(
                idx, objtype, [(varidx, setval) for varidx in varindices])
            vals = anno.get_values(idx, objtype)
            self.assertEqual(vals, [setval, setval, setval])

    def testSetGetRow(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.row
            rowidx = list(cpx.linear_constraints.add(names=["c1"]))[0]
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            anno.set_values(idx, objtype, rowidx, setval)
            val = anno.get_values(idx, objtype)
            self.assertEqual(val, [setval])

    def testSetGetSos(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.sos_constraint
            cpx.variables.add(names=["var1"])
            sosidx = cpx.SOS.add(name="sos1")
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            anno.set_values(idx, objtype, sosidx, setval)
            val = anno.get_values(idx, objtype)
            self.assertEqual(val, [setval])

    def testSetGetIndConstr(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.indicator_constraint
            varidx = list(cpx.variables.add(names=["var1"]))[0]
            indidx = cpx.indicator_constraints.add(
                indvar=varidx, name="ind1")
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            anno.set_values(idx, objtype, indidx, setval)
            val = anno.get_values(idx, objtype)
            self.assertEqual(val, [setval])

    def testSetGetQConstr(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._newCplex() as cpx:
            anno = self.get_interface(cpx)
            objtype = anno.object_type.quadratic_constraint
            cpx.variables.add(names=["var1"])
            qcidx = cpx.quadratic_constraints.add(name="qc1")
            idx = anno.add("anno1", defval)
            self.assertEqual(anno.get_num(), 1)
            anno.set_values(idx, objtype, qcidx, setval)
            val = anno.get_values(idx, objtype)
            self.assertEqual(val, [setval])

    def testSetBadIndex(self):
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        try:
            with self._newCplex() as cpx:
                anno = self.get_interface(cpx)
                objtype = anno.object_type.objective
                idx = anno.add("anno1", defval)
                badidx = idx + 1
                anno.set_values(badidx, objtype, 0, setval)
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_INDEX_RANGE)

    def testSetBadObjType(self):
        badobjtype = 999
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        try:
            with self._newCplex() as cpx:
                anno = self.get_interface(cpx)
                idx = anno.add("anno1", defval)
                anno.set_values(idx, badobjtype, 0, setval)
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_BAD_ARGUMENT)

    def testRead(self):
        contents = self.get_contents()
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._getTempFileName(ext='.ann', delete=True) as tmp:
            with open(tmp, 'w') as tmpfile:
                tmpfile.write(contents)
            with self._newCplex() as cpx:
                anno = self.get_interface(cpx)
                cpx.read_annotations(tmp)
                self.assertEqual(anno.get_num(), 1)
                self.assertEqual(anno.get_names(0), "anno1")
                self.assertEqual(anno.get_default_values(0), defval)
                objtype = anno.object_type.objective
                self.assertEqual(anno.get_values(0, objtype), [setval])

    def testWrite(self):
        annoname = "annowrite1"
        defval = self.cast(1.1)
        setval = self.cast(2.1)
        with self._getTempFileName(ext='.ann', delete=True) as tmp:
            with self._newCplex() as cpx:
                anno = self.get_interface(cpx)
                idx = anno.add(annoname, defval)
                anno.set_values(idx, anno.object_type.objective, 0, setval)
                cpx.write_annotations(tmp)
                self.assertTrue(os.path.exists(tmp))
            with self._newCplex() as cpx:
                anno = self.get_interface(cpx)
                self.assertEqual(anno.get_num(), 0)
                cpx.read_annotations(tmp)
                self.assertEqual(anno.get_num(), 1)
                self.assertEqual(anno.get_names(0), annoname)
                self.assertEqual(anno.get_default_values(0), defval)
                objtype = anno.object_type.objective
                self.assertEqual(anno.get_values(0, objtype), [setval])


class DoubleAnnotationTests(LongAnnotationTests):
    """We inherit all tests from LongAnnotationTests.

    Only a few methods need to be overridden.
    """

    @override(LongAnnotationTests)
    def get_interface(self, cpx):
        """Override for double annotations."""
        return cpx.double_annotations

    @override(LongAnnotationTests)
    def cast(self, x):
        """Override for double annotations."""
        return float(x)

    @override(LongAnnotationTests)
    def get_contents(self):
        """Override for double annotations."""
        return """\
<?xml version='1.0' encoding='utf-8'?>
<CPLEXAnnotations>
  <CPLEXAnnotation name='anno1' type='double' default='1.1'>
    <object type='0'>
      <anno index='0' value='2.1'/>
    </object>
  </CPLEXAnnotation>
</CPLEXAnnotations>
"""


class LongAnnotationEncodingTests(LongAnnotationTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


class DoubleAnnotationEncodingTests(DoubleAnnotationTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


def main():
    unittest.main()

if __name__ == '__main__':
    main()
