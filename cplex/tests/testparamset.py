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
Tests ParameterSet functionality.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase
from cplex import ParameterSet
from cplex.exceptions import CplexSolverError, error_codes
from cplex._internal import _constants as _const


class TParam():
    def __init__(self, param_id, param_obj, value):
        self.id_ = param_id
        self.obj = param_obj
        self.val = value
    def tearDown(self):
        pass


class ParameterSetTests(CplexTestCase):

    def setUp(self):
        cpx = self._newCplex()
        self.cpx = cpx
        self.int_param = TParam(_const.CPX_PARAM_ADVIND,
                                cpx.parameters.advance,
                                cpx.parameters.advance.values.none)
        self.lng_param = TParam(_const.CPX_PARAM_ITLIM,
                                cpx.parameters.simplex.limits.iterations,
                                1000)
        self.dbl_param = TParam(_const.CPX_PARAM_TILIM,
                                cpx.parameters.timelimit,
                                1000.0)
        self.str_param = TParam(_const.CPX_PARAM_WORKDIR,
                                cpx.parameters.workdir,
                                "mydir")

    def tearDown(self):
        self.cpx.end()

    def testEnd(self):
        with self.cpx.create_parameter_set() as ps:
            # Make sure nothing bad happens if called again.
            ps.end()
        # Make sure we get an exception if we try to use the ParameterSet
        # after it has been disposed.
        self.assertRaises(ValueError, ps.clear)

    def testCplexEnd(self):
        with self._newCplex() as cpx:
            with cpx.create_parameter_set() as ps:
                cpx.end()
                # Make sure we get an exception if we try to use the
                # ParameterSet after it's environment has been disposed.
                self.assertRaises(ValueError, ps.clear)

    def testDunderDel(self):
        with self.cpx.create_parameter_set() as ps:
            pass
        del ps

    def testLenEmpty(self):
        ps = self.cpx.create_parameter_set()
        self.assertEqual(len(ps), 0)

    def testGetEmpty(self):
        self.skipIfParamTesting(self.cpx)
        ps = self.cpx.create_parameter_set()
        for param in (self.int_param.id_, self.int_param.obj):
            with self.assertRaises(CplexSolverError) as cm:
                ps.get(param)
            self.assertEqual(cm.exception.args[2],
                             error_codes.CPXERR_BAD_PARAM_NUM)

    def testGetIDsEmpty(self):
        ps = self.cpx.create_parameter_set()
        self.assertEqual([], list(ps.get_ids()))

    def testDeleteEmpty(self):
        ps = self.cpx.create_parameter_set()
        for param in (self.int_param.id_, self.int_param.obj):
            ps.delete(param)  # no-op

    def testClearEmpty(self):
        ps = self.cpx.create_parameter_set()
        self.assertEqual(len(ps), 0)
        ps.clear()
        self.assertEqual(len(ps), 0)

    def testCopyEmpty(self):
        ps = self.cpx.create_parameter_set()
        self.assertEqual(len(ps), 0)
        copy = self.cpx.copy_parameter_set(ps)
        self.assertEqual(len(copy), 0)
        self.assertEqual(len(ps), 0)

    def checkAddOne(self, param):
        ps = self.cpx.create_parameter_set()
        for which in (param.id_, param.obj):
            ps.add(which, param.val)
            self.assertEqual(len(ps), 1)
            self.assertEqual(param.val, ps.get(which))
        for param_id in ps.get_ids():
            self.assertEqual(param_id, param.id_)

    def testAddOneInt(self):
        self.checkAddOne(self.int_param)

    def testAddOneLong(self):
        self.checkAddOne(self.lng_param)

    def testAddOneDouble(self):
        self.checkAddOne(self.dbl_param)

    def testAddOneString(self):
        self.checkAddOne(self.str_param)

    def get_test_params(self):
        return (self.int_param, self.lng_param, self.dbl_param,
                self.str_param)

    def testCopy(self):
        ps = self.cpx.create_parameter_set()
        test_params = self.get_test_params()
        for param in test_params:
            ps.add(param.obj, param.val)
        self.assertEqual(len(ps), len(test_params))
        copy = self.cpx.copy_parameter_set(ps)
        self.assertEqual(len(ps), len(copy))
        self.assertEqual(list(ps.get_ids()), list(copy.get_ids()))

    def testGet(self):
        ps = self.cpx.create_parameter_set()
        for param in self.get_test_params():
            ps.add(param.obj, param.val)
        for param in self.get_test_params():
            self.assertEqual(param.val, ps.get(param.obj))
            self.assertEqual(param.val, ps.get(param.id_))

    def testDelete(self):
        ps = self.cpx.create_parameter_set()
        for param in self.get_test_params():
            ps.add(param.obj, param.val)
        num = len(ps)
        for param in self.get_test_params():
            num -= 1
            ps.delete(param.obj)
            self.assertEqual(num, len(ps))
        self.assertEqual(0, len(ps))

    def testClear(self):
        ps = self.cpx.create_parameter_set()
        for param in self.get_test_params():
            ps.add(param.obj, param.val)
        ps.clear()
        self.assertEqual(0, len(ps))

    def testGetIDs(self):
        ps = self.cpx.create_parameter_set()
        for param in self.get_test_params():
            ps.add(param.obj, param.val)
        test_param_set = set(param.id_ for param in self.get_test_params())
        for param_id in ps.get_ids():
            test_param_set.remove(param_id)
        self.assertEqual(0, len(test_param_set))

    def testGetParamSetEmpty(self):
        self.skipIfParamTesting(self.cpx)
        ps = self.cpx.get_parameter_set()
        self.assertEqual(0, len(ps))

    def testGetParamSet(self):
        self.skipIfParamTesting(self.cpx)
        test_params = self.get_test_params()
        for param in test_params:
            param.obj.set(param.val)
        ps = self.cpx.get_parameter_set()
        self.assertEqual(len(ps), len(test_params))
        for param in test_params:
            self.assertEqual(ps.get(param.obj), param.val)

    def testSetParamSetEmpty(self):
        self.skipIfParamTesting(self.cpx)
        ps = self.cpx.create_parameter_set()
        self.cpx.set_parameter_set(ps)
        changed = self.cpx.parameters.get_changed()
        self.assertEqual(len(changed), 0)

    def testSetParamSet(self):
        self.skipIfParamTesting(self.cpx)
        ps = self.cpx.create_parameter_set()
        test_params = self.get_test_params()
        for param in test_params:
            ps.add(param.obj, param.val)
        self.cpx.set_parameter_set(ps)
        changed = self.cpx.parameters.get_changed()
        self.assertEqual(len(changed), len(ps))
        for param in test_params:
            self.assertEqual(param.obj.get(), param.val)

    def testSetParamSetValueError(self):
        cpx2 = self._newCplex()
        ps = cpx2.create_parameter_set()
        self.assertRaises(ValueError, self.cpx.set_parameter_set, ps)

    def testSetParamSetTypeError(self):
        self.assertRaises(TypeError, self.cpx.set_parameter_set, 0)

    def testCopyParamSetValueError(self):
        cpx2 = self._newCplex()
        ps = cpx2.create_parameter_set()
        self.assertRaises(ValueError, self.cpx.copy_parameter_set, ps)

    def testCopyParamSetTypeError(self):
        self.assertRaises(TypeError, self.cpx.copy_parameter_set, 0)

    def testReadParamSet(self):
        self.skipIfParamTesting(self.cpx)
        test_params = self.get_test_params()
        for param in test_params:
            param.obj.set(param.val)
        ps = self.cpx.create_parameter_set()
        with self._getTempFileName(ext='.prm') as tmp:
            self.cpx.parameters.write_file(tmp)
            ps.read(tmp)
        for param in test_params:
            self.assertEqual(ps.get(param.obj), param.val)

    def testWriteParamSet(self):
        self.skipIfParamTesting(self.cpx)
        ps = self.cpx.create_parameter_set()
        test_params = self.get_test_params()
        for param in test_params:
            ps.add(param.obj, param.val)
        with self._getTempFileName(ext='.prm') as tmp:
            ps.write(tmp)
            self.cpx.parameters.read_file(tmp)
        for param in test_params:
            self.assertEqual(ps.get(param.obj), param.obj.get())

    def testCompareParamFiles(self):
        self.skipIfParamTesting(self.cpx)
        test_params = self.get_test_params()
        for param in test_params:
            param.obj.set(param.val)
        ps = self.cpx.get_parameter_set()
        with self._getTempFileName(ext='.prm') as tmp:
            self.cpx.parameters.write_file(tmp)
            with open(tmp, 'r') as prmfile:
                prm1 = prmfile.readlines()
        # Depending on whether CPXDEBUG is defined or not, the datacheck
        # parameter has a different default value. And, the default value
        # of the datacheck parameter is always "on" in the Python API.
        # We just remove any entry that may or may not be there to ensure
        # that the parameter file from the environment has a chance to
        # match the parameter file from the paramter set.
        prm1[:] = [line for line in prm1
                   if "datacheck" not in line.lower()]
        with self._getTempFileName(ext='.prm') as tmp:
            ps.write(tmp)
            with open(tmp, 'r') as prmfile:
                prm2 = prmfile.readlines()
        self.compareLists(prm1, prm2)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
