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
Tests string input/output.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase

UTF8 = "UTF-8"


class StringTests(CplexTestCase):

    test_strings = [
        "foo",
        "啊☆€㐁ᠠﭖꀀༀU䨭抎駡郂",
        "𠀀𠀁𠀂𠀃𠀄𪛔𪛕𪛖",
        "ÇàâｱｲｳДфэأبتثअइउ",
        "鷗㐀葛渚噓𠀋𪆐𪚲か゚",
        "Çàâรื่ญูฏูक्",
        "Åéﾊﾟパ M¡ए￦𠀀",
        "啊☆€㐁ᠠﭖꀀༀU䨭抎駡郂𠀀𠀁𠀂𠀃𠀄𪛔𪛕𪛖ÇàâｱｲｳДфэأبتثअइउ"
    ]

    def testProbName(self):
        with self._newCplex() as cpx:
            for probname in StringTests.test_strings:
                cpx.set_problem_name(probname)
                self.assertEqual(probname, cpx.get_problem_name())

    def testBadProbName(self):
        with self._newCplex() as cpx:
            for probname in StringTests.test_strings:
                bytestring = probname.encode(UTF8)
                with self.assertRaises(TypeError):
                    cpx.set_problem_name(bytestring)

    def testVarNames(self):
        with self._newCplex() as cpx:
            cpx.variables.add(names=StringTests.test_strings)
            self.assertEqual(StringTests.test_strings,
                             cpx.variables.get_names())

    def testBadVarNames(self):
        byte_strings = [x.encode(UTF8)
                        for x in StringTests.test_strings]
        with self._newCplex() as cpx:
            with self.assertRaises(TypeError):
                cpx.variables.add(names=byte_strings)

    def testBadVarTypes(self):
        num = len(StringTests.test_strings)
        with self._newCplex() as cpx:
            # First, test with list of byte strings.
            with self.assertRaises(TypeError):
                cpx.variables.add(lb=[0.0] * num,
                                  types=[b"C"] * num)
            # Next, test with one byte string.
            with self.assertRaises(TypeError):
                cpx.variables.add(lb=[0.0] * num,
                                  types=b"C" * num)

    def testLinearConstrNames(self):
        num = len(StringTests.test_strings)
        with self._newCplex() as cpx:
            ind = list(cpx.variables.add(lb=[0.0] * num))
            val = list((float(x) for x in range(num)))
            cpx.linear_constraints.add(lin_expr=[[ind, val]
                                                 for _ in range(num)],
                                       rhs=[1.0] * num,
                                       names=StringTests.test_strings)
            self.assertEqual(StringTests.test_strings,
                             cpx.linear_constraints.get_names())

    def testAddLinearConstrWithVarNames(self):
        num = len(StringTests.test_strings)
        with self._newCplex() as cpx:
            cpx.variables.add(lb=[0.0] * num,
                              names=StringTests.test_strings)
            val = list((float(x) for x in range(num)))
            # Test that we can do name to index conversion correctly.
            cpx.linear_constraints.add(
                lin_expr=[[StringTests.test_strings, val]
                          for _ in range(num)],
                rhs=[1.0] * num)
            self.assertEqual(num, cpx.linear_constraints.get_num())

    def testBadLinearConstrNames(self):
        num = len(StringTests.test_strings)
        byte_strings = [x.encode(UTF8)
                        for x in StringTests.test_strings]
        with self._newCplex() as cpx:
            ind = list(cpx.variables.add(lb=[0.0] * num))
            val = list((float(x) for x in range(num)))
            with self.assertRaises(TypeError):
                cpx.linear_constraints.add(lin_expr=[[ind, val]
                                                     for _ in range(num)],
                                           rhs=[1.0] * num,
                                           names=byte_strings)

    def testBadLinearConstrSenses(self):
        num = len(StringTests.test_strings)
        with self._newCplex() as cpx:
            ind = list(cpx.variables.add(lb=[0.0] * num))
            val = list((float(x) for x in range(num)))
            # First, test with list of byte strings.
            with self.assertRaises(TypeError):
                cpx.linear_constraints.add(lin_expr=[[ind, val]
                                                     for _ in range(num)],
                                           senses=[b"L"] * num,
                                           rhs=[1.0] * num)
            # Next, test with one byte string.
            with self.assertRaises(TypeError):
                cpx.linear_constraints.add(lin_expr=[[ind, val]
                                                     for _ in range(num)],
                                           senses=b"L" * num,
                                           rhs=[1.0] * num)

    def testQuadObjWithVarNames(self):
        num = len(StringTests.test_strings)
        with self._newCplex() as cpx:
            cpx.variables.add(lb=[0.0] * num,
                              names=StringTests.test_strings)
            val = [1.0] * num
            # Test that we can do name to index conversion correctly.
            cpx.objective.set_quadratic([[StringTests.test_strings, val]
                                         for _ in range(num)])
            self.assertEqual(num * num,
                             cpx.objective.get_num_quadratic_nonzeros())

    def testWriteAndRead(self):
        with self._newCplex() as cpxw:
            cpxw.variables.add(names=StringTests.test_strings)
            with CplexTestCase._getTempFileName(ext='.sav',
                                                delete=True) as tmpfilename:
                cpxw.write(tmpfilename)
                with self._newCplex() as cpxr:
                    cpxr.read(tmpfilename)
                    self.assertEqual(cpxw.variables.get_names(),
                                     cpxr.variables.get_names())


def main():
    unittest.main()


if __name__ == '__main__':
    main()
