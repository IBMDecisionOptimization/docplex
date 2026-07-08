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
Tests the Python examples shipped with CPLEX.

This is nice for two reasons:
1) It can potentially improve our python coverage score (previously we
   did not include these tests when doing coverage analysis)
2) It's a sanity test for the example files when used as modules

No command line arguments are required.
"""
import os
import sys
import unittest

EXAMPLES_PATH = "../../src/python"

class ExampleTests(unittest.TestCase):
    ransetup = False

    def setUp(self):
        if not self.ransetup:
            # add the examples directory to Python's search path
            sys.path.append(EXAMPLES_PATH)
            self.ransetup = True

    def test_admipex1(self):
        import admipex1
        admipex1.admipex1("../../data/caso8.mps")
        del admipex1

    def test_admipex2(self):
        import admipex2
        admipex2.admipex2("../../data/p0033.mps")
        del admipex2

    def test_admipex3(self):
        import admipex3
        admipex3.admipex3("../../data/sosex3.lp")
        del admipex3

    def test_admipex4(self):
        import admipex4
        admipex4.admipex4("../../data")
        del admipex4

    def test_admipex5(self):
        import admipex5
        admipex5.admipex5(['admipex5.py'])
        admipex5.admipex5(['admipex5.py', '-table'])
        admipex5.admipex5(['admipex5.py', '-lazy'])
        admipex5.admipex5(['admipex5.py', '-no-cuts'])
        del admipex5

    def test_admipex6(self):
        import admipex6
        admipex6.admipex6("../../data/p0033.mps")
        del admipex6

    def test_admipex8(self):
        import admipex8
        admipex8.admipex8(datadir='../../data', from_table=False, lazy=False,
                          use_callback=True)
        admipex8.admipex8(datadir='../../data', from_table=True, lazy=False,
                          use_callback=True)
        admipex8.admipex8(datadir='../../data', from_table=False, lazy=True,
                          use_callback=True)
        admipex8.admipex8(datadir='../../data', from_table=False, lazy=False,
                          use_callback=False)
        del admipex8

    def test_admipex9(self):
        import admipex9
        admipex9.admipex9("../../data/p0033.mps")
        del admipex9

    def test_bendersatsp(self):
        import bendersatsp
        datafile = "../../data/atsp.dat"
        for i in ("0", "1"):
            bendersatsp.bendersATSP(i, datafile)
        self.assertRaises(ValueError, bendersatsp.bendersATSP, "x", datafile)
        del bendersatsp

    def test_bendersatsp2(self):
        import bendersatsp2
        datafile = "../../data/atsp.dat"
        for i in ("0", "1"):
            bendersatsp2.bendersatsp(i, datafile)
        self.assertRaises(ValueError, bendersatsp2.bendersatsp, "x", datafile)
        del bendersatsp2

    def test_blend(self):
        import blend
        blend.blend()
        del blend

    def test_cutstock(self):
        import cutstock
        cutstock.cutstock("../../data/cutstock.dat")
        del cutstock

    def test_conflictex1(self):
        import conflictex1
        conflictex1.main("../../data/infeasible.lp")
        del conflictex1

    def test_diet(self):
        import diet
        diet.diet("../../data/diet.dat", bycolumn=False,
                  int_vars=False, multiobj=False)
        diet.diet("../../data/diet.dat", bycolumn=True,
                  int_vars=False, multiobj=False)
        diet.diet("../../data/diet.dat", bycolumn=False,
                  int_vars=True, multiobj=True)
        self.assertRaises(IOError, diet.diet, "x")
        del diet

    def test_etsp(self):
        import etsp
        etsp.etsp("../../data/etsp.dat")
        del etsp

    def test_facility(self):
        import facility
        facility.facility("../../data/facility.dat", facility.NO_BENDERS)
        del facility

    def test_fixcost1(self):
        import fixcost1
        fixcost1.fixcost1()
        del fixcost1

    def test_fixnet(self):
        import fixnet
        fixnet.fixnet()
        del fixnet

    def test_foodmanu(self):
        import foodmanu
        foodmanu.foodmanu()
        del foodmanu

    def test_genericbranch(self):
        import genericbranch
        datafile = self._getResource("examples/data/noswot.mps")
        genericbranch.genericbranch(datafile)
        del genericbranch

    def test_globalqpex1(self):
        import globalqpex1
        datafile = "../../data/nonconvexqp.lp"
        globalqpex1.globalqpex1(datafile, "f")
        globalqpex1.globalqpex1(datafile, "g")
        globalqpex1.globalqpex1("../../data/nonconvexmiqp.lp", "g")
        self.assertRaises(ValueError, globalqpex1.globalqpex1, datafile, "x")
        del globalqpex1

    def test_indefqpex1(self):
        import indefqpex1
        indefqpex1.indefqpex1()
        del indefqpex1

    def test_inout1(self):
        import inout1
        inout1.inout1()
        del inout1

    def test_inout3(self):
        import inout3
        inout3.inout3()
        del inout3

    def test_lpex1(self):
        import lpex1
        for opt in ("r", "c", "n"):
            lpex1.lpex1(opt)
        self.assertRaises(ValueError, lpex1.lpex1, "x")
        del lpex1

    def test_lpex2(self):
        import lpex2
        datafile = "../../data/afiro.mps"
        for algo in ("o", "p", "d", "b", "h", "s", "c"):
            lpex2.lpex2(datafile, algo)
        self.assertRaises(ValueError, lpex2.lpex2, datafile, "x")
        del lpex2

    def test_lpex3(self):
        import lpex3
        lpex3.lpex3()
        del lpex3

    def test_lpex4(self):
        import lpex4
        lpex4.lpex4()
        del lpex4

    def test_lpex5(self):
        import lpex5
        lpex5.lpex5()
        del lpex5

    def test_lpex6(self):
        import lpex6
        lpex6.lpex6()
        del lpex6

    def test_lpex7(self):
        import lpex7
        datafile = "../../data/afiro.mps"
        for algo in ("o", "p", "d", "b", "h", "s", "c"):
            lpex7.lpex7(datafile, algo)
        self.assertRaises(ValueError, lpex7.lpex7, datafile, "x")
        del lpex7

    def test_mipex1(self):
        import mipex1
        for opt in ("r", "c", "n"):
            mipex1.mipex1(opt)
        self.assertRaises(ValueError, mipex1.mipex1, "x")
        del mipex1

    def test_mipex2(self):
        import mipex2
        mipex2.mipex2(self._getResource("examples/data/mexample.mps"))
        del mipex2

    def test_mipex3(self):
        import mipex3
        mipex3.mipex3()
        del mipex3

    def test_mipex4(self):
        import mipex4
        datafile = self._getResource("examples/data/noswot.mps")
        for opt in ("t", "l", "a"):
            mipex4.mipex4(datafile, opt)
        self.assertRaises(ValueError, mipex4.mipex4, datafile, "x")
        del mipex4

    def test_miqpex1(self):
        import miqpex1
        miqpex1.miqpex1()
        del miqpex1

    def test_multiobjex1(self):
        import multiobjex1
        multiobjex1.multiobjex1("../../data/multiobj.lp")
        multiobjex1.multiobjex1("../../data/UFL-biobjective.lp")
        del multiobjex1

    def test_populate(self):
        import populate
        populate.populate("../../data/location.lp")
        del populate

    def test_qcpdual(self):
        import qcpdual
        exitcode = qcpdual.qcpdual()
        self.assertEqual(None, exitcode)
        del qcpdual

    def test_qcpex1(self):
        import qcpex1
        qcpex1.qcpex1()
        del qcpex1

    def test_qpex1(self):
        import qpex1
        qpex1.qpex1()
        del qpex1

    def test_qpex2(self):
        import qpex2
        datafile = "../../data/qpex.lp"
        for algo in ("o", "p", "d", "n", "b"):
            qpex2.qpex2(datafile, algo)
        self.assertRaises(ValueError, qpex2.qpex2, datafile, "x")
        del qpex2

    def test_rates(self):
        import rates
        rates.rates("../../data/rates.dat")
        del rates

    def test_socpex1(self):
        import socpex1
        exitcode = socpex1.socpex1()
        self.assertEqual(0, exitcode)
        del socpex1

    def test_steel(self):
        import steel
        steel.steel()
        del steel

    def test_transport(self):
        import transport
        for i in (True, False):
            transport.transport(i)
        del transport

    def test_warehouse(self):
        import warehouse
        warehouse.warehouse()
        del warehouse

    def test_check_for_new_examples(self):
        untested_examples = []
        examples = ["admipex1.py",
                    "admipex2.py",
                    "admipex3.py",
                    "admipex4.py",
                    "admipex5.py",
                    "admipex6.py",
                    "admipex8.py",
                    "admipex9.py",
                    "benders.py",
                    "bendersatsp.py",
                    "bendersatsp2.py",
                    "blend.py",
                    "conflictex1.py",
                    "cutstock.py",
                    "diet.py",
                    "etsp.py",
                    "facility.py",
                    "fixcost1.py",
                    "fixnet.py",
                    "foodmanu.py",
                    "genericbranch.py",
                    "globalqpex1.py",
                    "indefqpex1.py",
                    "inout1.py",
                    "inout3.py",
                    "inputdata.py",
                    "lpex1.py",
                    "lpex2.py",
                    "lpex3.py",
                    "lpex4.py",
                    "lpex5.py",
                    "lpex6.py",
                    "lpex7.py",
                    "mipex1.py",
                    "mipex2.py",
                    "mipex3.py",
                    "mipex4.py",
                    "miqpex1.py",
                    "multiobjex1.py",
                    "populate.py",
                    "qcpdual.py",
                    "qcpex1.py",
                    "qpex1.py",
                    "qpex2.py",
                    "rates.py",
                    "socpex1.py",
                    "steel.py",
                    "transport.py",
                    "warehouse.py"]
        for dirpath, _, filenames in os.walk(EXAMPLES_PATH):
            for filename in filenames:
                if filename.lower().endswith(".py"):
                    if filename not in examples:
                        untested_examples.append(filename)
        self.assertEqual([], untested_examples,
                         "It looks like a new example was "
                         "added, but is not tested here: {0}"
                         .format(" ".join(untested_examples)))

def main():
    unittest.main()

if __name__ == '__main__':
    main()
