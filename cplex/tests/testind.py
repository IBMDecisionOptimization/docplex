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
Tests Indicator Constraint API.

No command line arguments are required.
"""
import unittest
import cplex
from cplex import SparsePair
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase, getTempLPFile
from interfacetestcase import InterfaceTestCase, override


class IndicatorConstraintTests(InterfaceTestCase, CplexTestCase):

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        return cpx.indicator_constraints

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        names = self.getTestNames(cpx)
        iface = self.get_interface(cpx)
        var = cpx.variables
        self.assertEqual(iface.get_num(), 0)
        self.assertEqual(var.get_num(), 0)
        var.add(names=['a', 'b'])
        sp = SparsePair(ind=['b'], val=[2.0])
        nitems = len(names)
        indices = iface.add_batch(
            lin_expr=[sp] * nitems,
            sense='G' * nitems,
            rhs=[1.0] * nitems,
            complemented=[0] * nitems,
            indvar=['a'] * nitems,
            name=names,
            indtype=[iface.type_.if_] * nitems)
        indices = list(indices)
        self.assertEqual(len(indices), nitems)
        self.assertEqual(iface.get_num(), len(indices))
        return names, indices

    @override(InterfaceTestCase)
    def addOne(self, cpx, name):
        iface = self.get_interface(cpx)
        [idx1, idx2] = cpx.variables.add(lb=[0.0, 0.0])
        idx = iface.add(lin_expr=SparsePair(ind=[idx2], val=[2.0]),
                        sense='G',
                        rhs=1.0,
                        indvar=idx1,
                        complemented=0,
                        name=name)
        return idx

    @override(InterfaceTestCase)
    def defaultName(self):
        return "i"

    # FIXME?: This isn't bad necessarily, but for indicators we get a
    # CPXERR_NO_NAMES here rather than a CPXERR_NAME_NOT_FOUND.  Thus,
    # we override this test to avoid a failure.  We should probably make
    # this consistent one way or the other.
    @override(InterfaceTestCase)
    def testGetIndicesNoNames(self):
        """RTC-31974"""
        try:
            with self._newCplex() as cpx:
                iface = self.get_interface(cpx)
                idx = iface.get_indices("bogus")
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_NAMES)

    def testAddAllTypes(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            # We need at least one variable to create an indicator
            # constraint.
            cpx.variables.add(lb=[0.0])
            for indtype in (iface.type_.if_,
                            iface.type_.onlyif,
                            iface.type_.iff):
                # Using defaults for all the other arguments.
                idx = iface.add(indtype=indtype)
                # Make sure we can query the type.
                self.assertEqual(indtype, iface.get_types(idx))
            cpx.write("indicator_types.lp")

    def testIndicatorTypesFromLPReader(self):
        model = """\
Minimize
 obj: 0 x1
Subject To
 i1: x1 = 1 ->  = 0
 i2: x1 = 1 <-  = 0
 i3: x1 = 1 <->  = 0
End
"""
        with self._newCplex() as cpx:
            with getTempLPFile(model) as tmp:
                cpx.read(tmp)
                self.assertEqual(cpx.indicator_constraints.get_num(), 3)
                indtypes = cpx.indicator_constraints.type_
                self.assertEqual(
                    cpx.indicator_constraints.get_types(),
                    [indtypes.if_, indtypes.onlyif, indtypes.iff])

    def testAddBadType(self):
        try:
            with self._newCplex() as cpx:
                iface = self.get_interface(cpx)
                cpx.variables.add(lb=[0.0])
                bogusindtype = 1000
                idx = iface.add(indtype=bogusindtype)
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_INDTYPE)

    def testAddBadSense(self):
        try:
            with self._newCplex() as cpx:
                iface = self.get_interface(cpx)
                cpx.variables.add(lb=[0.0])
                bogussense = 'Z'
                idx = iface.add(sense=bogussense)
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_SENSE)

    @unittest.skip("FIXME: We allow -1 for indvar but don't document it!")
    def testAddNoIndVar(self):
        try:
            with self._newCplex() as cpx:
                iface = self.get_interface(cpx)
                idx = iface.add(indvar=-1)
                # FIXME: We get an indicator with no indvar!
                #        i1: = 1 ->  = 0
                cpx.write("bogusindvar.lp")
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_INDEX_RANGE)

    def testAddBatchEmpty(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            indices = iface.add_batch()
            self.assertEqual(0, len(list(indices)))

    def testAddBatchNoIndVar(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            cpx.variables.add(lb=[0.0]*2)
            indices = iface.add_batch(
                lin_expr=[SparsePair(ind=[0], val=[1.0])],
                sense="L",
                rhs=[0.0],
                indvar=None,
                complemented=[0],
                name=['i1'],
                indtype=[iface.type_.if_])
            indices = list(indices)
            self.assertEqual(1, len(indices))
            # FIXME: Is it OK to use -1 for the default?
            self.assertEqual(
                iface.get_indicator_variables(indices[0]), -1)

    def testAddBatchNoComp(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            cpx.variables.add(lb=[0.0]*2)
            indices = iface.add_batch(
                lin_expr=[SparsePair(ind=[0], val=[1.0])],
                sense="L",
                rhs=[0.0],
                complemented=None,
                indvar=[1],
                name=['i1'],
                indtype=[iface.type_.if_])
            indices = list(indices)
            self.assertEqual(1, len(indices))
            self.assertEqual(iface.get_complemented(indices[0]), 0)

    def testAddBatchNoRHS(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            cpx.variables.add(lb=[0.0]*2)
            indices = iface.add_batch(
                lin_expr=[SparsePair(ind=[0], val=[1.0])],
                sense="L",
                rhs=None,
                complemented=[0],
                indvar=[1],
                name=['i1'],
                indtype=[iface.type_.if_])
            indices = list(indices)
            self.assertEqual(1, len(indices))
            self.assertEqual(iface.get_rhs(indices[0]), 0.0)

    def testAddBatchNoSense(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            cpx.variables.add(lb=[0.0]*2)
            indices = iface.add_batch(
                lin_expr=[SparsePair(ind=[0], val=[1.0])],
                sense=None,
                rhs=[0.0],
                complemented=[0],
                indvar=[1],
                name=['i1'],
                indtype=[iface.type_.if_])
            indices = list(indices)
            self.assertEqual(1, len(indices))
            self.assertEqual(iface.get_senses(indices[0]), 'E')

    def testAddBatchNoLinExpr(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            cpx.variables.add(lb=[0.0]*2)
            indices = iface.add_batch(
                lin_expr=None,
                sense="L",
                rhs=[0.0],
                complemented=[0],
                indvar=[1],
                name=['i1'],
                indtype=[iface.type_.if_])
            indices = list(indices)
            self.assertEqual(1, len(indices))
            sp = iface.get_linear_components(indices[0])
            self.assertEqual(len(sp.ind), 0)
            self.assertEqual(len(sp.val), 0)

    def testAddBatchNoType(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            cpx.variables.add(lb=[0.0]*2)
            indices = iface.add_batch(
                lin_expr=[SparsePair(ind=[0], val=[1.0])],
                sense="L",
                rhs=[0.0],
                indvar=[1],
                complemented=[0],
                name=['i1'],
                indtype=None)
            indices = list(indices)
            self.assertEqual(1, len(indices))
            self.assertEqual(iface.get_types(indices[0]),
                             iface.type_.if_)

    def testAddBatchNoName(self):
        with self._newCplex() as cpx:
            iface = self.get_interface(cpx)
            cpx.variables.add(lb=[0.0]*2)
            indices = iface.add_batch(
                lin_expr=[SparsePair(ind=[0], val=[1.0])],
                sense="L",
                rhs=[0.0],
                complemented=[0],
                indvar=[1],
                name=None,
                indtype=[iface.type_.if_])
            self.assertEqual(1, len(list(indices)))


class IndicatorConstraintEncodingTests(IndicatorConstraintTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


def main():
    unittest.main()

if __name__ == '__main__':
    main()
