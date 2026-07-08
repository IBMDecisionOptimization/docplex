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
Tests the Cplex.copylp() method.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase
from cplex import SparsePair
from cplex.exceptions import CplexError, CplexSolverError, error_codes

NaN = float('nan')
Inf = float('inf')


def rows2cmat(rows, ncols):
    """Convert a list of SparsePair objects (in row major form) to sparse
    column matrix arrays."""
    # First, convert to list of list format (column major form).
    columns = [[[], []] for _ in range(ncols)]
    for rowidx, row in enumerate(rows):
        ind, val = row.unpack()
        for i, v in zip(ind, val):
            columns[i][0].append(rowidx)
            columns[i][1].append(v)
    # Get rid of empty entries (i.e., variables that have no nonzeros).
    columns = [x for x in columns if x[0]]
    # Now convert to C-style arrays.
    cmatbeg, cmatcnt, cmatind, cmatval = [], [], [], []
    for col in columns:
        ind, val = col
        cmatbeg.append(len(cmatind))
        cmatcnt.append(len(ind))
        cmatind.extend(ind)
        cmatval.extend(val)
    return cmatbeg, cmatcnt, cmatind, cmatval


def copylp(orig, copy):
    """Make a copy an LP model using Cplex.copylp()."""
    # Query the original model for the LP data.
    numcols = orig.variables.get_num()
    numrows = orig.linear_constraints.get_num()
    objsense = orig.objective.get_sense()
    obj = orig.objective.get_linear()
    rhs = orig.linear_constraints.get_rhs()
    senses = orig.linear_constraints.get_senses()
    rows = orig.linear_constraints.get_rows()
    lb = orig.variables.get_lower_bounds()
    ub = orig.variables.get_upper_bounds()
    range_values = orig.linear_constraints.get_range_values()
    colnames = orig.variables.get_names()
    rownames = orig.linear_constraints.get_names()

    cmatbeg, cmatcnt, cmatind, cmatval = rows2cmat(rows, numcols)
    start = copy.get_time()
    copy.copylp(numcols,
                numrows,
                objsense,
                obj,
                rhs,
                senses,
                cmatbeg,
                cmatcnt,
                cmatind,
                cmatval,
                lb,
                ub,
                range_values,
                colnames,
                rownames)
    return copy


class CopyLPTests(CplexTestCase):

    def testAfiro(self):
        filename = "../../data/afiro.mps"
        with self._newCplex() as orig:
            orig.read(filename)
            with self._newCplex() as copy:
                copylp(orig, copy)
                self.compare(orig, copy)

    def testNoConstraints(self):
        with self._newCplex() as orig:
            orig.objective.set_sense(orig.objective.sense.maximize)
            orig.variables.add(obj=[1.0, 2.0, 3.0],
                               lb=[-10.0, -20.0, -30.0],
                               ub=[1.0e6, 2.0e6, 3.0e6],
                               names=['foo1', 'foo_2', 'fo__3'])
            with self._newCplex() as copy:
                copylp(orig, copy)
                self.compare(orig, copy)

    def testNoVariables(self):
        with self._newCplex() as orig:
            orig.objective.set_sense(orig.objective.sense.minimize)
            orig.linear_constraints.add(
                senses="LGER",
                rhs=[100.0, 200.0, 300.0, 400.0],
                range_values=[0.0, 0.0, 0.0, -100.0],
                names=['bar1', 'bar_2', 'ba__3', 'b4'])
            with self._newCplex() as copy:
                copylp(orig, copy)
                self.compare(orig, copy)

    def testCopyMIP(self):
        """Expecting that we can copy LP attributes from a MIP."""
        filename = "../../data/caso8.mps"
        with self._newCplex() as orig:
            orig.read(filename)
            self.assertEqual(orig.problem_type.MILP,
                             orig.get_problem_type())
            with self._newCplex() as copy:
                copylp(orig, copy)
                self.assertEqual(copy.problem_type.LP,
                                 copy.get_problem_type())
                self.compare(orig, copy)

    def testMinColArgs(self):
        """Test that we can specify the minimum number of column
        arguments.
        """
        with self._newCplex() as cpx:
            cpx.copylp(3, 0, obj=[1.0] * 3, lb=[0.0] * 3,
                       ub=[cplex.infinity] * 3)
            self.assertEqual(cpx.variables.get_num(), 3)

    def testMinRowArgs(self):
        """Test that we can specify the minimum number of row arguments."""
        with self._newCplex() as cpx:
            cpx.copylp(0, 3, rhs=[1.0, 1.0, 1.0], senses="LLL")
            self.assertEqual(cpx.linear_constraints.get_num(), 3)

    def testMinArgs(self):
        """Test that we can specify the minimum number of arguments if
        numcols > 0 and numrows > 0.

        No objsense, range_values, colnames, or rownames.
        """
        numcols = 3
        numrows = 3
        obj = [1.0] * numcols
        rhs = [1.0] * numrows
        senses = "L" * numrows
        lb = [0.0] * numcols
        ub = [cplex.infinity] * numcols
        lin_expr = [SparsePair(ind=[0], val=[1.0]),
                    SparsePair(ind=[1], val=[1.0]),
                    SparsePair(ind=[2], val=[1.0])]
        matbeg, matcnt, matind, matval = rows2cmat(lin_expr, numcols)
        with self._newCplex() as orig, self._newCplex() as copy:
            orig.variables.add(obj=obj, lb=lb, ub=ub)
            orig.linear_constraints.add(lin_expr=lin_expr,
                                        senses=senses,
                                        rhs=rhs)
            copy.copylp(numcols,
                        numrows,
                        obj=obj,
                        rhs=rhs,
                        senses=senses,
                        matbeg=matbeg,
                        matcnt=matcnt,
                        matind=matind,
                        matval=matval,
                        lb=lb,
                        ub=ub)
            self.compare(orig, copy)

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testWrongNumColArgs(self):
        good = 3
        bad = 2
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=good,
                           numrows=0,
                           obj=[1.0] * good,
                           lb=[0.0] * bad,
                           ub=[cplex.infinity] * good)
            self.assertIn("inconsistent argument lengths: obj, lb, ub, colnames",
                          str(err.exception))

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testWrongNumRowArgs(self):
        good = 3
        bad = 2
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=0,
                           numrows=good,
                           rhs=[1.0] * good,
                           senses="L" * bad)
            self.assertIn(
                "inconsistent argument lengths: rhs, senses, range_values, rownames",
                str(err.exception))

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testWrongNumMatBegArgs(self):
        ncols, nrows = 3, 2
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=ncols,
                           numrows=nrows,
                           obj=[1.0] * ncols,
                           rhs=[1.0] * nrows,
                           senses="L" * nrows,
                           matbeg=[0, 1, 2],
                           matcnt=[1], # wrong
                           matind=[0, 1, 0],
                           matval=[1.0, 1.0, 1.0],
                           lb=[0.0] * ncols,
                           ub=[cplex.infinity] * ncols)
            self.assertIn("inconsistent argument lengths: matbeg, matcnt",
                          str(err.exception))

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testWrongNumMatIndArgs(self):
        ncols, nrows = 3, 2
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=ncols,
                           numrows=nrows,
                           obj=[1.0] * ncols,
                           rhs=[1.0] * nrows,
                           senses="L" * nrows,
                           matbeg=[0, 1, 2],
                           matcnt=[1, 1, 1],
                           matind=[0, 1, 0],
                           matval=[1.0, 1.0], # wrong
                           lb=[0.0] * ncols,
                           ub=[cplex.infinity] * ncols)
            self.assertIn("inconsistent argument lengths: matind, matval",
                          str(err.exception))

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testWrongLengthMatInd(self):
        ncols, nrows = 3, 2
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=ncols,
                           numrows=nrows,
                           obj=[1.0] * ncols,
                           rhs=[1.0] * nrows,
                           senses="L" * nrows,
                           matbeg=[0, 1, 2],
                           matcnt=[1, 1, 1],
                           # Special case where len(matind) != sum(matcnt).
                           matind=[0, 1], # wrong nnz
                           matval=[1.0, 1.0], # wrong nnz
                           lb=[0.0] * ncols,
                           ub=[cplex.infinity] * ncols)
            self.assertIn("inconsistent arguments: len(matind) != sum(matcnt)",
                          str(err.exception))

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testWrongLengthMatVal(self):
        ncols, nrows = 3, 2
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=ncols,
                           numrows=nrows,
                           obj=[1.0] * ncols,
                           rhs=[1.0] * nrows,
                           senses="L" * nrows,
                           matbeg=[0, 1, 2],
                           matcnt=[1, 1, 1],
                           matind=[0, 1, 0],
                           matval=[1.0, 1.0], # wrong nnz
                           lb=[0.0] * ncols,
                           ub=[cplex.infinity] * ncols)
            self.assertIn("inconsistent argument lengths", str(err.exception))

    def testCopyOver(self):
        """Test that we can copy over an existing problem."""
        with self._newCplex() as caso8:
            caso8.read("../../data/caso8.mps")
            self.assertEqual(caso8.problem_type.MILP,
                             caso8.get_problem_type())
            with self._newCplex() as afiro:
                afiro.read("../../data/afiro.mps")
                self.assertEqual(afiro.problem_type.LP,
                                 afiro.get_problem_type())
                copylp(afiro, caso8)
                self.assertEqual(caso8.problem_type.LP,
                                 caso8.get_problem_type())
                self.compare(afiro, caso8)

    def testNegativeCols(self):
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=-1,
                           numrows=0,
                           obj=[1.0],
                           lb=[0.0],
                           ub=[cplex.infinity])
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_BAD_ARGUMENT)

    def testNegativeRows(self):
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=0,
                           numrows=-1,
                           rhs=[1.0],
                           senses="L")
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_BAD_ARGUMENT)

    def testBadObjSense(self):
        with self._newCplex() as cpx:
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=1,
                           numrows=0,
                           objsense=2,  # <- bad obj sense
                           obj=[1.0],
                           lb=[0.0],
                           ub=[cplex.infinity])
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_BAD_ARGUMENT)

    def testObjWithNaN(self):
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=1,
                           numrows=0,
                           obj=[NaN],  # <- NaN in obj
                           lb=[0.0],
                           ub=[cplex.infinity])
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_NAN)

    def testRhsWithNaN(self):
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=0,
                           numrows=1,
                           rhs=[NaN],  # <- NaN in rhs
                           senses="L")
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_NAN)

    def testBadRowSense(self):
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=0,
                           numrows=1,
                           rhs=[1.0],
                           senses="?")  # <- bad sense
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_BAD_SENSE)

    def testBadRange_Values(self):
        with self._newCplex() as cpx:
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=0,
                           numrows=1,
                           rhs=[1.0],
                           senses="R",
                           range_values=[Inf])  # <- Inf in range_values
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_DBL_MAX)

    def testBadRowIndex(self):
        ncols, nrows = 3, 2
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=ncols,
                           numrows=nrows,
                           obj=[1.0] * ncols,
                           rhs=[1.0] * nrows,
                           senses="L" * nrows,
                           matbeg=[0, 1, 2],
                           matcnt=[1, 1, 1],
                           matind=[0, 1, 2],  # <- bad row index
                           matval=[1.0, 1.0, 1.0],
                           lb=[0.0] * ncols,
                           ub=[cplex.infinity] * ncols)
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_ROW_INDEX_RANGE)

    def testCoefWithNan(self):
        ncols, nrows = 3, 2
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=ncols,
                           numrows=nrows,
                           obj=[1.0] * ncols,
                           rhs=[1.0] * nrows,
                           senses="L" * nrows,
                           matbeg=[0, 1, 2],
                           matcnt=[1, 1, 1],
                           matind=[0, 1, 0],
                           matval=[1.0, 1.0, NaN],  # <- bad coef
                           lb=[0.0] * ncols,
                           ub=[cplex.infinity] * ncols)
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_NAN)

    def testLBWithNaN(self):
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=1,
                           numrows=0,
                           obj=[1.0],
                           lb=[NaN],  # <- NaN in lb
                           ub=[cplex.infinity])
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_NAN)

    def testUBWithNaN(self):
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            with self.assertRaises(CplexSolverError) as err:
                cpx.copylp(numcols=1,
                           numrows=0,
                           obj=[1.0],
                           lb=[0.0],
                           ub=[NaN])  # <- NaN in ub
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_NAN)

    def testNumColsLenObj(self):
        ncols = 3
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=ncols,
                           numrows=0,
                           # Special case where numcols > len(obj)
                           obj=[1.0],  # <- len(obj) too small
                           lb=[0.0],
                           ub=[cplex.infinity])
            self.assertIn("inconsistent arguments: numcols > len(obj)",
                          str(err.exception))

    def testNumRowsLenRhs(self):
        nrows = 2
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError) as err:
                cpx.copylp(numcols=0,
                           numrows=nrows,
                           # Special case where numrows > len(rhs)
                           rhs=[1.0],  # <- len(rhs) too small
                           senses="L")
            self.assertIn("inconsistent arguments: numrows > len(rhs)",
                          str(err.exception))

    def compare(self, orig, copy):
        self.assertEqual(orig.variables.get_num(),
                         copy.variables.get_num())
        self.assertEqual(orig.linear_constraints.get_num(),
                         copy.linear_constraints.get_num())
        self.assertEqual(orig.objective.get_sense(),
                         copy.objective.get_sense())
        self.assertEqual(orig.objective.get_linear(),
                         copy.objective.get_linear())
        self.assertEqual(orig.linear_constraints.get_rhs(),
                         copy.linear_constraints.get_rhs())
        self.assertEqual(orig.linear_constraints.get_senses(),
                         copy.linear_constraints.get_senses())
        orig_rows = orig.linear_constraints.get_rows()
        copy_rows = copy.linear_constraints.get_rows()
        for orow, crow in zip(orig_rows, copy_rows):
            self.assertEqual(orow.ind, crow.ind)
            self.assertEqual(orow.val, crow.val)
        self.assertEqual(orig.variables.get_lower_bounds(),
                         copy.variables.get_lower_bounds())
        self.assertEqual(orig.variables.get_upper_bounds(),
                         copy.variables.get_upper_bounds())
        self.assertEqual(orig.linear_constraints.get_range_values(),
                         copy.linear_constraints.get_range_values())


def main():
    unittest.main()


if __name__ == '__main__':
    main()
