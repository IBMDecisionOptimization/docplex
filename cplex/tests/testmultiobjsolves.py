"""A collection of utilies for testing multiobjective optimization
in new cplex.

Several methods are implemented to create a multiobjective model from
a single objective one:
  - make a single blended objective by randomly breaking the original one.
  - Add objectives to make the solution lexmin.
  - Add random objectives (be careful that it may create unbounded objectives)

The command line tool with no arguments runs a collection of unittests.
Otherwise it can be passed a file with a model understood by Cplex
as argument and test it with one of these methods.

The tests can be skipped with option -s, a file with the multiobjective
model can be exported using the -e option.

It has four options:
       -h print this help
       --method=[dummy_blender, lexico, random] method to create multiobjective
         model
       -s skip the tests
       -e export model file with multiobjective.
       --nobj number of objective in tranformed model.
"""
import getopt
import unittest
from random import sample, uniform
import sys
import os
from cplex import infinity as cpxinfty
from cplextestcase import CplexTestCase


class MultiObjSolveTesting(CplexTestCase):
    """ Class for unittest of multiobjective solves done by creating
        various multiple objectives for a single objective model.
    """
    class ObjUtilException(Exception):
        """Exception when we are not able to build the objective"""
        pass

    def break_obj(self, obj, nobj=3, nonunitweights=True):
        """
        Break objective function (dense list of coefficient) into nobj
        lists by randomly putting non-zeroes into each lists.
        If nonunitweights is True also put some random weights for each of the
        objectives (scale down each objective so that the result
        remains the same)
        """
        objnnz = {i for i, v in enumerate(obj) if v != 0}
        if nobj > len(objnnz):
            raise self.ObjUtilException
        target_nnz = int(len(objnnz)/nobj)
        objs = list()
        if nonunitweights:
            weight = [int(uniform(10, 100))/10 for i in range(nobj)]
        else:
            weight = [1.]*nobj
        for i in range(nobj - 1):
            this_obj = sample(list(objnnz), target_nnz)
            objnnz = objnnz - set(this_obj)
            objs.append(sorted([(j, obj[j]/weight[i]) for j in this_obj],
                               key=lambda x: x[0]))
        objs.append(sorted([(j, obj[j]/weight[-1]) for j in objnnz],
                           key=lambda x: x[0]))
        obj_check = [0]*len(obj)
        for o, w in zip(objs, weight):
            for (i, v) in o:
                obj_check[i] = v*w
        self.assertListsAlmostEqual(obj_check, obj)
        return objs, weight

    def objective_split(self, c, nobj=3, priorities=None):
        """
        Function to split objective in several parts with different priorities
        Parameter 'priorities' should a list of size nobj of priorities
        """
        if not priorities:
            priorities = [1]*nobj
        if len(priorities) != nobj:
            raise Exception("Parameter priorities should be a list of "
                            "nobj priorities")
        # Break the objective
        objs, weight = self.break_obj(c.objective.get_linear(), nobj)

        # Zero out current objective
        c.multiobj.set_num(1)
        c.objective.set_linear(zip(range(c.variables.get_num()),
                                   [0]*c.variables.get_num()))

        # Set new number of objectives
        c.multiobj.set_num(nobj)

        # Fill out new objectives
        for i in range(nobj):
            c.multiobj.set_linear(i, objs[i])
            c.multiobj.set_priority(i, priorities[i])
            c.multiobj.set_weight(i, weight=weight[i])

    @staticmethod
    def rand_obj(n, k=1./3., low=-10, up=10):
        """ Define one random objective
        """
        obj = []
        for i in sample(range(n), int(n*k)):
            obj.append((i, uniform(low, up)))
        return obj

    def def_objs(self, c, nobj):
        """Define 'nobj' random objectives for the instance passed in.

        (Also define epsilon's but for now they are all 0).
        """
        n = c.variables.get_num()

        # Uncomment to set the same random seed as cplex. Why?
        # seed(c.parameters.randomseed.get())
        objs = [self.rand_obj(n) for i in range(nobj)]
        eps = [0 for i in range(nobj)]

        return (objs, eps)

    def add_rand_objs(self, c, nobj):
        """
        Add random objectives to Cplex instance.
        """
        objs, eps = self.def_objs(c, nobj=nobj)

        c.multiobj.set_num(nobj)
        for i in range(nobj-1):
            c.multiobj.set_linear(i+1, zip(range(c.variables.get_num()),
                                           [0]*c.variables.get_num()))
            c.multiobj.set_linear(i+1, objs[i])
            c.multiobj.set_priority(i+1, i+1)
            c.multiobj.set_weight(i+1, 1.)
            c.multiobj.set_reltol(i+1, eps[i])
        c.multiobj.set_priority(0, nobj*2+2)

    @staticmethod
    def add_lex_objs(c, nobj):
        """
        Add lexicographic order secondary objectives.

        Only minimize variables with a lower bound to avoid issues with
        infinity.
        """
        # Get a list of bounded variables
        bdvars = [i for i, lb in
                  enumerate(c.variables.get_lower_bounds())
                  if lb > - cpxinfty]
        nobj = min(nobj, len(bdvars))
        c.multiobj.set_num(nobj)
        bdvars = sorted(bdvars, reverse=True)
        sense = c.objective.get_sense()
        for i in range(nobj-1):
            c.multiobj.set_linear(i+1, zip(range(c.variables.get_num()),
                                           [0]*c.variables.get_num()))
            c.multiobj.set_linear(i+1, bdvars[i], 1.)
            c.multiobj.set_priority(i+1, i+1)
            if sense == c.objective.sense.minimize:
                c.multiobj.set_weight(i+1, 1.)
            else:
                c.multiobj.set_weight(i+1, -1.)
        c.multiobj.set_priority(0, nobj*2+2)

    def consistent_sols(self, c, cmulti):
        """Check that solutions are consistent:
           - c is a cplex model with single objective
           - cmulti is a cplex model with multiple but composite objective
             of highest priority is the same as c

           Both models should have the same objective value for the highest
           priority.
           Then, cmulti should be lexicographically smaller for the others.
        """
        # Solve original problem and get objective value
        c.solve()
        # Solve modified probem and get objective value
        cmulti.solve()

        # Now we need to get the priorities of the objective
        nobj = cmulti.multiobj.get_num()
        objpriority = [cmulti.multiobj.get_priority(i) for i in range(nobj)]
        priorities = set(objpriority)
        priorities = sorted(list(priorities), reverse=True)

        # Finally we can do our checks
        get_objval_by_prio = cmulti.solution.multiobj.get_objval_by_priority
        objval1 = c.solution.get_objective_value()
        objval2 = get_objval_by_prio(priorities[0])
        self.assertAlmostEqual(objval1, objval2)

        if len(priorities) < 2:
            # cmulti has only one objective, we are done
            return

        # Get solution vector to recompute other objectives for the
        # initial problem
        x = c.solution.get_values()
        for i in priorities[1:]:
            objval2 = cmulti.solution.multiobj.get_objval_by_priority(i)
            # We need to get the objective coefficient and recompute from the
            # solution vector x
            objval1 = 0
            error   = 1.
            for k in range(nobj):
                if objpriority[k] == i:
                    this_obj = cmulti.multiobj.get_offset(k)

                    lin = cmulti.multiobj.get_linear(k)
                    this_obj += sum([x[j]*lin[j]
                                     for j in range(c.variables.get_num())])
                    for j in range(c.variables.get_num()):
                        error += abs(x[j]*lin[j])

                    objval1 += cmulti.multiobj.get_weight(k)*this_obj
            objval1_tight = objval1 + 1e-10* error
            if objval2 > objval1_tight:
                print("Objective values are not consistent.")
                print("Multiobjective value {} should "
                      "be smaller than sinlge obj {}"
                      "(priority {})".format(objval2, objval1, i))
            self.assertLessEqual(objval2, objval1_tight)
            if objval1 > objval2 + 1e-12:
                break

    def reformulate_obj_and_compare(self, fname, nobj, method,
                                    export=False, skip_test=False):
        """Reformulate the model adding or splitting the objective"""
        methods = {'dummy_blender': self.objective_split,
                   'lexico': self.add_lex_objs,
                   'random': self.add_rand_objs}
        if method not in methods:
            print("Unknown method for creating multiobj.")
            print("Method should be one of:")
            print(methods)
            raise Exception("Unknown method")

        with self._newCplex(fname) as c:
            with self._newCplex(c) as cmulti:
                try:
                    methods[method](cmulti, nobj=nobj)

                    if export:
                        bname = ''.join(fname.split('/')[-1].split('.')[:-2])
                        pbtype = c.problem_type[c.get_problem_type()].lower()
                        if pbtype == 'milp':
                            pbtype = 'mip'
                        if fname.split('/')[0] == 'models':
                            if pbtype != fname.split('/')[1]:
                                pbtype_dir = ''.join(fname.split('/')[:1])
                                print("pbtype {} does not match file location"
                                      " {}".format(pbtype, pbtype_dir))
                                pbtype = fname.split('/')[1]
                        if method not in os.listdir('.'):
                            os.mkdir(method)
                        if pbtype not in os.listdir(method):
                            os.mkdir('{}/{}'.format(method, pbtype))
                        cmulti.write('{}/{}/{}.lp.gz'.
                                     format(method, pbtype, bname))
                    if not skip_test:
                        self.consistent_sols(c=c, cmulti=cmulti)
                except self.ObjUtilException as err:
                    print("Not enough coefficients to break objective")
                    raise err

    def testp0033_blend(self):
        self.reformulate_obj_and_compare("../../data/p0033.mps",
                                         5, "dummy_blender")

    def testp0033_lexico(self):
        self.reformulate_obj_and_compare("../../data/p0033.mps", 33, "lexico")

    def testp0033_random(self):
        self.reformulate_obj_and_compare("../../data/p0033.mps", 5, "random")

    def testcase1_blend(self):
        self.reformulate_obj_and_compare("../../data/case1.lp",
                                         2, "dummy_blender")

    # Should we expect user cuts and lazy constraints and user cuts to
    # work? So far it seems ok.
    def testcase1_lexico(self):
        self.reformulate_obj_and_compare("../../data/case1.lp", 10, "lexico")

    def testcase1_random(self):
        self.reformulate_obj_and_compare("../../data/case1.lp", 5, "random")

    def testcase5_blend(self):
        self.reformulate_obj_and_compare("../../data/case5.lp",
                                         2, "dummy_blender")

    def testcase5_lexico(self):
        self.reformulate_obj_and_compare("../../data/case5.lp", 10, "lexico")

    def testcase5_random(self):
        self.reformulate_obj_and_compare("../../data/case5.lp", 5, "random")

    def testlpprog_blend(self):
        self.reformulate_obj_and_compare(self._getResource("tests/data/lpprog.lp"), 5,
                                         "dummy_blender")

    def testlpprog_lexico(self):
        self.reformulate_obj_and_compare("../../data/lpprog.lp", 10, "lexico")

    def testlpprog_random(self):
        self.reformulate_obj_and_compare("../../data/lpprog.lp", 5, "random")


def usage():
    """ Print tool usage """
    print(__doc__)


def main():
    """ Main function of command line tool"""
    # Command line tool
    method = 'dummy_blender'
    export = False
    skip_test = False
    nobj = 10
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hes', ["method=", "nobj="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    if len(args) != 1:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            exit(0)
        if opt == '-e':
            export = True
        if opt == '-s':
            skip_test = True
        if opt == '--method':
            method = arg
        if opt == '--nobj':
            nobj = int(arg)

    try:
        testing = MultiObjSolveTesting()
        testing.reformulate_obj_and_compare(fname=args[0], nobj=nobj,
                                            method=method,
                                            export=export, skip_test=skip_test)
    except Exception as err:
        print("Error encountered")
        print(err)
        raise


if __name__ == '__main__':
    # If no argument run unittest
    if len(sys.argv) == 1:
        unittest.main()
    else:
        main()
