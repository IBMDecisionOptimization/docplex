'''
Created on Apr 30, 2015

@author: kong
'''
import contextlib
import errno
import filecmp
import os
import os.path
from os.path import abspath, join
from os.path import dirname as dn
import platform
import shutil
import sys
import tempfile
import unittest
from unittest import SkipTest

import traceback
import re

from docplex.util.status import JobSolveStatus

from docplex.mp.context import Context, BaseContext
from docplex.mp.cplex_engine import CplexEngine
from docplex.mp.model import Model
from docplex.mp.utils import get_logger
from docplex.mp.engine import IndexerEngine
from docplex.mp.sdetails import SolveDetails
from docplex.mp.error_handler import docplex_fatal

try:
    # try to locate cplex wrapper
    from cplex import Cplex

except ImportError:
    Cplex = None

def in_travis():
    return os.environ.get("TRAVIS") is not None

def make_abs_path(nb_ups, relative_path, filename=None):
    p = os.path.abspath(__file__)
    # go up (..) nb_up times
    for _ in range(nb_ups):
        p = dn(p)
    if filename:
        return os.path.join(p, relative_path, filename)
    else:
        return os.path.join(p, relative_path)


def append_to_sys_path(new_path):
    """Append ``new_path`` to sys.path if it is not already there
    """
    if new_path not in sys.path:
        print("appending %s" % new_path)
        sys.path.append(new_path)


def get_test_data_path():
    # Assume that docplex_tests_utils.py is in src/tests/python
    # data is  src/tests/data    from os.path import
    f = os.path.abspath(__file__)
    p = os.path.abspath(os.path.join(dn(dn(f)), "data"))
    return p


def get_settings_path():
    # Assume that docplex_tests_utils.py is in src/tests/python
    # data is  src/tests/data
    # f = os.path.abspath(__file__)
    # p = os.path.abspath(os.path.join(dn(dn(f)), "settings"))
    # return p
    return make_abs_path(nb_ups=2, relative_path="settings")


def get_samples_delivery_path():
    # home is either DCPLEX_HOME or 5 ups + "docplex"
    docplex_home = os.environ.get('DOCPLEX_HOME') or make_abs_path(nb_ups=5, relative_path="docplex")

    # samples can be in two places:
    # docplex_tests/../docplex/src/samples/examples/delivery on dev setup
    # target/libs/python/examples/delivery on jenkins
    # lets return the one that is sys.path (on jenkins, the dev delivery dir is not in the python path)
    #
    # first look for delivery in docplex/src:
    abs_pythonpath = [os.path.abspath(f).lower() for f in sys.path]
    src_samples = join(docplex_home, "src", "samples")
    if src_samples.lower() in abs_pythonpath:
        return src_samples
    # look in target
    delivery_in_target = make_abs_path(nb_ups=4, relative_path="target/docplex-mp-all-samples")
    if delivery_in_target.lower() in abs_pythonpath:
        return delivery_in_target
    return None

def make_abs_samples_root_dir():
    return make_abs_path(nb_ups=5, relative_path="docplex/src/samples/examples")

def make_abs_samples_data_path(filename):
    return make_abs_path(nb_ups=5, relative_path="docplex/src/samples/examples/data", filename=filename)

def make_abs_mreader_path(filename):
    return make_abs_path(nb_ups=2, relative_path="data/model_reader_tests", filename=filename)

def make_abs_cplex_data_path(filename):
    return make_abs_path(nb_ups=2, relative_path="data/cplex_samples", filename=filename)

def make_abs_cplex_data_dir():
    return make_abs_path(nb_ups=2, relative_path="data/cplex_samples")

def make_abs_mipstart_data_dir():
    return make_abs_path(nb_ups=2, relative_path="data/mip_starts")

def make_abs_cpo_lpmodels_dir():
    return make_abs_path(nb_ups=5, relative_path="docpo/UnitTests/lpmodels")

def get_test_temp_dir():
    # this is easily overriden with TMPDIR env variable
    return tempfile.gettempdir()

def make_test_temp_path(basename, extension=None):
    filename = basename
    if extension:
        filename += extension
    tempdir = tempfile.gettempdir()
    path = os.path.join(tempdir, filename)
    return path


def open_universal_newline(filename, mode):
    try:
        # try the python 3 syntax
        return open(filename, mode=mode, newline=None)
    except TypeError as te:
        te_s = str(te)
        if "'newline'" in te_s:
            # so open does not have a newline parameter -> python 2, use "U"
            # mode
            return open(filename, mode=mode + "U")
        else:
            # for other errors, just raise them
            raise


class StdoutGrabber(object):
    def __init__(self, io):
        self.io = io

    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self.io

    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_stdout


def silent_remove(path):
    if path is not None:
        try:
            os.remove(path)
        except OSError:
            pass


def read_and_check_with_cplex(case, amodel, ext_path, remove_after=True):
    """ Checks a model file (in whatever format) with Cplex.

        Args:
            case: the unit test case
            amodel: the instance of Docplex model from which the file was exported.
            ext_path:  the path
            remove_after: an optional boolean to indicate whether to remove the file after reading.

        """
    if not os.path.exists(ext_path):
        return

    if not Cplex:
        return

    cpx = Cplex()

    try:
        cpx.read(ext_path)
    except:
        raise Exception("File %s fails to load in CPLEX" % ext_path)

    cpx_nb_vars = cpx.variables.get_num()
    cpx_nb_linear_cts = cpx.linear_constraints.get_num()
    # build set of used vars
    used_var_indices = set()  # v._index for ct in amodel.iter_constraints() for v in ct.iter_variables())
    for ct in amodel.iter_constraints():
        for v in ct.iter_variables():
            used_var_indices.add(v._index)
    for ov in amodel.objective_expr.iter_variables():
        used_var_indices.add(ov._index)
    # piecewise vars are always used
    for pc in amodel.iter_pwl_constraints():
        for v, k in pc.expr.iter_terms():
            if k:
                used_var_indices.add(v._index)
    # sos vars are used
    for asos in amodel.iter_sos():
        for vsos in asos.iter_variables():
            used_var_indices.add(vsos.index)

    number_of_used_vars = len(used_var_indices)

    expected_nb_vars = number_of_used_vars + amodel.number_of_range_constraints

    #case.assertLessEqual(cpx_nb_vars, expected_nb_vars)

    expected_nb_constraints = amodel.number_of_linear_constraints
    # add quad cts which are not actually quadratic
    nb_of_false_quadcts = 0
    for qc in amodel.iter_quadratic_constraints():
        if not qc.has_net_quadratic_term():
            nb_of_false_quadcts += 1
    expected_nb_constraints += nb_of_false_quadcts

    case.assertEqual(cpx_nb_linear_cts, expected_nb_constraints)

    if amodel.solution is None:
        return

    # we have a solution, check it is the same
    cpx.solve()
    docplex_obj = amodel.objective_value
    cpx_obj = cpx.solution.get_objective_value()
    if docplex_obj == cpx_obj:
        pass
    else:
        abs_pyopl_obj = abs(docplex_obj)
        abs_cpx_obj = abs(cpx_obj)
        ok = True
        min_objs = min(abs_pyopl_obj, abs_cpx_obj)
        max_objs = max(abs_pyopl_obj, abs_cpx_obj)
        if min_objs == 0:
            if max_objs >= 1e-4:
                ok = False
        else:
            ok = (max_objs / min_objs) <= 1.01
        case.assertTrue(ok, "bad objectives, docplex is {0}, cplex is {1}".format(docplex_obj, cpx_obj))

        try:
            del cpx
        except:
            pass

        if remove_after:
            silent_remove(ext_path)


try:
    import docplex_wml.worker.solvehook

    class MockSolveHook(docplex_wml.worker.solvehook.SolveHook):
        """ Mock solve hook for tests
        """
        def notify_start_solve(self, mdl, model_statistics):
            keys = ["binary", "integer", "continuous",
                    "ler, req", "ge",
                    "range",
                    "indicator"]
            for k, c in zip(keys, model_statistics.as_tuple()):
                print("{0}: {1}".format(k, c))

        def notify_end_solve(self, mdl, has_solution, status, obj, var_value_dict):
            # Notifies the end of a solve
            # Args: has_solution: boolean, True if solve returned a solution
            #     status: an enumerated value of type JobSolveStatus
            #     obj: the objective value if solved ok, else irrelevant.
            #     attributes: a dictionary of variable names to values in solution.
            solve_verb = "succeeded" if has_solution else "failed"
            print("* solve {0}, status is: {1}".format(solve_verb, status))
            if has_solution:
                print("*  objective is {0}".format(obj))

        def update_solve_details(self, details):
            for detail_key, detail_value in details:
                print("* details: {0}: {1}".format(detail_key, detail_value))
except ImportError:
    pass


class TargetLocalSolver(object):
    def use_cloud(self):
        return False

    def get_context(self):
        return None



class RedirectedOutputContext(object):
    def __init__(self, new_out, error_handler=None):
        if new_out is not None:
            self._of = new_out
        else:
            self._of = sys.stdout
        self._saved_out = sys.stdout
        self.error_handler = error_handler

    def __enter__(self):
        self._saved_out = sys.stdout
        sys.stdout = self._of
        if self.error_handler:
            self.error_handler.suspend()
        return self._of

    # noinspection PyUnusedLocal
    def __exit__(self, atype, avalue, atraceback):
        sys.stdout = self._saved_out
        if self.error_handler:
            self.error_handler.flush()


class RedirectedOutputToStringContext(RedirectedOutputContext):
    def __init__(self, error_handler=None):
        # explicit !!!!
        from io import StringIO
        self._oss = StringIO()
        RedirectedOutputContext.__init__(self, new_out=self._oss, error_handler=error_handler)

    def __enter__(self):
        RedirectedOutputContext.__enter__(self)
        # return self as we need to extract the string after exit
        return self

    def get_str(self):
        return self._oss.getvalue()

    def __del__(self):
        # kill the stringio on deletion
        self._oss = None


# noinspection PyPep8Naming
def skipIfCplexCE(test_item):
    '''Skips the annotated test if CPLEX CE is detected.

    It also skips the tests if it run and that code=1016 is returned as a
    DOCplexException message.
    '''
    if not isinstance(test_item, type):
        def run_and_backtrack(*args, **kwargs):
            try:
                test_item(*args, **kwargs)
            except AssertionError:
                raise
            except Exception as de:
                try:
                    message = str(de)
                except AttributeError:
                    message = str(de)
                if 'code=1016' in message or 'error1016' in message:
                    raise SkipTest("Test skipped of CPLEX CE limitations")
                else:
                    raise

        return run_and_backtrack
    else:
        raise ValueError("This decorator works only on tests")


class DocplexAbstractTest(unittest.TestCase):
    GlobalEngineCode = 'cplex'

    def setUp(self):
        self.model = Model()

    def tearDown(self):
        self.model.end()
        self.model = None


def pprint_as_string(m):
    with RedirectedOutputToStringContext() as out:
        m.prettyprint()
    return out.get_str()


@contextlib.contextmanager
def working_directory(path):
    '''A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    Returns:
        The path
    '''
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev_cwd)


@contextlib.contextmanager
def temporary_directory(context=None, delete=True):
    '''Returns a tmp directory. If context is not None and
    ``context.docplex_test.tmp`` is defined, that directory is used. Otherwise
    a suitable tmp directory is created with ``tempfile.mkdtemp()``

    When the directory is created with mkdtemp, it will be destroyed by this
    context manager.

    Returns:
        The tmp dir
    '''
    dirname = context.docplex_tests.get("tmp") if context else None
    if dirname is None:
        name = tempfile.mkdtemp()
    else:
        name = dirname
    try:
        yield name
    finally:
        # delete only if the tmp was not one given in context
        if dirname is None and delete:
            shutil.rmtree(name)

try:
    from docplex_wml.worker.solvehook import get_solve_hook, set_solve_hook
except ImportError:
    def get_solve_hook():
        return None

    def set_solve_hook(h):
        pass


class OverrideHook(object):
    '''Allows to temporarly replace the global solve hook
    '''
    def __init__(self, hook, new_env=None):
        self.set_hook = hook
        self.set_env = new_env
        self.saved_hook = None
        self.saved_env = None

    def __enter__(self):
        self.saved_hook = get_solve_hook()
        set_solve_hook(self.set_hook)
        if self.set_env:
            import docplex.util.environment
            self.saved_env = docplex.util.environment.default_environment
            docplex.util.environment.default_environment = self.set_env
        else:
            self.saved_env = None

    def __exit__(self, type, value, traceback):
        set_solve_hook(self.saved_hook)
        if self.saved_env:
            import docplex.util.environment
            docplex.util.environment.default_environment = self.saved_env


def get_calling_test(reference=None, test_pattern_re='^test_.+'):
    '''Returns the name of the calling test.

    This method scan backwards a traceback, looking for a method matching
    test_pattern_re.

    This method returns (type(reference).__name__, method_name) if reference
    is not None, otherwise it just return the name of first method found in
    the stack.
    '''
    compiled_pattern = re.compile(test_pattern_re)
    rbb = traceback.extract_stack()
    for frame in reversed(rbb):
        method_name = frame[2]
        if re.match(compiled_pattern, method_name):
            return '%s.%s' % (type(reference).__name__, method_name) if reference else method_name


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def are_functional_vars_named():
    with Model() as m:
        pwl = m.piecewise(0, [(0, 0), (0.5, 1), (1, 1)], 0, name="fractional_part")
        x = m.continuous_var(name ="x")
        y = m.continuous_var(name = "y")
        fexp = pwl(x)
        fexp_fvar_name = fexp.functional_var.name
        return fexp_fvar_name and "fractional" in fexp_fvar_name


@contextlib.contextmanager
def check_against_reference(testcase, test_suite_basedir, tmpdir=None,
                            create_reference=True):
    '''This context manager makes it easy to make tests that compare some
    results to some reference values.

    Example:

        class BasicTesting(unittes.TestCase):
             def test_against_ref(self):
                with check_against_reference(self, 'BasicTesting') as path:
                    # writes to some <tmp>/BasicTesting.output/BasicTesting.test_against_ref.out
                    write_something_to(path)
                # at this point, output is compared to:
                # src/tests/reference_output/BasicTesting.reference/BasicTesting.test_against_ref.out
                # if reference does not exists, it is created

    Args:
        testcase: The TestCase
        test_suite_basedir: The basename for output. Typically this is
           basename(__file__).replace('.py', ''). Output of test is written
           at get_test_tempdir()/test_suite_basedir.output and
           reference files are expected to be in
           docplex_tests/src/tests/reference_output/test_suite_basedir.reference
        tmpdir: temp dir to use. If not specified, this will create a temp
           dir using tempfile.mkdtemp() and that temp dir will be deleted
           after the test is run, unless the test failed.
    Returns:
        Returns a path to a file were some output can be written, then
        that output will be checked against a reference.
    '''
    # find the calling test
    calling_test = get_calling_test(testcase)
    output_name = calling_test + '.out'
    print('calling_test = %s' % calling_test)
    my_tempdir = None
    failed = True
    try:
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp()

        test_suite_outdir = test_suite_basedir + '.output'
        test_suite_refdir = test_suite_basedir + '.reference'

        output_refdir = os.path.join(tmpdir, test_suite_outdir)
        if not os.path.isdir(output_refdir):
            mkdir_p(output_refdir)
        path = os.path.join(output_refdir, output_name)
        yield path
        # check against reference
        docplex_tests_src_tests = dn(dn(abspath(__file__)))
        reference_dir = join(docplex_tests_src_tests,
                             'reference_output')
        reference_file = join(reference_dir, test_suite_refdir, output_name)
        reference_fullname = join(reference_dir, reference_file)
        if create_reference and not os.path.isfile(reference_fullname):
            print('Making new reference of %s' % path)
            reference_dirname = os.path.dirname(reference_fullname)
            if not os.path.isdir(reference_dirname):
                mkdir_p(reference_dirname)
            shutil.copy(path, reference_fullname)
            print('New reference is: %s' % reference_fullname)
        print('Comparing %s %s' % (path, reference_fullname))
        # if there are some diffs, we want to print a line that can be easily
        # cut and past for diff
        diff_tool = 'winmergeu' if platform.system() == 'Windows' else 'diff'
        testcase.assertTrue(filecmp.cmp(path, reference_fullname),
                            'files differ: %s %s %s' % (diff_tool, path, reference_fullname))
        failed = False
    finally:
        if my_tempdir and not failed:
            shutil.rmtree(my_tempdir)


class ForceProceduralCplexContext(object):

    def __init__(self, forced):
        self._saved_procedural = False
        self._forced_procedural = forced

    def __enter__(self):
        forced = self._forced_procedural
        self._saved_procedural = CplexEngine.procedural
        if self._saved_procedural != forced:
            CplexEngine.procedural = forced
        return forced

    def __exit__(self, exc_type, exc_val, exc_tb):
        CplexEngine.procedural = self._saved_procedural

class TerminatedEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but always fail, and returns None.
    def terminate(self):
        docplex_fatal("model has been terminated, no solve is possible...")

    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover

    def name(self):
        return "exception"  # pragma: no cover

    def solve(self, mdl, parameters, lex_mipstart=None, parameter_sets=None):
        # solve fails equivalent to returning None
        self.terminate()
        return None  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        self.terminate()
        return None  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        self.terminate()
        return None  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.INFEASIBLE_SOLUTION  # pragma: no cover

    def get_solve_details(self):
        return SolveDetails.make_fake_details(time=0, feasible=False)


class RaiseErrorEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but always raises an exception, this is for testing

    @staticmethod
    def _simulate_error():
        docplex_fatal("simulate exception")

    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover

    @property
    def name(self):
        return "raise"  # pragma: no cover

    def solve(self, mdl, parameters, lex_mipstart=None, lex_timelimits=None, lex_mipgaps=None):
        # solve fails equivalent to returning None
        self._simulate_error()
        return None  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        self._simulate_error()
        return None  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        self._simulate_error()
        return None  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.INFEASIBLE_SOLUTION  # pragma: no cover

    def get_solve_details(self):
        return SolveDetails.make_fake_details(time=0, feasible=False)

