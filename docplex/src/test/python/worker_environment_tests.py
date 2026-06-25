"""This specifically test that docplex.util.environment.Environment works

Created on May 20, 2016

@author: kong
"""
import collections
import json
import logging
import os
import tempfile
import unittest
import time

from docplex.mp.context import Context, BaseContext, create_default_auto_publish_context
from docplex.mp.model import Model

from docplex.util.environment import get_environment, LocalEnvironment

try:
    import docplex_wml
    from docplex_wml.worker.environment import WorkerEnvironment
    from docplex_wml.worker.logging_utils import LoggerToDocloud
except ImportError:
    docplex_wml = None

from testutils import working_directory, temporary_directory, skipIfCplexCE

OUTPUT_ATTACHMENT_NAME = "worker_environment_tests.out"
EXPECTED_OUTPUT_DATA = "This is some output"
EXPECTED_NUMBER_OF_CORES = 5  # unlikely that a real hardware has 5 core but good for testing
ARBITRARY_ATTACHMENT_NAME = 'arbitrary.txt'
ARBITRARY_FILE_CONTENT = 'arbitrary file content'

VERBOSE = False

try:
    import pandas
except ImportError:
    pandas = None

the_hook = None
try:
    from docplex_wml.worker.solvehook import SolveHook
    from testutils import OverrideHook
    HOOK_AVAILABLE = True

    # set a custom solve hook to record things
    class MySolveHook(SolveHook):
        def __init__(self):
            super(MySolveHook, self).__init__()
            self.reset()

        def reset(self):
            self.solve_details = []
            self.attachments = {}  # current attachments
            self.attachments_history = []  # As a list. Each entry is a call to set_output_attachments.

        def set_output_attachments(self, attachments):
            self.attachments.update(attachments)
            self.attachments_history.append(attachments)

        def get_parameter_value(self, name):
            return SolveHook.get_parameter_value(self, name)

        def update_solve_details(self, details):
            self.solve_details.append(details)
            if VERBOSE:
                print("="*80)
                print(json.dumps(details, indent=3))
                print("="*80)

        def get_available_core_count(self):
            return EXPECTED_NUMBER_OF_CORES  # arbitrary esoteric number

    class SolveHookWithOutputAttachmentTimestamp(MySolveHook):
        def __init__(self):
            super().__init__()

        def set_output_attachments(self, attachments):
            self.attachments.update(attachments)
            now = time.time()
            acopy = attachments.copy()
            self.attachments_history.append((now, acopy))

        def search_attachments_in_history(self, name):
            """Returns all the (ts, attachment) couples in history where attachment name
            is specified
            """
            result = []
            for ts, attachments in self.attachments_history:
                for aname, afile in attachments.items():
                    if aname == name:
                        result.append((ts, {aname: afile}))
            # return None for empty results
            return result if result else None

    the_hook = MySolveHook()
except ImportError:
    the_hook = None



def open_universal_newline(filename, mode):
    try:
        # try the python 3 syntax
        return open(filename, mode=mode, newline=None)
    except TypeError as te:
        if "'newline'" in te.message:
            # so open does not have a newline parameter -> python 2, use "U"
            # mode
            return open(filename, mode=mode + "U")
        else:
            # for other errors, just raise them
            raise


class WorkerEnvironmentTest(unittest.TestCase):
    def setUp(self):
        if the_hook:
            the_hook.reset()

        self.context = Context(docplex_tests=BaseContext(),
                               docloud_api_tests=BaseContext())
        self.context.read_settings()

    def simulate_run(self, env, output_as_context_manager=True, threads=None,
                     auto_publish_details=None, auto_publish_solution=None,
                     auto_publish=None,
                     arbitrary_attachment_name=None,
                     solve_details=None,
                     update_interval=0):
        # simulate a solve, with solve details published 10 times
        # The output file contains EXPECTED_OUTPUT_DATA as a first line
        # then the number of threads as a second line
        # Returns the contents of the solution file, if it exists
        run_solution = None

        if solve_details is None:
            solve_details = ({'index': i} for i in range(10))

        # We want notify_start to be called before hand
        env.notify_start_solve(None)

        # While solve details were designed to be called only a few times,
        # like just after a solve, we also use them for kpis.
        # just use 'solve_details' as a generator and test it
        for i in solve_details:
            env.update_solve_details(i)
            if update_interval:
                time.sleep(update_interval)
        # actually create a zero model and solve to see how much threads we got
        context = Context()
        context.solver.agent = 'zero'
        if threads:
            context.cplex_parameters.threads = threads
        model = Model(context=context, agent='zero')

        # Prepare arguments
        context.solver.auto_publish = create_default_auto_publish_context(defaults=auto_publish)
        context.solver.auto_publish.solve_details = auto_publish
        context.solver.auto_publish.result_output = auto_publish
        context.solver.auto_publish.kpi_output = auto_publish

        if os.path.isfile("solution.json"):
            os.unlink("solution.json")
        model.solve()

        if not the_hook:
            print('reading solution as local file')
            # no hook, output attachments are saved as local file
            if auto_publish or auto_publish_solution:
                self.assertTrue(os.path.isfile("solution.json"))
                with open_universal_newline("solution.json", "r") as f:
                    run_solution = f.read()
            else:
                self.assertFalse(os.path.isfile("solution.json"))
        else:
            print('reading solution as attachment')
            # hook, attachments are posted using set_output_attachment
            att = the_hook.attachments.get('solution.json')
            if auto_publish or auto_publish_solution:
                self.assertTrue(att is not None)
                with open_universal_newline(att, "r") as f:
                    run_solution = f.read()
            else:
                self.assertTrue(att is None)

        solved_params = model.get_engine().last_solved_parameters or context.cplex_parameters

        def write_test_data(f):
            f.write(EXPECTED_OUTPUT_DATA.encode("utf-8"))
            f.write("\n".encode("utf-8"))
            f.write(("%s" % solved_params.threads.get()).encode("utf-8"))
            f.write("\n".encode("utf-8"))

        # publish output
        if output_as_context_manager:
            with env.get_output_stream(OUTPUT_ATTACHMENT_NAME) as f:
                write_test_data(f)
        else:
            file = None
            try:
                file = env.get_output_stream(OUTPUT_ATTACHMENT_NAME)
                write_test_data(file)
            finally:
                if file:
                    file.close()
        # publish arbitrary output with set_output_attachment
        if arbitrary_attachment_name:
            filename = tempfile.mktemp()
            with open(filename, 'w') as f:
                f.write(ARBITRARY_FILE_CONTENT)
            env.set_output_attachment(arbitrary_attachment_name, filename)

        return run_solution

    @unittest.skipIf(the_hook is None, "docplex_wml is needed")
    def test_local_env(self):
        with OverrideHook(None, LocalEnvironment()):
            env = get_environment()

            # in local, we expect worker_environment_tests.out to be writen in cwd.
            # let's change the cwd to a tmp dir so that we can check that
            with temporary_directory(self.context) as tmpd:
                print('Using tmp dir: %s' % tmpd)
                with working_directory(tmpd):
                    if os.path.isfile(OUTPUT_ATTACHMENT_NAME):
                        os.unlink(OUTPUT_ATTACHMENT_NAME)
                    self.simulate_run(env, auto_publish_details=False,
                                      arbitrary_attachment_name=ARBITRARY_ATTACHMENT_NAME)
                    self.assertTrue(os.path.isfile(OUTPUT_ATTACHMENT_NAME),
                                    "Expecting %s to be created in %s" % (OUTPUT_ATTACHMENT_NAME, os.getcwd()))
                    with open_universal_newline(OUTPUT_ATTACHMENT_NAME, "r") as file:
                        contents = file.readline().strip()
                        self.assertEqual(contents, EXPECTED_OUTPUT_DATA, "Wrong output contents")
                    with open_universal_newline(ARBITRARY_ATTACHMENT_NAME, 'r') as file:
                        contents = file.readline().strip()
                        self.assertEqual(contents, ARBITRARY_FILE_CONTENT, "Wrong data")

    @unittest.skipIf(not the_hook or not docplex_wml, "No hook available, skipping")
    def test_worker_env_threads_above_limit(self):
        """This test test that when we solve with a thread limit > env.get_available_cor_cont(),
        the effective limit is the one found in env.
        """
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):

            env = get_environment()

            # we set the threads to 8, but we should have EXPECTED_NUMBER_OF_CORES
            # because of the limits
            self.simulate_run(env, threads=8, auto_publish=False)

            # test output
            self.assertEqual(len(the_hook.attachments), 1, "Wrong number of attachments")
            with open_universal_newline(the_hook.attachments[OUTPUT_ATTACHMENT_NAME], "r") as file:
                contents = file.read().split('\n')
                self.assertEqual(contents[0], EXPECTED_OUTPUT_DATA, "Wrong output contents")
                self.assertEqual(contents[1], str(EXPECTED_NUMBER_OF_CORES), "Wrong number of cores")

    @unittest.skipIf(not the_hook, "No hook available, skipping")
    def test_worker_env_threads_below_limit(self):
        """This test test that when we solve with a thread limit < env.get_available_core_count(),
        the effective limit is the one that is set.
        """
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):
            # very simple test case
            env = get_environment()

            # we set the threads to 8, but we should have EXPECTED_NUMBER_OF_CORES
            # because of the limits
            self.simulate_run(env, threads=3, auto_publish=False)

            # test output
            self.assertEqual(len(the_hook.attachments), 1, "Wrong number of attachments")
            with open_universal_newline(the_hook.attachments[OUTPUT_ATTACHMENT_NAME], "r") as file:
                contents = file.read().split('\n')
                self.assertEqual(contents[0], EXPECTED_OUTPUT_DATA, "Wrong output contents")
                self.assertEqual(contents[1], str(3), "Wrong number of cores")

    @unittest.skipIf(not the_hook, "No hook available, skipping")
    def test_worker_env_output_as_context_manager(self):
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):
            env = get_environment()

            self.simulate_run(env, auto_publish=False)

            # test solve details
            # Solve details number is 10 because no automatic publish
            self.assertEqual(len(the_hook.solve_details), 10, "Wrong number of solve details")
            # test output
            self.assertEqual(len(the_hook.attachments), 1, "Wrong number of attachments")
            with open_universal_newline(the_hook.attachments[OUTPUT_ATTACHMENT_NAME], "r") as file:
                contents = file.read().split('\n')
                self.assertEqual(contents[0], EXPECTED_OUTPUT_DATA, "Wrong output contents")
                self.assertEqual(contents[1], str(EXPECTED_NUMBER_OF_CORES), "Wrong number of cores")

    @unittest.skipIf(not the_hook, "No hook available, skipping")
    def test_worker_env_output_with_close(self):
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):
            env = get_environment()

            self.simulate_run(env, output_as_context_manager=False, auto_publish=False,
                              arbitrary_attachment_name=ARBITRARY_ATTACHMENT_NAME)

            # test solve details
            # Solve details number is 10 because no automatic publish
            self.assertEqual(len(the_hook.solve_details), 10, "Wrong number of solve details: %s" % the_hook.solve_details)
            # test output
            self.assertEqual(len(the_hook.attachments), 2, "Wrong number of attachments")
            with open_universal_newline(the_hook.attachments[OUTPUT_ATTACHMENT_NAME], "r") as file:
                contents = file.read().split('\n')
                self.assertEqual(contents[0], EXPECTED_OUTPUT_DATA, "Wrong output contents")
                self.assertEqual(contents[1], str(EXPECTED_NUMBER_OF_CORES), "Wrong number of cores")
            with open_universal_newline(the_hook.attachments[ARBITRARY_ATTACHMENT_NAME], 'r') as file:
                        contents = file.readline().strip()
                        self.assertEqual(contents, ARBITRARY_FILE_CONTENT, "Wrong data")


    @unittest.skipIf(not the_hook, "No hook available, skipping")
    def test_stop_callback_value(self):
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):
            env = get_environment()

            # the worker calls us this way
            the_hook.stop_callback()

    @unittest.skipIf(not the_hook, "No hook available, skipping")
    def test_worker_env_auto_publish_details_and_solution(self):
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):
            env = get_environment()

            solution = self.simulate_run(env, auto_publish=True)
            print("solution = %s" % solution)
            #self.assertTrue("CPLEXSolution" in solution, "This does not look like a CPLEX solution")
            #self.assertTrue('"objectiveValue"' in solution, "This does not look like a CPLEX solution")
            #self.assertTrue('"problemName"' in solution, "This does not look like a CPLEX solution")

            # test solve details
            # 11 because 10 from our call + 1 from engine (automatic).
            # self.assertEqual(len(the_hook.solve_details), 11, "Wrong number of solve details")
            # test output
            # There must be 3 attachments: the solution.json + OUTPUT_ATTACHMENT_NAME + kpis.csv
            # But we actually only have 2: kpis.csv is not written since it is empty
            print('attachements = %s' % the_hook.attachments)
            self.assertEqual(len(the_hook.attachments), 3, "Wrong number of attachments") #stats, solution and test.out
            with open_universal_newline(the_hook.attachments[OUTPUT_ATTACHMENT_NAME], "r") as file:
                contents = file.read().split('\n')
            self.assertEqual(contents[0], EXPECTED_OUTPUT_DATA, "Wrong output contents")
            self.assertEqual(contents[1], str(EXPECTED_NUMBER_OF_CORES), "Wrong number of cores")

    @unittest.skipUnless(pandas, 'Skipping test since panda is not here')
    def df_io_tests(self):
        import numpy as np
        df = pandas.DataFrame(np.random.randn(50, 10))
        import docplex.util.environment as environment

        def cmp_df(df1, df2):
            for r1, r2 in zip(df1.itertuples(), df2.itertuples()):
                for x1, x2 in zip(r1, r2):
                    if not np.isclose(x1, x2):
                        print('Are different: ', x1, x2)
                        return False
            return True

        with temporary_directory(self.context) as tmpd:
            print('Using tmp dir: %s' % tmpd)
            with working_directory(tmpd):
                for ext in ('csv', 'msg'):
                    name = 'io_tests.%s' % ext
                    print('Testing format: %s' % ext)
                    environment.write_df(df, name)
                    rdf = environment.read_df(name)
                    self.assertTrue(cmp_df(df, rdf),
                                    'While testing %s io, those matrix should be equal +/- epsilon:\nMatrix 1:\n%s\nMatrix 2:\n%s' % (ext, df, rdf))

    @unittest.skipIf(not the_hook, "No hook available, skipping")
    def test_update_solve_details_no_update(self):
        '''Testing update of solve details in non update mode'''
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):
            env = get_environment()
            env.update_solve_details_dict = False

            solve_details_generator = ({'A': 1}, {'B': 2})

            solution = self.simulate_run(env, auto_publish=True,
                                         solve_details=solve_details_generator)

            # we should have 3 solve details
            # 1 with just 'A':
            sd = the_hook.solve_details
            self.assertEqual(len(sd[0]), 1, "first solve details is expected to have 1 element")
            self.assertIn('A', sd[0], 'Does not contain expected \'A\'')
            # 2 with just 'B':
            self.assertEqual(len(sd[1]), 1, "second solve details is expected to have 1 element")
            self.assertIn('B', sd[1], 'Does not contain expected \'B\'')
            # 3 has more elements, just check that it does not contain keys A & B
            self.assertNotIn('A', sd[2], 'Should not contain \'A\'')
            self.assertNotIn('B', sd[2], 'Should not contain \'B\'')

    @unittest.skipIf(not the_hook, "No hook available, skipping")
    def test_update_solve_details_with_update(self):
        '''Testing update of solve details in non update mode'''
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):
            env = get_environment()
            env.update_solve_details_dict = True

            solve_details_generator = ({'A': 1}, {'B': 2})

            solution = self.simulate_run(env, auto_publish=True,
                                         solve_details=solve_details_generator)

            # we should have 3 solve details
            # 1 with just 'A':
            sd = the_hook.solve_details
            print(sd)
            self.assertEqual(len(sd[0]), 1, "first solve details is expected to have 1 element")
            self.assertIn('A', sd[0], 'Does not contain expected \'A\'')
            # 2 contains A and B:
            self.assertEqual(len(sd[1]), 2, "second solve details is expected to have 2 element")
            self.assertIn('A', sd[1], 'Does not contain expected \'A\'')
            self.assertIn('B', sd[1], 'Does not contain expected \'B\'')
            # 3 has more elements, just check that it does not contain keys A & B
            self.assertGreater(len(sd[2]), 2, "last solve details is expected to have at least 2 element")
            self.assertIn('A', sd[2], 'Does not contain expected \'A\'')
            self.assertIn('B', sd[2], 'Does not contain expected \'B\'')

    @unittest.skipIf(not the_hook, "No hook available, skipping")
    def test_solve_details_with_history(self):
        '''Testing update of solve details with history'''
        with OverrideHook(the_hook, WorkerEnvironment(the_hook)):
            env = get_environment()
            env.update_solve_details_dict = True
            env.record_history_size = 5
            env.autoreset = False
            # force recording
            env.record_history_fields = ['PROGRESS_CURRENT_OBJECTIVE']

            solve_details_generator = ({'PROGRESS_CURRENT_OBJECTIVE': i} for i in range(12))
            start_ts = time.time()
            print('start timestamp = %s' % start_ts)
            solution = self.simulate_run(env, auto_publish=True,
                                         solve_details=solve_details_generator,
                                         update_interval=0.5)
            sd = the_hook.solve_details
            print(sd)
            for i,s in enumerate(sd):
                print('%s -> %s' % (i,s))
            # in the last sd, we should find a PROGRESS_CURRENT_OBJECTIVE.history
            last = the_hook.solve_details[-1]
            self.assertIn('PROGRESS_CURRENT_OBJECTIVE.history', last)
            histo = last.get('PROGRESS_CURRENT_OBJECTIVE.history')
            self.assertNotEqual(histo, None, 'There must be a PROGRESS_CURRENT_OBJECTIVE.history item')
            l = json.loads(histo)
            self.assertEqual(len(l), 5, 'history should have been limited to 5 items')
            self.assertEqual(l[-1][1], 11, 'history should contain last objective value')

            # solve a second time (tests that rests works)
            # we should only have one value in history after that
            env._reset_record_history(force=True)
            the_hook.solve_details = []
            solve_details_generator = [{'PROGRESS_CURRENT_OBJECTIVE': 314}]

            solution = self.simulate_run(env, auto_publish=True,
                                         solve_details=solve_details_generator,
                                         update_interval=0.5)

            # we should have 3 solve details
            # 1 with just 'A':
            sd = the_hook.solve_details
            print(sd)
            for i,s in enumerate(sd):
                print('%s -> %s' % (i,s))
            # in the last sd, we should find a PROGRESS_CURRENT_OBJECTIVE.history
            last = the_hook.solve_details[-1]
            self.assertIn('PROGRESS_CURRENT_OBJECTIVE.history', last)
            histo = last.get('PROGRESS_CURRENT_OBJECTIVE.history')
            self.assertNotEqual(histo, None, 'There must be a PROGRESS_CURRENT_OBJECTIVE.history item')
            l = json.loads(histo)
            self.assertEqual(len(l), 1, 'history should have 1 item: {0}'.format(l))
            self.assertEqual(l[-1][1], 314, 'history should contain last objective value (== 1)')

    def test_engine_log_level_and_debug_mode(self):
        # first, there must be no oaas.engineSolveLevel variable
        previous_value = None
        env_name = 'oaas.engineLogLevel'
        if env_name in os.environ:
            previous_value = os.environ[env_name]
            del os.environ[env_name]
        try:
            # at this point, there should be no oaas.engineLogLevel
            env = get_environment()
            self.assertEqual(env.get_engine_log_level(), None)
            self.assertFalse(env.is_debug_mode())
            # SEVERE
            os.environ[env_name] = 'SEVERE'
            self.assertEqual(env.get_engine_log_level(), logging.ERROR)
            self.assertFalse(env.is_debug_mode())
            # WARNING
            os.environ[env_name] = 'WARNING'
            self.assertEqual(env.get_engine_log_level(), logging.WARNING)
            self.assertFalse(env.is_debug_mode())
            # INFO
            os.environ[env_name] = 'INFO'
            self.assertEqual(env.get_engine_log_level(), logging.INFO)
            self.assertFalse(env.is_debug_mode())
            # CONFIG
            os.environ[env_name] = 'CONFIG'
            self.assertEqual(env.get_engine_log_level(), logging.INFO)
            self.assertFalse(env.is_debug_mode())
            # FINE
            os.environ[env_name] = 'FINE'
            self.assertEqual(env.get_engine_log_level(), logging.DEBUG)
            self.assertTrue(env.is_debug_mode())
            # FINER
            os.environ[env_name] = 'FINER'
            self.assertEqual(env.get_engine_log_level(), logging.DEBUG)
            self.assertTrue(env.is_debug_mode())
            # FINEST
            os.environ[env_name] = 'FINEST'
            self.assertEqual(env.get_engine_log_level(), logging.DEBUG)
            self.assertTrue(env.is_debug_mode())
            # ALL
            os.environ[env_name] = 'ALL'
            self.assertEqual(env.get_engine_log_level(), logging.DEBUG)
            self.assertTrue(env.is_debug_mode())
        finally:
            if previous_value is not None:
                os.environ[env_name] = previous_value

    @unittest.skipIf(the_hook is None, "docplex_wml is needed")
    @skipIfCplexCE
    def test_solve_details_with_ucp(self):
        """
        This tests that solve details are generated correctly for a given docplex.mp model
        :return:
        """
        timed_hook = SolveHookWithOutputAttachmentTimestamp()
        with OverrideHook(timed_hook, new_env=WorkerEnvironment(timed_hook)):
            # we make sure to make env believe it is working on docplex_wml
            # This will trigger the recording of history etc...
            restore = "IS_DODS" in os.environ
            if restore:
                saved_value = os.environ["IS_DODS"]
            os.environ["IS_DODS"] = "True"
            try:
                env = get_environment()

                env.update_solve_details_dict = True
                env.record_history_size = 10
                env.record_min_time = 0
                env.autoreset = False
                for expected in ['PROGRESS_BEST_OBJECTIVE', 'PROGRESS_CURRENT_OBJECTIVE', 'PROGRESS_GAP']:
                    self.assertIn(expected, env.record_history_fields)
                if VERBOSE:
                    print(env.record_history_fields)

                # with ucp, everything works as expected.
                from examples.modeling.generics.ucp_new import make_default_ucp_model as build_model
                # from examples.modeling.long_mip import build_long_mip as build_model

                m = build_model()
                m.context.solver.auto_publish = True

                m.solve(log_output=VERBOSE)
                # timed_hook.solve_details is the list of all generated solve details
                if VERBOSE:
                    print(len(timed_hook.solve_details))
                    for count, i in enumerate(timed_hook.solve_details):
                        print(f"solve_details[{count}]:")
                        print(json.dumps(i, indent=3))
                if VERBOSE:
                    print("Attachment history:")
                    for count, i in enumerate(timed_hook.attachments_history):
                        print(f"attachments_history[{count}]:")
                        print(json.dumps(i, indent=3))
                # this is the list of all generated kpis.csv (first element is time stamp)
                att_with_kpi = timed_hook.search_attachments_in_history('kpis.csv')
                kpi_ts = []
                if VERBOSE:
                    print("Number of kpis.csv attachments: %s" % len(att_with_kpi))
                    for count, i in enumerate(att_with_kpi):
                        print(f"{count}: {i}")
                        ts, att = i
                        kpi_ts.append(ts)
                        with open(att['kpis.csv'], "r") as r:
                            c = r.read()
                            print(c)
                att_with_stats = timed_hook.search_attachments_in_history('stats.csv')
                stats_ts = []
                if VERBOSE:
                    print("Number of stats.csv attachments: %s" % len(att_with_stats))
                    for count, i in enumerate(att_with_stats):
                        print(f"{count}: {i}")
                        ts, att = i
                        stats_ts.append(ts)
                        with open(att['stats.csv'], "r") as r:
                            c = r.read()
                            print(c)
                # solve details should have one more entry than attachments number
                # because of the final kpi publish in solve_local()
                self.assertEqual(len(timed_hook.solve_details) - 1, len(att_with_kpi),
                                 '# of solve details != published number of attachments with kpi')
                # solve details should have one more entry than attachments number
                # because of the final kpi publish in solve_local()
                self.assertEqual(len(timed_hook.solve_details) - 1, len(att_with_stats),
                                 '# of solve details != published number of attachments with stats')
                # we can do some tests on solve_details[1]:
                expected = {
                    "MODEL_DETAIL_INTEGER_VARS": 0,
                    "MODEL_DETAIL_CONTINUOUS_VARS": 3840,
                    "MODEL_DETAIL_CONSTRAINTS": 15455,
                    "MODEL_DETAIL_BOOLEAN_VARS": 3840,
                    "MODEL_DETAIL_KPIS": "[\"Total Fixed Cost\", \"Total Variable Cost\", \"Total Startup Cost\", \"Total CO2 Cost\", \"Total Nb Used\", \"Total Nb Starts\", \"Total Economic cost\"]",
                    "STAT.cplex.size.integerVariables": 0,
                    "STAT.cplex.size.continousVariables": 3840,
                    "STAT.cplex.size.linearConstraints": 15455,
                    "STAT.cplex.size.booleanVariables": 3840,
                    "STAT.cplex.size.constraints": 15455,
                    "STAT.cplex.size.quadraticConstraints": 0,
                    "STAT.cplex.size.variables": 7680,
                    "STAT.cplex.modelType": "MILP",
                    "MODEL_DETAIL_OBJECTIVE_SENSE": "minimize",
                    # "KPI.Total Fixed Cost": 213592.6629999966,
                    # "KPI.Total Variable Cost": 11059092.463149987,
                    # "KPI.Total Startup Cost": 4060.0,
                    # "KPI.Total CO2 Cost": 4658989.25,
                    # "KPI.Total Nb Used": 1910.0,
                    # "KPI.Total Nb Starts": 6.0,
                    # "KPI.Total Economic cost": 11276745.126149993,
                    "PROGRESS_CURRENT_OBJECTIVE": 15935734.37614998,
                    "PROGRESS_GAP": 0.11245153009885928,
                    "PROGRESS_BEST_OBJECTIVE": 14143736.662302924,
                    "STAT.cplex.solve.explored": 0,
                    "STAT.cplex.solve.opened": 1,
                    "STAT.cplex.solve.iterationCount": 1588,
                    # obviously we don't want to check time
                    # "STAT.cplex.solve.elapsedTime": 0.2649999999994179,
                    # do not check those as there might be some rounding errors, just check that they are here
                    # "PROGRESS_BEST_OBJECTIVE.history": "[[1594220217.9, 14143736.662302924]]",
                    # "PROGRESS_CURRENT_OBJECTIVE.history": "[[1594220217.9, 15935734.37614998]]",
                    # "PROGRESS_GAP.history": "[[1594220217.9, 0.11245153009885928]]"
                }
                sd = timed_hook.solve_details[1]
                for key, value in expected.items():
                    if isinstance(value, str):
                        self.assertEqual(value, sd[key], "We got a different value for %s" % key)
                    else:
                        eps_max = 4e-4 if key == "PROGRESS_GAP" else 2e-4
                        eps = max(eps_max, value * 1e-3)
                        self.assertAlmostEqual(value, sd[key], delta=eps, msg="We got a different value for %s" % key)
                self.assertIn("PROGRESS_BEST_OBJECTIVE.history", sd)
                self.assertIn("PROGRESS_CURRENT_OBJECTIVE.history", sd)
                self.assertIn("PROGRESS_GAP.history", sd)
                self.assertAlmostEqual(sd["STAT.cplex.solve.elapsedTime"], sd["KPI._time"])
                print("Timestamps for kpis:", kpi_ts)
                print("Timestamps for stats:", stats_ts)

            finally:
                if restore:
                    os.environ["IS_DODS"] = saved_value
                else:
                    del os.environ["IS_DODS"]

    @unittest.skipIf(the_hook is None, "docplex_wml is needed")
    @skipIfCplexCE
    def test_solve_details_with_cpo(self):
        """
        This tests that solve details are generated correctly for a given docplex.cp model
        :return:
        """
        timed_hook = SolveHookWithOutputAttachmentTimestamp()
        with OverrideHook(timed_hook, new_env=WorkerEnvironment(timed_hook)):
            # we make sure to make env believe it is working on docplex_wml
            # This will trigger the recording of history etc...
            restore = "IS_DODS" in os.environ
            if restore:
                saved_value = os.environ["IS_DODS"]
            os.environ["IS_DODS"] = "True"
            try:
                env = get_environment()

                env.update_solve_details_dict = True
                env.record_history_size = 10
                env.record_min_time = 0
                env.autoreset = False

                # build cpo model
                from docplex.cp.model import CpoModel
                import docplex.cp.solver.solver as solver
                from docplex.cp.utils import compare_natural
                from collections import deque

                # -----------------------------------------------------------------------------
                # Initialize the problem data
                # -----------------------------------------------------------------------------

                # Read problem data from a file and convert it as a list of integers
                src_tests = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                filename = src_tests + "/data/models/cpo/plant_location.data"
                data = deque()
                with open(filename, "r") as file:
                    for val in file.read().split():
                        data.append(int(val))

                # Read number of customers and locations
                nbCustomer = data.popleft()
                nbLocation = data.popleft()

                # Initialize cost. cost[c][p] = cost to deliver customer c from plant p
                cost = list([list([data.popleft() for l in range(nbLocation)]) for c in range(nbCustomer)])

                # Initialize demand of each customer
                demand = list([data.popleft() for c in range(nbCustomer)])

                # Initialize fixed cost of each location
                fixedCost = list([data.popleft() for p in range(nbLocation)])

                # Initialize capacity of each location
                capacity = list([data.popleft() for p in range(nbLocation)])

                # -----------------------------------------------------------------------------
                # Build the model
                # -----------------------------------------------------------------------------

                mdl = CpoModel()

                # Create variables identifying which location serves each customer
                cust = mdl.integer_var_list(nbCustomer, 0, nbLocation - 1, "CustomerLocation")

                # Create variables indicating which plant location is open
                open_loc = mdl.integer_var_list(nbLocation, 0, 1, "OpenLocation")

                # Create variables indicating load of each plant
                load = [mdl.integer_var(0, capacity[p], "PlantLoad_" + str(p)) for p in range(nbLocation)]

                # Associate plant openness to its load
                for p in range(nbLocation):
                    mdl.add(open_loc[p] == (load[p] > 0))

                # Add constraints
                mdl.add(mdl.pack(load, cust, demand))

                # Add objective
                obj = mdl.scal_prod(fixedCost, open_loc)
                for c in range(nbCustomer):
                    obj += mdl.element(cust[c], cost[c])
                mdl.add(mdl.minimize(obj))

                # Add KPIs
                sol_version = solver.get_solver_version()
                if compare_natural(sol_version, '12.9') >= 0:
                    mdl.add_kpi(mdl.sum(demand) / mdl.scal_prod(open_loc, capacity), "Occupancy")
                    mdl.add_kpi(mdl.min([load[l] / capacity[l] + (1 - open_loc[l]) for l in range(nbLocation)]),
                                "Min occupancy")

                msol = mdl.solve(TimeLimit=10, trace_log=False)
                # timed_hook.solve_details is the list of all generated solve details
                if VERBOSE:
                    print(len(timed_hook.solve_details))
                    for count, i in enumerate(timed_hook.solve_details):
                        print(f"solve_details[{count}]:")
                        print(json.dumps(i, indent=3))
                if VERBOSE:
                    print("Attachment history:")
                    for count, i in enumerate(timed_hook.attachments_history):
                        print(f"attachments_history[{count}]:")
                        print(json.dumps(i, indent=3))
                # this is the list of all generated kpis.csv (first element is time stamp)
                att_with_kpi = timed_hook.search_attachments_in_history('kpis.csv')
                kpi_ts = []
                if VERBOSE:
                    print("Number of kpis.csv attachments: %s" % len(att_with_kpi))
                for count, i in enumerate(att_with_kpi):
                    if VERBOSE:
                        print(f"{count}: {i}")
                    ts, att = i
                    kpi_ts.append(ts)
                    if VERBOSE:
                        with open(att['kpis.csv'], "r") as r:
                            c = r.read()
                            print(c)
                att_with_stats = timed_hook.search_attachments_in_history('stats.csv')
                stats_ts = []
                if VERBOSE:
                    print("Number of stats.csv attachments: %s" % len(att_with_stats))
                for count, i in enumerate(att_with_stats):
                    if VERBOSE:
                        print(f"{count}: {i}")
                    ts, att = i
                    stats_ts.append(ts)
                    if VERBOSE:
                        with open(att['stats.csv'], "r") as r:
                            c = r.read()
                            print(c)
                att_with_solution = timed_hook.search_attachments_in_history('solution.json')
                solution_ts = []
                if VERBOSE:
                    print("Number of solution.json attachments: %s" % len(att_with_solution))
                for count, i in enumerate(att_with_solution):
                    ts, att = i
                    solution_ts.append(ts)

                '''
                # solve details should have one more entry than attachments number
                # because of the final kpi publish in solve_local()
                self.assertEqual(len(timed_hook.solve_details) - 1, len(att_with_kpi),
                                 '# of solve details != published number of attachments with kpi')
                # solve details should have one more entry than attachments number
                # because of the final kpi publish in solve_local()
                self.assertEqual(len(timed_hook.solve_details) - 1, len(att_with_stats),
                                 '# of solve details != published number of attachments with stats')
                '''
                if VERBOSE:
                    print("Timestamps for kpis:", kpi_ts)
                    print("Timestamps for stats:", stats_ts)

                    print("Number of solve details:", len(timed_hook.solve_details))
                    print("Number of kpis.csv", len(kpi_ts))
                    print("Number of stats.csv", len(stats_ts))
                    print("Number of solution.json", len(solution_ts))

                # for cp, we must have the same number of solve details, kpis, stats, and solution
                num_details = len(timed_hook.solve_details)
                self.assertEqual(num_details, len(kpi_ts), "# of solve details != # of kpi attachments")
                self.assertEqual(num_details, len(stats_ts), "# of solve details != # of stats attachments")
                self.assertEqual(num_details, len(solution_ts), "# of solve details != # of solution attachments")
            finally:
                if restore:
                    os.environ["IS_DODS"] = saved_value
                else:
                    del os.environ["IS_DODS"]

    @unittest.skipIf(the_hook is None, "docplex_wml is needed")
    def test_logging_issue_gh312(self):
        ''' Non regression test for https://github.ibm.com/IBMDecisionOptimization/docplex/issues/312
        '''
        with OverrideHook(the_hook, new_env=WorkerEnvironment(the_hook)):
            # we make sure to make env believe it is working on docplex_wml
            # This will trigger the recording of history etc...
            restore = "IS_DODS" in os.environ
            if restore:
                saved_value = os.environ["IS_DODS"]
            os.environ["IS_DODS"] = "True"
            try:
                env = get_environment()

                # The logger on the python worker is a logger that converts python strings to JSON string.
                # It behaves as. this.
                # first, this transforms a log level + message to a dict, then JSON serialize the dict for the message
                def logSystem(level, message):
                    command = collections.OrderedDict()
                    command["@command"] = "LogJUL"
                    command["levelName"] = level  # string
                    command["message"] = message  # string
                    return json.dumps(command)

                # Then the logger is there to translate python logging messages to java logging messages
                # it normally calls `send()` on the `channel`. To reproduce the error, we just append the message
                # string
                class docloudlogger:
                    def __init__(self):
                        self.channel = []

                    def info(self, msg):
                        self.channel.append(logSystem("INFO", msg))

                    def warning(self, msg):
                        self.channel.append(logSystem("WARNING", msg))

                    def error(self, msg):
                        self.channel.append(logSystem("SEVERE", msg))

                    def fine(self, msg):
                        self.channel.append(logSystem("FINE", msg))

                worker_logger = docloudlogger()
                env.logger = LoggerToDocloud(worker_logger)

                from examples.delivery.modeling.diet import build_diet_model
                # from examples.modeling.long_mip import build_long_mip as build_model

                m = Model()
                build_diet_model(m)
                m.context.solver.auto_publish = True

                m.solve(log_output=VERBOSE)
                m.end()
            except TypeError as te:
                self.fail("Should not have raised error \"" + str(te) + "\"")
            finally:
                if restore:
                    os.environ["IS_DODS"] = saved_value
                else:
                    del os.environ["IS_DODS"]


if __name__ == "__main__":
    unittest.main(verbosity=3)
