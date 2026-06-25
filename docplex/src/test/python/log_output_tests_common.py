'''
Created on Feb 15, 2016

@author: kong
'''
import os
import re
import unittest
import uuid

try:
    import tee
except ImportError:
    tee = None

from docplex.mp.context import Context
from io import StringIO

from examples.modeling.love_hearts import build_hearts

from testutils import temporary_directory, make_test_temp_path, RedirectedOutputToStringContext


# we can add more log output recognition pattern if needed
LOG_PATTERNS = ['Tried aggregator [0-9]+ times?\.']
COMPILED_PATTERNS = [re.compile(p) for p in LOG_PATTERNS]


class LogOutputTestsCommon(unittest.TestCase):

    def create_model(self, **kwargs):
        return build_hearts(r=9, **kwargs)

    def get_context(self):
        context = Context.make_default_context()
        context.cplex_parameters.timelimit = 30
        return context

    def check_that_log_is_ok(self, log_lines):
        """ Returns true if s looks like it is a log output from nurses
        """
        ok = False
        lines = log_lines.split('\n')
        if len(lines) <= 20:
            # Nurses triggers at least 20 lines of logs
            return False

        for l in lines:
            # check that at least one line starts with that
            for cp in COMPILED_PATTERNS:
                match = cp.search(l)
                ok = ok or match

        return ok

    def test_override_at_construction(self):
        """ Test that when we provide a log_output to the model's constructor
        kwarg, it is set into the model's context
        """

        with self.create_model(log_output=True) as mdl:
            with RedirectedOutputToStringContext() as oss:
                mdl.solve()
            log = oss.get_str()
        self.assertTrue(self.check_that_log_is_ok(log),
                        "Should have a log output but got: [%s]" % log)
        self.assertTrue(mdl.context.solver.log_output)

    def test_override_true_at_solve(self):
        """ Test that we can override the log_output value at solve time.
        """
        with self.create_model(log_output=False) as mdl:
            with RedirectedOutputToStringContext() as oss:
                mdl.solve(log_output=True)
            log = oss.get_str()
        self.assertTrue(self.check_that_log_is_ok(log),
                        "Should have a log output but got: [%s]" % log)
        self.assertFalse(mdl.log_output)

    def test_override_false_at_solve(self):
        """ Test that we can override the log_output value to False
        """
        with self.create_model(log_output=True) as mdl:
            with RedirectedOutputToStringContext() as oss:
                mdl.solve(log_output=False)
            log = oss.get_str()
        self.assertEqual('', log)
        self.assertTrue(mdl.log_output)

    def test_with_filename(self):
        """ Test that we can set a filename and that this works.
        """
        context = self.get_context()
        with temporary_directory(context) as outputdir:
            logfile_path = os.path.join(outputdir, "test.log")
            with self.create_model(log_output=False) as mdl:
                mdl.solve(log_output=logfile_path)
            self.assertTrue(os.path.exists(logfile_path))

            with open(logfile_path, "r") as logs:
                l = logs.read()
            self.assertFalse(mdl.log_output)
            self.assertTrue('Found incumbent' in l, 'Could not found line with "Found incumbent" in text: %s' % l)
            self.assertTrue('Tried aggregator 1 time' in l,
                            'Could not found line with "Tried aggregator 1 time" in text: %s' % l)

    def test_set_in_context_before_build(self):
        """ Test that when log_output is set in the context before Model init
        everything is fine.
        """
        context = self.get_context()
        context.solver.log_output = True
        with self.create_model(context=context) as mdl:
            with RedirectedOutputToStringContext() as oss:
                mdl.solve()
            log = oss.get_str()
            self.assertTrue(mdl.log_output)
            self.assertTrue(self.check_that_log_is_ok(log),
                        "Should have a log output but got: [%s]" % log)

    def test_set_in_context_after_build(self):
        """ Test that when log_output is set between Model init and solve(),
        everything is fine.
        """
        context = self.get_context()
        with self.create_model(context=context) as mdl:
            context.solver.log_output = True
            self.assertTrue(mdl.log_output)
            with RedirectedOutputToStringContext() as oss:
                mdl.solve()
            log = oss.get_str()
            self.assertTrue(self.check_that_log_is_ok(log),
                            "Should have a log output but got: [%s]" % log)

    def test_log_output_setter_false(self):
        """ Testing that when set to false, it works too
        """
        with self.create_model(log_output=False) as mdl:
            mdl.log_output = True
            mdl.error_handler.set_output_level("warning")
            with RedirectedOutputToStringContext() as oss:
                mdl.log_output = False
                mdl.solve()
            log = oss.get_str()
        self.assertEqual('', log)

    def test_value_after_override(self):
        """ Tests that after solve(log_output=a_value), log_output is set
        back to its previous value.
        """
        context = self.get_context()
        context.solver.log_output = False
        mdl = self.create_model(context=context)

        # This one should capture something

        with RedirectedOutputToStringContext() as oss1:
            mdl.solve(log_output=True)
        log1 = oss1.get_str()
        self.assertTrue(self.check_that_log_is_ok(log1),
                        "Should have a log output but got: [%s]" % log1)

        self.assertEqual(mdl.context.solver.log_output, False,
                         "mdl.context.solver.log_output property should be set")

        # This one should capture nothing

        with RedirectedOutputToStringContext() as oss2:
            mdl.solve(clean_before_solve=True)
        log2 = oss2.get_str()
        self.assertEqual('', log2)

    @unittest.skipUnless(tee, ' this test requires module tee')
    def test_tee(self):
        with self.create_model() as mdl:
            filename = "temp_solve_tee_{0}".format(uuid.uuid1())
            temp_path = make_test_temp_path(basename=filename, extension=".log")
            mytee =  tee.StdoutTee(temp_path, buff=1024)
            with RedirectedOutputToStringContext() as oss:
                with mytee:
                    mdl.solve(log_output=mytee)
                    mytee.flush()
                    log_out = oss.get_str()
                self.assertIn('incumbent', log_out)
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, 'r') as logf:
                log_file = logf.read()
                self.assertEqual(log_file, log_out)

if __name__ == "__main__":
    unittest.main(verbosity=3)
