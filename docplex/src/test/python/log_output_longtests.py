'''
Created on Feb 15, 2016

@author: kong
'''
import sys
import unittest
import requests



from docplex.mp.context import Context, BaseContext
from docplex.mp.utils import get_logger
from io import StringIO

from examples.delivery.modeling.nurses import *
from examples.modeling.love_hearts import build_hearts


from testutils import get_settings_path, StdoutGrabber

from log_output_tests_common import LogOutputTestsCommon

from requests.packages.urllib3.exceptions import InsecureRequestWarning

class LogOutputLongTest(LogOutputTestsCommon):
    def create_model(self, **kwargs):
        if 'context' not in kwargs:
            kwargs['context'] = self.get_context()
        return build_hearts(r=12, **kwargs)

    def get_context(self):
        logger = get_logger("log_output_longtests", True)
        context = Context(docplex_tests=BaseContext(),
                          docloud_api_tests=BaseContext())
        context.read_settings(logger=logger)
        context.solver.agent = 'local'
        # needed because sometimes solve of nurses can be fast
        context.solver.docloud.log_poll_interval = 2
        context.cplex_parameters.timelimit = 25
        return context

    def setUp(self):
        # makes sure that src/tests/settings are in the path
        sys.path.append(get_settings_path())

        # RTC-32079: Disable warning for certificate that can't be verified
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    def populate_simple_model(self, model):
        # populates a simple model with very simple stuf
        x = model.integer_var(name="x")
        y = model.integer_var(name="y")
        model.add_constraint(x == 5)
        model.add_constraint((x + y) == 11)
        model.minimize(y)
        return model

    def populate_solve_check_simple_model(self, model, **kwargs):
        out = StringIO()
        self.populate_simple_model(model)
        with StdoutGrabber(out):
            sol = model.solve(**kwargs)
            self.assertEqual(sol['y'], 6)
        v = out.getvalue()
        self.assertTrue('Found incumbent of value 6.000000' in v)
        self.assertTrue('integer optimal solution (101)' in v)
        print("--- LOG CONTENTS ---\n%s\n-------" % v)



if __name__ == "__main__":
    unittest.main(verbosity=3)
