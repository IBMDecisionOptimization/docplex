'''This test suite test that docplex.mp supports unicode correctly.
In particular, regular tests only tests the default python string type.

This test suite adds the testing of actual unicode strings in py2.
It also make sure that unicode strings with non ASCII chars are correctly
supported

@author: kong
'''
import unittest
import requests

from docplex.mp.model import Model
from docplex.mp.context import Context, BaseContext
from docplex.mp.utils import get_logger

from unicode_support_tests import UnicodeSupportTests
from testutils import get_settings_path, append_to_sys_path

from requests.packages.urllib3.exceptions import InsecureRequestWarning

class UnicodeSupportLongtests(UnicodeSupportTests):
    def setUp(self):
        append_to_sys_path(get_settings_path())

        logger = get_logger("unicode_support_longtests", True)
        self.context = Context(docplex_tests=BaseContext(),
                               docloud_api_tests=BaseContext())
        self.context.read_settings(logger=logger)
        # RTC-32079: Disable warning for certificate that can't be verified
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    def create_empty_model(self):
        return Model(solver_agent=self.context.solver.agent,
                     context=self.context)

    @unittest.skip("Local only")
    def test_model_save_lp(self):
        pass

    @unittest.skip("Local only")
    def test_save_solution_json(self):
        pass
    
    @unittest.skip("Local only")
    def test_save_solution_xml(self):
        pass

def load_tests(loader, tests, pattern):
    '''load_tests protocol
    '''
    return unittest.defaultTestLoader.loadTestsFromTestCase(UnicodeSupportLongtests)


if __name__ == "__main__":
    unittest.main()