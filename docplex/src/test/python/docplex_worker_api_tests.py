'''
Created on Apr 22, 2016

@author: kong
'''
import unittest
import shutil
import tempfile
import socket
import os
from collections import OrderedDict

from docplex.mp.context import Context, BaseContext
from docplex.mp.utils import get_logger

try:
    from docplex_wml.worker.clientapi import set_output_attachments, get_available_core_count
except ImportError:
    set_output_attachments = None

from testutils import get_settings_path, temporary_directory


@unittest.skipIf(set_output_attachments is None,
                 "Skipped because no docplex.worler.clientapi found")
class DocplexWorkerAPITests(unittest.TestCase):
    """The test suite to test the docplex.docplex_wml.worker client API
    """
    def setUp(self):
        file_list = ["cplex_config.py",
                     "cplex_config_" + socket.gethostname() + ".py"]
        abs_file_list = map(lambda x: os.path.join(get_settings_path(), x),
                            file_list)
        logger = get_logger("docplex_worker_api_tests", True)
        self.context = Context(docplex_tests=BaseContext(),
                               docloud_api_tests=BaseContext())
        self.context.read_settings(file_list=abs_file_list,
                                   logger=logger)
        # accelerator for some very often used keys

    def test_get_available_core_count(self):
        """Test the get_available_core_count() method()
        """
        cores = get_available_core_count()
        # we can safely assume that the number of available cores is greater than 0...
        self.assertGreater(cores, 0, "Should be greater than 0")

    def test_output_attachments(self):
        """Test that the set_output_attachments methods work.

        Obviously, this can only test the local methods. *.py running on
        DOcplexcloud will need their own tests, and we will need a tests for
        notebooks (if this makes sense)
        """
        with temporary_directory(self.context) as tmpdir:

            data_csv = os.path.join(tmpdir, "data.csv")
            data_json = os.path.join(tmpdir, "data.json")

            outputs = OrderedDict()
            with open(data_csv, "w") as csv:
                csv.write("1,2,3\n")
            with open(data_json, "w") as jso:
                jso.write("{'att':0}")
            outputs['data.csv'] = data_csv
            outputs['data.json'] = data_json
            set_output_attachments(outputs)
            # TODO: for the moment, just call the method.
            # TODO: Need to do more tests when the API is fully specified



if __name__ == "__main__":
    unittest.main()