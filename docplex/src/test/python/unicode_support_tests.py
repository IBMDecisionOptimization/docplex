'''This test suite test that docplex.mp supports unicode correctly.
In particular, regular tests only tests the default python string type.

This test suite adds the testing of actual unicode strings in py2.
It also make sure that unicode strings with non ASCII chars are correctly
supported

@author: kong
'''
import os
import unittest

from docplex.mp.model import Model
from docplex.mp.context import Context, BaseContext
from docplex.mp.utils import get_logger

from testutils import get_settings_path, append_to_sys_path, temporary_directory

CHI = u'\u0FBE'  # Tibetan Ku Ru Kha, looks like 'X'
LITTLE_Z = u'\u0291'  # Latin Small Letter Z with Curl, like 'z'
TURNED_Y = u'\u028E'  # Latin Small Letter Turned Y
CONSTRAINT = u"\u0188\u014F\u0273\u1E67\u1E6F\u027D\u0201\u1F76\u1F22\u0567"  # leet version of constraint
FILENAME = u"\u1E1F\u01D0\u1E3B\u0451"  # leet version of "file"
MODEL = u"\u1E3F\u00F3\u1E0B\u0207\u013A"  # leet version of "model"


class UnicodeSupportTests(unittest.TestCase):
    def setUp(self):
        append_to_sys_path(get_settings_path())

        logger = get_logger("unicode_support_tests", True)
        self.context = Context(docplex_tests=BaseContext(),
                               docloud_api_tests=BaseContext())
        self.context.read_settings(logger=logger)
        self.context.solver.agent = 'cplex'

    def create_empty_model(self):
        return Model(solver_agent='cplex')

    def create_model_with_non_ascii_names(self):
        m = self.create_empty_model()
        self.populate_model_with_non_ascii_names(m)
        return m

    def populate_model_with_non_ascii_names(self, m):
        x = m.continuous_var(name=CHI)
        y = m.continuous_var(name=TURNED_Y)
        m.add_constraint(x >= 18)
        m.add_constraint(x >= y + 17, CONSTRAINT)
        m.minimize(y)

    def test_model_with_non_ascii_names(self):
        m = self.create_model_with_non_ascii_names()
        ict = m.get_constraint_by_name(CONSTRAINT)
        self.assertEqual(u"%s: %s >= %s+17" % (CONSTRAINT, CHI, TURNED_Y), str(ict))
        m.solve()
        x = m.get_var_by_name(CHI)
        self.assertGreaterEqual(float(x), 18)

    def test_model_save_lp(self):
        # Create local model
        m = Model(name=MODEL, solver_agent='cplex')
        self.populate_model_with_non_ascii_names(m)
        # tmpdir & all
        write_ok = False
        try:
            with temporary_directory(self.context) as tmpdir:
                # save the model as lp, with exotic chars in filename
                lp_name = os.path.join(tmpdir, FILENAME + ".lp")
                print("exporting file as (utf-8): %s" % lp_name.encode("utf-8"))
                m.export_as_lp(lp_name)
                # now save with no filename
                expected_filename = os.path.join(tmpdir, MODEL + ".lp")
                print("exporting with path = tmpdir, no basename, should use model name (utf-8): %s" % expected_filename.encode("utf-8"))
                m.export_as_lp(path=tmpdir)
                write_ok = True
        except WindowsError:
            # with some mounted volumes, we need that hack to prevent filesystem
            # errors
            if not write_ok:
                raise

    def test_save_solution_json(self):
        # Create local model
        m = Model(name=MODEL, solver_agent='cplex')
        self.populate_model_with_non_ascii_names(m)
        m.solve()
        solution = m.solution
        # tmpdir & all
        write_ok = False
        try:
            with temporary_directory(self.context) as tmpdir:
                wb_filename = os.path.join(tmpdir, FILENAME + "_solution.json")
                print("exporting file as (utf-8): %s" % wb_filename.encode("utf-8"))
                with open(wb_filename, "wb") as oss:
                    solution.export(oss, format='json')
                write_ok = True
        except WindowsError:
            # with some mounted volumes, we need that hack to prevent filesystem
            # errors
            if not write_ok:
                raise

    def test_save_solution_xml(self):
        # Create local model
        m = Model(name=MODEL, solver_agent='cplex')
        self.populate_model_with_non_ascii_names(m)
        m.solve()
        solution = m.solution
        # tmpdir & all
        write_ok = False
        try:
            with temporary_directory(self.context) as tmpdir:
                wb_filename = os.path.join(tmpdir, FILENAME + "_solution.xml")
                print("exporting file as (utf-8): %s" % wb_filename.encode("utf-8"))
                with open(wb_filename, "wb") as oss:
                    solution.export(oss, format='json')
                write_ok = True
        except WindowsError:
            # with some mounted volumes, we need that hack to prevent filesystem
            # errors
            if not write_ok:
                raise

if __name__ == "__main__":
    unittest.main()