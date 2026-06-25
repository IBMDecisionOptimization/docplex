import os
import tempfile
import unittest

from testutils import DocplexAbstractTest
from docplex.mp.model_reader import ModelReader
from docplex.cp.model import CpoModel

class TestBuildOplModel(DocplexAbstractTest):
    """Unit tests for build_opl_model() for both CPO (CP Optimizer) and CPLEX (MP) models."""

    def setUp(self):
        """Prepare temporary mod/dat files and model contents."""
        self.tmpdir = tempfile.TemporaryDirectory()

        # ------------------------------------------------------------------
        # CPLEX OPL Model (Mathematical Programming)
        # ------------------------------------------------------------------
        self.mod_cplex_content = """
            // CPLEX OPL Model for linear optimization
            int MAXVAL = ...;   
            int LIMIT  = ...;   

            dvar int x in 0..MAXVAL;
            dvar int y in 0..MAXVAL;

            // Objective
            maximize x * y;

            // Constraints
            subject to {
            x + y <= LIMIT;
            x >= y;
            }
        """

        self.dat_cplex_content = """
            MAXVAL = 10;
            LIMIT  = 12;
        """

        self.dat_json_cplex_content = """
            {
            "MAXVAL": 10,
            "LIMIT": 12
            }
        """   

        # ------------------------------------------------------------------
        # CPO OPL Model (Constraint Programming)
        # ------------------------------------------------------------------
        self.mod_cpo_content = """
            // CPO OPL Model using CP Optimizer
            using CP;
            int MAXVAL = ...;   
            int LIMIT  = ...;   

            dvar int x in 0..MAXVAL;
            dvar int y in 0..MAXVAL;

            // Objective
            maximize x * y;

            // Constraints
            subject to {
            x + y <= LIMIT;
            x >= y;
            }
        """

        self.dat_cpo_content = """
            MAXVAL = 10;
            LIMIT  = 12;
        """

        self.dat_json_cpo_content = """
            {
            "MAXVAL": 10,
            "LIMIT": 12
            }
        """   

        # Write to temp files for path-based tests
        self.cplex_mod_path = os.path.join(self.tmpdir.name, "model_cplex.mod")
        self.cplex_dat_path = os.path.join(self.tmpdir.name, "data_cplex.dat")
        self.cplex_dat_json_path = os.path.join(self.tmpdir.name, "data_cplex_json.json")

        self.cpo_mod_path = os.path.join(self.tmpdir.name, "model_cpo.mod")
        self.cpo_dat_path = os.path.join(self.tmpdir.name, "data_cpo.dat")
        self.cpo_dat_json_path = os.path.join(self.tmpdir.name, "data_cpo_json.json")

        with open(self.cplex_mod_path, "w") as f:
            f.write(self.mod_cplex_content)
        with open(self.cplex_dat_path, "w") as f:
            f.write(self.dat_cplex_content)
        with open(self.cplex_dat_json_path, "w") as f:
            f.write(self.dat_json_cplex_content)

        with open(self.cpo_mod_path, "w") as f:
            f.write(self.mod_cpo_content)
        with open(self.cpo_dat_path, "w") as f:
            f.write(self.dat_cpo_content)
        with open(self.cpo_dat_json_path, "w") as f:
            f.write(self.dat_json_cpo_content)

    def tearDown(self):
        """Cleanup temporary files."""
        self.tmpdir.cleanup()

    # ----------------------------------------------------------------------
    # TESTS FOR CPLEX MODELS (ModelReader.build_opl_model)
    # ----------------------------------------------------------------------

    def test_cplex_build_from_paths(self):
        """Build CPLEX model from .mod and .dat file paths."""
        mdl = ModelReader.build_opl_model(self.cplex_mod_path, self.cplex_dat_path)
        self.assertIsNotNone(mdl)

    def test_cplex_build_from_strings(self):
        """Build CPLEX model from mod/dat strings."""
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, self.dat_cplex_content)
        self.assertIsNotNone(mdl)

    def test_cplex_build_with_dict(self):
        """Build CPLEX model using dictionary data."""
        data = {"MAXVAL": 10, "LIMIT": 12}
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, data)
        self.assertIsNotNone(mdl)

    def test_cplex_build_with_kwargs(self):
        """Build CPLEX model with keyword arguments."""
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, MAXVAL=10, LIMIT=12)
        self.assertIsNotNone(mdl)

    def test_cplex_build_with_mix_data_and_kwargs(self):
        """Build CPLEX model with dict and kwargs merged."""
        data = {"MAXVAL": 10}
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, data, LIMIT=12)
        self.assertIsNotNone(mdl)

    def test_cplex_build_with_json_data(self):
        """ Build CPLEX model from .mod and .json file paths. """
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, self.cplex_dat_json_path)
        self.assertIsNotNone(mdl)

    def test_cplex_build_with_json_string(self):
        """ Build CPLEX model from .mod and JSON string. """
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, self.dat_json_cplex_content)
        self.assertIsNotNone(mdl)

    # ----------------------------------------------------------------------
    # TESTS FOR CPO MODELS (CpoModel.build_opl_model)
    # ----------------------------------------------------------------------

    def test_cpo_build_from_paths(self):
        """Build CPO model from .mod and .dat file paths."""
        mdl = CpoModel()
        mdl.build_opl_model(self.cpo_mod_path, self.cpo_dat_path)
        self.assertIsNotNone(mdl)
        self.assertIsInstance(mdl, CpoModel)

    def test_cpo_build_from_strings(self):
        """Build CPO model from mod/dat strings."""
        mdl = CpoModel()
        mdl.build_opl_model(self.mod_cpo_content, self.dat_cpo_content)
        self.assertIsNotNone(mdl)
        self.assertIsInstance(mdl, CpoModel)

    def test_cpo_build_with_dict(self):
        """Build CPO model using dictionary data."""
        mdl = CpoModel()
        data = {"MAXVAL": 10, "LIMIT": 12}
        mdl.build_opl_model(self.mod_cpo_content, data)
        self.assertIsNotNone(mdl)

    def test_cpo_build_with_kwargs(self):
        """Build CPO model with keyword arguments."""
        mdl = CpoModel()
        mdl.build_opl_model(self.mod_cpo_content, MAXVAL=10, LIMIT=12)
        self.assertIsNotNone(mdl)

    def test_cpo_build_with_mix_data_and_kwargs(self):
        """Build CPO model with dict and kwargs merged."""
        mdl = CpoModel()
        data = {"MAXVAL": 10}
        mdl.build_opl_model(self.mod_cpo_content, data, LIMIT=12)
        self.assertIsNotNone(mdl)
    
    def test_cpo_build_with_json_data(self):
        """ Build CPO model from .mod and .json file paths. """
        mdl = CpoModel()
        mdl.build_opl_model(self.mod_cpo_content, self.cpo_dat_json_path)
        self.assertIsNotNone(mdl)

    def test_cpo_build_with_json_string(self):
        """ Build CPO model from .mod and JSON string. """
        mdl = CpoModel()
        mdl.build_opl_model(self.mod_cpo_content, self.dat_json_cpo_content)
        self.assertIsNotNone(mdl)

    # ----------------------------------------------------------------------
    # ERROR HANDLING TESTS
    # ----------------------------------------------------------------------

    def test_invalid_mod_file(self):
        """Missing .mod file should raise FileNotFoundError."""
        bad_path = os.path.join(self.tmpdir.name, "missing.mod")
        mdl = CpoModel()
        with self.assertRaises(FileNotFoundError):
            mdl.build_opl_model(bad_path)
    
    def test_invalid_data_type(self):
        """Invalid data input should raise TypeError."""
        mdl = CpoModel()
        with self.assertRaises(TypeError):
            mdl.build_opl_model(self.mod_cpo_content, data=1234)

    def test_invalid_mod_syntax(self):
        """Invalid OPL syntax should raise a RuntimeError with parse error message."""
        # Create temporary invalid .mod content
        bad_mod_content = """
            using CP;

            int MAXVAL = 10;
            int LIMIT  = 5;

            // Syntax error here: 'in' used twice
            dvar in x in 0..MAXVAL;

            maximize x;
            subject to {
                x <= LIMIT;
            }
        """
        mdl = CpoModel()

        with self.assertRaises(RuntimeError) as ctx:
            mdl.build_opl_model(bad_mod_content)
        err_msg = str(ctx.exception)
        # Optional: verify that error message contains parse info
        self.assertIn("ERROR[PARSE", str(ctx.exception))
        self.assertIn("syntax error", str(ctx.exception))
        self.assertIn("Impossible to load model", str(ctx.exception))
        self.assertRegex(err_msg, r"at\s+\d+:")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
