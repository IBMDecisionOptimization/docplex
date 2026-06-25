import os
import unittest
from docplex.cp.model import CpoModel
from docplex.mp.model_reader import ModelReader


class TestGetOplVar(unittest.TestCase):
    """Unit tests for get_opl_var() on both CPO and CPLEX models."""

    @classmethod
    def setUpClass(cls):
        # --- CPO model content ---
        cls.mod_cpo_content = """
            using CP;
            int NbMachines = ...;
            int NbTasks = ...;

            range Machines = 1..NbMachines;
            range Tasks = 1..NbTasks;

            dvar interval operations[Machines][Tasks] size 1..2;
            dvar sequence SequenceVar_135 in all(m in Machines, t in Tasks) operations[m][t];
            dvar interval workovers[Machines];

            minimize sum(m in Machines, t in Tasks) lengthOf(operations[m][t]);
            subject to {
              forall(m in Machines)
                endBeforeStart(operations[m][1], workovers[m]);
            }
        """

        cls.dat_cpo_content = """
            NbMachines = 2;
            NbTasks = 3;
        """

        # --- CPLEX model content ---
        cls.mod_cplex_content = """
            int NbMachines = ...;
            int NbTasks = ...;

            range Machines = 1..NbMachines;
            range Tasks = 1..NbTasks;

            dvar float+ operations[Machines][Tasks];
            dvar float+ workovers[Machines];

            maximize sum(m in Machines, t in Tasks) operations[m][t];
            subject to {
                forall(m in Machines)
                    sum(t in Tasks) operations[m][t] + workovers[m] <= 10;
            }
        """

        cls.dat_cplex_content = """
            NbMachines = 2;
            NbTasks = 3;
        """

    # -----------------------------------------------------------
    # Variable name extraction
    # -----------------------------------------------------------
    def test_variable_name_extraction_cpo(self):
        """Ensure variable name extraction works as expected for CPO model."""
        mdl = CpoModel()
        mdl.build_opl_model(self.mod_cpo_content, self.dat_cpo_content)

        allVar = mdl.get_opl_var()
        var_names = allVar.variable_names

        self.assertIn("operations", var_names)
        self.assertIn("workovers", var_names)
        # SequenceVar_135 is derived; not a direct decision variable
        self.assertNotIn("SequenceVar_135", var_names)

    def test_variable_name_extraction_cplex(self):
        """Ensure variable name extraction works as expected for CPLEX model."""
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, self.dat_cplex_content)

        allVar = mdl.get_opl_var()
        var_names = allVar.variable_names

        self.assertIn("operations", var_names)
        self.assertIn("workovers", var_names)
    
    # --------------------------------------------------------------------
    # Confirm all model variables are accessible and check their structure
    # --------------------------------------------------------------------
    def test_variable_extraction_and_structure_cpo(self):
        """Ensure allVar contains correct variable names and structure for cpo model."""
        mdl = CpoModel()
        mdl.build_opl_model(self.mod_cpo_content, self.dat_cpo_content)

        allVar = mdl.get_opl_var()
        expected_names = ['operations', 'workovers']

        # Validate variable names
        self.assertEqual(allVar.variable_names, expected_names)

        # Check presence and type
        self.assertTrue(hasattr(allVar, 'operations'))
        self.assertTrue(hasattr(allVar, 'workovers'))

        # Validate nested structure: allVar.operations[m][t] is CpoIntervalVar
        for m in [1, 2]:
            for t in [1, 2, 3]:
                var_obj = allVar.operations[m][t]
                self.assertIsInstance(var_obj.__class__.__name__, str)  # It’s a CpoIntervalVar
                self.assertIn("CpoIntervalVar", str(var_obj.__class__))

    def test_variable_extraction_and_structure_cplex(self):
        """Ensure allVar contains correct variable names and structure for cplex model."""
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, self.dat_cplex_content)
        allVar = mdl.get_opl_var()
        expected_names = ['operations', 'workovers']

        # Validate variable names
        self.assertEqual(allVar.variable_names, expected_names)

        # Check presence and type
        self.assertTrue(hasattr(allVar, 'operations'))
        self.assertTrue(hasattr(allVar, 'workovers'))

        # Validate nested structure: allVar.operations[m][t] is CpoIntervalVar
        for m in [1, 2]:
            for t in [1, 2, 3]:
                var_obj = allVar.operations[m][t]
                self.assertIsInstance(var_obj.__class__.__name__, str)  
                self.assertIn("docplex.mp.dvar.Var", str(var_obj.__class__))

    # -----------------------------------------------------------
    # Solution extraction from from_solution()
    # -----------------------------------------------------------
    def test_from_solution_interval_values_cpo(self):
        """Ensure from_solution correctly maps to IntervalVarValue objects."""
        mdl = CpoModel()
        mdl.build_opl_model(self.mod_cpo_content, self.dat_cpo_content)
        sol = mdl.solve(LogVerbosity="Quiet")

        self.assertIsNotNone(sol)

        allVar = mdl.get_opl_var()
        allVarSol = allVar.from_solution(sol)

        # Variable names should remain the same
        self.assertEqual(allVarSol.variable_names, ['operations', 'workovers'])

        # Verify structure mirrors allVar
        for m in [1, 2]:
            for t in [1, 2, 3]:
                sol_val = allVarSol.operations[m][t]
                self.assertIn("IntervalVarValue", str(sol_val))
                # Start, end, and size should be integers
                self.assertTrue(hasattr(sol_val, "start"))
                self.assertTrue(hasattr(sol_val, "end"))
                self.assertTrue(hasattr(sol_val, "size"))
                
    def test_from_solution_cplex(self):
        """Test from_solution() extracts variable values for a solved CPLEX model."""
        mdl = ModelReader.build_opl_model(self.mod_cplex_content, self.dat_cplex_content)
        sol = mdl.solve(log_output=False)
        self.assertIsNotNone(sol)

        allVar = mdl.get_opl_var()
        allVarSol = allVar.from_solution(sol)

        # Variable names should remain the same
        self.assertEqual(allVarSol.variable_names, ['operations', 'workovers'])
        operations_expected_values = {1: {1: 0, 2: 0, 3: 10.0}, 2: {1: 0, 2: 0, 3: 10.0}}
        self.assertEqual(operations_expected_values, allVarSol.operations)

if __name__ == "__main__":
    unittest.main()
