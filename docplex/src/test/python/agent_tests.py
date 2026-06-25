import unittest
import six

from docplex.mp.utils import DOcplexException
from docplex.mp.model import Model


class DummyEngine():
    def __init__(self, mdl, **kwargs):
        pass

    @property
    def name(self):
        return "dummy"

    def set_streams(self, os):
        pass

    def set_objective_sense(self, sense):
        pass

    def set_objective_expr(self, x, y):
        pass

    def end(self):
        pass


class SolverAgentTests(unittest.TestCase):

    def test_agent_unknown(self):
        six.assertRaisesRegex(self, DOcplexException,
                              "Unexpected agent name: spirou", lambda: Model(agent="spirou"))

    def test_agent_class_bad_ctor(self):
        class Dumb():
            pass
        six.assertRaisesRegex(self, DOcplexException,
                              "failed to create instance from model", lambda: Model(agent=Dumb))

    def test_agent_class_ok(self):

        with Model(agent=DummyEngine) as md:
            self.assertEqual("dummy", md.solves_with)
            # solver agent is a class, not a string....
            self.assertEqual(DummyEngine, md.solver_agent)


        with Model(agent="agent_tests.DummyEngine") as md:
            self.assertEqual("dummy", md.solves_with)
            self.assertEqual("agent_tests.DummyEngine", md.solver_agent)


    def test_agent_nosolve(self):
        with Model(agent="nosolve") as nm:
            self.assertEqual(nm.solves_with, "nosolve")


if __name__ == "__main__":
    unittest.main()
