from docplex.mp.model import Model
import unittest
import timeit as t
try:
    import numpy as np
except ImportError:
    np = None

class DOcplexPerformanceTests(unittest.TestCase):

    def setUp(self):
        pass

    @unittest.skipIf(np is None, "No numpy")
    def test_model_creation_time(self):
        execution_times = t.repeat('mdl=Model("create a model")',
                                   setup="from docplex.mp.model import Model",
                                   repeat=5,
                                   number=500)
        self.assertLessEqual(np.mean(execution_times),1,msg='creating 500 models takes longer than 1 second - average time to create 500 models was: {}'.format(np.mean(execution_times)))


if __name__ == "__main__":
    unittest.main()