import unittest

import numpy as np

from models.observation_models.urn_observational_model import UrnModel


class UrnModelTestCase(unittest.TestCase):
    # TODO: Find good asserts
    def test_data_simulation_one_cell_uniform_c(self):
        """
        Simulate data and expect x_n to be largest at site with largest gc for large diff of gc_min and gc_max.
        :return:
        """
        N = 1
        M = 10
        A = 5
        c = np.ones((N, M), dtype=int) * 2

        R_0 = 100
        min_gc = 0.1
        max_gc = 100.0

        urn_model = UrnModel(N, M, A)
        x, R, gc, phi = urn_model.simulate_data(R_0, min_gc, max_gc, c)

        gc_max_idx = np.argmax(gc)
        x_max_idx = np.argmax(x)
        self.assertEqual(gc_max_idx, x_max_idx, msg="Largest x not at site with largest gc for uniform c.")

