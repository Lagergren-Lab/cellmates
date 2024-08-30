import unittest
from random import random

import networkx as nx

from inference.em import em_alg, jcb_em_alg
from models.copy_tree import CopyTree
from models.observation_models.read_counts_models import QuadrupletSpecificPoissonModel
from models.observation_models.normalized_read_counts_models import (
    QuadrupletSpecificCellbaselineAndPrecisionModel,
)
import numpy as np

from models.quadruplet import Quadruplet


class QuadrupletTestCase(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(seed=101)

    def test_quadruplet_data_simulation(self):
        M = 100
        A = 5
        eps_a = 5.0
        eps_b = M * 1.0
        eps_0 = 0.05
        mu_v = 1.0
        mu_w = 1.0
        tau_v = 100.0
        tau_w = 100.0
        obs_param_v = {'mu': mu_v, 'tau': tau_v}
        obs_param_w = {'mu': mu_w, 'tau': tau_w}
        obs_model = QuadrupletSpecificCellbaselineAndPrecisionModel(M, mu_v, mu_w, tau_v, tau_w)

        quadruplet = Quadruplet(M, A, CN_model=None, obs_model=obs_model)
        out = quadruplet.simulate_data(eps_a, eps_b, eps_0, obs_param_v, obs_param_w)

        expected_Y_v = out['c_v'] * out['obs_param_v']['mu']
        expected_Y_w = out['c_w'] * out['obs_param_w']['mu']

        # All observations within 5 standard deviations of the mean
        self.assertTrue(np.allclose(out['data_v'], expected_Y_v, atol=5. / tau_v ** (1 / 2)))
        self.assertTrue(np.allclose(out['data_w'], expected_Y_w, atol=5. / tau_w ** (1 / 2)))

    def test_quadruplet_inference(self):
        # Simulate data using the quadruplet graph
        M = 200
        A = 5
        eps_a = 5.0
        eps_b = M * 1.0
        eps_0 = 0.05
        lambda_v = 1.0
        lambda_w = 1.0
        obs_model = QuadrupletSpecificPoissonModel(M, lambda_v, lambda_w)
        cn_model = CopyTree(M, A, nx.DiGraph([(0, 1), (1, 2), (1, 3)]))
        quadruplet = Quadruplet(M, A, CN_model=cn_model, obs_model=obs_model)

        out = quadruplet.simulate_data(eps_a, eps_b, eps_0, lambda_v, lambda_w)

        # Run EM
        r = np.vstack((out['data_v'], out['data_w'])).transpose()
        ctr_table = em_alg(r)

        print(f"Inferred d(r, u):  {ctr_table[0, 1]}")
        print(f"True d(r,u): {quadruplet.CN_model.true_tree.edges[0, 1]['weight']}")

        assert np.isclose(ctr_table[0, 1], quadruplet.CN_model.true_tree.edges[0, 1]['weight'], rtol=0.2)
