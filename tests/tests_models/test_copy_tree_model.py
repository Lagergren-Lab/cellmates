import unittest

import networkx as nx
import numpy as np

from models.copy_tree import CopyTree, p_delta_change, p_delta_trans_mat, p_delta_start_prob


class CopyTreeTestCase(unittest.TestCase):
    # TODO: Find good asserts

    def test_p_delta_change(self):
        n_states = 5
        # edge cases
        l = 0
        p_ddp = p_delta_change(n_states, l, change=True)
        self.assertAlmostEqual(p_ddp, 0)
        p_dd = p_delta_change(n_states, l, change=False)
        self.assertAlmostEqual(p_dd, 1)

        l = np.inf
        p_ddp = p_delta_change(n_states, l, change=True)
        self.assertAlmostEqual(p_ddp, 1 / n_states)
        p_dd = p_delta_change(n_states, l, change=False)
        self.assertAlmostEqual(p_dd, 1 / n_states)

        l = 5.
        p_ddp = p_delta_change(n_states, l, change=True)
        p_dd = p_delta_change(n_states, l, change=False)

        self.assertEqual(p_dd + (n_states - 1) * p_ddp, 1)

    def test_p_delta_trans_mat(self):
        n_states = 5
        l = 5.
        mat = p_delta_trans_mat(n_states, l)
        self.assertEqual(mat.shape, (n_states, n_states, n_states, n_states))
        self.assertTrue(np.allclose(np.sum(mat, axis=0), np.ones((n_states,) * 3)))

    def test_p_delta_start_prob(self):
        n_states = 5
        l = 5.
        mat = p_delta_start_prob(n_states, l)
        self.assertEqual(mat.shape, (n_states, n_states))
        self.assertTrue(np.allclose(np.sum(mat, axis=0), np.ones(n_states)))

    def test_data_simulation_small_tree(self):
        N = 3
        M = 50
        A = 3
        tree = nx.DiGraph()
        tree.add_edge(0, 1)
        tree.add_edge(0, 2)
        tree.add_edge(2, 3)
        copy_tree = CopyTree(M, A, true_tree=tree)

        eps_a = 1.0
        eps_b = 20.0
        eps_0 = 0.05
        eps, c = copy_tree.simulate_data(eps_a, eps_b, eps_0)
        print(f"eps: {eps} \n c: {c}")

    def test_data_simulation_medium_tree(self):
        N = 30
        M = 50
        A = 3
        tree = nx.random_tree(N, create_using=nx.DiGraph)
        copy_tree = CopyTree(M, A, true_tree=tree)

        eps_a = 1.0
        eps_b = 20.0
        eps_0 = 0.05
        eps, c = copy_tree.simulate_data(eps_a, eps_b, eps_0)
        for u, v in tree.edges:
            print(f"eps: {eps[u, v]}")

    def test_data_simulation_large_tree(self):
        N = 1000
        M = 50
        A = 3
        tree = nx.random_tree(N, create_using=nx.DiGraph)
        copy_tree = CopyTree(M, A, true_tree=tree)

        eps_a = 1.0
        eps_b = 20.0
        eps_0 = 0.05
        eps, c = copy_tree.simulate_data(eps_a, eps_b, eps_0)

    def assert_no_transition_from_absorbing_state(self, c, tree):
        M = c.shape[1]
        for u, v in tree.edges:
            for m in range(M):
                if c[u, m] == 0:
                    self.assertEqual(
                        c[v, m],
                        0,
                        msg=f"Parent node {u} copy number 0 at m {m} but child has {c[v, m]}",
                    )
