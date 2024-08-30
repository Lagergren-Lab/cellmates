import itertools
import logging
import random
import unittest

import networkx as nx
import numpy as np
from dendropy.calculate import treecompare

from simulation.datagen import rand_dataset, get_ctr_table
from inference.em import jcb_em_alg
from utils.tree_utils import convert_networkx_to_dendropy

from inference.em import em_alg, build_tree


def _generate_obs(noise=0):
    # 10 sites, 5 cells
    obs = np.array([
        [200] * 5 + [300] * 5,
        [100] * 5 + [200] * 5,
        [100] * 3 + [200] * 2 + [300] * 5,
        [200] * 9 + [100],
        [400] * 2 + [300] * 2 + [200] * 3 + [100] * 3
    ]).transpose()
    eps = np.ones((5, 5))
    print("cn:\n")
    print((obs / 100).astype(int).transpose())
    noise = np.round(np.random.normal(size=obs.shape) * noise).astype(int)
    return obs + noise, eps


class EMTestCase(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(101)
        np.random.seed(seed=101)
        # TODO: add logging lever

    def test_em_alg(self):
        # generate toy data
        obs, eps = _generate_obs(noise=10)
        # run em
        ctr_table = em_alg(obs)
        # assert epsilons
        for v, w in itertools.combinations(range(obs.shape[1]), r=2):
            print(f"eps({v},{w}) = {ctr_table[v, w]:.3f}")
            print(np.round((obs[:, [v, w]] / 100)).astype(int).transpose())
            print(" ------- ")
        # print(ctr_table)

    def test_tree_inference_toy(self):
        # generate toy data
        obs, eps = _generate_obs(noise=10)
        # run em
        # ctr_table = em_alg(obs)
        ctr_table = jcb_em_alg(obs)
        print(ctr_table)
        # build tree
        em_tree = build_tree(ctr_table)
        print(em_tree)
        nx.write_network_text(em_tree, sources=['r'])
        assert nx.is_tree(em_tree)

    def test_tree_inference_synth(self):
        logging.basicConfig(level=logging.WARNING)
        n_states = 6
        n_sites = 30
        n_cells = 6
        data = rand_dataset(n_cells, n_states, n_sites, alpha=1., obs_type='pois')
        print("Generated tree")
        data['tree'].print_plot()
        for node in data['tree'].preorder_node_iter():
            print(node.label, node.edge_length)

        print("Observations")
        print(data['obs'][:20, :])

        ctr_table = jcb_em_alg(data['obs'])
        print(ctr_table)

        em_tree = build_tree(ctr_table)
        # relabel tree nodes with data taxon labels
        nx.write_network_text(em_tree, sources=['r'])

        # compare with true tree using RF-distance (unweighted)
        dendropy_tree = convert_networkx_to_dendropy(em_tree, data['tree'].taxon_namespace)
        dendropy_tree.print_plot()
        rf_distance_jcb = treecompare.symmetric_difference(data['tree'], dendropy_tree)
        print(rf_distance_jcb)

    def test_two_cells(self):
        # seed for reproducibility
        random.seed(101)
        alpha = 3.
        data = rand_dataset(4, 5, 100, alpha=alpha, obs_type='pois')
        print("Generated tree")
        data['tree'].print_plot(plot_metric='length')
        print(f"True CTR table")
        true_ctr_table = get_ctr_table(data)
        print(true_ctr_table)
        # for each node print the edge length
        for node in data['tree'].preorder_node_iter():
            print(f"Node {node.label} edge length: {node.edge_length}")
        # tree to newick
        print(data['tree'].as_string('newick'))
        print(data['cn'])
        # compare with true tree using RF-distance (unweighted)
        print("--- TRUE TREE ---")
        data['tree'].print_plot(plot_metric='length')

        # pick two leaves whose CTR is not the root
        c1, c2 = 0, 1
        for r, s in itertools.combinations(range(data['obs'].shape[1]), 2):
            if data['tree'].mrca(taxon_labels=['c' + str(r), 'c' + str(s)]) != data['tree'].seed_node:
                c1, c2 = r, s
                break
        print(f"Leaves with non-root CTR: {c1}, {c2}")
        # run EM
        ctr_table = jcb_em_alg(data['obs'][:, [c1, c2]])
        print(f"Estimated CTR table")
        print(ctr_table)
        self.assertAlmostEqual(ctr_table[0, 1], true_ctr_table[c1, c2])
        # build tree
        em_tree = build_tree(ctr_table)


        dendropy_tree = convert_networkx_to_dendropy(em_tree, data['tree'].taxon_namespace)
        print("--- EM TREE ---")
        dendropy_tree.print_plot()

        rf_distance_jcb = treecompare.symmetric_difference(data['tree'], dendropy_tree)
        print(f'Robinson-Fould distance: {rf_distance_jcb}')


