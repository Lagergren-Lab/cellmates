import itertools
import random
import unittest
import numpy as np

from simulation.datagen import rand_dataset, get_ctr_table


class MyTestCase(unittest.TestCase):
    def test_rand_dataset(self):
        random.seed(1234)
        np.random.seed(1234)
        n_cells = 50
        n_states = 7
        n_sites = 200

        data = rand_dataset(n_cells, n_states, n_sites, obs_type='pois')

        self.assertEqual(data['obs'].shape, (n_sites, n_cells))
        self.assertEqual(data['cn'].shape, (2 * n_cells - 1, n_sites))
        # check that all nodes are labeled with unique integers and leaves labels are between 0 and n_cells
        self.assertTrue({n.label for n in data['tree'].nodes()} == set(range(2 * n_cells - 1)))
        self.assertTrue(all(0 <= n.label < n_cells for n in data['tree'].leaf_node_iter()))

    def test_centroid_table(self):
        random.seed(1234)
        np.random.seed(1234)
        n_cells = 10
        n_states = 7
        n_sites = 20

        data = rand_dataset(n_cells, n_states, n_sites, obs_type='pois')
        ctr_table = get_ctr_table(data)
        for r, s in itertools.combinations(range(n_cells), 2):
            centroid = data['tree'].mrca(taxon_labels=['c' + str(r), 'c' + str(s)])
            if centroid != data['tree'].seed_node:
                self.assertGreater(centroid.edge_length, 0.)
                self.assertGreater(ctr_table[r, s], 0.)



if __name__ == '__main__':
    unittest.main()
