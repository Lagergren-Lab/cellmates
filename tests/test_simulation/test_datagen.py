import itertools
import random
import unittest
import numpy as np

from cellmates.simulation.datagen import rand_dataset, get_ctr_table_int, simulate_quadruplet, rand_ann_dataset


class DatagenTestCase(unittest.TestCase):

    def test_rand_dataset(self):
        random.seed(1234)
        np.random.seed(1234)
        n_cells = 50
        n_states = 7
        n_sites = 200

        data = rand_dataset(n_states, n_sites, obs_model='poisson', n_cells=n_cells)

        self.assertEqual(data['obs'].shape, (n_sites, n_cells))
        self.assertEqual(data['cn'].shape, (2 * n_cells - 1, n_sites))
        # check that all nodes are labeled with unique integers and leaves labels are between 0 and n_cells
        self.assertTrue({n.label for n in data['tree'].nodes()} == set(map(str, range(2 * n_cells - 1))))
        self.assertTrue(all(0 <= int(n.label) < n_cells for n in data['tree'].leaf_node_iter()))

    def test_centroid_table(self):
        random.seed(1234)
        np.random.seed(1234)
        n_cells = 10
        n_states = 7
        n_sites = 20

        data = rand_dataset(n_states, n_sites, obs_model='poisson', n_cells=n_cells)
        ctr_table = get_ctr_table_int(data['tree'])
        for r, s in itertools.combinations(range(n_cells), 2):
            centroid = data['tree'].mrca(taxon_labels=[str(r), str(s)])
            if centroid != data['tree'].seed_node:
                self.assertGreater(centroid.edge_length, 0.)
                self.assertTrue(np.all(ctr_table[r, s, :] != -1))
                self.assertTrue(np.all(ctr_table[r, s, :] > 0))

    # test rand_dataset random seed
    def test_rand_dataset_seed(self):
        seed = 1234
        n_cells = 50
        n_states = 7
        n_sites = 200

        data1 = rand_dataset(n_states, n_sites, obs_model='poisson', n_cells=n_cells, seed=seed)
        data2 = rand_dataset(n_states, n_sites, obs_model='poisson', n_cells=n_cells, seed=seed)

        self.assertTrue(np.all(data1['obs'] == data2['obs']))
        self.assertTrue(np.all(data1['cn'] == data2['cn']))
        self.assertTrue(data1['tree'].as_string('newick') == data2['tree'].as_string('newick'))

    def test_simulate_quadruplet(self):
        n_states = 7
        n_sites = 200
        data = simulate_quadruplet(n_sites, n_states=n_states)
        self.assertEqual(data['obs'].shape, (n_sites, 2))
        self.assertEqual(data['cn'].shape, (4, n_sites))
        self.assertTrue({n.label for n in data['tree'].nodes()} == set(map(str, range(4))))

    def test_anndata(self):
        n_cells = 10
        n_states = 7
        n_sites = 200
        adata = rand_ann_dataset(n_cells, n_states, n_sites)
        self.assertIn('tree', adata.uns, msg="tree not found in anndata")
        self.assertIn('state', adata.layers, msg="copy number state layer not found in anndata")
        self.assertEqual(adata.n_obs, n_cells)
        self.assertEqual(adata.n_vars, n_sites)


if __name__ == '__main__':
    unittest.main()
