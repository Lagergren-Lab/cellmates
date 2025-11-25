import random
import unittest

import dendropy.calculate.treecompare
import numpy as np
import skbio
from dendropy.calculate.phylogeneticdistance import PhylogeneticDistanceMatrix

from cellmates.simulation.datagen import rand_dataset


class NJTestCase(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(101)
        np.random.seed(seed=101)

    def test_tree_true_distances(self):
        n_states = 7
        n_sites = 200
        n_cells = 10
        data = rand_dataset(n_states, n_sites, obs_model='poisson', evo_model='jcb',
                            n_cells=n_cells, p_change=.1, seed=101)

        # Extract the true tree from the data
        true_tree = data['tree']

        # Generate the distance matrix using the l_v + l_w distances
        dm = PhylogeneticDistanceMatrix.from_tree(true_tree)
        distances = np.zeros((n_cells, n_cells))
        for i, taxon1 in enumerate(true_tree.taxon_set):
            for j, taxon2 in enumerate(true_tree.taxon_set):
                distances[i, j] = dm.distance(taxon1, taxon2)

        # Run Neighbor Joining on the distance matrix
        skbio_dm = skbio.DistanceMatrix(distances, ids=[n.label for n in true_tree.taxon_set])
        neighbor_joining = skbio.tree.nj(skbio_dm)
        nj_tree_dendropy = dendropy.Tree.get(data=str(neighbor_joining), schema="newick", taxon_namespace=true_tree.taxon_namespace)

        # Compare the inferred tree with the true tree - Expect fully correct inference
        rf_distance = dendropy.calculate.treecompare.robinson_foulds_distance(true_tree, nj_tree_dendropy)
        print(f"RF distance: {rf_distance}")

        nj_tree_dendropy.print_plot(plot_metric='length')
        true_tree.print_plot(plot_metric='length')
        self.assertAlmostEqual(rf_distance, 0, delta=0.1)

