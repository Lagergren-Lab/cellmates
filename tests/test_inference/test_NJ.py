import random
import unittest

import dendropy.calculate.treecompare
import networkx as nx
import numpy as np
import skbio
from dendropy.calculate.phylogeneticdistance import PhylogeneticDistanceMatrix

from cellmates.inference.neighbor_joining import std_nj_root, rooted_nj, lm_to_rooted_tree, lm_to_tree, get_root_dist_from_tripledist
from cellmates.simulation.datagen import rand_dataset
from cellmates.utils.tree_utils import get_ctr_table_int, convert_networkx_to_dendropy, label_tree, get_ctr_table


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
        self.assertAlmostEquals(rf_distance, 0, delta=0.1)

    def test_lm_to_rooted_tree_basic(self):
        """
        Single-row linkage matrix:
        parent = max(0,1,len(lm)+1)=2, so edges 2->0 and 2->1 with given lengths.
        """
        lm = np.array([[0, 1, 0.5, 0.6]])
        # tree = lm_to_rooted_tree(lm, edge_attr="length")
        tree = lm_to_tree(lm, edge_attr="length")

        self.assertIsInstance(tree, nx.DiGraph)
        self.assertTrue(nx.is_tree(tree), "The resulting graph should be a tree")
        self.assertTrue(nx.is_arborescence(tree), "The resulting tree should be rooted (an arborescence)")
        self.assertEqual(tree.number_of_edges(), 2)
        self.assertIn((2, 0), tree.edges)
        self.assertIn((2, 1), tree.edges)
        self.assertAlmostEqual(tree[2][0]["length"], 0.5, places=7)
        self.assertAlmostEqual(tree[2][1]["length"], 0.6, places=7)

    def test_std_nj_small_returns_graph_with_weights(self):
        """
        Run standard NJ on a small symmetric distance matrix and assert expected properties.
        For N=3, the linkage produces 2 rows -> 4 edges in the converted tree (2 edges per row).
        """
        dm = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ], dtype=float)

        tree = std_nj_root(dm, root_dist=np.array([2., 3., 1.5]), taxa=['A', 'B', 'C'], collapsed_root=True)

        self.assertIsInstance(tree, nx.DiGraph)
        # Each linkage row produces two edges in lm_to_rooted_tree -> for N=3 expect 4 edges
        self.assertEqual(tree.number_of_edges(), 4)
        self.assertTrue(nx.is_tree(tree), "The resulting tree should be a tree")
        self.assertTrue(nx.is_arborescence(tree), "The resulting tree should be rooted (an arborescence)")
        # All edges should carry the 'weight' attribute
        for _, _, data in tree.edges(data=True):
            self.assertIn("weight", data)
            self.assertIsInstance(data["weight"], (int, float))

    def test_rooted_nj_small_returns_graph_with_custom_attr(self):
        """
        Build a small triplet distance matrix for 3 taxa and ensure rooted_nj produces
        a DiGraph with the requested edge attribute set (here 'length').
        """
        N = 3
        tdm = np.zeros((N, N, 3), dtype=float)

        # Fill symmetric triplets for i < j
        for i in range(N):
            for j in range(i + 1, N):
                # arbitrary but consistent values
                h = 1.0
                v = 0.4
                w = 0.6
                tdm[i, j, 0] = h
                tdm[i, j, 1] = v
                tdm[i, j, 2] = w
                # mirror with swapped roles for leaves
                tdm[j, i, 0] = h
                tdm[j, i, 1] = w
                tdm[j, i, 2] = v

        tree = rooted_nj(tdm, edge_attr="length")

        self.assertIsInstance(tree, nx.DiGraph)
        # For N=3 expect 2 linkage rows -> 4 edges
        self.assertEqual(tree.number_of_edges(), 4)
        for _, _, data in tree.edges(data=True):
            self.assertIn("length", data)
            self.assertIsInstance(data["length"], (int, float))

    def test_std_nj_small(self):
        # make small tree with 4 leaves
        newick = "((A:0.2,B:0.3)5:0.7,((C:0.4,D:0.1)6:0.6,E:0.6)7:0.6)8;"
        # load with dendropy and get triplet distance matrix
        tnamespace = dendropy.TaxonNamespace(['B', 'A', 'C', 'D', 'E'], label='taxa')
        tree = dendropy.Tree.get(data=newick, schema='newick', taxon_namespace=tnamespace)
        label_tree(tree)
        tree.is_rooted = True
        print("\n")
        tree.print_plot(plot_metric='length')
        tdm, taxa = get_ctr_table(tree, full=True)
        print("taxa order: ", taxa)
        # check all >= 0
        print(np.transpose(tdm, [2, 0, 1]))
        for i in range(tdm.shape[0]):
            for j in range(i+1, tdm.shape[1]):
                self.assertGreaterEqual(tdm[i, j, 0], 0)
                self.assertGreaterEqual(tdm[i, j, 1], 0)
                self.assertGreaterEqual(tdm[i, j, 2], 0)
        # run std_nj
        root_dist = get_root_dist_from_tripledist(tdm)
        nj_tree = std_nj_root(tdm[:, :, 1] + tdm[:, :, 2], root_dist=root_dist, edge_attr='length', taxa=taxa)
        print("root dist:", root_dist)
        print(nj_tree.edges())
        self.assertIsInstance(nj_tree, nx.DiGraph)
        # convert to dendropy
        dj_tree_dpy = convert_networkx_to_dendropy(nj_tree, taxon_namespace=tnamespace, edge_length='length', internal_nodes_label='int')
        # compute rf distance
        rf_distance = dendropy.calculate.treecompare.robinson_foulds_distance(tree, dj_tree_dpy)
        print(f"RF distance: {rf_distance}")
        dj_tree_dpy.print_plot(plot_metric='length')
        self.assertAlmostEquals(rf_distance, 0, delta=0.1)

    def test_rooted_nj_small(self):
        # make small tree with 4 leaves
        newick = "((0:0.5,1:0.5)4:0.7,(2:0.6,3:0.6)5:0.6)6;"
        # load with dendropy and get triplet distance matrix
        tnamespace = dendropy.TaxonNamespace(['0', '1', '2', '3'], label='taxa')
        tree = dendropy.Tree.get(data=newick, schema='newick', taxon_namespace=tnamespace)
        label_tree(tree)
        tree.is_rooted = True
        print("\n")
        tree.print_plot(plot_metric='length')
        tdm = get_ctr_table_int(tree, full=True)
        # check all >= 0
        print(tdm.transpose(2, 0, 1))
        for i in range(tdm.shape[0]):
            for j in range(i+1, tdm.shape[1]):
                self.assertGreaterEqual(tdm[i, j, 0], 0)
                self.assertGreaterEqual(tdm[i, j, 1], 0)
                self.assertGreaterEqual(tdm[i, j, 2], 0)
        # run rooted_nj
        nj_tree = rooted_nj(tdm, taxa=['0', '1', '2', '3'])
        print(nj_tree.edges())
        self.assertIsInstance(nj_tree, nx.DiGraph)
        # convert to dendropy
        dj_tree_dpy = convert_networkx_to_dendropy(nj_tree, taxon_namespace=tnamespace, edge_length='weight')
        # compute rf distance
        rf_distance = dendropy.calculate.treecompare.robinson_foulds_distance(tree, dj_tree_dpy)
        print(f"RF distance: {rf_distance} - symmetric diff: {(symrf:=dendropy.calculate.treecompare.symmetric_difference(tree, dj_tree_dpy))}")
        dj_tree_dpy.print_plot(plot_metric='length')
        # self.assertAlmostEquals(rf_distance, 0, delta=0.1)
        self.assertEqual(symrf,0)  # trees should be identical



