import itertools
import random
import unittest

import networkx as nx
import numpy as np

from inference import neighbor_joining
from simulation.datagen import rand_dataset
from inference.em import EM, jcb_em_alg
from utils.tree_utils import convert_networkx_to_dendropy

from inference.em import em_alg, build_tree


class NJTestCase(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(101)
        np.random.seed(seed=101)
        # TODO: add logging lever

    def test_tree_true_distances(self):
        n_states = 6
        n_sites = 30
        n_cells = 10
        data = rand_dataset(n_cells, n_states, n_sites, alpha=.02, obs_type='pois')

        # Extract the true tree from the data

        # Generate the distance matrix using the l_v + l_w distances
        distance_matrix = np.zeros((n_cells, n_cells))

        # Run Neighbor Joining on the distance matrix
        neighbor_joining.reconstruct_tree(distance_matrix)

        # Compare the inferred tree with the true tree - Expect fully correct inference
