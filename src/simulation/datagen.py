"""
Synthetic data generation functions.
"""
import logging
from itertools import combinations
from typing import TypedDict

import numpy as np
import scipy.stats as ss
import dendropy as dpy
import random
import anndata

from models.copy_tree import p_delta_change
from utils.math_utils import l_from_p
from utils.tree_utils import random_binary_tree, get_node2node_distance, label_tree


class Dataset(TypedDict):
    """
    Data structure for synthetic data.
    """
    obs: np.ndarray
    tree: dpy.Tree
    cn: np.ndarray


def simulate_quadruplet(n_states, n_sites, alpha=1., l_mean=None) -> Dataset:
    """
    Simulate a quadruplet tree with 2 leaves, one internal node and a root.
    The tree is rooted and the edge lengths are generated from an exponential distribution if l_mean is None.
    The copy number profiles are simulated from the tree and the observations are emitted from the leaves.
    Indices are r, u, v, w = 3, 2, 0, 1.
    Parameters
    ----------
    n_states: int, number of copy number states
    n_sites: int, number of sites
    alpha: float, alpha parameter for evolution model
    l_mean: float, mean of the exponential distribution for edge lengths

    Returns
    -------
    dict with keys 'obs', 'tree', 'cn'

    """
    # generate dendropy tree with 2 leaves, one internal node and root
    tree = dpy.Tree.get(data="((0,1)2)3;", schema='newick', taxon_namespace=dpy.TaxonNamespace(['0', '1']))
    label_tree(tree)
    tree.is_rooted = True
    # generate edge lengths
    l_true = np.empty(3)
    if l_mean is None:
        l_true[:] = np.array([0.01, 0.03, 0.008])
    else:
        for i in range(3):
            l_true[i] = ss.expon(scale=l_mean).rvs()
    r, u, v, w = tuple(map(str, [3, 2, 0, 1]))
    for edge in tree.preorder_edge_iter():
        # centroid to root
        if edge.head_node.label == u:
            edge.length = l_true[0]
        # centroid to v
        elif edge.head_node.label == v:
            edge.length = l_true[1]
        # centroid to w
        elif edge.head_node.label == w:
            edge.length = l_true[2]

    # and cn profiles
    cn = simulate_cn(tree, n_sites, n_states, alpha=alpha)
    # emit observations from tree leaves
    obs = np.empty((n_sites, 2))
    for t in tree.leaf_node_iter():
        cell_id = int(t.label)
        obs[:, cell_id] = emit_raw_obs(cn[cell_id, :])
    return {
        'obs': obs,
        'tree': tree,
        'cn': cn
    }


def simulate_cn_seq(prev_cn, n_states, l, alpha=1.):
    node_cn = np.empty_like(prev_cn)
    # scale l if needed
    pdd = p_delta_change(n_states, l, change=False, alpha=alpha)
    # simulate first copy number
    u = random.random()
    if u < pdd:
        node_cn[0] = prev_cn[0]
    else:
        node_cn[0] = random.choice([j for j in range(n_states) if j != prev_cn[0]])

    for m in range(1, len(prev_cn)):
        u = random.random()
        no_change_cn = prev_cn[m] - prev_cn[m - 1] + node_cn[m - 1]
        if prev_cn[m] == 0:
            # 0 absorption
            no_change_cn = 0

        if 0 <= no_change_cn < n_states:
            if u < pdd:
                node_cn[m] = no_change_cn
            else:
                node_cn[m] = random.choice([j for j in range(n_states) if j != no_change_cn])
        elif no_change_cn < 0:
            node_cn[m] = 0
        else:
            node_cn[m] = random.choice([j for j in range(n_states)])
    return node_cn


def emit_normalized_obs(cn_seq, mu=1.0, scale=1.0):
    eps = ss.norm(loc=0., scale=scale).rvs(size=len(cn_seq))
    return np.clip(cn_seq / mu + eps, a_min=0., a_max=None)


def emit_raw_obs(cn_seq, lam=100.):
    return ss.poisson.rvs(mu=np.clip(cn_seq, a_min=.01, a_max=None) * lam, size=len(cn_seq))


def simulate_cn(tree, n_sites, n_states, alpha=1.):
    cn = np.empty((len(tree.nodes()), n_sites))
    cn[int(tree.seed_node.label), :] = 2
    # tree needs index-labeled node
    assert tree.seed_node.label is not None
    for n in tree.preorder_node_iter():
        if n != tree.seed_node:
            cn[int(n.label)] = simulate_cn_seq(cn[int(n.parent_node.label)], n_states, n.edge_length, alpha=alpha)
    return cn


def get_root_distance(centroid):
    root_distance = 0
    while centroid.parent_node is not None:
        root_distance += centroid.edge_length
        centroid = centroid.parent_node
    return root_distance


def get_ctr_table(tree: dpy.Tree) -> np.ndarray:
    n_cells = len(tree.leaf_nodes())
    assert n_cells == len(tree.taxon_namespace)
    ctr_table = np.zeros((n_cells, n_cells, 3))
    for r, s in combinations(range(n_cells), 2):
        assert r < s, "r must be less than s to ensure upper triangular matrix"
        # most recent common ancestor
        centroid = tree.mrca(taxon_labels=[str(r), str(s)])
        ctr_table[r, s, 0] = get_root_distance(centroid)
        ctr_table[r, s, 1] = get_node2node_distance(tree, centroid.label, str(r))
        ctr_table[r, s, 2] = get_node2node_distance(tree, centroid.label, str(s))

    return ctr_table


def rand_ann_dataset(n_cells: int, n_states: int, n_sites: int, **kwargs):
    #   using different hmms for each chromosome
    # kwargs = [alpha, obs_type]
    # minimum n_sites = 200 because bins are shared among 23 chromosomes
    # in the human genome
    if n_sites < 200:
        logging.debug(f"requested bins number {n_sites} is too low for human genome, setting n_sites = 200")
        n_sites = 200
    anndataset = anndata.AnnData()
    # TODO: implement
    #   using different hmms for each chromosome
    data = rand_dataset(n_cells, n_states, n_sites, **kwargs)

    return anndataset


def rand_dataset(n_cells: int, n_states: int, n_sites: int, alpha=1., obs_type='norm', p_change: float = .2,
                 seed=None) -> Dataset:
    # generate random sc binary tree
    tree = random_binary_tree(n_cells, length_mean=l_from_p(p_change, n_states), seed=seed)
    # set tree to rooted
    tree.is_rooted = True
    # simulate copy number chains
    cn = simulate_cn(tree, n_sites, n_states, alpha=alpha)
    # emit observations from tree leaves
    obs = np.empty((n_sites, n_cells))
    for t in tree.leaf_node_iter():
        cell_id = int(t.label)
        if obs_type == 'pois':
            obs[:, cell_id] = emit_raw_obs(cn[cell_id])
        elif obs_type == 'norm':
            obs[:, cell_id] = emit_normalized_obs(cn[cell_id], scale=.7)
        else:
            logging.debug(f"type {obs_type} not supported for obs model")

    # return dict with observations and all latent variables
    data = {
        'obs': obs,
        'tree': tree,  # contains lengths as edges
        'cn': cn
    }
    return data
