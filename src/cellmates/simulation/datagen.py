"""
Synthetic data generation functions.
"""
import logging
from typing import TypedDict

import numpy as np
import pandas as pd
import scipy.stats as ss
import dendropy as dpy
import random
import anndata

from cellmates.models.evo import EvoModel, CopyTree, JCBModel, SimulationEvoModel
from cellmates.models.obs import ObsModel, NormalModel, PoissonModel
from cellmates.utils import tree_utils
from cellmates.utils.math_utils import l_from_p, p_from_l
from cellmates.utils.tree_utils import random_binary_tree, label_tree, get_root_distance, \
    get_ctr_table_int


class Dataset(TypedDict):
    """
    Data structure for synthetic data.
    """
    obs: np.ndarray  # shape (n_sites, n_cells)
    tree: dpy.Tree  # contains edge _lengths
    cn: np.ndarray  # shape (2*n_cells-1, n_sites)


def simulate_quadruplet(n_sites,
                        obs_model: ObsModel | str = 'poisson',
                        evo_model: EvoModel | SimulationEvoModel | str = 'jcb',
                        gamma_params: tuple | list[tuple] = (1, 1),
                        edge_lengths: np.ndarray = None,
                        n_states: int = None, seed: int = None, return_adata=False) -> Dataset | anndata.AnnData:
    """
    Simulate a quadruplet tree with 2 leaves, one internal node and a root.
    The tree is rooted and the edge_lengths are generated from an exponential distribution if l_mean is None,
    otherwise they are set to a fixed value: [0.01, 0.03, 0.008].
    The copy number profiles are simulated from the tree and the observations are emitted from the leaves.
    Indices are r, u, v, w = 3, 2, 0, 1.
    Parameters
    ----------
    n_sites: int, number of sites
    alpha: float, alpha parameter for evolution model
    l_mean: float, mean of the exponential distribution for edge _lengths
    gamma_params: tuple or list of tuples, parameters for gamma distribution for edge _lengths
        if tuple, the same parameters are used for all edges, otherwise a list of tuples is expected, one for each edge
    n_states: int, number of copy number states (if None, it will use the evolution model parameters)

    Returns
    -------
    dict with keys 'obs', 'tree', 'cn'

    """
    # generate tree
    # generate dendropy tree with 2 leaves, one internal node and root
    tree = dpy.Tree.get(data="((0,1)2)3;", schema='newick', taxon_namespace=dpy.TaxonNamespace(['0', '1']))
    label_tree(tree)
    tree.is_rooted = True

    if isinstance(gamma_params, tuple):
        gamma_params = [gamma_params] * 3
    if isinstance(gamma_params, list):
        if len(gamma_params) == 1:
            gamma_params = gamma_params * 3
        elif len(gamma_params) == 2:
            gamma_params.append(gamma_params[1])
        elif len(gamma_params) > 3:
            logging.error(f"too many gamma_params provided, using the first 3")
            gamma_params = gamma_params[:3]
    assert len(gamma_params) == 3 and all(len(gamma_params[i]) == 2 for i in range(3)), "gamma_params must be a tuple or list of 3 tuples"
    gamma_params = np.stack(gamma_params)

    # simulate edge_lengths (or epsilon param for 'copytree')
    edge_lengths = ss.gamma.rvs(gamma_params[:, 0], scale=gamma_params[:, 1]) if edge_lengths is None else edge_lengths
    if isinstance(evo_model, SimulationEvoModel):
        edge_lengths = np.ones(3)
    elif isinstance(evo_model, CopyTree):
        edge_lengths = p_from_l(edge_lengths, n_states=n_states)

    for edge in tree.preorder_edge_iter():
        # root to centroid u
        if edge.head_node.label == '2':
            edge.length = edge_lengths[0]
        # centroid to v
        elif edge.head_node.label == '0':
            edge.length = edge_lengths[1]
        # centroid to w
        elif edge.head_node.label == '1':
            edge.length = edge_lengths[2]

    out_dataset = rand_dataset(n_states, n_sites, evo_model=evo_model, obs_model=obs_model, tree=tree, seed=seed)
    if return_adata:
        out_dataset = _from_data_to_adata(out_dataset)
    return out_dataset

def emit_normalized_obs(cn_seq, mu=1.0, scale=1.0):
    eps = ss.norm(loc=0., scale=scale).rvs(size=len(cn_seq))
    return np.clip(cn_seq / mu + eps, a_min=0., a_max=None)


def emit_raw_obs(cn_seq, lam=100.):
    return ss.poisson.rvs(mu=np.clip(cn_seq, a_min=.01, a_max=None) * lam, size=len(cn_seq))


def _get_full_distance_matrix_from_tree(tree_dpy, matrix_idx: int = 0):
    # matrix_idx: which distance to use from ctr table
    # 0: centroid to root
    # 1: centroid to cell 1
    # 2: centroid to cell 2
    assert matrix_idx in [0, 1, 2], "matrix_idx must be 0, 1, or 2"
    # get ctr table from tree
    # the ctr table is upper triangular, with shape (n_cells, n_cells, 3)
    # make it full and fill diagonal with node to root distances
    n_cells = len(tree_dpy.leaf_nodes())
    ctr_triul_matrix = get_ctr_table_int(tree_dpy)[..., matrix_idx]
    cell_dist = {int(t.label): get_root_distance(t) for t in tree_dpy.leaf_node_iter()}
    ctr_full_matrix = np.empty_like(ctr_triul_matrix)
    for c, d in cell_dist.items():
        for c2 in range(n_cells):
            if c < c2:
                ctr_full_matrix[c, c2] = ctr_triul_matrix[c, c2]
            elif c > c2:
                ctr_full_matrix[c, c2] = ctr_triul_matrix[c2, c]
        ctr_full_matrix[c, c] = d
    return ctr_full_matrix

def rand_ann_dataset(n_cells: int, n_states: int, n_sites: int, n_chrom: int = 1, **kwargs):
    #   using different hmms for each chromosome
    # kwargs = [alpha, obs_type]
    # minimum n_sites = 200 because bins are shared among 23 chromosomes
    # in the human genome
    # TODO: implement
    #   using different hmms for each chromosome
    data = rand_dataset(n_states, n_sites, n_cells=n_cells, **kwargs)
    return _from_data_to_adata(data)

def _from_data_to_adata(data: Dataset) -> anndata.AnnData:
    n_sites = data['obs'].shape[0]
    cn_matrix = np.empty_like(data['obs'].T)
    for t in data['tree'].leaf_node_iter():
        cell_id = int(t.label)
        cn_matrix[cell_id] = data['cn'][cell_id]
    adata = anndata.AnnData(
        X=data['obs'].T,
        var=pd.DataFrame(data={
            'chr': ['1'] * cn_matrix.shape[1],
            'start': [i * 100 + 1 for i in range(n_sites)],
            'end': [(i + 1) * 100 for i in range(n_sites)]
        }),
    )
    adata.layers['state'] = cn_matrix
    adata.uns['tree'] = data['tree'].as_string('newick')

    adata.obsm['ctr-distance-matrix'] = _get_full_distance_matrix_from_tree(data['tree'])
    return adata

def rand_dataset(n_states: int, n_sites: int,
                 evo_model: EvoModel | SimulationEvoModel | str = 'jcb',
                 obs_model: ObsModel | str = 'normal',
                 alpha=1., p_change: float = .2,
                 n_cells: int = None,
                 tree: dpy.Tree = None, seed=None) -> Dataset:
    # generate random sc binary tree
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        dpy.utility.GLOBAL_RNG.seed(seed)

    # parse evo_model and obs_model params
    if isinstance(evo_model, str):
        evo_model = JCBModel(n_states, alpha=alpha) if evo_model == 'jcb' else PoissonModel(n_states)
    if isinstance(obs_model, str):
        obs_model = PoissonModel(n_states) if obs_model == 'poisson' else NormalModel(n_states)

    if n_cells is None:
        assert tree is not None, "n_cells must be provided if tree is not given"
        n_cells = len(tree.leaf_nodes())
    else:
        if tree is not None:
            assert n_cells == len(tree.leaf_nodes()), "n_cells must match the number of leaf nodes in the tree"
        else:
            tree = random_binary_tree(n_cells, length_mean=l_from_p(p_change, n_states) / alpha, seed=None)

    # set tree to rooted
    tree.is_rooted = True
    # simulate copy number chains
    cn = evo_model.simulate_cn(tree, n_sites)
    # emit observations from tree leaves
    obs = np.empty((n_sites, n_cells), dtype=np.float64)
    for t in tree.leaf_node_iter():
        cell_id = int(t.label)
        obs[:, cell_id] = obs_model.sample(cn[cell_id][None, :]).ravel()

    # return dict with observations and all latent variables
    data = {
        'obs': obs,
        'tree': tree,  # contains _lengths as edges
        'cn': cn
    }
    return data
