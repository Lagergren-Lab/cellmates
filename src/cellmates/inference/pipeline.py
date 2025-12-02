import os, time, pickle, logging

import networkx as nx
import numpy as np
import anndata
from tqdm import tqdm

from cellmates.common_helpers.cnasim_data import correct_readcounts
from cellmates.inference.neighbor_joining import rooted_nj0
from cellmates.models.obs import NormalModel, JitterCopy, ObsModel
from cellmates.models.evo import JCBModel, EvoModel
from cellmates.inference.em import EM, estimate_theta_from_cn
from cellmates.utils.hmm import viterbi_decode_pomegranate
from cellmates.utils.tree_utils import write_cells_to_tree, relabel_name_to_int_mapping, nxtree_to_newick, newick_to_nx

logger = logging.getLogger(__name__)

def obs_from_adata(adata, layer_name='copy', normal_annotation=None):
    """
    Extract observations and chromosome ends from AnnData object.
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data.
    layer_name : str, optional
        Name of the layer to use for observations. If None, use adata.X.
    normal_annotation : str, optional
        Annotation in adata.obs to identify normal cells. (normal cells are not used in distance estimation)
    Returns
    -------
    obs : np.ndarray
        Observation matrix (n_bins x n_cells).
    chromosome_ends : list
        List of indices where chromosomes end.
    """
    chromosome_ends = np.where(adata.var['chr'].cat.codes.diff() != 0)[0] + 1
    # return dat.T, chromosome_ends.tolist()
    adata_tumor = adata
    if normal_annotation is not None:
        assert normal_annotation in adata.obs, f"Normal annotation '{normal_annotation}' not found in adata.obs"
        adata_tumor = adata[~adata.obs[normal_annotation]]
    dat = adata_tumor.layers[layer_name] if layer_name else adata_tumor.X

    return dat.T, chromosome_ends.tolist(), adata_tumor.obs_names.tolist()

def load_and_prepare_adata(adata_path, use_copynumbers):
    logger.info(f"Reading AnnData file {adata_path}")
    adata = anndata.read_h5ad(adata_path)
    if 'cnasim-params' in adata.uns:
        logger.info("Found CNAsim parameters; applying correction.")
        correct_readcounts(adata)
    elif 'copy' not in adata.layers and not use_copynumbers:
        raise ValueError("Missing 'copy' layer or CNAsim parameters.")
    elif use_copynumbers and 'state' not in adata.layers:
        raise ValueError("Missing 'state' layer for copy number states.")
    return adata


def prepare_observations(adata, n_states, tau, learn_obs_params, use_copynumbers, normal_annotation, layer_name=None, jitter=0.1):
    if not use_copynumbers:
        obs, chromosome_ends, cell_names = obs_from_adata(adata, normal_annotation=normal_annotation)
        obs_model = NormalModel(
            n_states=n_states, mu_v_prior=1.0, tau_v_prior=tau, train=learn_obs_params
        )
    else:
        layer_name = 'state' if layer_name is None else layer_name
        logger.info("Using copy number states directly.")
        obs, chromosome_ends, cell_names = obs_from_adata(
            adata, layer_name=layer_name, normal_annotation=None
        )
        obs_model = JitterCopy(n_states=n_states, error_rate=jitter)  # default jitter (0.1)
    return obs, chromosome_ends, cell_names, obs_model

def predict_cn_profiles(obs: np.ndarray, nx_tree: nx.DiGraph, cell_names: list,
                        evo_model: EvoModel, leaf_obs_model: ObsModel,
                        zero_absorption: bool = True) -> tuple[np.ndarray, list]:
    """
    Predict copy number profiles for all nodes in the tree using Viterbi algorithm.
    Parameters
    ----------
    obs : np.ndarray (n_bins x n_cells)
        Observation matrix.
    nx_tree : nx.DiGraph
        Phylogenetic tree in NetworkX DiGraph format.
    cell_names : list
        List of cell names corresponding to the leaves of the tree.
    evo_model : EvoModel
        Evolutionary model for copy number changes.
    leaf_obs_model : ObsModel
        Observation model for leaf nodes.
    zero_absorption : bool, optional
        Whether to enforce zero-absorbing states.
    Returns
    -------
    predicted_cn : np.ndarray (n_nodes x n_bins)
        Predicted copy number profiles for all nodes in the tree.
    node_labels : list
        List of node labels corresponding to the rows of predicted_cn.
    """
    n_bins, n_cells = obs.shape
    n_states = evo_model.n_states
    n_nodes = n_cells * 2 - 1
    int_nx_tree, full_mapping = relabel_name_to_int_mapping(nx_tree, cell_names)
    # create node labels to map cn profiles array back to names
    node_labels = [None] * n_nodes
    for name, idx in full_mapping.items():
        node_labels[idx] = name

    cn_obs_model = JitterCopy(n_states=n_states, jitter=1e-3)  # FIXME: JitterCopy is a temporary solution
    root = [n for n,d in int_nx_tree.in_degree() if d==0][0]
    # print(f"Root node is {root}")
    cn_matrix = np.zeros((n_nodes, n_bins), dtype=int) - 1
    cn_matrix[root, :] = 2
    # save internal nodes log_probs
    log_p = np.zeros((n_nodes, n_bins, n_states)) - np.inf
    log_p[root, :, 2] = 0.0  # root is diploid
    # add log_p for observed leaves
    for i in range(n_cells):
        log_p[i, :, :] = leaf_obs_model.log_emission_split(obs[:, [i, 0]])[0]
    visited = {root}
    # traverse the tree postorder and use viterbi to predict copy numbers
    for u in tqdm(nx.dfs_postorder_nodes(int_nx_tree), desc="Predicting CN profiles", total=n_nodes):
        # operate on median nodes
        if u not in visited and int_nx_tree.out_degree(u) != 0:
            vw = list(int_nx_tree.successors(u))  # if binary tree, there are two children
            # make transition matrix
            evo_model.theta = [
                nx.path_weight(int_nx_tree, nx.shortest_path(int_nx_tree, root, u), weight='length'),
                int_nx_tree.edges[u, vw[0]]['length'],
                int_nx_tree.edges[u, vw[1]]['length']
            ]
            # at least one is an internal node
            log_emissions = np.zeros((n_bins, n_states, n_states))
            if not visited.intersection(vw):
                obs_vw = obs[:, vw]
                # both are leaves
                log_emissions[...] = leaf_obs_model.log_emission(obs_vw)
            else:
                log_emissions_single = []
                for v in vw:
                    if v not in visited:
                        # it must be a leaf node
                        assert int_nx_tree.out_degree(v) == 0
                        log_emissions_single.append(leaf_obs_model.log_emission_split(obs[:,[v, 0]])[0])
                    else:
                        cn_obs = cn_matrix[[v, root], :].T
                        log_emission_cn = cn_obs_model.log_emission_split(cn_obs)[0]
                        log_emissions_single.append(log_emission_cn)

                # combine
                log_emissions[...] = log_emissions_single[0][:, :, None] + log_emissions_single[1][:, None, :]

            # FIXME: change to evo_model.viterbi_decode when available
            path = viterbi_decode_pomegranate(log_emissions, evo_model.trans_mat, evo_model.start_prob)
            # FIXME: temporary fix for zero-absorbing states
            if zero_absorption:
                # change u cn if they are zero and children are non-zero
                child_zeros = np.any(path[1:, :] == 0, axis=0)
                parent_zeros = path[0, :] == 0
                fix_cn = (~child_zeros) & parent_zeros
                for i in np.where(fix_cn)[0]:
                    path[0, i] = path[0, i-1] if i > 0 else 2

            cn_matrix[u, :] = path[0, :]
            for i in range(2):
                cn_matrix[vw[i], :] = path[i+1, :]
            # add to visited
            visited.update({u, vw[0], vw[1]})
    return cn_matrix, node_labels

def run_em_inference(obs,
                     chromosome_ends,
                     n_states,
                     alpha,
                     jc_correction,
                     hmm_alg, max_iter, rtol, num_processors, obs_model, verbose, save_diag, out_path,
                     cn_profiles: np.ndarray = None):
    evo_model = JCBModel(n_states=n_states, chromosome_ends=chromosome_ends,
                         jc_correction=jc_correction, alpha=alpha, hmm_alg=hmm_alg)
    em = EM(n_states=n_states, evo_model=evo_model, obs_model=obs_model,
            verbose=verbose, diagnostics=save_diag)
    theta_init = None
    if cn_profiles is not None:
        theta_init = estimate_theta_from_cn(cn_profiles, n_states=n_states, evo_model=evo_model)
        logger.info("Initialized evolutionary parameters from copy number profiles.")
        logger.info("First 5 pairs of theta_init: " + ", ".join([str(t.tolist()) for t in theta_init[0, :5, :]]))

    start = time.time()
    em.fit(obs, max_iter=max_iter, rtol=rtol, num_processors=num_processors, checkpoint_path=out_path, theta_init=theta_init)
    elapsed = time.time() - start
    logger.info(f"EM inference completed in {elapsed:.2f}s")

    if save_diag:
        diag_path = os.path.join(out_path, 'em_diagnostics.pkl')
        with open(diag_path, 'wb') as f:
            pickle.dump(em.diagnostic_data, f)
        logger.info(f"Saved EM diagnostics → {diag_path}")
        # wipe checkpoint data to save space (all _checkpoint_*.pkl files)
        for fname in os.listdir(out_path):
            if fname.startswith('_checkpoint_') and fname.endswith('.pkl'):
                os.remove(os.path.join(out_path, fname))
        logger.info("Removed redundant EM checkpoint files")

    return em


def save_results(em, out_path, cell_names, tree_nx):
    os.makedirs(out_path, exist_ok=True)
    dist_path = os.path.join(out_path, 'distance_matrix.npy')
    tree_path = os.path.join(out_path, 'tree.nwk')
    cell_names_path = os.path.join(out_path, 'cell_names.txt')

    np.save(dist_path, em.distances)
    logger.info(f"Saved distance matrix → {dist_path}")

    nwk_str = nxtree_to_newick(tree_nx, weight='length')
    with open(tree_path, 'w') as f:
        f.write(nwk_str + '\n')
    logger.info(f"Saved tree in Newick format → {tree_path}")

    with open(cell_names_path, 'w') as f:
        f.writelines(f"{n}\n" for n in cell_names)

    logger.info(f"Saved cell names → {cell_names_path}")
    return {"distances": dist_path, "tree": tree_path, "cells": cell_names_path}

def save_cn_profiles(predicted_cn_tuple, out_path):
    cn_path = os.path.join(out_path, 'predicted_copy_numbers.npz')
    np.savez(cn_path, data=predicted_cn_tuple[0], labels=predicted_cn_tuple[1])
    logger.info(f"Saved predicted copy number profiles → {cn_path}")
    return cn_path

def run_inference_pipeline(
    input,
    output=None,
    n_states=7,
    max_iter=30,
    verbose=0,
    num_processors=1,
    alpha=1.0,
    jc_correction=False,
    rtol=1e-3,
    learn_obs_params=False,
    numpy=False,
    use_copynumbers=False,
    tau=10.0,
    save_diagnostics=False,
    normal_annotation=None,
    init_from_cn=False,
    predict_cn=True,
    layer_name=None,
    jitter=0.1
):
    out_path = output or "."
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    hmm_alg = "broadcast" if numpy else "pomegranate"

    cn_layer = 'state'
    if use_copynumbers and layer_name is None:
        cn_layer = 'state'
    elif use_copynumbers and layer_name is not None:
        cn_layer = layer_name
    adata = load_and_prepare_adata(input, use_copynumbers)
    obs, chrom_ends, cell_names, obs_model = prepare_observations(
        adata, n_states, tau, learn_obs_params, use_copynumbers, normal_annotation, layer_name=cn_layer, jitter=jitter
    )
    cn_profiles = None

    if init_from_cn:
        try:
            cn_profiles = adata[cell_names].layers[cn_layer]
        except KeyError:
            raise KeyError(f"Cannot initialize from copy numbers, layer {cn_layer} is not in AnnData obj")
    logger.info("adata shape: " + str(adata.shape))
    logger.info("Observation shape: " + str(obs.shape))
    logger.info("CN shape for initialization: " + str(cn_profiles.shape) if cn_profiles is not None else "No CN profiles for initialization.")

    em = run_em_inference(
        obs=obs,
        chromosome_ends=chrom_ends,
        n_states=n_states,
        alpha=alpha,
        jc_correction=jc_correction,
        hmm_alg=hmm_alg,
        max_iter=max_iter,
        rtol=rtol,
        num_processors=num_processors,
        obs_model=obs_model,
        verbose=verbose,
        save_diag=save_diagnostics,
        out_path=out_path,
        cn_profiles=cn_profiles
    )

    logger.info("Building tree from distance matrix...")
    tree = rooted_nj0(em.distances)
    tree_relab = write_cells_to_tree(tree, cell_names=cell_names)  # relabel tree with cell names and put ancestor names
    # save before predicting cn profiles (which may take time)
    res_paths = save_results(em, out_path, cell_names, tree_relab)
    # infer cn profiles using the tree and the distance matrix
    cn_path = None
    if predict_cn:
        logger.info("Predicting copy number profiles for all nodes in the tree...")
        predicted_cn_tuple = predict_cn_profiles(obs, tree_relab, cell_names, em.evo_model, leaf_obs_model=em.obs_model, zero_absorption=True)
        cn_path = save_cn_profiles(predicted_cn_tuple, out_path)
    res_paths['predicted_copy_numbers'] = cn_path

    return res_paths

def run_prediction_from_output(adata_path, output_path, tau, n_states, use_copynumbers=False, diagnostics_path=None, normal_annotation=None):
    # either normal or jittercopy model
    # if diagnostics_path is not None:
    #     with open(diagnostics_path, 'rb') as f:
    #         diag_data = pickle.load(f)
    #         psi = {p: diag_data[p]['psis'][-1] for p in diag_data}  # last iteration psi for each pair of cells
    adata = load_and_prepare_adata(adata_path, use_copynumbers)
    tree = newick_to_nx(open(os.path.join(output_path, 'tree.nwk')).read().strip(), edge_attr='length')
    evo_model = JCBModel(n_states=n_states)
    obs, chrom_ends, cell_names, obs_model = prepare_observations(
        adata, n_states, tau, False, use_copynumbers, normal_annotation)

    predicted_cn_tuple = predict_cn_profiles(obs, tree, cell_names, evo_model, leaf_obs_model=obs_model, zero_absorption=True)
    cn_path = os.path.join(output_path, 'predicted_copy_numbers.npz')
    np.savez(cn_path, data=predicted_cn_tuple[0], labels=predicted_cn_tuple[1])
    logger.info(f"Saved predicted copy number profiles → {cn_path}")
    return cn_path
