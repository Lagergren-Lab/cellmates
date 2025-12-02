import argparse
import itertools
import json
import logging
import os
import pickle
import time
import timeit

import anndata
import numpy as np
from dendropy.calculate import treecompare
import dendropy as dpy
from matplotlib import pyplot as plt

from cellmates.inference import neighbor_joining
from cellmates.inference.em import EM
from cellmates.models.evo import SimulationEvoModel, JCBModel
from cellmates.models.obs import NormalModel, JitterCopy
from cellmates.other_methods import dice_api, medicc2_api
from cellmates.simulation import datagen
from cellmates.utils import visual, tree_utils, testing


def simulate_data(N, M, K, args,
                  haplotype_aware=True,
                  out_dir='./simulated_data'):
    root_CN = 1 if haplotype_aware else 2

    nCN, nfCN, CN_lr = args.num_CN, args.num_fCN, args.CN_length_ratio

    if args.CN_prob is not None and args.fCN_prob is not None:
        CN_prob = args.CN_prob
        fCN_prob = args.fCN_prob
        evo_model_sim = SimulationEvoModel(clonal_CN_prob=CN_prob, focal_prob=fCN_prob)
    elif args.CN_prob is not None:
        CN_prob = args.CN_prob
        evo_model_sim = SimulationEvoModel(clonal_CN_prob=CN_prob, n_focal_events=nfCN)
    elif args.fCN_prob is not None:
        fCN_prob = args.fCN_prob
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=nCN, focal_prob=fCN_prob)
    else:
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=nCN,
                                           n_focal_events=nfCN)
    evo_model_sim.root_cn = root_CN
    evo_model_sim.clonal_CN_length_ratio = CN_lr
    obs_model = JitterCopy(K, args.jitter_error) if args.jitter_error is not None else NormalModel(K)
    out_sim_hap_a = datagen.rand_dataset(K, M, evo_model_sim, obs_model=obs_model, n_cells=N)
    true_tree = out_sim_hap_a['tree']
    if haplotype_aware:
        out_sim_hap_b = datagen.rand_dataset(K, M, evo_model_sim,
                                             obs_model=obs_model, n_cells=N, tree=true_tree)
        cnps_hap_a = out_sim_hap_a['cn']
        cnps_hap_b = out_sim_hap_b['cn']
        cnps = np.stack((cnps_hap_a, cnps_hap_b), axis=-1)  # shape (2*n_cells-1, n_bins, 2)
        x_hap_a = out_sim_hap_a['obs'].T
        x_hap_b = out_sim_hap_b['obs'].T
        x = np.stack((x_hap_a, x_hap_b), axis=-1)  # shape (n_cells, n_bins, 2)
    else:
        cnps = out_sim_hap_a['cn']  # shape (2*n_cells-1, n_bins)
        x = out_sim_hap_a['obs'].T    # shape (n_cells, n_bins)

    # Save simulated data for visualization
    fig, ax = plt.subplots(2, 1)
    visual.plot_cn_profile(cnps_hap_a, cell_labels=np.arange(0, N), ax=ax[0], title="Hap A")
    visual.plot_cn_profile(cnps_hap_b, cell_labels=np.arange(0, N), ax=ax[1], title="Hap B")
    fig.suptitle("CN profiles - no noise")
    fig.savefig(out_dir + '/cn_profile.png')

    # Save noisy data
    fig, ax = plt.subplots(2, 1)
    visual.plot_cn_profile(x_hap_a, cell_labels=np.arange(0, N), ax=ax[0], title="Hap A")
    visual.plot_cn_profile(x_hap_b, cell_labels=np.arange(0, N), ax=ax[1], title="Hap B")
    jitter_error = 0 if args.jitter_error is None else args.jitter_error
    fig.suptitle(f"CN profiles with jitter error = {jitter_error}")
    fig.savefig(out_dir + '/cn_profile_noisy.png')

    # Save true tree figure and Newick
    visual.plot_tree_phylo(true_tree, out_dir=out_dir, filename='true_tree', title='True tree')
    true_tree_nwk_file_path = out_dir + '/true_tree.nwk'
    with open(true_tree_nwk_file_path, 'w') as f:
        f.write(true_tree.as_string(schema='newick'))

    return cnps, true_tree, x

def convert_data(dataset_path, cnps_obs=None, out_dir=None, haplotype_aware=True):
    out_dir = out_dir if out_dir is not None else dataset_path
    if cnps_obs is None:
        # Load dataset
        asd = 1
        # cnps = load
    M = cnps_obs.shape[1]
    chr_ends_idx = [M // 2, M - 1]
    bin_length = 1000
    dice_api.convert_to_dice_tsv(cnps_obs, chr_ends_idx, bin_length, out_dir + '/dice_states.tsv')
    dice_api.convert_dice_tsv_to_medicc2(out_dir + '/dice_states.tsv', out_dir, out_filename='medicc2_states.tsv',
                                         totalCN=not haplotype_aware)

def run_dice(dataset_path, dice_out_dir=None):
    dice_out_dir = dice_out_dir if dice_out_dir is not None else dataset_path
    dice_api.run_dice(dataset_path + '/dice_states.tsv', dice_out_dir, method='star', tree_rec='balME')
    logging.info(f'Dice results are saved to {dice_out_dir}')

def run_medicc2(medicc2_input_path, medicc2_out_dir=None, num_proc=1):
    medicc2_out_dir = medicc2_out_dir if medicc2_out_dir is not None else medicc2_input_path
    medicc2_api.run_medicc2(medicc2_input_path + '/medicc2_states.tsv', medicc2_out_dir, num_proc=num_proc)
    logging.info(f'Medicc2 results are saved to {medicc2_input_path}/{medicc2_out_dir}')

def run_cellmates(x, K, cellmates_out_dir, num_proc=1):
    evo_model = JCBModel(K)
    obs_model = JitterCopy(K)
    em_alg = EM(K, obs_model, evo_model)
    em_alg.fit(x, max_iter=20, rtol=1e-4, num_processors=num_proc)
    distances = em_alg.distances
    CM_tree_nx = neighbor_joining.rooted_nj0(distances)
    nw_tree = tree_utils.nxtree_to_newick(CM_tree_nx)

    # Save Cellmates tree
    N = x.shape[0]
    cell_names = [n for n in range(N)]
    tree_utils.write_newick(CM_tree_nx, cell_names, out_path=cellmates_out_dir)
    logging.info(f"Cellmates results are saved to: {cellmates_out_dir}")


def run_cellmates_ideal(cnps, K, haplotype_aware, true_tree: dpy.Tree,
                        cellmates_out_dir=None,
                        obs_model=None):
    # Compare with ideal Cellmates inference
    # Setup Cellmates model
    N = (cnps.shape[0] + 1) // 2
    true_tree_nx = tree_utils.convert_dendropy_to_networkx(true_tree)
    evo_model = JCBModel(n_states=K)
    obs_model = obs_model if obs_model is not None else NormalModel(K)
    if haplotype_aware:
        cnps_hap_a = cnps[:, :, 0]  # shape (2*n_cells-1, n_bins)
        cnps_hap_b = cnps[:, :, 1]  # shape (2*n_cells-1, n_bins)
        cnps_cellmates = np.concatenate((cnps_hap_a, cnps_hap_b), axis=1)  # shape (n_cells, 2*n_bins)
    else:
        cnps_cellmates = cnps  # shape (n_cells, n_bins)
    cell_pairs = list(itertools.combinations(range(N), r=2))
    psi_init = {'mu_v': 1.0, 'tau_v': 50.0, 'mu_w': 1.0, 'tau_w': 50.0}
    results, D, Dp = testing.run_ideal_cellmates_em_from_cnps(cnps_cellmates,
                                                              cnps_cellmates,
                                                              true_tree_nx, cell_pairs, K,
                                                              evo_model, obs_model, psi_init)

    distances, iterations, loglikelihoods = -np.ones((N, N, 3)), -np.ones((N, N)), -np.ones((N, N))
    for (u, v), l_i, loglik, it, _, _ in results:
        distances[u, v, :] = l_i
        iterations[(u, v)] = it
        loglikelihoods[(u, v)] = loglik

    # Build tree from distances
    CM_tree_nx = neighbor_joining.rooted_nj0(distances)
    CM_tree_dp = tree_utils.convert_networkx_to_dendropy(CM_tree_nx,
                                                         taxon_namespace=true_tree.taxon_namespace)
    tree_utils.label_tree(CM_tree_dp)
    # Save Cellmates tree
    #cell_names = [n for n in range(N)]
    #tree_utils.write_newick(CM_tree_nx, cell_names, out_path=cellmates_out_dir)
    return CM_tree_dp


def compare_trees(true_tree, dice_tree, medicc2_tree, cellmates_tree, out_dir):
    # Calculate RF distances
    norm_rf_dist_dice = tree_utils.normalized_rf_distance(true_tree, dice_tree)
    norm_rf_dist_medicc2 = tree_utils.normalized_rf_distance(true_tree, medicc2_tree)
    norm_rf_dist_CM = tree_utils.normalized_rf_distance(true_tree, cellmates_tree)
    rf_dist_DICE = treecompare.symmetric_difference(true_tree, dice_tree)
    rf_dist_Medicc2 = treecompare.symmetric_difference(true_tree, medicc2_tree)
    rf_dist_CM = treecompare.symmetric_difference(true_tree, cellmates_tree)
    logging.info(f"Normalized RF distances:\nDICE: {norm_rf_dist_dice}\nMEDICC2: {norm_rf_dist_medicc2}\nCellmates: {norm_rf_dist_CM}")
    logging.info(f"RF distances:\nDICE: {rf_dist_DICE}\nMEDICC2: {rf_dist_Medicc2}\nCellmates: {rf_dist_CM}")
    # Save RF distances to file
    rf_dist = {'DICE': {'normalized': norm_rf_dist_dice, 'absolute': rf_dist_DICE},
               'MEDICC2': {'normalized': norm_rf_dist_medicc2, 'absolute': rf_dist_Medicc2},
               'Cellmates': {'normalized': norm_rf_dist_CM, 'absolute': rf_dist_CM}}
    with open(out_dir + '/rf_distances.json', 'w') as fp:
        json.dump(rf_dist, fp)

    # Save tree figures and Newick files
    visual.plot_tree_phylo(dice_tree, out_dir=out_dir, filename='dice_tree', title='DICE tree')
    visual.plot_tree_phylo(medicc2_tree, out_dir=out_dir, filename='medicc2_tree', title='MEDICC2 tree')
    visual.plot_tree_phylo(cellmates_tree, out_dir=out_dir, filename='cellmates_tree', title='Cellmates tree')
    dice_tree_nwk_file_path = out_dir + '/dice_tree.nwk'
    medicc2_tree_nwk_file_path = out_dir + '/medicc2_tree.nwk'
    cellmates_tree_nwk_file_path = out_dir + '/cellmates_tree.nwk'
    with open(dice_tree_nwk_file_path, 'w') as f:
        f.write(dice_tree.as_string(schema='newick'))
    with open(medicc2_tree_nwk_file_path, 'w') as f:
        f.write(medicc2_tree.as_string(schema='newick'))
    with open(cellmates_tree_nwk_file_path, 'w') as f:
        f.write(cellmates_tree.as_string(schema='newick'))

    logging.info(f"RF distances:\nDICE: {rf_dist['DICE']['normalized']}, MEDICC2: {rf_dist['MEDICC2']['normalized']}, Cellmates: {rf_dist['Cellmates']['normalized']}")
    print(f"RF distances:"
          f"\nDICE: {rf_dist['DICE']['normalized']},"
          f" MEDICC2: {rf_dist['MEDICC2']['normalized']},"
          f" Cellmates: {rf_dist['Cellmates']['normalized']}")
    print(f"RF distances:"
          f"\nDICE: {rf_dist['DICE']['absolute']},"
          f" MEDICC2: {rf_dist['MEDICC2']['absolute']},"
          f" Cellmates: {rf_dist['Cellmates']['absolute']}")


def load_trees(out_dir, taxon_namespace, N):
    # Load DICE tree
    cell_names = ['cell_'+str(i) for i in range(N)]
    dice_tree = dice_api.load_dice_tree(out_dir + '/standard_root_balME_tree.nwk', taxon_namespace, cell_names)
    medicc2_tree = medicc2_api.load_medicc2_tree(out_dir + '/medicc2_states_final_tree.new', taxon_namespace, N)
    return dice_tree, medicc2_tree


def run(args):
    out_dir = args.out_dir
    if os.path.exists(out_dir):
        logging.warning(f"Output directory {out_dir} already exists. Contents may be overwritten.")
    os.makedirs(out_dir, exist_ok=True)

    haplotype_aware = args.haplotype_aware
    if args.dataset_path is None:
        N, M, K = args.num_cells, args.num_bins, args.num_states
        out_dir += f"/N{N}_M{M}_K{K}"
        nCN, nfCN = args.num_CN, args.num_fCN
        if args.CN_prob is not None:
            CN_prob = args.CN_prob
            out_dir += f'pCN{CN_prob}'
        else:
            out_dir += f'nCN{nCN}'
        if args.fCN_prob is not None:
            fCN_prob = args.fCN_prob
            out_dir += f'pfCN{fCN_prob}'
        else:
            out_dir += f'nfCN{nfCN}'
        out_dir += f'CNlength{args.CN_length_ratio}'
        out_dir += f"err{args.jitter_error}" if args.jitter_error is not None else ""
        os.makedirs(out_dir, exist_ok=True)

        cnps, true_tree, x = simulate_data(N, M, K, args, out_dir=out_dir)
        cnps_obs = cnps[:N]  # observed cells only
        if args.jitter_error is None:
            cnps_noise_obs = cnps_obs # pure CNPs
            cnps_noise = cnps
        else:
            cnps_noise_obs = cnps_obs
            cnps_noise = cnps
            cnps_noise[:N] = cnps_noise_obs
        convert_data(None, cnps_obs=cnps_noise_obs, out_dir=out_dir, haplotype_aware=haplotype_aware)
    else:
        # Load dataset from path
        dataset_path = args.dataset_path
        adata = anndata.load_h5ad(dataset_path)
        convert_data(dataset_path, out_dir, haplotype_aware=haplotype_aware)
        # Get N, M, K, nCN, nfCN

    # Run DICE
    time_dice_start = time.time()
    run_dice(out_dir, out_dir)
    time_dice = time.time() - time_dice_start
    # Run MEDICC2
    time_medicc2_start = time.time()
    run_medicc2(out_dir, num_proc=args.num_proc)
    time_medicc2 = time.time() - time_medicc2_start
    # Run Cellmates
    time_cellmates_start = time.time()
    cellmates_tree = run_cellmates_ideal(cnps_noise, K, haplotype_aware, true_tree)
    time_cellmates = time.time() - time_cellmates_start
    #run_cellmates(cnps, K)

    # Evaluation
    taxon_namespace = true_tree.taxon_namespace
    dice_tree, medicc2_tree = load_trees(out_dir, taxon_namespace, N)
    compare_trees(true_tree, dice_tree, medicc2_tree, cellmates_tree, out_dir)
    print(f"DICE: {time_dice:.2f}, MEDICC2: {time_medicc2:.2f}, Cellmates: {time_cellmates:.2f}")


if __name__ == '__main__':
    cli = argparse.ArgumentParser("Run Cellmates, DICE and Medicc2 on simulated data")

    cli.add_argument('--out-dir', type=str, required=True,)
    cli.add_argument('--dataset-path', type=str, default=None,)
    cli.add_argument('--num-cells', type=int, default=10,)
    cli.add_argument('--num-bins', type=int, default=1000,)
    cli.add_argument('--num-states', type=int, default=5,)
    cli.add_argument('--num-CN', type=int, default=1,)
    cli.add_argument('--num-fCN', type=int, default=1,)
    cli.add_argument('--CN-length-ratio', type=float, default=0.1,)
    cli.add_argument('--CN-prob', type=float, default=None)
    cli.add_argument('--fCN-prob', type=float, default=None)
    cli.add_argument('--jitter-error', type=float, default=None)
    cli.add_argument('--num-proc', type=int, default=1,)
    cli.add_argument('--haplotype-aware', action='store_true', default=True)
    args = cli.parse_args()

    run(args)



