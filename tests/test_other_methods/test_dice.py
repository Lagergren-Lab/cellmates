import itertools
import logging
import unittest
import os

import numpy as np
from Bio import Phylo
import dendropy as dpy
from dendropy.calculate import treecompare
from matplotlib import pyplot as plt

from cellmates.inference import neighbor_joining
from cellmates.inference.em import EM
from cellmates.models.evo import SimulationEvoModel, JCBModel
from cellmates.models.obs import NormalModel
from cellmates.other_methods import dice_api
from cellmates.simulation import datagen
from cellmates.utils import testing, visual, tree_utils
from cellmates.utils.tree_utils import label_tree


# skip if DICE is not installed (executable not found)
@unittest.skipUnless(dice_api.is_dice_installed(), "DICE is not installed")
class DiceAPITestCase(unittest.TestCase):

    def setUp(self):
        self.test_out_dir_rel_path = '../../output/tests/test_other_methods/test_dice'
        self.test_data_dir_rel_path = '../testdata/dice'

    def test_run_dice(self):
        logging.basicConfig(level=logging.INFO)
        path_to_testdata = self.test_data_dir_rel_path + "/sampleProfiles.tsv"
        out_dir = self.test_out_dir_rel_path + '/test_run_dice'
         # Run DICE on the test data
        dice_api.run_dice(path_to_testdata, out_path=out_dir)
        self.assertTrue(os.path.exists(out_dir))

    def test_toy_data_and_run_dice(self):
        # Simulate a small AnnData object
        n_cells = 10
        n_bins = 50
        X = np.random.randint(0, 5, size=(n_cells, n_bins))
        states = np.random.randint(0, 3, size=(n_cells, n_bins))
        cnps = np.stack((states, states), axis=-1)  # shape (n_cells, n_bins, 2)

        # Save simulated data tsv-file to temporary directory
        dataset_path = self.test_out_dir_rel_path + '/test_simulate_data_and_run_dice/simulated_data'
        chr_ends_idx = [10, 20, 35, n_bins-1]
        bin_length = 1000
        dice_api.convert_to_dice_tsv(cnps, chr_ends_idx, bin_length, dataset_path + '_states.tsv')

        # Run DICE on the simulated data
        dice_input_path = dataset_path + '_states.tsv'
        dice_out_dir = self.test_out_dir_rel_path + '/test_toy_data_and_run_dice'
        dice_api.run_dice(dice_input_path, dice_out_dir)

    def test_simulate_data_and_run_dice(self):
        """
        Simulates data using the SimulationEvoModel, converts it to DICE format, and runs DICE.
        Then loads the DICE output and checks if it is as expected.
        """
        testing.set_seed(0)
        run_CM_inference = True # Warning: takes long time for larger datasets
        N, M, K = 5, 500, 7  # number of cells, bins, states
        M_tot = 2*M # total bins for both haplotypes
        n_clonal_events_per_edge = 3
        n_focal_events_per_edge = 3
        clonal_CN_length_ratio = 0.1
        root_CN = 1 # haplotype-specific root copy number
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                            n_focal_events=n_focal_events_per_edge,
                                            clonal_CN_length_ratio=clonal_CN_length_ratio,
                                           root_cn=root_CN)
        out_sim_hap_a = datagen.rand_dataset(K, M, evo_model_sim, obs_model='normal', n_cells=N)
        true_tree = out_sim_hap_a['tree']
        out_sim_hap_b = datagen.rand_dataset(K, M, evo_model_sim, obs_model='normal', n_cells=N, tree=true_tree)
        cnps_hap_a = out_sim_hap_a['cn']
        cnps_hap_b = out_sim_hap_b['cn']
        cnps = np.stack((cnps_hap_a, cnps_hap_b), axis=-1)  # shape (2*n_cells-1, n_bins, 2)
        cnps_obs = cnps[:N, :, :] # take only leaf nodes (n_cells, n_bins, 2)

        # Save simulated data for visualization
        out_dir = testing.create_output_test_folder(sub_folder_name=f"N{N}_M{M}_K{K}_CN{n_clonal_events_per_edge}_fCN{n_focal_events_per_edge}")
        fig, ax = plt.subplots(2, 1)
        visual.plot_cn_profile(cnps_hap_a, cell_labels=np.arange(0, N), ax=ax[0], title="Hap A")
        visual.plot_cn_profile(cnps_hap_b, cell_labels=np.arange(0, N), ax=ax[1], title="Hap B")
        fig.savefig(out_dir + '/cn_profile.png')

        # Save true tree figure and Newick
        visual.plot_tree_phylo(true_tree, out_dir=out_dir, filename='true_tree', show=False)
        true_tree_nwk_file_path = out_dir + '/true_tree.nwk'
        with open(true_tree_nwk_file_path, 'w') as f:
            f.write(true_tree.as_string(schema='newick'))

        # Save simulated data tsv-file to temporary directory
        dataset_path = out_dir + '/simulated_data'
        chr_ends_idx = [M//2, M-1]
        bin_length = 1000
        dice_api.convert_to_dice_tsv(cnps_obs, chr_ends_idx, bin_length, dataset_path + '_states.tsv')

        # Run DICE on the simulated data
        out_dir_rel_test_dir = out_dir.partition('/test_dice')[2]
        dice_out_dir = self.test_out_dir_rel_path + out_dir_rel_test_dir
        dice_input_path = dice_out_dir + '/simulated_data_states.tsv'
        dice_api.run_dice(dice_input_path, dice_out_dir, method='star', tree_rec='balME')

        # Load and check DICE output
        # Load DICE tree
        cell_names = ['cell_'+str(i) for i in range(N)]
        taxon_namespace = true_tree.taxon_namespace
        dice_nwk_file_path = dice_out_dir + '/standard_root_balME_tree.nwk'
        dice_tree_dpy2 = dice_api.load_dice_tree(dice_nwk_file_path, taxon_namespace=taxon_namespace, cell_names=cell_names)

        # Compare with ideal Cellmates inference
        # Setup Cellmates model
        true_tree_nx = tree_utils.convert_dendropy_to_networkx(true_tree)
        evo_model = JCBModel(n_states=K)
        obs_model = NormalModel(n_states=K)
        cnps_cellmates = np.concatenate((cnps_hap_a, cnps_hap_b), axis=1)  # shape (n_cells, 2*n_bins)
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

        if run_CM_inference:
            em_alg = EM(n_states=K, evo_model=evo_model, obs_model=obs_model)
            x_hap_a = out_sim_hap_a['obs']
            x_hap_b = out_sim_hap_b['obs']
            x_cellmates = np.concatenate((x_hap_a, x_hap_b), axis=0)  # shape (2*n_bins, n_cells)
            em_alg.fit(x_cellmates, max_iter=20, rtol=1e-4, num_processors=1, psi_init=psi_init)
            distances_cm_inf = em_alg.distances

        # Build tree from inferred distances
        CM_tree_nx = neighbor_joining.rooted_nj0(distances)
        CM_tree_dp = tree_utils.convert_networkx_to_dendropy(CM_tree_nx, taxon_namespace=true_tree.taxon_namespace)
        if run_CM_inference:
            CM_inf_tree_nx = neighbor_joining.rooted_nj0(distances_cm_inf)
            CM_inf_tree_dp = tree_utils.convert_networkx_to_dendropy(CM_inf_tree_nx,
                                                                     edge_length='length',
                                                                     taxon_namespace=true_tree.taxon_namespace)
            tree_utils.label_tree(CM_inf_tree_dp)
            # Save CM inferred tree figure
            visual.plot_tree_phylo(CM_inf_tree_dp, out_dir=out_dir, filename='CM_inferred_tree', show=False)

        # Compare trees
        norm_rf_dist_dice = tree_utils.normalized_rf_distance(true_tree, dice_tree_dpy2)
        norm_rf_dist_CM = tree_utils.normalized_rf_distance(true_tree, CM_tree_dp)
        rf_dist_CM = treecompare.symmetric_difference(true_tree, CM_tree_dp)
        rf_dist_DICE = treecompare.symmetric_difference(true_tree, dice_tree_dpy2)

        if run_CM_inference:
            norm_rf_dist_CM_inf = tree_utils.normalized_rf_distance(true_tree, CM_inf_tree_dp)
            rf_dist_CM_inf = treecompare.symmetric_difference(true_tree, CM_inf_tree_dp)

        print(f"Normalized RF distance DICE: {norm_rf_dist_dice}")
        print(f"Normalized RF distance CM: {norm_rf_dist_CM}")
        print(f"RF dist CM: \n {rf_dist_CM}")
        print(f"RF dist DICE: \n {rf_dist_DICE}")
        if run_CM_inference:
            print(f"Normalized RF distance CM inference: {norm_rf_dist_CM_inf}")
            print(f"RF dist CM inference: \n {rf_dist_CM_inf}")


    def test_simulate_data_and_convert_to_dice_tsv_and_medicc2_tsv(self):
        """
        Simulates data using the SimulationEvoModel, converts it to DICE format, and also to MEDICC2 format.
        """
        testing.set_seed(0)
        N, M, K = 10, 100, 5  # number of cells, bins, states
        n_clonal_events_per_edge = 2
        n_focal_events_per_edge = 2
        clonal_CN_length_ratio = 0.1
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                            n_focal_events=n_focal_events_per_edge,
                                            clonal_CN_length_ratio=clonal_CN_length_ratio)
        out_sim_hap_a = datagen.rand_dataset(K, M, evo_model_sim, obs_model='normal', n_cells=N)
        true_tree = out_sim_hap_a['tree']
        out_sim_hap_b = datagen.rand_dataset(K, M, evo_model_sim, obs_model='normal', n_cells=N, tree=true_tree)
        cnps_hap_a = out_sim_hap_a['cn']
        cnps_hap_b = out_sim_hap_b['cn']
        cnps = np.stack((cnps_hap_a, cnps_hap_b), axis=-1)  # shape (2*n_cells-1, n_bins, 2)
        cnps_obs = cnps[:N, :, :] # take only leaf nodes (n_cells, n_bins, 2)

        # Save simulated data tsv-file to temporary directory
        out_dir = testing.create_output_test_folder(sub_folder_name=f"N_{N}_M{M}_K{K}_CN{n_clonal_events_per_edge}_fCN{n_focal_events_per_edge}_to_dice_and_medicc2")
        chr_ends_idx = [M//2, M-1]
        bin_length = 1000

        # Convert to DICE format
        dice_tsv_path = out_dir + '/dice_input.tsv'
        dice_api.convert_to_dice_tsv(cnps_obs, chr_ends_idx, bin_length, dice_tsv_path)

        # Convert to MEDICC2 format
        medicc2_output_path = out_dir
        dice_api.convert_dice_tsv_to_medicc2(dice_tsv_path, medicc2_output_path, totalCN=False)

    def test_convert_dice_tsv_to_medicc2(self):
        """
        Tests the conversion of a DICE TSV file to MEDICC2 format.
        """
        # Create a small DICE-format TSV file
        dataset = 'N_25_M500_K7_CN3_fCN3'
        dice_tsv_path = self.test_data_dir_rel_path + '/' + dataset + '/simulated_data_states.tsv'
        medicc2_output_path = self.test_data_dir_rel_path
        medicc2_filename = dataset + '/' + dataset + '_medicc2_input.tsv'
        dice_api.convert_dice_tsv_to_medicc2(dice_tsv_path, medicc2_output_path, medicc2_filename, totalCN=False)

    def test_load_anndata_and_convert_to_dice_tsv(self):
        """
        Tests loading an AnnData object and converting it to DICE TSV format.
        """
        path_to_anndata_file = self.test_data_dir_rel_path + '/adata_filt.h5ad'
        out_dir = self.test_out_dir_rel_path + '/test_load_anndata_and_convert_to_dice_tsv'
        dice_api.load_anndata_and_convert_to_dice_tsv(path_to_anndata_file, out_dir)




