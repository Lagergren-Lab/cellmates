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
from cellmates.models.evo import SimulationEvoModel, JCBModel
from cellmates.models.obs import NormalModel
from cellmates.other_methods import dice_api, medicc2_api
from cellmates.simulation import datagen
from cellmates.utils import testing, visual, tree_utils
from cellmates.utils.tree_utils import label_tree

@unittest.skip("Skipping MEDICC2 tests by default. Run manually if needed.")
class Medicc2APITestCase(unittest.TestCase):

    def setUp(self):
        self.test_out_dir_rel_path = '../../output/tests/test_other_methods/test_medicc2'
        self.test_data_dir_rel_path = '../testdata/medicc2'

    def test_run_medicc2(self):
        logging.basicConfig(level=logging.INFO)
        path_to_testdata = self.test_data_dir_rel_path + "/sampleProfiles.tsv"
        out_dir = self.test_out_dir_rel_path + '/test_run_dice'
         # Run DICE on the test data
        medicc2_api.run_medicc2(path_to_testdata, out_dir_path=out_dir)
        self.assertTrue(os.path.exists(out_dir))

    def test_toy_data_and_run_medicc2(self):
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
        out_dir = testing.create_output_test_folder()
        medicc2_tsv_path = self.test_out_dir_rel_path + '/test_simulate_data_and_run_dice/medicc2_tsv'
        dice_api.convert_dice_tsv_to_medicc2(dataset_path + '_states.tsv', self.test_out_dir_rel_path)

        # Run DICE on the simulated data
        medicc2_input_path = dataset_path + '_states.tsv'
        medicc2_out_dir = self.test_out_dir_rel_path + '/test_toy_data_and_run_dice'
        medicc2_api.run_medicc2(medicc2_input_path, medicc2_out_dir)

    def test_simulate_data_and_run_dice(self):
        """
        Simulates data using the SimulationEvoModel, converts it to DICE format, and runs DICE.
        Then loads the DICE output and checks if it is as expected.
        """
        testing.set_seed(0)
        N, M, K = 25, 500, 7  # number of cells, bins, states
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
        taxon_namespace = true_tree.taxon_namespace # dpy.TaxonNamespace(cell_names)
        dice_nwk_file_path = dice_out_dir + '/standard_root_balME_tree.nwk'
        newick_str = open(dice_nwk_file_path).read().strip()
        dice_tree_bio = Phylo.read(dice_nwk_file_path, 'newick')

        dice_tree_dpy: dpy.Tree = dpy.Tree.get(data=newick_str, schema='newick', taxon_namespace=taxon_namespace)
        label_tree(dice_tree_dpy)

        dice_tree_nx = tree_utils.convert_dendropy_to_networkx(dice_tree_dpy)
        # Root at healthy cell
        dice_api.add_root(dice_tree_nx, healthy_cell_name='cell_0')
        dice_tree_nx = tree_utils.relabel_name_to_int(dice_tree_nx, cell_names)
        dice_tree_dpy2 = tree_utils.convert_networkx_to_dendropy(dice_tree_nx, taxon_namespace=taxon_namespace)

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

        distances = -np.ones((N, N, 3))
        iterations = -np.ones((N, N))
        loglikelihoods = -np.ones((N, N))
        # collect results
        for (u, v), l_i, loglik, it, _, _ in results:
            distances[u, v, :] = l_i
            iterations[(u, v)] = it
            loglikelihoods[(u, v)] = loglik

        # Build tree from inferred distances
        CM_tree_nx = neighbor_joining.rooted_nj0(distances)
        CM_tree_dp = tree_utils.convert_networkx_to_dendropy(CM_tree_nx, taxon_namespace=true_tree.taxon_namespace)

        # Compare trees
        norm_rf_dist_dice = tree_utils.normalized_rf_distance(true_tree, dice_tree_dpy2)
        norm_rf_dist_CM = tree_utils.normalized_rf_distance(true_tree, CM_tree_dp)
        rf_dist_CM = treecompare.symmetric_difference(true_tree, CM_tree_dp)
        rf_dist_DICE = treecompare.symmetric_difference(true_tree, dice_tree_dpy2)
        print(f"Normalized RF distance DICE: {norm_rf_dist_dice}")
        print(f"Normalized RF distance CM: {norm_rf_dist_CM}")
        print(f"RF dist CM: \n {rf_dist_CM}")
        print(f"RF dist DICE: \n {rf_dist_DICE}")

        #true_tree.print_plot()
        #dice_tree_dpy2.print_plot()

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

    def test_medicc2_rf_dist(self):
        "Temporary test to check MEDICC2 output tree RF distance. To be removed later."
        dataset = 'N_25_M500_K7_CN1_fCN3'
        true_tree_nw_file_path = f'../testdata/medicc2/{dataset}/true_tree.nwk'
        true_tree_nw = open(true_tree_nw_file_path).read().strip()
        true_tree_dp = dpy.Tree.get(data=true_tree_nw,
                                        schema='newick')
        medicc2_nwk_file_path = f'../testdata/medicc2/{dataset}/{dataset}_medicc2_input_final_tree.new'
        medicc2_tree_dpy2 = medicc2_api.load_medicc2_tree(medicc2_nwk_file_path, true_tree_dp.taxon_namespace, N_cells=25, remove_diploid=True)

        norm_rf_dist_medicc2 = tree_utils.normalized_rf_distance(true_tree_dp, medicc2_tree_dpy2)
        rf_dist_medicc2 = treecompare.symmetric_difference(true_tree_dp, medicc2_tree_dpy2)
        print(f"Normalized RF distance MEDICC2: {norm_rf_dist_medicc2}")
        print(f"RF dist MEDICC2: \n {rf_dist_medicc2}")

        out_dir = testing.create_output_test_folder(sub_folder_name=dataset)

        visual.plot_tree_phylo(medicc2_tree_dpy2, out_dir=out_dir, filename='medicc2_tree', show=False)
        visual.plot_tree_phylo(true_tree_dp, out_dir=out_dir, filename='true_tree', show=False)



