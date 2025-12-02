import itertools
import os
import random
import unittest
from unittest.mock import MagicMock

import anndata
import skbio
from dendropy.calculate import treecompare

from cellmates.inference import neighbor_joining
from cellmates.inference.em import EM, fit_quadruplet
from cellmates.inference.pipeline import run_inference_pipeline, predict_cn_profiles
from cellmates.models.obs import NormalModel
from cellmates.utils import tree_utils, testing, visual

import dendropy
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from cellmates.simulation.datagen import rand_dataset, rand_ann_dataset, simulate_quadruplet
from cellmates.models.evo import JCBModel, SimulationEvoModel
from cellmates.utils.math_utils import l_from_p
from cellmates.utils.testing import create_output_test_folder
from cellmates.utils.tree_utils import convert_dendropy_to_networkx


class CellmatesTestCase(unittest.TestCase):
    """
    Tests for running Cellmates EM inference and tree reconstruction on simulated data.
    """

    def setUp(self) -> None:
        self.seed = 0
        random.seed(self.seed)
        np.random.seed(seed=self.seed)
        dendropy.utility.GLOBAL_RNG.seed(self.seed)

    #@unittest.skip("Long test, run manually")
    def test_cellmates_given_c(self):
        # Inference parameters
        max_iter = 20
        rtol = 1e-4

        # Simulation parameters
        n_sites = 1000
        n_cells = 20
        n_states = 7
        n_clonal_events_per_edge = 5
        n_focal_events_per_edge = 5
        clonal_CN_length = n_sites // 20
        obs_model_sim = 'normal'
        sim_evo_model = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           n_focal_events=n_focal_events_per_edge,
                                           clonal_CN_length=clonal_CN_length)
        data = rand_dataset(n_sites=n_sites, n_cells=n_cells, n_states=n_states,
                            obs_model=obs_model_sim,
                            evo_model=sim_evo_model)
        x = data['obs']
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)

        out_dir = testing.create_output_test_folder(sub_folder_name=f"M{n_sites}_N{n_cells}_A{n_states}")
        fig, ax = plt.subplots()
        visual.plot_cn_profile(cnps, cell_labels=np.arange(0, n_cells), ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # --------- Setup Models and Mocks ---------
        obs_model = NormalModel(n_states=n_states)
        psi_init = {'mu_v': 1.0, 'tau_v': 50.0, 'mu_w': 1.0, 'tau_w': 50.0}
        evo_model = JCBModel(n_states=n_states)
        cell_pairs = list(itertools.combinations(range(n_cells), r=2))

        # --------- Run Ideal Cellmates EM inference with Mocks ---------
        results, D, Dp = testing.run_ideal_cellmates_em_from_cnps(x, cnps, tree_nx, cell_pairs, n_states,
                                                                  evo_model, obs_model, psi_init)
        exp_distances, exp_pairwise_distances = testing.get_expected_distances(D, Dp, n_states, cell_pairs)

        if out_dir is not None:
            fig, ax = plt.subplots()
            visual.plot_cell_pairwise_heatmap(exp_pairwise_distances, label=np.arange(0, n_cells), ax=ax)
            fig.savefig(out_dir + '/exp_pairwise_distances.png')

        distances = -np.ones((n_cells, n_cells, 3))
        iterations = -np.ones((n_cells, n_cells))
        loglikelihoods = -np.ones((n_cells, n_cells))
        # collect results
        for (u, v), l_i, loglik, it, _, _ in results:
            distances[u, v, :] = l_i
            iterations[(u, v)] = it
            loglikelihoods[(u, v)] = loglik

        L1_diff = np.zeros((n_cells, n_cells, 3))
        for v,w in cell_pairs:
            L1_diff[v,w,:] = abs(distances[v,w,:] - exp_distances[v, w])

        tot_L1_diff = L1_diff.sum(axis=(0,1))
        print(f"Total L1 diff: \n {tot_L1_diff}")

        # Build tree from inferred distances
        tree_res_nx = neighbor_joining.rooted_nj0(distances)
        tree_res_dp = tree_utils.convert_networkx_to_dendropy(tree_res_nx, taxon_namespace=tree_dp.taxon_namespace)
        # Compare with standard NJ tree
        pairwise_distances = distances[:, :, 1] + distances[:, :, 2]
        # symmetrize
        pairwise_distances = np.triu(pairwise_distances) + np.triu(pairwise_distances, k=1).T
        np.fill_diagonal(pairwise_distances, 0)
        skbio_dm = skbio.DistanceMatrix(pairwise_distances, ids=[str(i) for i in range(n_cells)])
        tree_nj_skbio = skbio.tree.nj(skbio_dm)
        tree_nj_skbio = tree_nj_skbio.root_at_midpoint()
        tree_nj_dp = dendropy.Tree.get(data=str(tree_nj_skbio), schema="newick",
                                                          taxon_namespace=tree_dp.taxon_namespace)
        tree_utils.label_tree(tree_nj_dp, method='int')
        tree_nj_nx = tree_utils.convert_dendropy_to_networkx(tree_nj_dp)

        # Save the pairwise distance matrix
        fig, ax = plt.subplots()
        visual.plot_cell_pairwise_heatmap(pairwise_distances, label=np.arange(0, n_cells), ax=ax)
        fig.savefig(out_dir + '/inferred_pairwise_distances.png')

        #print(f"Inferred tree:")
        #nx.write_network_text(tree_res_nx)
        #print(f"True tree:")
        #nx.write_network_text(tree_nx)

        # Compare trees
        rf_dist = treecompare.symmetric_difference(tree_dp, tree_res_dp)
        print(f"RF dist: \n {rf_dist}")
        rf_dist_nj = treecompare.symmetric_difference(tree_dp, tree_nj_dp)
        print(f"RF dist_nj: \n {rf_dist_nj}")

        fig, axs = plt.subplots(1, 3, figsize=(15, 10))
        _, ax1 = visual.draw_graph(tree_nx, ax=axs[0])
        _, ax2 = visual.draw_graph(tree_res_nx, ax=axs[1])
        _, ax3 = visual.draw_graph(tree_nj_nx, ax=axs[2])
        axs[0].set_title('True tree')
        axs[1].set_title('Inferred tree')
        axs[2].set_title('NJ tree')
        fig.savefig(out_dir + '/true_inferred_and_NJ_tree.png')

        self.assertLessEqual(rf_dist, rf_dist_nj)

    def test_cellmates_simple_tree(self):
        # Inference parameters
        max_iter = 20
        tol = 1e-4

        # Simulation parameters
        n_sites = 200
        n_cells = 10
        n_states = 5
        n_clonal_events_per_edge = 1
        n_focal_events_per_edge = 3
        clonal_CN_length = 20
        obs_model_sim = 'normal'
        sim_evo_model = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           n_focal_events=n_focal_events_per_edge,
                                           clonal_CN_length=clonal_CN_length)

        data = rand_dataset(n_sites=n_sites, n_cells=n_cells, n_states=n_states,
                            obs_model=obs_model_sim,
                            evo_model=sim_evo_model)
        x = data['obs']
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)

        out_dir = testing.create_output_test_folder(sub_folder_name=f"Cellmates_EM_M{n_sites}_N{n_cells}_A{n_states}")
        fig, ax = plt.subplots()
        visual.plot_cn_profile(cnps, ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # --------- Setup Models ---------
        obs_model = obs_model_sim
        evo_model = JCBModel(n_states=n_states)
        # --------- Run Cellmates EM inference ---------
        em_alg = EM(n_states, evo_model=evo_model, obs_model=obs_model)
        em_alg.fit(x, max_iter=max_iter, rtol=tol, num_processors=1)

        distances = em_alg.distances
        print(f"Distance matrix: \n {distances[0, ...]}")

        # Get the inferred tree
        tree_res_nx = neighbor_joining.rooted_nj0(distances, internal_indexing=True)

        nx.write_network_text(tree_nx)
        nx.write_network_text(tree_res_nx)

        tree_res_dp = tree_utils.convert_networkx_to_dendropy(tree_res_nx, taxon_namespace=tree_dp.taxon_namespace)

        # Compare trees
        rf_dist = treecompare.symmetric_difference(tree_dp, tree_res_dp)
        print(f"RF dist: \n {rf_dist}")

        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        _, ax1 = visual.draw_graph(tree_nx, ax=axs[0])
        _, ax2 = visual.draw_graph(tree_res_nx, ax=axs[1])
        axs[0].set_title('True tree')
        axs[1].set_title('Inferred tree')
        fig.savefig(out_dir + '/true_inferred_and_NJ_tree.png')

    def test_dice_benchmark_PoC_data_on_expected_lengths(self):
        datasets = ["A1_0"]#, "D2_0", "D3_0", "D4_0", "D5_0", "D6_0", "D7_0", "D8_0"]
        dataseeds = [0]#, 1, 2]
        path_to_data = "../../data/CNAsim/results"
        if not os.path.exists(path_to_data):
            self.skipTest(f"CNASim data not found at {path_to_data}, skipping test.")

        n_datasets = len(datasets)
        n_seeds = len(dataseeds)
        rf_dist_matrix = np.zeros((n_datasets, n_seeds, 3))

        for i, dataset in enumerate(datasets):
            for dataseed in dataseeds:
                dataset_name = f"{dataset}/{dataseed}"
                print(f"Running test on dataset: {dataset_name}")

                # Load CNASim anndata
                adata = anndata.read_h5ad(os.path.join(path_to_data, dataset_name, "anndata.h5ad"))

                out_dir = testing.create_output_test_folder(sub_folder_name=f"{dataset_name}")
                out_prep = self.prepare_cnasim_data(adata, dataset_name, out_dir, K_max=10)
                cnp_tot, cnp_hap = out_prep['cnp_tot'], out_prep['cnp_hap']
                cnasim_tree_dp, cnasim_tree_nx = out_prep['cnasim_tree_dp'], out_prep['cnasim_tree_nx']
                n_cells = out_prep['n_cells']
                cell_names = out_prep['cell_names']

                # --------- Setup Models and Mock Expected changes D, D' ---------
                n_states = 7
                max_iter = 20
                tol = 1e-4
                # Use CNPs from CNASim construct D, D' matrices
                tree_dp, dist_matrix = tree_utils.make_gt_tree_dist(ad=adata, n_states=n_states, cell_names=cell_names)
                cell_pairs = list(itertools.combinations(range(n_cells), r=2))
                D, Dp = testing.get_expected_changes(cnp_hap, cnasim_tree_nx, cell_pairs)
                l_quad_exp, l_pair_exp = testing.get_expected_distances(D, Dp, n_states, cell_pairs)

                distances = -np.ones((n_cells, n_cells, 3))
                for v, w in cell_pairs:
                    distances[v, w, :] = l_quad_exp[v, w]
                tree_res_nx = neighbor_joining.rooted_nj0(distances, internal_indexing=True)
                tree_res_nx2 = neighbor_joining.rooted_nj0(dist_matrix, internal_indexing=True)
                tree_nj_skbio = tree_utils.skbio_neighbour_joining_from_pairwise_distances(pairwise_distances=l_pair_exp)

                tree_nj_dp = dendropy.Tree.get(data=str(tree_nj_skbio), schema="newick",
                                               taxon_namespace=cnasim_tree_dp.taxon_namespace)
                tree_utils.label_tree(tree_nj_dp, method='int')

                tree_res_dp = tree_utils.convert_networkx_to_dendropy(tree_res_nx,
                                                                      taxon_namespace=cnasim_tree_dp.taxon_namespace)
                tree_res_dp2 = tree_utils.convert_networkx_to_dendropy(tree_res_nx2,
                                                                       taxon_namespace=cnasim_tree_dp.taxon_namespace)

                # Compare trees
                rf_dist = tree_utils.normalized_rf_distance(cnasim_tree_dp, tree_res_dp)
                rf_dist2 = tree_utils.normalized_rf_distance(cnasim_tree_dp, tree_res_dp2)
                print(f"RF dist: \n {rf_dist}")
                print(f"RF dist2: \n {rf_dist2}")
                rf_dist_nj = tree_utils.normalized_rf_distance(cnasim_tree_dp, tree_nj_dp)
                print(f"RF dist_nj: \n {rf_dist_nj}")

                # Save metrics in numpy array
                rf_out = np.array([rf_dist, rf_dist2, rf_dist_nj])
                np.save(os.path.join(out_dir, 'rf_distances.npy'), rf_out)
                rf_dist_matrix[i, dataseed, :] = rf_out

        # Save summary of all results
        avg_dataset_rf = rf_dist_matrix.mean(axis=1)
        avg_all = rf_dist_matrix.mean(axis=(0,1))
        print(f"Average RF distances per dataset:\n {avg_dataset_rf}")
        print(f"Average RF distances overall:\n {avg_all}")
        np.save(os.path.join(out_dir, 'avg_dataset_rf.npy'), avg_dataset_rf)
        np.save(os.path.join(out_dir, 'avg_all_rf.npy'), avg_all)

        for i in range(n_datasets):
            self.assertLessEqual(avg_dataset_rf[i, 0], avg_dataset_rf[i, 2])  # Centroid based should be better than NJ

    @unittest.skip("Under development")
    def test_dice_benchmark_PoC_data(self):
        # Load CNASim anndata
        path_to_data = "../../data/CNAsim/results"
        dataset_name = "B5_0/0"
        adata = anndata.read_h5ad(os.path.join(path_to_data, dataset_name, "anndata.h5ad"))

        cnasim_tree_dp, cnasim_tree_nx, cnp_concat, n_cells = self.prepare_cnasim_data(adata, dataset_name)

        # --------- Setup Models and Mock Expected changes D, D' ---------
        n_states = 7
        max_iter = 20
        tol = 1e-4
        # Use CNPs from CNASim construct D, D' matrices
        cell_pairs = list(itertools.combinations(range(n_cells), r=2))
        D, Dp = testing.get_expected_changes(cnp_concat, cnasim_tree_nx, cell_pairs)
        l_quad_exp, l_pair_exp = testing.get_expected_distances(D, Dp, n_states, cell_pairs)

        evo_model_temp = JCBModel(n_states=7)
        evo_model = JCBModel(n_states=7)
        evo_model_temp.new = MagicMock(return_value=evo_model)  # bypass new model
        obs_model = NormalModel(n_states=7, mu_v=1.0, tau_v=10.0, mu_w=1.0, tau_w=10.0)

        psi_init = {'mu_v': 1.0, 'tau_v': 10.0, 'mu_w': 1.0, 'tau_w': 10.0}

        em_alg = EM(n_states, evo_model=evo_model_temp, obs_model=obs_model)

        # Run Cellmates EM inference
        distances = -np.ones((n_cells, n_cells, 3))
        for i, (v,w) in enumerate(cell_pairs):
            theta_init = np.array([0.25, 0.25, 0.25])
            pC1_v = testing.get_marginals_from_cnp(cnp_concat[v], n_states)[0]
            pC1_w = testing.get_marginals_from_cnp(cnp_concat[w], n_states)[0]
            evo_model.get_one_slice_marginals = MagicMock(return_value=(pC1_v, pC1_w))
            evo_model._expected_changes = MagicMock(return_value=(D[v,w], Dp[v,w], -1.0, None, None))
            out_quad = fit_quadruplet(v, w, cnp_concat,
                                      theta_init=theta_init,
                                      psi_init=psi_init, max_iter=max_iter, rtol=tol,
                                      evo_model_template=evo_model, obs_model_template=em_alg.obs_model)
            (v, w), theta_vw, loglik, it = out_quad
            distances[v, w, :] = theta_vw

        tree_res_nx = neighbor_joining.rooted_nj0(distances, internal_indexing=True)

        tree_res_dp = tree_utils.convert_networkx_to_dendropy(tree_res_nx, taxon_namespace=cnasim_tree_dp.taxon_namespace)

        # Compare trees
        rf_dist = treecompare.symmetric_difference(cnasim_tree_dp, tree_res_dp)
        print(f"RF dist: \n {rf_dist}")

    def prepare_cnasim_data(self, adata, dataset_name, out_dir, K_max=10):
        # Get observed CNPs and meta data
        cnp_obs = adata.layers['state']
        cnp_obs_A = adata.layers['Astate']
        cnp_obs_B = adata.layers['Bstate']
        cnp_obs_concat = np.concatenate([cnp_obs_A, cnp_obs_B], axis=1)
        n_cells = cnp_obs.shape[0]
        n_sites = cnp_obs.shape[1]
        C_max = cnp_obs.max()
        n_states = min(C_max, K_max)

        anc_names = adata.uns['ancestral-names']
        anc_mapping = {name: i for i, name in enumerate(anc_names)}
        anc_mapping_tree = {name: n_cells+i for i, name in enumerate(anc_names)}

        # Get true tree
        cnasim_tree_nw = adata.uns['cell-tree-newick']
        cnasim_tree_nx = tree_utils.newick_to_nx(cnasim_tree_nw)
        cell_names = [f"cell{i}" for i in range(1, n_cells + 1)]
        cnasim_tree_nx = tree_utils.relabel_name_to_int(cnasim_tree_nx, cell_names, anc_mapping_tree)
        cnasim_tree_dp = tree_utils.convert_networkx_to_dendropy(cnasim_tree_nx)

        # Get ancestral CNPs
        cnp_anc = adata.uns['ancestral-cn']
        cnp_anc_sorted = cnp_anc[[anc_mapping[name] for name in anc_names], :]
        cnp_tot_concat = np.concatenate([cnp_obs, cnp_anc_sorted], axis=0)

        cnp_anc_A = adata.uns['ancestral-cnA']
        cnp_anc_B = adata.uns['ancestral-cnB']
        cnp_anc_concat = np.concatenate([cnp_anc_A, cnp_anc_B], axis=1)
        cnp_anc_concat_sorted = cnp_anc_concat[[anc_mapping[name] for name in anc_names], :]
        cnp_hap_concat = np.concatenate([cnp_obs_concat, cnp_anc_concat_sorted], axis=0)
        n_ancestors = cnp_anc.shape[0]

        print(f"Number of cells: {n_cells}, number of ancestors: {n_ancestors}, number of sites: {n_sites},"
              f" max CN state: {C_max}, n_states for inference: {n_states}")

        # save visualizations
        fig, ax = visual.draw_graph(cnasim_tree_nx, save_path=out_dir + '/cnasim_tree.png',
                                    node_size=30, with_labels=True)
        fig.savefig(out_dir + '/cnasim_tree.png')
        plt.close()
        fig, ax = plt.subplots()
        ax = visual.plot_cn_profile(cnp_hap_concat, ax=ax)
        fig.savefig(out_dir + '/cnasim_haplotype_cn_profile.png')
        plt.close()
        fig, ax = plt.subplots()
        ax = visual.plot_cn_profile(cnp_tot_concat, ax=ax)
        fig.savefig(out_dir + '/cnasim_total_cn_profile.png')
        plt.close()
        out_dict = {
            'cnp_hap': cnp_hap_concat,
            'cnp_tot': cnp_tot_concat,
            'cnasim_tree_dp': cnasim_tree_dp,
            'cnasim_tree_nx': cnasim_tree_nx,
            'n_cells': n_cells,
            'cell_names': cell_names
        }
        return out_dict

    def test_run_inference_pipeline_saves_and_deletes_files(self):
        # Create small AnnData (5 cells x 10 features) with a `copy` layer and `normal` annotation
        n_cells = 5
        n_states = 5
        n_sites = 20
        adata = rand_ann_dataset(n_cells, n_states, n_sites)
        adata.layers['copy'] = adata.X.copy()
        test_dir = create_output_test_folder()

        h5ad_path = os.path.join(test_dir, "test_data.h5ad")
        adata.write_h5ad(str(h5ad_path))

        results = run_inference_pipeline(
            input=str(h5ad_path),
            output=test_dir,
            n_states=n_states,
            max_iter=3,
            numpy=True,
            learn_obs_params=True,
            use_copynumbers=False,
            save_diagnostics=True,
            init_from_cn=True
        )

        # Verify returned paths exist and content is plausible
        dist_path = results["distances"]
        tree_path = results["tree"]
        cells_path = results["cells"]

        assert os.path.exists(dist_path), "distance file not created"
        assert os.path.exists(tree_path), "tree file not created"
        assert os.path.exists(cells_path), "cell names file not created"

        # Check distance matrix shape matches number of cells
        loaded = np.load(dist_path)
        assert loaded.shape == (n_cells, n_cells, 3)

        # Basic check for cell names contents
        with open(cells_path, "r") as f:
            lines = [l.strip() for l in f.readlines()]
        assert len(lines) == n_cells

        # Read tree and check number of leaves
        tree_dp = dendropy.Tree.get(path=tree_path, schema="newick")
        assert len(tree_dp.leaf_nodes()) == n_cells

        # Clean up the created files explicitly
        os.remove(dist_path)
        os.remove(tree_path)
        os.remove(cells_path)

    @unittest.skip("Under development")
    def test_predict_cn_triplet(self):
        # FIXME: the triplet is not actually a proper binary tree
        seed = 0
        n_sites, n_states = 100, 5
        p_changes = np.array([0.05, 0.05, 0.05])
        edge_lengths = l_from_p(p_changes)
        evo_model = JCBModel(n_states=n_states)
        obs_model = NormalModel(n_states=n_states, mu_v=1.0, tau_v=5.0)
        data = simulate_quadruplet(n_sites, obs_model=obs_model, n_states=n_states, seed=0, edge_lengths=edge_lengths)
        cell_names = ['cell0', 'cell1']
        obs = data['obs']
        cn_true = data['cn']
        nx_tree = convert_dendropy_to_networkx(data['tree'], edge_attr='length')
        cn_pred, ancestor_names = predict_cn_profiles(obs, nx_tree, cell_names, evo_model, obs_model)
        self.assertEqual(cn_pred.shape, cn_true)
        # Check that predicted CNs are within valid range
        self.assertTrue(np.all(cn_pred >= 0))
        self.assertTrue(np.all(cn_pred < n_states))
        # Check that predicted CNs are reasonably close to true CNs, compute MAE
        mae = np.mean(np.abs(cn_pred - cn_true), axis=1)
        print(f"Mean Absolute Error per cell: {mae}")

    def test_predict_cn(self):
        # randomly generate a small dataset
        n_sites = 20
        n_cells = 7
        n_states = 5
        evo_model = JCBModel(n_states=n_states)
        obs_model = NormalModel(n_states=n_states, mu_v=1.0, tau_v=5.0)
        data = rand_dataset(n_sites=n_sites, n_cells=n_cells, n_states=n_states,
                            obs_model=obs_model, evo_model=evo_model)
        obs = data['obs']
        cn_true = data['cn']
        print(f"True CN profiles:\n {cn_true}")
        nx_tree = convert_dendropy_to_networkx(data['tree'], edge_attr='length')
        cell_names = list(range(n_cells))
        evo_model = JCBModel(n_states=n_states)
        obs_model = NormalModel(n_states=n_states, mu_v=1.0, tau_v=5.0)
        cn_matrix, labels = predict_cn_profiles(obs, nx_tree, cell_names, evo_model, obs_model)
        self.assertTrue(all(c in labels for c in cell_names))
        print(f"Predicted CN profiles:\n {cn_matrix}")
        self.assertEqual(cn_matrix.shape, cn_true.shape)
        # Check that predicted CNs are within valid range
        root = [n for n,d in nx_tree.in_degree() if d==0][0]
        self.assertTrue(root in labels)
        self.assertTrue(np.all(cn_matrix[root] == 2))
        self.assertTrue(np.all(cn_matrix >= 0))
        self.assertTrue(np.all(cn_matrix < n_states))
        # Check that predicted CNs are reasonably close to true CNs, compute MAE
        mae = np.mean(np.abs(cn_matrix - cn_true), axis=1)
        # check that cells cn is correctly estimated
        print(f"Mean Absolute Error per cell: {mae[:n_cells]}")
        self.assertTrue(np.all(mae[:n_cells] < 0.1), "Large errors in predicted CN for cells")
        print(f"Mean Absolute Error per internal nodes and depth:\n\t{mae[n_cells:]}\n\t{[len(nx.shortest_path(nx_tree, root, n)) for n in nx_tree.nodes if n >= n_cells]}")
        print(f"Depth of ")
        # FIXME: internal nodes predictions can be very off, this test is relaxed for now
        #   several fixes include: better length over tree, proper zero absorption
        self.assertTrue(np.all(mae[n_cells:] < 1.0), "Large errors in predicted CN for internal nodes")
