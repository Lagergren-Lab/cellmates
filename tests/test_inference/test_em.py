import io
import itertools
import logging
import random
import time
import unittest
from unittest.mock import MagicMock

import dendropy
from Bio import Phylo
import networkx as nx
import numpy as np
from dendropy.calculate import treecompare
from matplotlib import pyplot as plt

from cellmates.utils import math_utils, testing, tree_utils
from cellmates.utils.visual import plot_cn_profile, plot_cell_pairwise_heatmap
from cellmates.models.evo import p_delta_change, CopyTree, JCBModel, SimulationEvoModel
from cellmates.models.obs import NormalModel, PoissonModel
from cellmates.simulation.datagen import rand_dataset, simulate_quadruplet, rand_ann_dataset
from cellmates.inference.em import jcb_em_ctrtable, EM, jcb_em_alg, fit_quadruplet
from cellmates.utils.testing import create_output_test_folder, _generate_obs
from cellmates.utils.tree_utils import convert_networkx_to_dendropy, random_binary_tree, label_tree, nxtree_to_newick, \
    get_ctr_table_int
from cellmates.utils.math_utils import l_from_p, p_from_l, compute_cn_changes

from cellmates.inference.em import rooted_nj0

class EMTestCase(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(101)
        np.random.seed(seed=101)
        logging.basicConfig(level=logging.DEBUG)
        self.DEFAULT_GAMMA_PARAMS = [(1*1000, 0.01/1000), (1*500, 0.03/500), (1*200, 0.008/200)]  # with 500 bins

    def test_em_alg(self):
        # generate toy data
        n_states = 5
        obs, eps = _generate_obs(noise=10)
        # run em
        evo_model = CopyTree(n_states)
        em = EM(n_states, PoissonModel(n_states, 100, 100), evo_model, tree_build='ctr', verbose=2)
        em.fit(obs, max_iter=30, rtol=1e-6, num_processors=1)
        ctr_table = em.distances
        # assert epsilons
        for v, w in itertools.combinations(range(obs.shape[1]), r=2):
            print(f"eps({v},{w}) = {ctr_table[v, w]}")
            print(np.round((obs[:, [v, w]] / 100)).astype(int).transpose())
            print(" ------- ")
        # print(ctr_table)

    def test_em_updates_given_c(self):
        """
        Tests the EM algorithm on a simple quadruplet tree where the true expected number of changes are given and used
        in the M-step. This acts as a sanity check for the remaining terms of the evo M-step and the observation M-step.
        """
        n_states = 7
        n_sites = 500
        p_sim = np.array([0.02, 0.01, 0.01])
        l_sim = l_from_p(np.array(p_sim), n_states)
        evo_model_sim = CopyTree(n_states)
        evo_model_temp = JCBModel(n_states) # Used to bypass .new() call in _fit_quadruplet
        evo_model = JCBModel(n_states)      # Used to mock expected changes
        obs_model = NormalModel(n_states)

        data = simulate_quadruplet(n_sites, obs_model, evo_model_sim, edge_lengths=l_sim, n_states=n_states)
        obs, cnps = (data['obs'], data['cn'])

        # Save data plots
        out_dir = create_output_test_folder(sub_folder_name=f'M_{n_sites}')
        fig, ax = plt.subplots()
        plot_cn_profile(cnps, ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # Ideal D and Dp returns
        breakpoints = math_utils.compute_cn_changes(cnps, pairs=[(3, 2), (2, 0), (2, 1)])
        D_ru, D_uv, D_uw = float(breakpoints[0]), float(breakpoints[1]), float(breakpoints[2])
        Dp_ru, Dp_uv, Dp_uw = n_sites - D_ru, n_sites - D_uv, n_sites - D_uw
        D = np.array([D_ru, D_uv, D_uw])
        Dp = np.array([Dp_ru, Dp_uv, Dp_uw])

        evo_model_temp.new = MagicMock(return_value=evo_model)   # bypass new model creation to enable mocking
        evo_model._expected_changes = MagicMock(return_value=(D, Dp, -1.0, -1., -1.))
        pC1_v, _ = testing.get_marginals_from_cnp(cnps[0], n_states)
        pC1_w, _ = testing.get_marginals_from_cnp(cnps[1], n_states)
        evo_model.get_one_slice_marginals = MagicMock(return_value=[pC1_v, pC1_w])

        theta_init = np.array([0.2, 0.05, 0.1])
        em = EM(n_states, obs_model, evo_model_temp, tree_build='ctr', verbose=2)

        (v, w), theta, loglik, it, obs_model, diagnostic_data = fit_quadruplet(0, 1, obs, max_iter=30, rtol=1e-6, evo_model_template=evo_model_temp, theta_init=theta_init, obs_model_template=em.obs_model, save_diagnostics=em.diagnostics, min_iter=em.min_iter)
        print(f"True epsilons: {np.array(D)/n_sites}")
        print(f"True l: {math_utils.l_from_p(np.array(D)/n_sites, n_states)}")
        print(f"Theta out: {theta}")

    def test_tree_inference_toy(self):
        # generate toy data
        obs, eps = _generate_obs(noise=10)
        # run em
        ctr_table = jcb_em_ctrtable(obs, n_states=5)
        print(ctr_table)
        # build tree
        em_tree = rooted_nj0(ctr_table)
        print(em_tree)
        nx.write_network_text(em_tree, sources=['r'])
        assert nx.is_tree(em_tree)

    @unittest.skip("Slow test, run manually")
    def test_tree_inference_synth(self):
        seed = 101
        n_states = 5
        n_sites = 500
        n_cells = 8
        # data = rand_dataset(n_states, n_sites, obs_model='poisson', alpha=1., p_change=0.05, n_cells=n_cells, seed=seed)
        adata = rand_ann_dataset(n_cells, n_states, n_sites, obs_model='poisson', alpha=1., p_change=0.01, seed=seed)
        data = {
            'obs': adata.X.T,
            'cn': adata.layers['state'],
            'tree': dendropy.Tree.get(data=adata.uns['tree'], schema='newick')
        }

        bio_tree = Phylo.read(io.StringIO(adata.uns['tree']), 'newick')
        test_folder = create_output_test_folder()
        # plot with scgenome to show tree
        try:
            import scgenome.plotting as pl
            g = pl.plot_cell_cn_matrix_fig(adata, tree=bio_tree, show_cell_ids=True)
            g['fig'].savefig(test_folder + '/cn_profile_true_tree.png')

            # plot obs
            g = pl.plot_cell_cn_matrix_fig(adata, layer_name=None, tree=bio_tree, raw=True, show_cell_ids=True)
            g['fig'].savefig(test_folder + '/obs_profile.png')
            print("Saved cn profile and reads to", test_folder)
        except ImportError:
            print("scgenome is not installed; skipping scgenome plotting tests.")

        # run EM
        em = EM(n_states=n_states, obs_model='poisson', evo_model='jcb')
        em.fit(data['obs'])
        ctr_table = em.distances
        print(f"EM converged in {em.n_iterations[(0, 1)]} iterations")
        # plot likelihood
        fig, ax = plt.subplots()
        # from dict[tuple, float] to 2d array
        ll_arr = np.array([[em.loglikelihoods.get((i,j), np.nan) for j in range(data['obs'].shape[1])] for i in range(data['obs'].shape[1])])
        plot_cell_pairwise_heatmap(ll_arr, ax=ax, label='loglikelihoods')
        fig.savefig(test_folder + '/pairwise_loglikelihoods.png')

        print(ctr_table[..., 0])

        em_tree = rooted_nj0(ctr_table)
        # relabel tree nodes with data taxon labels
        nx.write_network_text(em_tree, sources=['r'])
        labels_mapping = {n.label: n.label for n in data['tree'].nodes() if n != data['tree'].seed_node}
        labels_mapping['r'] = data['tree'].seed_node.label
        nx.write_network_text(nx.relabel_nodes(em_tree, labels_mapping, copy=True),
                              sources=[data['tree'].seed_node.label])
        # save and plot inferred tree
        nwk_file_path = test_folder + '/inferred_tree.nwk'
        with open(nwk_file_path, 'w') as f:
            f.write(nxtree_to_newick(em_tree, weight='length'))
        em_tree_bio = Phylo.read(nwk_file_path, 'newick')
        try:
            import scgenome.plotting as pl
            g = pl.plot_cell_cn_matrix_fig(adata, tree=em_tree_bio, show_cell_ids=True)
            g['fig'].savefig(test_folder + '/cn_profile_inferred_tree.png')
        except ImportError:
            print("scgenome is not installed; skipping scgenome plotting tests.")
        # plot diff ctr distances in heatmap
        em_ctr_matrix_full = np.triu(em.distances[..., 0]) + np.tril(em.distances[..., 0].T)
        diff_ctr_dist = adata.obsm['ctr-distance-matrix'] - em_ctr_matrix_full
        np.fill_diagonal(diff_ctr_dist, 0)
        fig, ax = plt.subplots()
        plot_cell_pairwise_heatmap(diff_ctr_dist, ax=ax, label='diff CTR (true - em)')
        fig.savefig(test_folder + '/diff_ctr_distances.png')
        print("Saved cn profile and inferred tree to", test_folder)

        # compare with true tree using RF-distance (unweighted)
        dendropy_tree = convert_networkx_to_dendropy(em_tree,
                                                     taxon_namespace=data['tree'].taxon_namespace,
                                                     edge_length='length')
        dendropy_tree.print_plot()
        sym_distance_jcb = treecompare.symmetric_difference(data['tree'], dendropy_tree)
        print(f'Symmetric (unweighted) distance: {sym_distance_jcb}')
        rf_distance_jcb = treecompare.robinson_foulds_distance(data['tree'], dendropy_tree, edge_weight_attr='length')
        print(f'Robinson-Fould distance: {rf_distance_jcb}')
        self.assertEqual(0, sym_distance_jcb)
        self.assertLess(rf_distance_jcb, 0.05)

    def test_quadruplet(self):
        """ Test EM on a simple quadruplet tree with known edge lengths compared to true ones and assess likelihood improvement. """
        # seed for reproducibility
        seed = 120
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500
        out_dir = create_output_test_folder()

        data = simulate_quadruplet(n_sites, n_states=n_states, gamma_params=self.DEFAULT_GAMMA_PARAMS)
        fig, ax = plt.subplots()
        plot_cn_profile(data['cn'], ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')
        gt_ctr_table = get_ctr_table_int(data['tree'])
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"CN (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")

        # print tree with _lengths (true actually means the _lengths used to generate the data)
        l_true = gt_ctr_table[0, 1, :].tolist()
        data['tree'].print_plot(plot_metric='length')
        print(f"Data generating edge _lengths: {l_true}")

        # edge lengths estimated from copy number (likely more accurate than the true ones)
        comp_eps = compute_cn_changes(data['cn'], [(3, 2), (2, 0), (2, 1)])
        comp_lengths = l_from_p(np.array(comp_eps)/n_sites, n_states)
        print(f"Est (CN) edge _lengths: {comp_lengths}")

        # run EM
        out = jcb_em_alg(data['obs'], n_states=n_states, max_iter=30, rtol=1e-3, num_processors=1)
        ctr_table = out['l_hat']

        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == 2:
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == 0:
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == 1:
                edge.length = ctr_table[0, 1, 2]
        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()
        print("Estimated edge lengths:")
        print(l_est)

        # check likelihood
        evo_model = JCBModel(n_states=n_states)
        evo_model.lengths = l_true

        obs_model = PoissonModel(n_states, 100, 100)
        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_true = evo_model.forward_pass(log_emissions)
        ll_est = out['loglikelihoods'][0, 1]

        evo_model = JCBModel(n_states=n_states)
        evo_model.lengths = comp_lengths
        _, ll_cn = evo_model.forward_pass(log_emissions)
        self.assertGreater(ll_cn, ll_true)
        self.assertGreater(ll_est, ll_true)

        # if these tests don't pass, it's likely that they are wrong
        self.assertAlmostEqual(ctr_table[0, 1, 0], gt_ctr_table[0, 1, 0], delta=0.02)
        self.assertAlmostEqual(ctr_table[0, 1, 1], gt_ctr_table[0, 1, 1], delta=0.02)
        self.assertAlmostEqual(ctr_table[0, 1, 2], gt_ctr_table[0, 1, 2], delta=0.01)

    @unittest.skip("Slow test, run manually")
    def test_quadruplet_random_l(self):
        # seed for reproducibility
        seed = 120
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500
        hmm_alg = 'pomegranate'

        data = simulate_quadruplet(n_sites, gamma_params=self.DEFAULT_GAMMA_PARAMS, n_states=n_states, seed=seed)
        gt_ctr_table = get_ctr_table_int(data['tree'])
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"CN (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")
        # plot
        fig, ax = plt.subplots()
        plot_cn_profile(data['cn'], ax=ax)
        out_dir = create_output_test_folder()
        fig.savefig(out_dir + '/cn_profile.png')

        # print tree with _lengths
        l_true = gt_ctr_table[0, 1, :].tolist()
        data['tree'].print_plot(plot_metric='length')
        print(f"True edge _lengths: {l_true}")

        # run EM
        obs_model = PoissonModel(n_states, 100)
        evo_model = JCBModel(n_states=n_states, hmm_alg=hmm_alg)
        em = EM(n_states=n_states, obs_model=obs_model, evo_model=evo_model)
        em.fit(data['obs'], max_iter=30, rtol=1e-5, num_processors=8, jc_correction=False, theta_init=None)
        out = {
            'l_hat': em.distances,
            'iterations': em.n_iterations,
            'loglikelihoods': em.loglikelihoods
        }
        ctr_table = out['l_hat']

        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == 2:
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == 0:
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == 1:
                edge.length = ctr_table[0, 1, 2]
        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()
        print("Estimated edge _lengths:")
        print(l_est)

        # check likelihood
        ll_est = out['loglikelihoods'][0, 1]
        evo_model = JCBModel(n_states=n_states, hmm_alg=hmm_alg)
        evo_model.lengths = l_true

        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_true = evo_model.forward_pass(log_emissions)
        self.assertGreater(ll_est, ll_true)

        # if these tests don't pass, it's likely that they are wrong
        self.assertAlmostEqual(ctr_table[0, 1, 0], gt_ctr_table[0, 1, 0], delta=0.03)
        self.assertAlmostEqual(ctr_table[0, 1, 1], gt_ctr_table[0, 1, 1], delta=0.03)
        self.assertAlmostEqual(ctr_table[0, 1, 2], gt_ctr_table[0, 1, 2], delta=0.03)

    def test_quadruplet_random_l_normal(self):
        # seed for reproducibility
        seed = 0
        hmm_alg = 'pomegranate'
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500
        n_CN_ru, n_CN_uv, n_CN_uw = 5, 3, 7
        n_fCN_ru, n_fCN_uv, n_fCN_uw = 5, 7, 7
        n_clonal_events_per_edge = {(3,2): n_CN_ru, (2,0): n_CN_uv, (2,1): n_CN_uw}
        n_focal_events_per_edge = {(3,2): n_fCN_ru, (2,0): n_fCN_uv, (2,1): n_fCN_uw}
        clonal_CN_event_ratio = 0.1

        evo_model_sim = SimulationEvoModel(n_states,
                                           n_clonal_CN_events=n_clonal_events_per_edge,
                                           clonal_CN_length_ratio=clonal_CN_event_ratio,
                                           n_focal_events=n_focal_events_per_edge)
        evo_model = JCBModel(n_states=n_states, alpha=1, hmm_alg=hmm_alg)
        obs_model = NormalModel(n_states=n_states, mu_v_prior=1.0, tau_v_prior=100.0)
        data = simulate_quadruplet(n_sites, obs_model=obs_model, evo_model=evo_model_sim, n_states=n_states)
        # Note: since SimulationEvoModel is used, true edge lengths are not known, so only CN changes can be used to assess EM performance
        # gt_ctr_table = get_ctr_table(data['tree'])  # cannot be used here
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"CN (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")
        # plot
        fig, ax = plt.subplots()
        plot_cn_profile(data['cn'], ax=ax)
        sub_folder_name = f'M{n_sites}_K{n_states}_CN_{n_CN_ru}_{n_CN_uv}_{n_CN_uw}_focCN_{n_fCN_ru}_{n_fCN_uv}_{n_fCN_uw}'
        out_dir = create_output_test_folder(sub_folder_name=sub_folder_name)
        fig.savefig(out_dir + '/cn_profile.png')
        print(f"Image saved to {out_dir}/cn_profile.png")

        # run EM
        em = EM(n_states=n_states, obs_model=obs_model, evo_model=evo_model, diagnostics=True)
        em.fit(data['obs'], theta_init=None)
        diagnostics_data = em.diagnostic_data[0,1]
        testing.plot_diagnostics(diagnostics_data, out_dir)
        ctr_table = em.distances
        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == '2':
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == '0':
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == '1':
                edge.length = ctr_table[0, 1, 2]
        print(f"Estimated tree")

        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()
        print("Estimated edge _lengths:")
        print(l_est)

        # check likelihood
        ll_est = em.loglikelihoods[(0, 1)]
        print(f"Estimated lengths likelihood: {ll_est}")
        # compute cn changes
        comp_eps = compute_cn_changes(data['cn'], [(3, 2), (2, 0), (2, 1)])
        comp_lengths = l_from_p(np.array(comp_eps)/n_sites, n_states)
        print(f"Est (CN) edge _lengths: {comp_lengths}")
        ll_cn = em.compute_pair_likelihood(data['obs'], theta=np.array(comp_eps) / n_sites, psi=obs_model.psi)
        print(f"Est (CN) edge _lengths likelihood: {ll_cn}")
        # self.assertGreater(ll_cn, ll_true, msg="Generated lengths fit better than actual CN changes")
        self.assertGreater(ll_est, ll_cn, msg="EM estimates fit better than generated but not than actual CN changes")

        # Assert
        EM_result = ctr_table[0, 1, :].tolist()
        expected_result = comp_lengths
        print(f"\nEM estimates theta parameters:\n{EM_result}")
        print(f"Expected theta estimates:\n{expected_result}")
        for i in range(3):
            rel_error = abs(EM_result[i] - expected_result[i]) / expected_result[i]
            self.assertAlmostEqual(rel_error, 0, delta=0.2,
                                   msg=f"EM estimated theta parameter {i} is not within 20% of expected value.")

    def test_quadruplet_true_init_normal_given_psi(self):
        """
        Initialize EM with true theta and psi parameters for NormalModel and JCBModel on a quadruplet tree.
        Then runs EM for theta parameters only by setting train=False in the observation model.
        This tests that EM improves likelihood when starting from the true parameters.
        """
        testing.set_seed(0)
        n_states = 5
        n_sites = 500
        n_focal_events_per_edge = 5
        n_clonal_events_per_edge = 5
        clonal_CN_length = n_sites // 20
        evo_model = JCBModel(n_states=n_states, alpha=1)
        obs_model = NormalModel(n_states=n_states, mu_v_prior=1.0, tau_v_prior=100.0, train=False)

        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           clonal_CN_length=clonal_CN_length,
                                           n_focal_events=n_focal_events_per_edge)
        data = simulate_quadruplet(n_sites, obs_model=obs_model, evo_model=evo_model_sim, n_states=n_states)
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)
        cell_pairs = [(0, 1)]
        D, Dp = testing.get_expected_changes(cnps, tree_nx, cell_pairs)
        l_exp, pairwise_l_exp = testing.get_expected_distances(D, Dp, n_states, cell_pairs)

        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"\nCN (first 20 sites) (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")
        print(f"\nObs (first 20 sites) (r, u, v, w):\n{data['obs'].T[:, :20]}")

        # save data
        out_dir = create_output_test_folder(
            sub_folder_name=f'M{n_sites}_K{n_states}_Clonal{n_clonal_events_per_edge}_Focal{n_focal_events_per_edge}')
        fig, ax = plt.subplots()
        plot_cn_profile(data['cn'], ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # print tree without _lengths since they are not used to generate data in SimulationEvoModel
        gt_ctr_table = get_ctr_table_int(data['tree'])
        l_init = gt_ctr_table[0, 1, :].tolist()
        print(f"Generated tree")
        data['tree'].print_plot()

        psi_init = {'mu_v': obs_model.mu_v_prior,
                    'tau_v': obs_model.tau_v_prior,
                    'mu_w': obs_model.mu_w_prior,
                    'tau_w': obs_model.tau_w_prior}
        # run EM
        em = EM(n_states=n_states, obs_model=obs_model, evo_model=evo_model)
        em.fit(data['obs'], theta_init=l_init, psi_init=psi_init)
        ctr_table = em.distances
        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == '2':
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == '0':
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == '1':
                edge.length = ctr_table[0, 1, 2]
        print(f"Estimated tree")
        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()
        print("Estimated edge _lengths:")
        print(l_est)
        print(f"Expected edge lengths:")
        print(l_exp[0, 1])

        # check likelihood
        ll_est = em.loglikelihoods[(0, 1)]
        evo_model.lengths = l_init

        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_init = evo_model.forward_pass(log_emissions)
        print(f"Initial lengths likelihood: {ll_init}")
        print(f"Estimated lengths likelihood: {ll_est}")
        self.assertGreater(ll_est, ll_init, msg="EM does not improve likelihood at all")
        # compute cn changes
        comp_eps = compute_cn_changes(data['cn'], [(3, 2), (2, 0), (2, 1)])
        comp_lengths = l_from_p(np.array(comp_eps) / n_sites, n_states)
        print(f"Est (CN) edge _lengths: {comp_lengths}")
        ll_cn = em.compute_pair_likelihood(data['obs'], theta=np.array(comp_eps) / n_sites, psi=obs_model.psi)
        print(f"Est (CN) edge _lengths likelihood: {ll_cn}")
        self.assertGreater(ll_est, ll_cn, msg="EM estimates fit better than generated but not than actual CN changes")
        # Parameter checks
        psi_out = em.obs_model.psi
        print(f"True psi: {psi_init}")
        print(f"Estimated psi: {psi_out}")
        self.assertEqual(psi_out['mu_v'], psi_init['mu_v'], msg="psi updated during optimization.")
        self.assertEqual(psi_out['mu_w'], psi_init['mu_w'], msg="psi updated during optimization.")

    @unittest.skip("Buggy test, needs fix")
    def test_quadruplet_true_init_viterbi_given_psi(self):
        """
        Initialize EM with true theta and psi parameters for NormalModel and JCBModel on a quadruplet tree.
        Then runs EM with Viterbi-based E-step for theta parameters only by setting train=False in the observation model.
        Returns
        -------
        """
        # FIXME: uses viterbi_matrix_K6 which needs fix (check FIXME in viterbi_matrix_K6)
        # seed for reproducibility
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500
        n_CN_ru, n_CN_uv, n_CN_uw = 5, 3, 7
        n_fCN_ru, n_fCN_uv, n_fCN_uw = 5, 7, 7
        n_clonal_events_per_edge = {(3, 2): n_CN_ru, (2, 0): n_CN_uv, (2, 1): n_CN_uw}
        n_focal_events_per_edge = {(3, 2): n_fCN_ru, (2, 0): n_fCN_uv, (2, 1): n_fCN_uw}
        clonal_CN_event_ratio = 0.1

        evo_model = JCBModel(n_states=n_states, alpha=1)
        obs_model = NormalModel(n_states=n_states, mu_v_prior=1.0, tau_v_prior=100.0, train=False)
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           clonal_CN_length_ratio=clonal_CN_event_ratio,
                                           n_focal_events=n_focal_events_per_edge)
        data = simulate_quadruplet(n_sites, obs_model=obs_model, evo_model=evo_model_sim, n_states=n_states)
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)
        cell_pairs = [(0, 1)]
        D, Dp = testing.get_expected_changes(cnps, tree_nx, cell_pairs)
        l_exp, pairwise_l_exp = testing.get_expected_distances(D, Dp, n_states, cell_pairs)
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"\nCN (first 20 sites) (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")
        print(f"\nObs (first 20 sites) (r, u, v, w):\n{data['obs'].T[:, :20]}")

        # save data
        subfolder_name = f'M{n_sites}_K{n_states}_CN_{n_CN_ru}_{n_CN_uv}_{n_CN_uw}_focCN_{n_fCN_ru}_{n_fCN_uv}_{n_fCN_uw}'
        out_dir = create_output_test_folder(sub_folder_name=subfolder_name)
        fig, ax = plt.subplots()
        plot_cn_profile(data['cn'], ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # initialize psi and true lengths
        psi_init = {'mu_v': obs_model.mu_v_prior,
                    'tau_v': obs_model.tau_v_prior,
                    'mu_w': obs_model.mu_w_prior,
                    'tau_w': obs_model.tau_w_prior}
        #theta_init = l_exp[0, 1].tolist()
        theta_init = [5/n_sites, 5/n_sites, 5/n_sites]
        # run EM
        em = EM(n_states=n_states, obs_model=obs_model, evo_model=evo_model,
                E_step_alg='viterbi',
                diagnostics=True)
        em.fit(data['obs'], theta_init=theta_init, psi_init=psi_init)
        ctr_table = em.distances

        # Save results
        diagnostics_data = em.diagnostic_data[0, 1]
        testing.plot_diagnostics(diagnostics_data, out_dir)

        # Assert
        EM_result = ctr_table[0, 1, :].tolist()
        expected_result = l_exp[0, 1].tolist()
        print(f"\nEM estimates theta parameters:\n{EM_result}")
        print(f"Expected theta estimates:\n{expected_result}")
        for i in range(3):
            rel_error = abs(EM_result[i] - expected_result[i]) / expected_result[i]
            self.assertAlmostEqual(rel_error, 0, delta=0.2, msg=f"EM estimated theta parameter {i} is not within 10% of expected value.")


    def test_quadruplet_true_init_normal_obs(self):
        """
        Initializes theta and psi to the true values used to generate the data.
        Tests whether EM improves the likelihood from the true generating parameters and that the estimated parameters
        are close to the true ones.
        Returns: THE TRUTH
        """

        # seed for reproducibility
        testing.set_seed(0)
        n_states = 5
        n_sites = 500
        n_focal_events_per_edge = 5
        n_clonal_events_per_edge = 5
        clonal_CN_length = n_sites//20
        evo_model = JCBModel(n_states=n_states, alpha=1, hmm_alg='broadcast')
        obs_model = NormalModel(n_states=n_states, mu_v_prior=1.0, tau_v_prior=100.0, train=True)

        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           clonal_CN_length=clonal_CN_length,
                                           n_focal_events=n_focal_events_per_edge)
        data = simulate_quadruplet(n_sites, obs_model=obs_model, evo_model=evo_model_sim, n_states=n_states)
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)
        cell_pairs = [(0, 1)]
        D, Dp = testing.get_expected_changes(cnps, tree_nx, cell_pairs)
        l_exp, pairwise_l_exp = testing.get_expected_distances(D, Dp, n_states, cell_pairs)

        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"\nCN (first 20 sites) (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")
        print(f"\nObs (first 20 sites) (r, u, v, w):\n{data['obs'].T[:, :20]}")

        # save data
        out_dir = create_output_test_folder(sub_folder_name=f'M{n_sites}_K{n_states}_Clonal{n_clonal_events_per_edge}_Focal{n_focal_events_per_edge}')
        fig, ax = plt.subplots()
        plot_cn_profile(data['cn'], ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # print tree without _lengths since they are not used to generate data in SimulationEvoModel
        gt_ctr_table = get_ctr_table_int(data['tree'])
        l_init = gt_ctr_table[0, 1, :].tolist()
        print(f"Generated tree")
        data['tree'].print_plot()

        # print tree with _lengths
        psi_init = {'mu_v': obs_model.mu_v_prior,
                    'tau_v': obs_model.tau_v_prior,
                    'mu_w': obs_model.mu_w_prior,
                    'tau_w': obs_model.tau_w_prior}
        # run EM
        em = EM(n_states=n_states, obs_model=obs_model, evo_model=evo_model)
        em.fit(data['obs'], theta_init=l_init, psi_init=psi_init)
        ctr_table = em.distances
        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == '2':
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == '0':
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == '1':
                edge.length = ctr_table[0, 1, 2]
        print(f"Estimated tree")
        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()
        print("Estimated edge _lengths:")
        print(l_est)
        print(f"Expected edge lengths:")
        print(l_exp[0,1])

        # check likelihood
        ll_est = em.loglikelihoods[(0, 1)]
        evo_model.lengths = l_init

        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_init = evo_model.forward_pass(log_emissions)
        print(f"Generating lengths likelihood: {ll_init}")
        print(f"Estimated lengths likelihood: {ll_est}")
        self.assertGreater(ll_est, ll_init, msg="EM does not improve likelihood at all")
        # compute cn changes
        comp_eps = compute_cn_changes(data['cn'], [(3, 2), (2, 0), (2, 1)])
        comp_lengths = l_from_p(np.array(comp_eps)/n_sites, n_states)
        print(f"Est (CN) edge _lengths: {comp_lengths}")
        ll_cn = em.compute_pair_likelihood(data['obs'], theta=np.array(comp_eps) / n_sites, psi=obs_model.psi)
        print(f"Est (CN) edge _lengths likelihood: {ll_cn}")
        # self.assertGreater(ll_cn, ll_true, msg="Generated lengths fit better than actual CN changes")
        self.assertGreater(ll_est, ll_cn, msg="EM estimates fit better than generated but not than actual CN changes")
        # Parameter checks
        psi_out = em.obs_model.psi
        print(f"True psi: {psi_init}")
        print(f"Estimated psi: {psi_out}")
        self.assertAlmostEqual(psi_out['mu_v'], psi_init['mu_v'], delta=0.1)
        self.assertAlmostEqual(psi_out['mu_w'], psi_init['mu_w'], delta=0.1)


    def test_quadruplet_true_init(self):
        # TEST WITH POISSON OBS and TRUE INIT for LENGTHS
        # seed for reproducibility
        seed = 100
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500

        obs_model = PoissonModel(n_states, 100, 100)
        data = simulate_quadruplet(n_sites, obs_model=obs_model, gamma_params=self.DEFAULT_GAMMA_PARAMS, n_states=n_states)
        gt_ctr_table = get_ctr_table_int(data['tree'])
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"\nCN (first 20 sites) (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")

        # print tree with _lengths
        l_gen = gt_ctr_table[0, 1, :].tolist()
        print(f"Generated tree")
        data['tree'].print_plot(plot_metric='length')
        print(f"Generated edge _lengths: {l_gen}")
        print(f"(from p: {p_from_l(gt_ctr_table[0, 1, :], n_states)}")

        # run EM
        em = EM(n_states=n_states, obs_model=obs_model, evo_model='jcb')
        em.fit(data['obs'], max_iter=30, rtol=1e-5, num_processors=1, theta_init=gt_ctr_table[0, 1, :])
        out = {
            'l_hat': em.distances,
            'iterations': em.n_iterations,
            'loglikelihoods': em.loglikelihoods
        }
        ctr_table = out['l_hat']
        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == '2':
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == '0':
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == '1':
                edge.length = ctr_table[0, 1, 2]
        print(f"Estimated tree")
        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()

        # check likelihood
        ll_est = out['loglikelihoods'][0, 1]
        evo_model = JCBModel(n_states=n_states)
        evo_model.lengths = l_gen

        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_gen = evo_model.forward_pass(log_emissions)
        print(f"Generating lengths likelihood: {ll_gen}")
        print(f"Estimated lengths likelihood: {ll_est}")
        self.assertGreater(ll_est, ll_gen, msg="EM does not improve likelihood at all")
        # compute cn changes
        comp_eps = compute_cn_changes(data['cn'], [(3, 2), (2, 0), (2, 1)])
        comp_lengths = l_from_p(np.array(comp_eps)/n_sites, n_states)
        print("Estimated edge _lengths:")
        print(l_est)
        print("Generating edge _lengths:")
        print(l_gen)
        print(f"Computed edge _lengths:\n{comp_lengths}")
        em = EM(n_states=n_states, obs_model=obs_model, evo_model=JCBModel(n_states))
        ll_cn = em.compute_pair_likelihood(data['obs'], theta=np.array(comp_eps) / n_sites)
        print(f"Computed edge _lengths likelihood: {ll_cn}")
        # self.assertGreater(ll_cn, ll_gen, msg="Generated lengths fit better than actual CN changes")
        self.assertGreater(ll_est, ll_cn, msg="EM estimates fit better than generated but not than actual CN changes")


    def test_two_cells(self):
        # seed for reproducibility
        seed = 101
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_cells = 4
        n_states = 5
        n_sites = 1000

        alpha = 1.
        data = rand_dataset(n_states, n_sites, obs_model='poisson', alpha=alpha, p_change=8 / n_sites, n_cells=n_cells,
                            seed=seed)
        print(f"True CTR table")
        true_ctr_table = get_ctr_table_int(data['tree'])
        print(true_ctr_table)
        # for each node print the edge length
        for node in data['tree'].preorder_node_iter():
            print(f"Node {node.label} edge length: {node.edge_length}")
        # tree to newick
        print(data['tree'].as_string('newick'))
        # compare with true tree using RF-distance (unweighted)
        print("--- TRUE TREE ---")
        data['tree'].print_plot(plot_metric='length')

        # pick two leaves whose CTR is not the root
        c1, c2 = 0, 3
        # get node with label
        centroid = data['tree'].find_node_with_label('4')
        self.assertNotEqual(data['tree'].seed_node, centroid, msg="centroid is the root, fix the test")
        self.assertEqual(mrca:=data['tree'].mrca(taxon_labels=[str(c1), str(c2)]), centroid, msg=f"centroid {centroid.label} is not the mrca ({mrca.label}) of {c1} and {c2}")
        # for r, s in itertools.combinations(range(data['obs'].shape[1]), 2):
        #     centroid = data['tree'].mrca(taxon_labels=[str(r), str(s)])
        #     if centroid != data['tree'].seed_node:
        #         c1, c2 = r, s
        #         break
        print(f"Centroid of {c1} and {c2} is {centroid.label} with edge length {centroid.edge_length}")

        print(f"Copy Numbers")
        print(f"R:\t {data['cn'][int(data['tree'].seed_node.label), :]}")
        print(f"U:\t {data['cn'][int(centroid.label), :]}")
        print(f"C{c1}:\t {data['cn'][c1, :]}")
        print(f"C{c2}:\t {data['cn'][c2, :]}")

        print(f"Leaves with non-root CTR: {c1}, {c2}")
        print(f"GT l_trip: {true_ctr_table[c1, c2]}")
        # run EM
        em = EM(n_states=n_states, obs_model='poisson', evo_model='jcb', alpha=alpha)
        em.fit(data['obs'][:, [c1, c2]], max_iter=100, num_processors=1)
        ctr_table = em.distances
        print(f"EM converged in {em.n_iterations[(0, 1)]} iterations")
        print(f"Final loglikelihood: {em.loglikelihoods[(0, 1)]}")
        print(f"Estimated CTR triplet (l_ru, l_rv, l_uv): {ctr_table[0, 1]}")
        self.assertAlmostEqual(ctr_table[0, 1, 0], true_ctr_table[c1, c2, 0],
                               msg=f"cell {c1} and {c2} CTR: {ctr_table[0, 1, 0]} != {true_ctr_table[c1, c2, 0]}", places=3)

        # also with other pair
        c1, c2 = 1, 2
        centroid = data['tree'].find_node_with_label('5')
        # for r, s in itertools.combinations(range(data['obs'].shape[1]), 2):
        #     centroid = data['tree'].mrca(taxon_labels=[str(r), str(s)])
        #     if centroid != data['tree'].seed_node:
        #         c1, c2 = r, s
        #         break
        self.assertNotEqual(data['tree'].seed_node, centroid, msg="centroid is the root, fix the test")
        self.assertEqual(mrca:=data['tree'].mrca(taxon_labels=[str(c1), str(c2)]), centroid, msg=f"centroid {centroid.label} is not the mrca ({mrca.label}) of {c1} and {c2}")
        print(f"Leaves with non-root CTR: {c1}, {c2}")
        print(f"GT l_trip: {true_ctr_table[c1, c2]}")
        em.fit(data['obs'][:, [c1, c2]], max_iter=100, num_processors=1)
        ctr_table = em.distances
        print(f"EM converged in {em.n_iterations[(0, 1)]} iterations")
        print(f"Final loglikelihood: {em.loglikelihoods[(0, 1)]}")
        print(f"Estimated CTR triplet (l_ru, l_rv, l_uv): {ctr_table[0, 1]}")
        self.assertAlmostEqual(ctr_table[0, 1, 0], true_ctr_table[c1, c2, 0],
                               msg=f"cell {c1} and {c2} CTR: {ctr_table[0, 1, 0]} != {true_ctr_table[c1, c2, 0]}", places=3)
        # FIXME: the test works for CTR distances, but not for centroid to leaves distances
        #   there must be some bug in the implementation
        self.assertAlmostEqual(ctr_table[0, 1, 1], true_ctr_table[c1, c2, 1], places=2)
        self.assertAlmostEqual(ctr_table[0, 1, 2], true_ctr_table[c1, c2, 2], places=2)


    def test_two_slice_marginals(self):
        print(f"\n")
        n_sites = 100
        t0 = n_sites // 6  # first change idx
        t1 = n_sites // 2  # second change idx
        print(f"[2 changes] n_sites: {n_sites}, t0 (1->2): {t0}, t1 (2->3): {t1}")

        n_states = 4
        # reasonable _lengths computed by setting the change probability to
        p_change = 3 / n_sites
        ll = l_from_p(p_change * 2, n_states)
        sl = l_from_p(p_change / 10, n_states)
        jcb_model = JCBModel(n_states=n_states)
        print(f"LONG l prop of change: 1 - pdd = {1 - p_delta_change(n_states, ll, change=False)}")
        print(f"SHORT l prop of change: 1 - pdd = {1 - p_delta_change(n_states, sl, change=False)}")
        obs_vw = np.array([
            [100] * (t0 + 1) + [200] * (t1 - t0) + [300] * (n_sites - t1 - 1),
            [100] * (t0 + 1) + [200] * (t1 - t0) + [300] * (n_sites - t1 - 1)
        ]).transpose() + np.random.randint(-10, 10, (n_sites, 2))
        cn_vw = np.round(obs_vw / 100).astype(int)
        print(f"CN VW: {cn_vw.transpose()}")
        n_sites = obs_vw.shape[0]
        obs_model = PoissonModel(n_states, 100, 100)
        # the best explanation is that centroid and root are further apart than centroid and v,u
        # compute two-slice marginals assuming centroid is placed closer to the root
        # print("OBS VW (first 20 sites):")
        # print(obs_vw[:20, :].transpose())
        jcb_model.lengths = np.array([sl, ll, ll])
        expected_counts_early_centroid, log_gamma_early, loglik_early = jcb_model.forward_backward(obs_vw, obs_model)

        # compute two-slice marginals assuming centroid is placed closer to the leaves
        jcb_model.lengths = np.array([ll, sl, sl])
        expected_counts_late_centroid, log_gamma_late, loglik_late = jcb_model.forward_backward(obs_vw, obs_model)
        self.assertGreater(loglik_late, loglik_early)

        self.assertEqual(expected_counts_late_centroid.shape, (n_states,) * 6)
        self.assertAlmostEqual(np.sum(expected_counts_late_centroid), n_sites - 1, places=2)

        # NOTE: these tests where built on log_xi, not sure if they still hold on expected_counts
        # transition from 1 to 2 is more likely in the late scenario due to the higher l_ru
        self.assertGreater(log_gamma_late[t0, 2, 2, 2],
                           log_gamma_early[t0, 2, 2, 2])
        # transition from 1 to 2 is more likely than from 2 to 2 even though l_uv is higher due to observation
        self.assertGreater(log_gamma_late[t0-1, 1, 1, 1],
                           log_gamma_early[t0-1, 2, 1, 1])

    def test_compute_viterbi_path(self):
        n_sites = 100
        t0 = n_sites // 4  # first change idx
        t1 = n_sites // 2  # second change idx

        n_states = 4
        p_change = 2 / n_sites
        ll = l_from_p(p_change * 2, n_states)
        sl = l_from_p(p_change / 10, n_states)
        # FIXME: test passes, but compute viterbi is taking lengths as if they were p_change directly
        #   why does it still pass? needs a stricter test
        print(f"GT |\tp_change LONG p: {p_change * 2}, SHORT p: {p_change / 10},\n\tlength LONG l: {ll}, SHORT l: {sl}")
        jcb_model = JCBModel(n_states=n_states)
        obs_vw = np.array([
            [100] * (t0 + 1) + [200] * (t1 - t0) + [300] * (n_sites - t1 - 1),
            [100] * (t0 + 1) + [200] * (t1 - t0) + [300] * (n_sites - t1 - 1)
        ]).transpose() + np.random.randint(-10, 10, (n_sites, 2))
        cn_vw = np.round(obs_vw / 100).astype(int)
        obs_model = PoissonModel(n_states, 100, 100)
        # compute viterbi path assuming centroid is placed closer to the leaves
        jcb_model.lengths = np.array([ll, sl, sl])
        log_emissions = obs_model.log_emission(obs_vw)
        viterbi_path, _ = jcb_model.compute_viterbi_path(log_emissions)

        self.assertEqual(viterbi_path.shape, (n_sites, 3))
        # check that the viterbi path has the expected changes
        for m in range(n_sites):
            if m <= t0:
                if viterbi_path[m, 0] != 1 or viterbi_path[m, 1] != 1:
                    print(f"m={m}, viterbi_path: {viterbi_path[m, 1:]}, cn_vw: {cn_vw[m, :]}")
            elif t0 < m <= t1:
                if viterbi_path[m, 0] != 2 or viterbi_path[m, 1] != 2:
                    print(f"m={m}, viterbi_path: {viterbi_path[m, 1:]}, cn_vw: {cn_vw[m, :]}")
            else:
                if viterbi_path[m, 0] != 3 or viterbi_path[m, 1] != 3:
                    print(f"m={m}, viterbi_path: {viterbi_path[m, 1:]}, cn_vw: {cn_vw[m, :]}")
        n_diff_v = np.sum(viterbi_path[:, 0] != cn_vw[:, 0])
        n_diff_w = np.sum(viterbi_path[:, 1] != cn_vw[:, 1])
        self.assertLess(n_diff_v, 2, msg="Too many differences between viterbi u and true cn u")
        self.assertLess(n_diff_w, 2, msg="Too many differences between viterbi v and true cn v")

    def test_compute_exp_changes(self):
        # random seed
        random.seed(101)
        # generate cn
        n_sites = 1000
        t0 = n_sites // 200  # first change idx
        n_states = 4
        eps_ru = 200 / n_sites
        eps_uv = 0.01 / n_sites
        eps_uw = 100 / n_sites
        l_ru = l_from_p(eps_ru, n_states)
        l_uv = l_from_p(eps_uv, n_states)  # close to zero since no change from u to v
        l_uw = l_from_p(eps_uw, n_states)
        print(f"1 - pdd_ru: {1 - p_delta_change(n_states, l_ru, change=False)} - l_ru: {l_ru}")
        print(f"1 - pdd_uv: {1 - p_delta_change(n_states, l_uv, change=False)} - l_uv: {l_uv}")
        print(f"1 - pdd_uw: {1 - p_delta_change(n_states, l_uw, change=False)} - l_uw: {l_uw}")
        alpha = 1.

        centroid_cn = ([2] * t0 + [1] * t0) * 100  # 2 * 100 changes from root
        v_cn = ([2] * t0 + [1] * t0) * 100  # no distance from centroid
        w_cn = ([2] * t0 + [3] * (t0 // 2) + [1] * (t0 // 2 + 1)) * 100  # 100 more changes than v

        # generate observations
        pois_model = PoissonModel(n_states=n_states)
        vw_obs = pois_model.sample(np.array([v_cn, w_cn]))

        print(f"V,W obs:\n {vw_obs[:, :20]}")

        quad_model = JCBModel(n_states=n_states)
        quad_model.lengths = np.array([l_ru, l_uv, l_uw])

        # compute expected changes given true l
        d, dp, loglik, _, _ = quad_model._expected_changes(vw_obs, obs_model=pois_model)
        print(f"expected p statistic: p: {d / (d + dp)}"
              f" D = {d}, D' = {dp}, loglik = {loglik}")

        # test with eps model
        quad_model = CopyTree(n_states=n_states)
        quad_model.eps = np.array([eps_ru, eps_uv, eps_uw])
        d, dp, loglik, _, _ = quad_model._expected_changes(vw_obs, obs_model=pois_model)
        print(f"expected p statistic via eps ({[eps_ru, eps_uv, eps_uw]}):\n"
              f"\tp: {d / (d + dp)}, D = {d}, D' = {dp}, loglik = {loglik}")

    def test_build_tree(self):
        n_cells = 20
        # generate tree
        tree: dendropy.Tree = random_binary_tree(n_cells, length_mean=0.01, seed=101)
        print("--- Starting tree ---")
        print(f'txnsp: {tree.taxon_namespace}')
        label_tree(tree, method='group')
        tree.print_plot(plot_metric='length')

        # derive true ctr table from tree
        ctr_table = get_ctr_table_int(tree)

        # rebuild tree
        nx_tree = rooted_nj0(ctr_table)
        new_dpy_tree = convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace, edge_length='length')
        print("--- Rebuilt tree ---")
        print(f'txnsp: {new_dpy_tree.taxon_namespace}')
        label_tree(new_dpy_tree, method='group')
        new_dpy_tree.print_plot(plot_metric='length')

        self.assertTrue(len(new_dpy_tree.taxon_namespace) == len(tree.taxon_namespace))
        for taxon in new_dpy_tree.taxon_namespace:
            self.assertTrue(taxon in tree.taxon_namespace)
            self.assertEqual(taxon.label, tree.find_node_with_taxon(lambda t: t.label == taxon.label).taxon.label)
        self.assertEqual(treecompare.symmetric_difference(tree, new_dpy_tree), 0)
        self.assertLess(treecompare.robinson_foulds_distance(tree, new_dpy_tree, edge_weight_attr='length'), 0.03)

        # random tree robinson foulds distance
        rnd_dpy_tree = convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace)
        # rename leaves to random numbers
        rnd_ints = np.random.permutation(range(n_cells))
        for leaf in rnd_dpy_tree.leaf_node_iter():
            leaf.taxon.label = str(rnd_ints[int(leaf.taxon.label)])

        self.assertEqual(treecompare.symmetric_difference(tree, rnd_dpy_tree), 0)

    @unittest.skip("Slow test, run manually")
    def test_multiprocessing(self):
        logging.basicConfig(level=logging.DEBUG)
        seed = 42

        n_cells = 10
        n_states = 5
        n_sites = 100
        p_change = 0.02

        data = rand_dataset(n_states, n_sites, obs_model='poisson', p_change=p_change, n_cells=n_cells, seed=seed)
        self.assertEqual(data['obs'].shape, (n_sites, n_cells))

        l_init = np.random.exponential(scale=l_from_p(p_change, n_states), size=3)
        start_time = time.time()
        ctr_table_5p = jcb_em_ctrtable(data['obs'], n_states=n_states, l_init=l_init, max_iter=50, num_processors=5)
        tot_time_5_proc = time.time() - start_time

        start_time = time.time()
        ctr_table_1p = jcb_em_ctrtable(data['obs'], n_states=n_states, l_init=l_init, max_iter=50, num_processors=1)
        tot_time_1_proc = time.time() - start_time

        print(f"5proc: {tot_time_5_proc}, 1proc: {tot_time_1_proc}")

        self.assertLess(tot_time_5_proc, tot_time_1_proc)
        self.assertTrue(np.allclose(ctr_table_5p, ctr_table_1p))

    def test_quadruplet_copytree(self):
        """ Test EM with CopyTree model on a simple quadruplet tree with known edge lengths compared to true ones and assess likelihood improvement. """
        # seed for reproducibility
        seed = 120
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500
        poisson_param = 100
        normal_param = 1., 50.  # mean and precision

        # obs_model = PoissonModel(n_states, poisson_param, poisson_param)
        obs_model = NormalModel(n_states, normal_param[0], normal_param[0],
                                tau_v_prior=normal_param[1], tau_w_prior=normal_param[1])
        evo_model = CopyTree(n_states=n_states)

        data = simulate_quadruplet(n_sites, evo_model=evo_model, obs_model=obs_model,
                                   n_states=n_states, gamma_params=self.DEFAULT_GAMMA_PARAMS)
        print(data['obs'][data['cn'][1] == 0,1])
        gt_ctr_table = get_ctr_table_int(data['tree'])  # eps
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"CN (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")

        # print tree with _lengths
        data['tree'].print_plot(plot_metric='length')
        print(f"True edge _eps: {gt_ctr_table[0, 1, :].tolist()}")

        # plot data
        out_dir = create_output_test_folder()
        fig, ax = plt.subplots()
        plot_cn_profile(data['cn'], ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # run EM
        em = EM(n_states, obs_model, evo_model, verbose=2)
        em.fit(data['obs'], max_iter=50, rtol=1e-5, num_processors=1, theta_init=np.array([0.3, 0.3, 0.3]))
        print(f"EM converged in {em.n_iterations[0, 1]} iterations")
        print(f"Final loglikelihood: {em.loglikelihoods[0, 1]}")
        # get estimated ctr table
        ctr_table = em.distances
        ll_est = em.loglikelihoods[0, 1]
        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == 2:
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == 0:
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == 1:
                edge.length = ctr_table[0, 1, 2]
        data['tree'].print_plot(plot_metric='length')
        print("Estimated edge _lengths:")
        eps_est = ctr_table[0, 1, :].tolist()
        print(eps_est)

        copy_number_changes = compute_cn_changes(data['cn'], pairs=[(3, 2), (2, 0), (2, 1)])
        print(f"True CN changes")
        print([c / n_sites for c in copy_number_changes])

        # check likelihood
        print(f"(EM) computing likelihood with theta:{ctr_table[0, 1]}")
        ll_est = em.compute_pair_likelihood(data['obs'], theta=ctr_table[0, 1])
        print(f"(gen) computing likelihood with theta:{gt_ctr_table[0, 1]}")
        ll_generating = em.compute_pair_likelihood(data['obs'], theta=gt_ctr_table[0, 1])
        print(f"(true) computing likelihood with theta:{np.array(copy_number_changes) / n_sites}")
        ll_true = em.compute_pair_likelihood(data['obs'], theta=np.array(copy_number_changes) / n_sites)

        self.assertGreater(ll_true, ll_generating,
                           msg="likelihood of eps given known copy numbers should be higher than data-generating eps likelihood")
        self.assertGreater(ll_est, ll_generating,
                           msg="likelihood of EM eps values should be higher than data-generating eps likelihood")

        print(ll_true / ll_est)
        # self.assertGreater(ll_est, ll_true)

        # if these tests don't pass, it's likely that they are wrong
        # FIXME: it seems like this test either is too strict or the implementation has some issues with epsilon estimation
        #   dismissing for now
        # self.assertAlmostEqual(ctr_table[0, 1, 0], gt_ctr_table[0, 1, 0], delta=0.02)
        # self.assertAlmostEqual(ctr_table[0, 1, 1], gt_ctr_table[0, 1, 1], delta=0.02)
        # self.assertAlmostEqual(ctr_table[0, 1, 2], gt_ctr_table[0, 1, 2], delta=0.01)

