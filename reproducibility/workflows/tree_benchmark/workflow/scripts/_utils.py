"""
Compare performance of Cellmates tree algorithm vs. neighbor-joining in RF distance
on simulated data.
"""
import os
import time
import random

import anndata
import dendropy
import numpy as np
import pandas as pd
from dendropy import Tree
from skbio import DistanceMatrix
from skbio.tree import nj
from Bio import Phylo
import subprocess
import io

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cellmates.inference.neighbor_joining import rooted_nj0, std_nj_root, rooted_nj, get_root_dist_from_tripledist, \
    extend_dm
from cellmates.models.evo import JCBModel
from cellmates.simulation.datagen import rand_dataset
from cellmates.utils.math_utils import l_from_p, p_from_l, cn_changes_from_healthy
from cellmates.utils.testing import get_expected_changes
from cellmates.utils.tree_utils import convert_dendropy_to_networkx, normalized_rf_distance, \
    label_tree, convert_networkx_to_dendropy, random_binary_tree, nxtree_to_newick

N_STATES = 8
N_SITES = 1000


def simulate_tree(n_cells, p_change, tree_type='balanced'):
    # simulate tree and data
    match tree_type:
        case 'balanced':
            return simulate_tree_nonultra(n_cells, p_change)
        case 'balanced-ultra':
            return simulate_tree_ultra(n_cells, p_change)
        case 'unbalanced':
            return simulate_tree_unbalanced(n_cells, p_change)  # non-ultrametric unbalanced
        case _:
            raise ValueError(f"Unknown tree_type: {tree_type}")


def simulate_tree_unbalanced(n_cells, p_change):
    # hardcode imbalance level
    birth_rate, death_rate = 1.0, 0.5
    tns = dendropy.TaxonNamespace([dendropy.Taxon(str(i)) for i in range(n_cells)], label='taxa')
    tree = dendropy.treesim.treesim.birth_death_tree(birth_rate, death_rate, num_extant_tips=n_cells, taxon_namespace=tns)
    # scale tree to desired total p_change
    max_dist = tree.max_distance_from_root()
    desired_max_dist = l_from_p(p_change, N_STATES)
    scale_factor = desired_max_dist / max_dist
    for e in tree.edges():
        if e.length is not None:
            e.length *= scale_factor
    label_tree(tree)
    evo_model = JCBModel(n_states=N_STATES)
    dat = rand_dataset(N_STATES, N_SITES, evo_model, n_cells=n_cells, tree=tree)
    return dat['tree'], dat['cn']


def simulate_tree_nonultra(n_cells, p_change):
    evo_model = JCBModel(n_states=N_STATES)
    dat = rand_dataset(N_STATES, N_SITES, evo_model, n_cells=n_cells, p_change=p_change)
    return dat['tree'], dat['cn']


def simulate_tree_ultra(n_cells, tot_p_change):
    evo_model = JCBModel(n_states=N_STATES)
    full_length = l_from_p(tot_p_change, N_STATES)
    tree = random_binary_tree(n_cells, seed=None, full_length=full_length)
    dat = rand_dataset(N_STATES, N_SITES, evo_model, n_cells=n_cells, tree=tree)
    return dat['tree'], dat['cn']


def compute_triplet_distance_matrix(true_tree, cnp, n_cells):
    n_bins = cnp.shape[1]
    d, _ = get_expected_changes(cnp, convert_dendropy_to_networkx(true_tree, edge_attr='length'))
    dist_mat = np.zeros((n_cells, n_cells, 3))
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            p = d[i, j] / n_bins
            dist_mat[i, j] = l_from_p(p, N_STATES)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat


def neighbor_joining(dist_matrix, taxon_namespace):
    ids = [str(i) for i in range(dist_matrix.shape[0])]
    # use luv + luw distances
    dm = DistanceMatrix(dist_matrix[:, :, 1] + dist_matrix[:, :, 2], ids)
    nj_tree = nj(dm).root_at_midpoint()
    # convert to dendropy tree
    dpy_tree = Tree.get(data=str(nj_tree), schema="newick", taxon_namespace=taxon_namespace)
    label_tree(dpy_tree, method='int')
    dpy_tree.is_rooted = True
    # print("NJ LEAVES ", [(l.taxon.label, l.label) for l in dpy_tree.leaf_nodes()])
    return dpy_tree


def save_distmatrix(dist_matrix, file_name):
    dist_matrix = dist_matrix[:, :, 1] + dist_matrix[:, :, 2]
    # clip very small values to the minimum representable float in 5 decimal places
    n = dist_matrix.shape[0]
    dist_matrix = np.clip(dist_matrix, 0.00001, None)
    max_char = len(str(n - 1)) + 1
    with open(file_name, 'w+') as f:
        f.write(str(n) + '\n')
        for i in range(n):
            cell = str(i)
            x = np.array2string(dist_matrix[i], formatter={'float_kind': lambda x: "%.5f" % x})[1:-1].replace('\n', '')
            sep_char = ' ' * (max_char - len(cell))
            f.write(cell + sep_char + x + '\n')


def fast_me(dist_matrix, taxon_namespace, suffix=""):
    # run balanced minimum evolution (fast ME)
    # timestamp to make unique file names
    if suffix == "":
        suffix = f'_{random.randint(0,1000000)}'
    file_name = f'dist_mat{suffix}.PHYLIP'
    save_distmatrix(dist_matrix, file_name)
    tree_prefix = f'tree{suffix}.nwk'
    call = subprocess.run(['fastme', '-i', file_name, '-o', tree_prefix, '-m', 'B', '-s'], capture_output=True,
                          text=True)
    # wait for process to finish
    if call.returncode != 0:
        print("FASTME ERROR: ", call.stderr)
        raise RuntimeError("FASTME failed")
    # open in dpy
    newick_str = ''
    with open(tree_prefix, 'r') as f:
        newick_str = f.read().strip()
    os.remove(tree_prefix)
    dpy_tree = Tree.get(data=newick_str, schema="newick", taxon_namespace=taxon_namespace)
    label_tree(dpy_tree, method='int')
    dpy_tree.is_rooted = True
    # print("FASTME LEAVES ", [(l.taxon.label, l.label) for l in dpy_tree.leaf_nodes()])
    # clean up
    os.remove(file_name)
    return dpy_tree


def cellmates_infer(dist_matrix, taxon_namespace):
    nx_rec_tree = rooted_nj0(dist_matrix)
    cellmates_tree = convert_networkx_to_dendropy(nx_rec_tree, taxon_namespace=taxon_namespace, edge_length='length')
    return cellmates_tree


def cellmates_rnj_infer(dist_matrix, taxon_namespace):
    nx_rec_tree = rooted_nj(dist_matrix, edge_attr='length')
    rnj_tree = convert_networkx_to_dendropy(nx_rec_tree, taxon_namespace=taxon_namespace, edge_length='length')
    return rnj_tree


def std_njx(dist_matrix, taxon_namespace, root_dist=None):
    # standard neighbor-joining with correction for rooting
    # root_dist = np.sum(dist_matrix[:, :, 0] + dist_matrix[:, :, 1] + dist_matrix[:, :, 2], axis=0) / (dist_matrix.shape[0] - 1)
    if root_dist is None:
        root_dist = get_root_dist_from_tripledist(dist_matrix, agg_func='max')
    d1d2 = dist_matrix[:, :, 1] + dist_matrix[:, :, 2]
    nx_rec_tree = std_nj_root(d1d2, root_dist=root_dist, edge_attr='length',
                              taxa=[str(i) for i in range(dist_matrix.shape[0])])
    std_nj_tree = convert_networkx_to_dendropy(nx_rec_tree, taxon_namespace=taxon_namespace, edge_length='length',
                                               internal_nodes_label='int')
    return std_nj_tree


def std_njx_phylo(trip_dist, taxon_namespace, root_dist=None):
    # standard neighbor-joining with correction for rooting
    # root_dist = np.sum(dist_matrix[:, :, 0] + dist_matrix[:, :, 1] + dist_matrix[:, :, 2], axis=0) / (dist_matrix.shape[0] - 1)
    if root_dist is None:
        root_dist = get_root_dist_from_tripledist(trip_dist, agg_func='mean')
    distance_matrix = trip_dist[:, :, 1] + trip_dist[:, :, 2]
    distance_matrix = extend_dm(distance_matrix, root_dist)  # extend distance matrix with root distances (last row/col)
    root_idx = distance_matrix.shape[0]
    nj_tree = nj(
        DistanceMatrix(distance_matrix, ids=[str(i) for i in range(distance_matrix.shape[0])])).root_by_outgroup(
        outgroup=str(root_idx))
    dpy_tree = Tree.get(data=str(nj_tree), schema="newick", taxon_namespace=taxon_namespace)
    label_tree(dpy_tree, method='int')
    dpy_tree.is_rooted = True
    return dpy_tree


def plot_comparison(df, pdf_path):
    """Generate comparison plots and store them in a single PDF."""

    sns.set_theme(style="whitegrid")

    # Method renaming (safe replace)
    method_labels = {
        'rnj0': "RNJ0", 'nj-mid': "NJ-mid",
        'bme': "balME", 'rnj1': "RNJ*", 'nj-root': "NJ-root"
    }
    tree_type_labels = {
        'balanced': "Balanced",
        'balanced-ultra': "Balanced Ultrametric",
        'unbalanced': "Unbalanced"
    }
    df_plot = df.copy()
    df_plot['method'] = df_plot['method'].map(method_labels).fillna(df_plot['method'])
    df_plot['tree_type'] = df_plot['tree_type'].map(tree_type_labels).fillna(df_plot['tree_type'])

    with PdfPages(pdf_path) as pdf:
        # ---- Plot 1: RF vs p_change ----
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='p_change', y='normalized_rf', hue='method', data=df_plot, palette="Set2", ax=ax)
        sns.stripplot(x='p_change', y='normalized_rf', hue='method', data=df_plot,
                      dodge=True, alpha=0.3, color='white', palette="Set2", legend=False, ax=ax)
        ax.set_title("RF distance vs. p_change")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Plot 2: RF vs n_cells ----
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='n_cells', y='normalized_rf', hue='method', data=df_plot, palette="Set2", ax=ax)
        sns.stripplot(x='n_cells', y='normalized_rf', hue='method', data=df_plot,
                      dodge=True, alpha=0.3, color='white', palette="Set2", legend=False, ax=ax)
        ax.set_title("RF distance vs. number of cells")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Plot 3: Faceted view ----
        df_facet = df_plot.copy()
        df_facet['CNAs'] = (df_facet['p_change'] * N_SITES).round().astype(int)

        g = sns.catplot(
            x='n_cells', y='normalized_rf', hue='method',
            row='CNAs', col='tree_type', data=df_facet,
            kind='box', height=2, aspect=2, palette="Set2", margin_titles=True
        )
        g.set_axis_labels("N", "Normalized RF")
        g.set_titles(row_template="#CNAs = {row_name}", col_template="{col_name} Tree")
        sns.move_legend(g, title=None, loc='lower center',
                        ncol=df_facet['method'].nunique(),
                        frameon=False, bbox_to_anchor=(0.5, -0.05))

        # Save entire FacetGrid to PDF
        pdf.savefig(g.figure)
        plt.close(g.figure)

    return pdf_path


# def plot_cell_cn_profiles(cnp, title="", outfile=None):
#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     cn_colors = pl.cn_colors.map_cn_colors(cnp)
#     ax.imshow(cn_colors, aspect="auto", interpolation="none")
#     ax.set_ylabel(f"cells")
#     ax.set_xlabel(f"bins")
#     ax.set_xticks([])
#     ax.set_yticks([])
#     fig.tight_layout()
#     if title:
#         fig.suptitle(title)
#     if outfile is not None:
#         fig.savefig(outfile, dpi=300)
#     plt.close(fig)


def plot_cell_cn_tree(tree: Tree, cnp, title="", outfile=None):
    import scgenome.plotting as pl
    ad = anndata.AnnData(X=cnp,
                         var=pd.DataFrame(dict(
                             chr=['1'] * cnp.shape[1],
                             start=range(cnp.shape[1]),
                             end=range(1, cnp.shape[1] + 1))
                         ))
    ad.obs_names = [str(i) for i in range(cnp.shape[0])]
    for n in tree.preorder_internal_node_iter():
        n.label = None
    biotree = Phylo.read(io.StringIO(tree.as_string(schema="newick")), "newick")
    g = pl.plot_cell_cn_matrix_fig(ad, tree=biotree, layer_name=None)

    g['fig'].suptitle(title)
    if outfile is not None:
        g['fig'].savefig(outfile, dpi=300)
    plt.close(g['fig'])


def build_tree(dist_matrix, taxon_namespace, method='rnj0'):
    match method:
        case 'nj-mid':
            return neighbor_joining(dist_matrix, taxon_namespace)
        case 'bme':
            return fast_me(dist_matrix, taxon_namespace)
        case 'rnj0':
            return cellmates_infer(dist_matrix, taxon_namespace)
        case 'rnj1':
            return cellmates_rnj_infer(dist_matrix, taxon_namespace)
        case 'nj-root':
            return std_njx(dist_matrix, taxon_namespace)
        case _:
            raise ValueError(f"Unknown method: {method}")


def benchmark_method(dist_matrix, true_tree, n, p, seed, tree_type, method) -> dict:
    print(f"Running method: {method}")
    max_l = true_tree.max_distance_from_root()
    start = time.time()
    tree = build_tree(dist_matrix, true_tree.taxon_namespace, method=method)
    elapsed_time = time.time() - start
    rf = normalized_rf_distance(true_tree, tree)
    print(f"{method} tree:")
    tree.print_plot(plot_metric='length')
    row = {
        "n_cells": n,
        "p_change": p,
        "seed": seed,
        "method": method,
        "normalized_rf": rf,
        'time': elapsed_time,
        'max_length': max_l,
        'max_p': p_from_l(max_l, N_STATES),
        'tree_type': tree_type
    }
    return row


def benchmark_tree_inference(dist_matrix, true_tree, n, p, seed, tree_type, methods=None):
    if methods is None:
        # all methods
        methods = ['bme', 'rnj0', 'rnj1', 'nj-mid', 'nj-root']
    rows = []
    print("True tree:")
    true_tree.print_plot(plot_metric='length')
    for m in methods:
        row = benchmark_method(dist_matrix, true_tree, n, p, seed, tree_type, m)
        rows.append(row)
    return rows


def main():
    # make trees of varying sizes and branch lengths
    rerun = True
    n_cells = [10]
    # n_cells = [10, 20, 50, 100, 200]
    # p_changes = [0.001, 0.002, 0.005, 0.01, 0.1, 0.2]
    # p_changes = [0.01, 0.05, 0.1, 0.2, 0.3]
    # p_changes = [0.01, 0.02, 0.05, 0.1, 0.2]
    p_changes = [0.1]
    methods = ['nj-mid', 'rnj0', 'nj-root']
    tree_types = ['unbalanced']
    num_seeds = 2
    out_dir = "../../../../experiments/rf_benchmark_plots_new"
    # df_path = out_dir + "/tab.csv"
    df_path = out_dir + "/tab.csv"
    rows = []
    df_found = False
    results_df = None
    if not rerun and os.path.exists(df_path):
        results_df = pd.read_csv(df_path)
        n_cells = []
        df_found = True
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for n in n_cells:
        for p in p_changes:
            for seed in range(num_seeds):
                for tree_type in ['balanced', 'balanced-ultra', 'unbalanced']:
                    ## simulate tree and data
                    print(f"Simulating n={n}, p_change={p}, seed={seed}, tree_type={tree_type}")
                    np.random.seed(seed)
                    true_tree, cnp = simulate_tree(n, p, tree_type=tree_type)
                    if seed == 0:
                        true_tree.print_plot(plot_metric='length')
                        # print(f"leaves cnp[:10, :10]:\n{cnp[:min(10, n), :20]}")
                        # print("Max length: ", max_l, " -> p=", p_from_l(max_l, N_STATES))
                    dist_matrix = compute_triplet_distance_matrix(true_tree, cnp, n)
                    root_dist = l_from_p(np.array(cn_changes_from_healthy(cnp[:n])) / N_SITES, N_STATES)
                    ## infer trees and compute RF distances
                    rows = rows + benchmark_tree_inference(dist_matrix, true_tree, n, p, seed, tree_type=tree_type, methods=methods)
                    # print rf
                    print(pd.DataFrame(rows).tail(len(['cellmates', 'nj', 'rnj', 'njx']))[
                              ['method', 'normalized_rf', 'time']])
            # print avg times and rf
            avg_times = pd.DataFrame(rows).groupby('method')['time'].mean()
            avg_rf = pd.DataFrame(rows).groupby('method')['normalized_rf'].mean()
            print("Average times:\n", avg_times)
            print("Average RF:\n", avg_rf)

    ## save results
    results_df = pd.DataFrame(rows) if not df_found else results_df
    ##print stats
    print(results_df.groupby(['n_cells', 'p_change', 'method'])['normalized_rf'].describe())
    results_df.to_csv(df_path, index=False)
    print("Results saved to ", df_path)
    # plot comparison
    out_dirs = plot_comparison(results_df, os.path.join(out_dir, "rf_comparison_plots.pdf"))
    print(f"Plots saved to {out_dirs}")


if __name__ == "__main__":
    main()
    # rewrite_df_file()
