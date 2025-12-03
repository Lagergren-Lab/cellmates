"""
Compare performance of Cellmates tree algorithm vs. neighbor-joining in RF distance
on simulated data.
"""
import os
import time

import numpy as np
import pandas as pd
from dendropy import Tree
from skbio import DistanceMatrix
from skbio.tree import nj
import subprocess

import seaborn as sns
import matplotlib.pyplot as plt

from cellmates.inference.neighbor_joining import build_tree
from cellmates.models.evo import JCBModel
from cellmates.simulation.datagen import rand_dataset
from cellmates.utils.math_utils import l_from_p, p_from_l
from cellmates.utils.testing import get_expected_changes
from cellmates.utils.tree_utils import convert_dendropy_to_networkx, normalized_rf_distance, \
    label_tree, convert_networkx_to_dendropy, random_binary_tree

N_STATES = 8
N_SITES = 1000

def simulate_tree(n_cells, p_change):
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
    max_char = len(str(n-1)) + 1
    with open(file_name, 'w+') as f:
        f.write(str(n) + '\n')
        for i in range(n):
            cell = str(i)
            x = np.array2string(dist_matrix[i], formatter={'float_kind':lambda x: "%.5f" % x})[1:-1].replace('\n', '')
            sep_char = ' ' * (max_char-len(cell))
            f.write(cell + sep_char + x + '\n')

def fast_me(dist_matrix, taxon_namespace):
    # run balanced minimum evolution (fast ME)
    file_name = 'dist_mat_nonultra.PHYLIP'
    save_distmatrix(dist_matrix, file_name)
    tree_prefix = 'tree_nonultra.nwk'
    call = subprocess.run(['fastme', '-i', file_name, '-o', tree_prefix, '-m', 'B', '-s'], capture_output=True, text=True)
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
    return dpy_tree


def cellmates_infer(dist_matrix, taxon_namespace):
    nx_rec_tree = build_tree(dist_matrix)
    # print("cells: ", [n for n in nx_rec_tree.nodes() if nx_rec_tree.out_degree(n) == 0])
    cellmates_tree = convert_networkx_to_dendropy(nx_rec_tree, taxon_namespace=taxon_namespace, edge_length='length')
    # print("CELLMATES LEAVES: ", [(l.taxon.label, l.label) for l in cellmates_tree.leaf_nodes()])
    # cellmates_tree = label_tree(cellmates_tree)
    return cellmates_tree


def plot_comparison(melt_df, suff=""):
    # RF distance for the two methods, two plots (p_change vs RF distance and n_cells vs RF distance)
    out_paths = [f"rf_vs_p_change.png", f"rf_vs_n_cells.png"]
    sns.set_theme(style="whitegrid")
    # Map method labels
    method_labels = {'cellmates': "Cellmates", 'nj': "NJ", 'bme': "balME"}
    melt_df['method'] = melt_df['method'].map(method_labels)

    # plot rf vs p_change (boxplot and swarmplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='p_change', y='normalized_rf', hue='method', data=melt_df, palette="Set2")
    sns.stripplot(x='p_change', y='normalized_rf', hue='method', color='white', data=melt_df, dodge=True, alpha=0.3, palette="Set2",
                  legend=False)
    plt.title("rf distance vs. p_change")
    plt.savefig(out_paths[0])
    plt.close()

    # plot rf vs n_cells (boxplot and swarmplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='n_cells', y='normalized_rf', hue='method', data=melt_df, palette="Set2", legend=False)
    sns.stripplot(x='n_cells', y='normalized_rf', hue='method', color='white', data=melt_df, dodge=True, alpha=0.3, palette="Set2",
                  legend=False)
    plt.title("rf distance vs. number of cells")
    plt.savefig(out_paths[1])
    plt.close()

    g = sns.catplot(x='n_cells', y='normalized_rf', hue='method', col='p_change',
        data=melt_df, kind='box', height=3, aspect=0.7, palette="Set2")
    g.set_axis_labels("N", "Normalized RF")
    g.set_titles("p = {col_name}")
    sns.move_legend(g, title=None, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    # place legend to the bottom right inside the figure and smaller
    g.savefig(f"rf_vs_n_cells_facet{suff}.png", dpi=300)

    return out_paths



def main():
    # make trees of varying sizes and branch lengths
    rerun = True
    n_cells = [10, 20, 50, 100, 200]
    # n_cells = [ 100]
    # p_changes = [0.001, 0.002, 0.005, 0.01, 0.1, 0.2]
    # p_changes = [0.01, 0.05, 0.1, 0.2, 0.3]
    p_changes = [0.01, 0.02, 0.05, 0.1, 0.2]
    num_seeds = 20
    df_path = "rf_benchmark_results_nonultra.csv"
    rows = []
    df_found = False
    results_df = None
    if not rerun and os.path.exists(df_path):
        results_df = pd.read_csv(df_path)
        n_cells = []
        df_found = True

    for n in n_cells:
        for p in p_changes:
            for seed in range(num_seeds):
                rf = {}
                times = {}
                print(f"Simulating n={n}, p_change={p}, seed={seed}")
                np.random.seed(seed)
                # true_tree, cnp = simulate_tree(n, p)
                true_tree, cnp = simulate_tree_ultra(n, p)
                if seed == 0:
                    true_tree.print_plot(plot_metric='length')
                    print(f"leaves cnp[:10, :10]:\n{cnp[:min(10, n), :20]}")
                    print("Max length: ", max_l:=true_tree.max_distance_from_root(), " -> p=", p_from_l(max_l, N_STATES))
                dist_matrix = compute_triplet_distance_matrix(true_tree, cnp, n)
                # print("max dist: ", np.max(dist_matrix), " should be <= l(1 - 1/K)", l_from_p(1 - (1 / (N_STATES-1)), N_STATES))

                start = time.time()
                nj_tree = neighbor_joining(dist_matrix, true_tree.taxon_namespace)
                times['nj'] = time.time() - start
                rf['nj'] = normalized_rf_distance(true_tree, nj_tree)
                print(f"NJ normalized RF: {rf['nj']:.4f}")

                start = time.time()
                bme_tree = fast_me(dist_matrix, true_tree.taxon_namespace)
                times['bme'] = time.time() - start
                rf['bme'] = normalized_rf_distance(true_tree, bme_tree)
                print(f"balME normalized RF: {rf['bme']:.4f}")

                start = time.time()
                cellmates_tree = cellmates_infer(dist_matrix, true_tree.taxon_namespace)
                times['cellmates'] = time.time() - start
                rf['cellmates'] = normalized_rf_distance(true_tree, cellmates_tree)
                print(f"Cellmates normalized RF: {rf['cellmates']:.4f}")

                for method, rf_value in rf.items():
                    rows.append({"n_cells": n, "p_change": p, "seed": seed, "method": method, "normalized_rf": rf_value, 'time': times[method], 'max_length': max_l, 'max_p': p_from_l(max_l, N_STATES)})
    # save results
    results_df = pd.DataFrame(rows) if not df_found else results_df
    # print stats
    print(results_df.groupby(['n_cells', 'p_change', 'method'])['normalized_rf'].describe())
    results_df.to_csv(df_path, index=False)
    print("Results saved to ", df_path)
    # plot comparison
    out_dirs = plot_comparison(results_df, suff="_nonultra")
    print(f"Plots saved to {out_dirs}")

def rewrite_df_file():
    df_path = "rectree_vs_nj_results.csv"
    results_df = pd.read_csv(df_path)
    melt_df = results_df.melt(id_vars=['n_cells', 'p_change', 'seed'], value_vars=['rf_nj', 'rf_cellmates'],
                              var_name='method', value_name='normalized_rf')
    melt_df.to_csv("rf_benchmark_results.csv", index=False)

if __name__=="__main__":
    main()
    # rewrite_df_file()