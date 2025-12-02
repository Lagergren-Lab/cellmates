"""
Try Cellmates on CNAsim dataset
"""
import os.path
import argparse

import anndata
import dendropy as dpy
import numpy as np

from cellmates.inference.em import jcb_em_alg
from cellmates.inference.neighbor_joining import rooted_nj0
from cellmates.utils.tree_utils import convert_networkx_to_dendropy


def load_cnasim_data(adata_path: str) -> anndata.AnnData:
    # this adata provides the following attributes:
    # X (raw read counts)
    # layers:
    #   - copy (normalized read counts)
    #   - state (copy number states)
    # obs:
    #   - clone (cell type)
    #   - clone-founder (ancestor of cell type)
    # uns:
    #   - clonal-tree-newick (newick string of clonal tree)
    #   - cnasim-params (dict with coverage, bin_length, region_length, window_size)
    adata = anndata.read(adata_path)
    adata.layers['state'] = adata.layers['state'].astype(int)
    return adata


def run(data_path, num_processors=4, **kwargs):
    # folders for output files
    out_dir = kwargs.get('out_dir', os.path.dirname(data_path))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        # make file with metadata
        with open(os.path.join(out_dir, 'metadata.txt'), 'w') as f:
            f.write(f"Data path: {data_path}\n")
            f.write(f"Number of processors: {num_processors}\n")
            for k, v in kwargs.items():
                f.write(f"{k}: {v}\n")

    print("Running Cellmates on CNAsim dataset")
    adata = load_cnasim_data(data_path)
    n_cells = adata.n_obs
    print(f"Loaded CNAsim data: n_cells={adata.n_obs}, n_bins={adata.n_vars}")
    # get true tree dendropy
    print("tree newick:", adata.uns['clonal-tree-newick'])
    true_tree = dpy.Tree.get(data=adata.uns['clonal-tree-newick'], schema='newick',
                             taxon_namespace=dpy.TaxonNamespace())
    true_tree.is_rooted = True

    # get expected number of reads from cnasim params
    params = adata.uns['cnasim-params']
    pois_mean_percopy = (params['coverage'] * params['region_length'] * params['window_size']
                         / (2 * params['bin_length']))
    print(f"Expected mean reads per copy: {pois_mean_percopy:.2f}")

    distances_path: str | None = kwargs.get('distances_path', None)
    if distances_path is not None:
        print("Distances file provided")
        distances_tensor = np.load(distances_path)
    else:
        # run cellmates
        print("Running EM...")
        max_iter = kwargs.get('max_iter', 40)
        em_out_dict = jcb_em_alg(adata.X.T, n_states=7, max_iter=max_iter, num_processors=num_processors,
                                 lam=pois_mean_percopy)
        print("EM done")
        distances_tensor = em_out_dict['l_hat']
        l_hat_path = os.path.join(out_dir, 'em_l_hat.npy')
        np.save(l_hat_path, distances_tensor)
        # loglikelihoods and iterations to numpy matrices
        ll_arr = np.zeros((n_cells, n_cells))
        it_arr = np.zeros((n_cells, n_cells))
        for k in em_out_dict['loglikelihoods']:
            ll_arr[k[0], k[1]] = em_out_dict['loglikelihoods'][k]
            it_arr[k[0], k[1]] = em_out_dict['iterations'][k]
        np.save(os.path.join(out_dir, 'loglikelihoods.npy'), ll_arr)
        np.save(os.path.join(out_dir, 'iterations.npy'), it_arr)
        print(f"EM output saved: {l_hat_path}, loglikelihoods.npy, iterations.npy")

    # build tree from EM output
    print("Building tree from EM output")
    nx_em_tree = rooted_nj0(ctr_table=distances_tensor)
    em_tree = convert_networkx_to_dendropy(nx_em_tree, taxon_namespace=true_tree.taxon_namespace, edge_length='length')
    # compare trees (draw and compute RF distance)
    print("Comparing trees")
    print("True tree")
    true_tree.print_plot(plot_metric='length')
    print("EM tree")
    em_tree.print_plot(plot_metric='length')
    print("RF distance between true and EM tree:", true_tree.robinson_foulds_distance(em_tree))
    # save tree
    em_tree.write(path=os.path.join(out_dir, 'em_tree.nwk'), schema='newick')
    print(f"EM tree saved: {os.path.join(out_dir, 'em_tree.nwk')}")
    print("Done")


if __name__ == '__main__':

    cli = argparse.ArgumentParser(
        description="Run Cellmates on CNAsim dataset"
    )
    cli.add_argument('-i', '--input-data', type=str, help="path to CNAsim data in h5ad format")
    cli.add_argument('-o', '--out-dir', type=str, default=None, help="output directory")
    cli.add_argument('-p', '--num-processors', type=int, default=4, help="number of processors to use")
    cli.add_argument('-m', '--max-iter', type=int, default=40, help="maximum number of EM iterations")
    cli.add_argument('-D', '--distances-path', type=str, default=None, help="path to distances file (npy)")
    args = cli.parse_args()

    run(data_path=args.input_data, num_processors=args.num_processors, max_iter=args.max_iter, out_dir=args.out_dir,
        distances_path=args.distances_path)
