import argparse
import io

from Bio import Phylo

from cellmates.inference.pipeline import predict_cn_profiles
from cellmates.models.evo import JCBModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process cellmate inference output.")
    parser.add_argument("--input-adata", type=str, required=True, help="Path to the input file containing cellmate inference results.")
    parser.add_argument('--cellmates-outdir', type=str, required=True, help="Directory containing cellmate inference output files. Contains 'distance_matrix.npy', 'tree.nwk', 'cell_names.txt'.")
    parser.add_argument("--output-file", type=str, required=False, help="Path to the output file to save processed results. If not provided, results will be added to anndata object in layer 'cellmates-cn'.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    import numpy as np
    import pandas as pd
    import anndata as ad

    # Load the input anndata object
    adata = ad.read_h5ad(args.input_adata)
    n_states

    # Load cell names
    cell_names = open(f"{args.cellmates_outdir}/cell_names.txt").read().strip().splitlines()
    assert len(cell_names) == adata.n_obs, "Number of cell names does not match number of observations in anndata."

    # Load distance matrix
    distance_matrix = np.load(f"{args.cellmates_outdir}/distance_matrix.npy")
    assert distance_matrix.shape == (adata.n_obs, adata.n_obs, 3), "Distance matrix shape is incorrect."

    # Load tree
    tree_relab = Phylo.read(io.StringIO(f"{args.cellmates_outdir}/tree.nwk"), "newick")
    evo_model = JCBModel()

    predict_cn_profiles(adata.layers['copy'], tree_relab, cell_names, em.evo_model, leaf_obs_model=em.obs_model, zero_absorption=True)
