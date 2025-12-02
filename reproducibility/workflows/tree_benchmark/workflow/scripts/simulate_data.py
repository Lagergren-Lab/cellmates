import numpy as np

from _utils import simulate_tree, compute_triplet_distance_matrix


def main(n_cells, p_change, seed, tree_type, dist_mat_file, tree_nwk_file):
    np.random.seed(seed)
    tree, cnp = simulate_tree(n_cells, p_change, tree_type=tree_type)
    tree.write(path=tree_nwk_file, schema='newick')
    dist_mat = compute_triplet_distance_matrix(tree, cnp, n_cells)
    with open(dist_mat_file, 'wb') as f:
        np.save(f, dist_mat)
    return


if __name__=='__main__':
    # snakemake object available
    main(
        n_cells=snakemake.params['n_cells'],
        p_change=snakemake.params['p_change'],
        seed=snakemake.params['seed'],
        tree_type=snakemake.params['tree_type'],
        dist_mat_file=snakemake.output['dist_mat'],
        tree_nwk_file=snakemake.output['tree_nwk'],
    )
