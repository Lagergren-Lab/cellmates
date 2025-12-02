import dendropy
import numpy as np

from _utils import benchmark_method


def main(n_cells, p_change, tree_type, method, seed, dist_mat_file, tree_nwk_file, output_csv_file):
    dist_mat = np.load(dist_mat_file)
    true_tree = dendropy.Tree.get(path=tree_nwk_file, schema="newick")
    res = benchmark_method(dist_mat, true_tree, n_cells, p_change, seed, tree_type, method)
    # save res dict values to csv with header
    with open(output_csv_file, 'w') as f:
        f.write(','.join(res.keys()) + '\n')
        f.write(','.join([str(res[k]) for k in res.keys()]) + '\n')
    return


if __name__ == '__main__':
    # snakemake object available
    main(
        n_cells=snakemake.params['n_cells'],
        p_change=snakemake.params['p_change'],
        tree_type=snakemake.params['tree_type'],
        method=snakemake.params['method'],
        seed=snakemake.params['seed'],
        dist_mat_file=snakemake.input['dist_mat'],
        tree_nwk_file=snakemake.input['tree_nwk'],
        output_csv_file=snakemake.output[0]
    )
