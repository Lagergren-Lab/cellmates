import numpy as np

from _utils import simulate_tree, plot_cell_cn_tree


def main(n_cells, p_change, seed, tree_type, outfile):
    np.random.seed(seed)
    tree, cnp = simulate_tree(n_cells, p_change, tree_type=tree_type)
    plot_cell_cn_tree(tree, cnp, outfile=outfile)
    return


if __name__=='__main__':
    # snakemake object available
    main(
        n_cells=snakemake.params['n_cells'],
        p_change=snakemake.params['p_change'],
        seed=snakemake.params.get('seed', 0),
        tree_type=snakemake.params['tree_type'],
        outfile=snakemake.output[0],
    )
