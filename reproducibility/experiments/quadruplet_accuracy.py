import os
import itertools
import time
from io import StringIO

import numpy as np
import multiprocessing as mp
import scgenome.plotting as pl
from Bio import Phylo

from cellmates import ROOT_DIR

from cellmates.inference.em import EM
from cellmates.models.evo import JCBModel
from cellmates.models.obs import PoissonModel, NormalModel
from cellmates.simulation.datagen import simulate_quadruplet, _from_data_to_adata
from cellmates.utils.tree_utils import get_ctr_table_int
from cellmates.utils.math_utils import p_from_l, l_from_p


def parse_p_change_list(p_change):
    if len(p_change) == 1:
        p_change_u = p_change_v = p_change_w = p_change[0]
    elif len(p_change) == 2:
        p_change_u = p_change[0]
        p_change_v = p_change[1]
        p_change_w = p_change[1]
    else:
        p_change_u = p_change[0]
        p_change_v = p_change[1]
        p_change_w = p_change[2]
    return p_change_u, p_change_v, p_change_w

def run_experiment(n_sites, length_size, seed, max_iter, file_name, n_states, gamma_params_dict, variance, obs_model, out_dir=None):
    i = seed
    print(f"n_sites: {n_sites}")
    print(f"length sizes: {length_size}")
    p_change = []
    for s in length_size:
        a, b = gamma_params_dict[s]
        mean_length = a * b
        p_change.append(p_from_l(mean_length, n_states=n_states))
        print(f"mean: {mean_length:.3f} (p = {p_from_l(mean_length, n_states):.3f}), var: {a * b ** 2}", end=" --- ")
    p_change_u, p_change_v, p_change_w = parse_p_change_list(p_change)
    print(f"p_change_u: {p_change_u:.5f}, p_change_v: {p_change_v:.5f}, p_change_w: {p_change_w:.5f}")
    length_size_str = '-'.join(length_size)
    evo_model = JCBModel(n_states=n_states)
    # obs_model = PoissonModel(n_states=n_states)
    gamma_params = [gamma_params_dict[s] for s in length_size]

    # generate data
    data = simulate_quadruplet(n_sites=n_sites, obs_model=obs_model, evo_model=evo_model,
                               gamma_params=gamma_params, seed=i)
    if out_dir:
        # save data plot
        if not os.path.exists(out_dir + "/data"):
            os.makedirs(out_dir + "/data")
        adata = _from_data_to_adata(data)
        bio_tree = Phylo.read(StringIO(adata.uns['tree']), 'newick')
        g = pl.plot_cell_cn_matrix_fig(adata, show_cell_ids=True, tree=bio_tree)
        g['fig'].savefig(out_dir + f"/data/{length_size_str}.png")

    # ground truth (data-generating length means)
    gt_ctr_table = get_ctr_table_int(data['tree'])

    # run EM
    em = EM(n_states=n_states, obs_model=obs_model, evo_model=evo_model, tree_build='ctr')
    start_time = time.time()
    # initialize with the true lengths
    em.fit(data['obs'], max_iter=max_iter, num_processors=1, rtol=1e-8, l_init=gt_ctr_table[0, 1])
    exec_time = time.time() - start_time

    true_param_ll = em.compute_pair_likelihood(data['obs'], theta=gt_ctr_table[0, 1], psi=obs_model.psi)

    ctr_table_delta = em.distances - gt_ctr_table
    print(
        f"[{i}] time: {exec_time:.2f}, average delta ctr_table_distance: {np.mean(np.abs(ctr_table_delta[0, 1])):.5f}")
    obs_model_name = "NA"
    obs_model_variance = np.nan
    if isinstance(obs_model, NormalModel):
        obs_model_name = 'normal'
        obs_model_variance = 1/obs_model.tau_v_prior
    elif isinstance(obs_model, PoissonModel):
        obs_model_name = 'poisson'
        obs_model_variance = obs_model.lambda_v_prior
    else:
        raise ValueError(f"Unknown obs_model: {obs_model}")
    with open(file_name, 'a') as f:
        f.write(
            f"{i},{n_sites},{n_states},{length_size_str},{p_change_u},{p_change_v},{p_change_w},{em.distances[0, 1, 0]},{em.distances[0, 1, 1]},{em.distances[0, 1, 2]},"
            f"{ctr_table_delta[0, 1, 0]},{ctr_table_delta[0, 1, 1]}," f"{ctr_table_delta[0, 1, 2]},{exec_time:.2f},{em._n_iterations[(0, 1)]},{em._loglikelihoods[(0, 1)]},{true_param_ll},{variance},{obs_model_name},{obs_model_variance}\n")


def main():
    # parameters
    max_iter = 60
    parallel_experiments = 32 # cpus
    n_states = 7
    n_sites_list = [
        200,
        500,
        1000
    ]
    obs_model_list = [
        # PoissonModel(n_states=n_states, lambda_v_prior=100),
        NormalModel(n_states=n_states, mu_v_prior=1., tau_v_prior=50),
        NormalModel(n_states=n_states, mu_v_prior=1., tau_v_prior=100),
        NormalModel(n_states=n_states, mu_v_prior=1., tau_v_prior=300),
    ]
    # define a range of p_change values (xs, s, m, l, xl)
    p_change_arr = np.array([0.005, 0.01, 0.02, 0.08, 0.2])
    mean_l = l_from_p(p_change_arr, n_states=n_states)  # mean edge length
    # variance proportional to mean so that smaller lengths have smaller variance and do not shift to other size categories
    base_variance = 0.0001
    var_ = base_variance * mean_l  # variance
    scale_arr = var_ / mean_l
    shape_arr = mean_l / scale_arr
    gamma_params_dict = [(shape.item(), scale.item()) for shape, scale in zip(shape_arr, scale_arr)]
    sizes = ['xs', 's', 'm', 'l', 'xl']
    gamma_params_dict = dict(zip(sizes, gamma_params_dict))

    length_sizes_list = (
            [ [p] for p in sizes ] +
            [[sizes[0], sizes[4]], [sizes[1], sizes[3]] ] + # small r->u, large u->v,w
            [[sizes[4], sizes[0]], [sizes[3], sizes[1]] ] + # large r->u, small u->v,w
            [[sizes[2], sizes[0], sizes[4]], [sizes[2], sizes[1], sizes[3]]] + # medium r->u, small u->v, large u->w
            [[sizes[4], sizes[2], sizes[0]], [sizes[3], sizes[2], sizes[1]]] # large r->u, medium u->v, small u->w
    )
    out_dir = ROOT_DIR + '/output/reproducibility/experiments/quadruplet_accuracy'

    num_replicates = 20
    print(f"with n_states = {n_states}, alpha = 1.")
    # make filename unique to avoid overwriting
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_path = out_dir + f'/quadruplet_accuracy_{time.strftime("%y%m%d%H%M%S")}.csv'
    with open(file_path, 'w') as f:
        f.write('seed,n_sites,n_states,length_params,p_change_u,p_change_v,p_change_w,lu_em,lv_em,lw_em,lu_err,lv_err,lw_err,exec_time,n_iter,loglik,true_ll,base_variance,obs_model,obs_var\n')
    args = [(n_sites, length_size, seed, max_iter, file_path, n_states, gamma_params_dict, base_variance, obs_model, out_dir) for n_sites, length_size, seed, obs_model in itertools.product(n_sites_list, length_sizes_list, range(num_replicates), obs_model_list)]
    if parallel_experiments == 1:
        for arg in args:
            run_experiment(*arg)
    else:
        # parallelize with multiprocessing
        with mp.Pool(parallel_experiments) as pool:
            # main loop
            pool.starmap(run_experiment, args)
    print(f"Results saved in {file_path}")


if __name__ == '__main__':
    main()
