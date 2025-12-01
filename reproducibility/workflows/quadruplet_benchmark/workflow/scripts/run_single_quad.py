import json
import sys
import time
# from snakemake.script import snakemake

import numpy as np

from cellmates.inference.em import EM
from cellmates.models.evo import JCBModel
from cellmates.models.obs import PoissonModel, NormalModel
from cellmates.simulation.datagen import simulate_quadruplet, Dataset
from cellmates.utils.tree_utils import get_ctr_table_int
from cellmates.utils.math_utils import p_from_l, l_from_p, compute_cn_changes


def get_gamma_params(sizes, base_variance):
    # sizes: dict of size categories {"xs": 0.005, ...} with p_change values
    gamma_params_dict = {}
    for size, p_change in sizes.items():
        mean_l = l_from_p(p_change, n_states=7)
        scale = (base_variance * mean_l) / mean_l
        shape = mean_l / scale
        gamma_params_dict[size] = (shape, scale)
    return gamma_params_dict


def main(snakemake):
    n_states = snakemake.params["n_states"]
    max_iter = snakemake.params["max_iter"]
    seed = snakemake.params["seed"]
    n_sites = snakemake.params["n_sites"]
    sizes_dict = snakemake.params["sizes_dict"]
    base_variance = snakemake.params["base_variance"]
    obs_model = snakemake.params["obs_model"]
    length_size = snakemake.params["length_size"]
    num_processors = snakemake.threads
    out_file = snakemake.output[0]
    print(f"params: {snakemake.params}")
    print(f"threads: {num_processors}")
    print(f"out: {out_file}")

    gamma_params_dict = get_gamma_params(sizes_dict, base_variance)

    # observation model
    if obs_model["type"] == "normal":
        obs_model = NormalModel(n_states=n_states,
                                mu_v_prior=obs_model["mu_v_prior"],
                                tau_v_prior=obs_model["tau_v_prior"])
    elif obs_model["type"] == "poisson":
        obs_model = PoissonModel(n_states=n_states,
                                 lambda_v_prior=obs_model["lambda_v_prior"])
    else:
        raise ValueError("Unknown obs_model")

    # p_change
    mean_lengths = [gamma_params_dict[s][0]*gamma_params_dict[s][1] for s in length_size]
    p_change = [p_from_l(m, n_states=n_states) for m in mean_lengths]
    if len(p_change) == 1:
        p_u = p_v = p_w = p_change[0]
    elif len(p_change) == 2:
        p_u, p_v = p_change
        p_w = p_v
    else:
        p_u, p_v, p_w = p_change

    evo_model = JCBModel(n_states=n_states)
    gamma_seq = [gamma_params_dict[s] for s in length_size]
    data: Dataset = simulate_quadruplet(n_sites=n_sites, obs_model=obs_model,
                               evo_model=evo_model, gamma_params=gamma_seq,
                               seed=seed)

    gt_ctr = get_ctr_table_int(data['tree'])  # shape: (2, 2, 3), but only 3 values are stored (0,1,:3)
    em = EM(n_states=n_states, obs_model=obs_model, evo_model=evo_model, tree_build='ctr')
    start = time.time()
    em.fit(data['obs'], max_iter=max_iter, num_processors=num_processors,
           rtol=1e-8, l_init=gt_ctr[0,1])
    exec_time = time.time() - start
    # generating lengths likelihood
    gen_ll = em.compute_pair_likelihood(data['obs'], theta=gt_ctr[0,1], psi=obs_model.psi)
    # actual lengths likelihood (from copy number changes)
    tree_nwk = data['tree'].as_string(schema='newick').strip()
    assert np.array(data['cn'][3] == 2).all(), "bug: index 3 is assumed to be the root, but tree is: " + tree_nwk  # root
    true_eps = compute_cn_changes(data['cn'], [(3, 2), (2, 0), (2, 1)])
    true_lengths = l_from_p(np.array(true_eps) / n_sites, n_states)
    true_ll = em.compute_pair_likelihood(data['obs'], theta=true_lengths)
    # delta
    # delta = em.distances - gt_ctr
    delta = em.distances - true_lengths

    obs_name = "normal" if isinstance(obs_model, NormalModel) else "poisson"
    obs_var = (1/obs_model.tau_v_prior if obs_name=="normal" else obs_model.lambda_v_prior)

    with open(out_file, "w") as f:
        f.write("seed,n_sites,n_states,length_params,p_change_u,p_change_v,p_change_w,"
                "lu_em,lv_em,lw_em,lu_err,lv_err,lw_err,exec_time,n_iter,loglik,"
                "true_ll,gen_ll,base_variance,obs_model,obs_var\n")
        f.write(f"{seed},{n_sites},{n_states},{'-'.join(length_size)},"
                f"{p_u},{p_v},{p_w},{em.distances[0,1,0]},{em.distances[0,1,1]},"
                f"{em.distances[0,1,2]},{delta[0,1,0]},{delta[0,1,1]},"
                f"{delta[0,1,2]},{exec_time:.2f},{em._n_iterations[(0,1)]},"
                f"{em._loglikelihoods[(0,1)]},{true_ll},{gen_ll},{base_variance},"
                f"{obs_name},{obs_var}\n")

class Params:
    def __init__(self, args):
        for arg in args:
            key, value = arg.split('=')
            try:
                # try to evaluate as a Python literal (e.g., list, dict)
                # lists are passed as space-separated strings, convert to list
                # dicts are passed as key:value pairs separated by spaces, convert to dict
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1].split()
                elif value.startswith('{') and value.endswith('}'):
                    items = value[1:-1].split()
                    value = {k: eval(v) for k,v in (item.split(':') for item in items)}
                else:
                    value = eval(value)
            except:
                pass
            setattr(self, key, value)

if __name__ == "__main__":
    main(snakemake)

