import sys
import random
import time
import logging

import numpy as np
from dendropy.calculate import treecompare

from cellmates.inference.em import jcb_em_alg
from cellmates.inference.neighbor_joining import build_tree
from cellmates.simulation.datagen import rand_dataset
from cellmates.utils.tree_utils import convert_networkx_to_dendropy, get_ctr_table_int

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    # generate 10 datasets for each of data parameters (p_change = 0.05, 0.01) and (n_cells = 10, 20, 50, 100),
    # with n_sites = 500, n_states = 7, alpha = 1.
    # save records for each run with: seed, time, tot_likelihood, avg_likelihood_pair,
    #   ctr_table distance, tree distance, tree rf distance
    # save records in a csv file
    quick_test = False
    # take num processors as input
    if len(sys.argv) == 2:
        num_processors = int(sys.argv[1])
        print(f"Using {num_processors} processor(s)")
    elif len(sys.argv) > 2:
        print("Usage: python synth_performance.py [<num_processors>]")
        sys.exit(1)
    else:
        print("Using 1 processor")
        num_processors = 1

    # data parameters
    max_iter = 40
    n_sites = 500
    n_states = 7
    alpha = 1.
    p_change_list = [0.001]
    n_cells_list = [100]
    n_datasets = 10
    if quick_test:
        n_datasets = 2
        n_states = 4
        n_sites = 100
        n_cells_list = [10]
        p_change_list = [0.05]
        max_iter = 10

    # output file
    timestamp = time.strftime('%y%m%d%H%M%S')
    output_file = f'synth_performance{timestamp}.csv'
    with open(output_file, 'w') as f:
        f.write('seed,n_cells,p_change,time,tot_likelihood,avg_likelihood_pair,avg_iterations_pair,ctr_table_distance,'
                'tree_distance,tree_rf_distance,num_processors,n_states,n_sites,max_iter,alpha\n')

    for p_change in p_change_list:
        for n_cells in n_cells_list:
            for i in range(n_datasets):
                seed = random.randint(0, 100000)
                data = rand_dataset(n_states, n_sites, obs_model='poisson', alpha=alpha, p_change=p_change,
                                    n_cells=n_cells, seed=seed)
                true_ctr_table = get_ctr_table_int(data['tree'])

                start = time.time()
                out_dict = jcb_em_alg(data['obs'], n_states, max_iter=max_iter, num_processors=num_processors)
                end = time.time()
                ctr_table = out_dict['l_hat']
                likelihoods = out_dict['loglikelihoods']
                iterations = out_dict['iterations']
                nx_em_tree = build_tree(ctr_table)
                em_tree = convert_networkx_to_dendropy(nx_em_tree, taxon_namespace=data['tree'].taxon_namespace,
                                                       edge_length='length')

                tot_likelihood = np.sum([c for c in likelihoods.values()]).item()
                avg_likelihood_pair = np.mean([c for c in likelihoods.values()]).item()
                avg_iterations_pair = np.mean([c for c in iterations.values()]).item()
                ctr_table_distance = np.linalg.norm(ctr_table - true_ctr_table)
                tree_distance = treecompare.symmetric_difference(data['tree'], em_tree)
                tree_rf_distance = treecompare.robinson_foulds_distance(data['tree'], em_tree, edge_weight_attr='length')
                with open(output_file, 'a') as f:
                    f.write(f"{seed},{n_cells},{p_change},{end-start},{tot_likelihood},{avg_likelihood_pair},"
                            f"{avg_iterations_pair},{ctr_table_distance},{tree_distance},{tree_rf_distance},"
                            f"{num_processors},{n_states},{n_sites},{max_iter},{alpha}\n")



