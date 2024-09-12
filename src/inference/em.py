import itertools
import logging
import operator
import time
from typing import Optional, Union

import numpy as np
from dendropy.calculate.treecompare import (
    robinson_foulds_distance,
    unweighted_robinson_foulds_distance,
    symmetric_difference,
)
import scipy.stats as stats
from scipy.special import comb
from scipy.special import logsumexp
import networkx as nx

from models.quadruplet import Quadruplet
from simulation.datagen import rand_dataset, get_ctr_table

from src.models.copy_tree import h_eps, get_zipping_mask, get_zipping_mask0, p_delta_trans_mat, p_delta_start_prob
from utils.math_utils import l_from_p
from utils.tree_utils import convert_networkx_to_dendropy


# CURRENTLY UNUSED
# TODO: wrap EM function `jcb_em_alg` in class (sklearn style)
class EM():
    """
    Runs the EM-algorithm for a quadruplet. Requires the copy number sequence of the root and observations of a pair
    of leaves.
    """

    def __init__(self, quadruplet: Quadruplet):
        self.quadruplet = quadruplet

    def run_hmmlearn(self):
        yv = self.quadruplet.data_v
        yw = self.quadruplet.data_v
        A = self.quadruplet.A
        y = np.concatenate([yv, yw])
        lengths = [len(yv), len(yw)]
        eps_trans_matrix_prior = np.ones((A, A, A)) / A * 0.05

        lambdas_prior = np.ones((A, 2))
        init_prior = np.zeros((3, A))
        init_prior[:, 2] = 1.
        model = hmm.PoissonHMM(n_components=(A, A, A),
                               startprob_prior=init_prior,
                               transmat_prior=eps_trans_matrix_prior,
                               lambdas_prior=lambdas_prior,
                               lambdas_weight=0.0,
                               algorithm='viterbi',
                               random_state=None,
                               n_iter=10,
                               tol=0.01,
                               verbose=False,
                               params='stl', init_params='', implementation='log')
        model.startprob_ = init_prior
        model.fit(y, lengths)


def compute_log_emissions(obs_vw: np.ndarray, n_states, pois_mean_eps=1e-5) -> np.ndarray:
    """
    returns the log probability of the observations for each site, and each pair of copy number states (v, w)
    Parameters
    ----------
    obs_vw
    n_states

    Returns
    -------
    array of shape (n_sites, n_states, n_states), log p(y_m^{vw} | C_m^v = i, C_m^w = j) for each m, i, j
    """
    lam = 100.
    assert obs_vw.shape[1] == 2
    n_sites = obs_vw.shape[0]
    log_emissions = np.empty((n_sites, n_states, n_states))
    for m, i, j in itertools.product(range(n_sites), range(n_states), range(n_states)):
        # log p(y_m^v | . ) + log p(y_m^w | . )
        log_emissions[m, i, j] = stats.poisson.logpmf(obs_vw[m],
                                                      np.clip(lam * np.array([i, j]), a_min=pois_mean_eps,
                                                              a_max=None)).sum()

    return log_emissions


def compute_eps_trans_mat(eps_trip, n_states) -> np.ndarray:
    trans_mat = np.empty((n_states,) * 6)
    # transition from r -> u (fix r = 2)
    a_ru = h_eps(n_states, eps=eps_trip[0])[:, :, 2, 2]
    a_uv = h_eps(n_states, eps=eps_trip[1])
    a_uw = h_eps(n_states, eps=eps_trip[2])
    # results is state i, j, k -> x, y, z
    trans_mat[...] = np.einsum('xi,yjxi,zkxi->ijkxyz', a_ru, a_uv, a_uw)
    return trans_mat


def compute_l_trans_mat(l_trip, n_states, alpha: float = 1.) -> np.ndarray:
    trans_mat = np.empty((n_states,) * 6)
    # transition from r -> u (fix r = 2)
    a_ru = p_delta_trans_mat(n_states, l_trip[0], alpha=alpha)[:, :, 2, 2]
    a_uv = p_delta_trans_mat(n_states, l_trip[1], alpha=alpha)
    a_uw = p_delta_trans_mat(n_states, l_trip[2], alpha=alpha)
    # results is state (m-1) i, j, k -> (m) x, y, z
    trans_mat[...] = np.einsum('xi,yjxi,zkxi->ijkxyz', a_ru, a_uv, a_uw)
    return trans_mat


def compute_eps_start_prob(eps_trip, n_states):
    # TODO: uniform prob, but not implemented properly yet
    return np.ones((n_states,) * 3) / n_states ** 3


def compute_l_start_prob(l_trip, n_states, alpha: float = 1.):
    trip_start = np.empty((n_states,) * 3)

    # transition from r -> u (fix r = 2)
    a_ru = p_delta_start_prob(n_states, l_trip[0], alpha=alpha)[:, 2]
    # transition from u -> v, w
    a_uv = p_delta_start_prob(n_states, l_trip[1], alpha=alpha)
    a_uw = p_delta_start_prob(n_states, l_trip[2], alpha=alpha)
    # results is state i, j, k
    trip_start[...] = np.einsum('i, ij, ik -> ijk', a_ru, a_uv, a_uw)
    return trip_start


def _forward_pass_likelihood(obs_vw, start_prob, trans_mat, log_emissions, eps=1e-5) -> (np.ndarray, float):
    """
    Compute the forward pass of the hidden markov model with three latent chains and return the forward probabilities
    as well as the log likelihood of the observations. See wrapper function `forward_pass` for more details.
    """
    # transmat idx = (i, j, k, x, y, z) = (m-1, m)
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    alpha = np.empty((n_sites, n_states, n_states, n_states))

    # init alpha[0] = start_prob * emission[0] (in log-form)
    alpha[0, ...] = np.log(start_prob.clip(eps)) + log_emissions[0, np.newaxis, ...]
    alpha[0] -= logsumexp(alpha[0])

    log_likelihood = 0.
    for m in range(1, n_sites):
        # [ \sum_{x,y,z} alpha_{m-1}(x, y, z) p(i, j, k | x, y, z) ] p(y_m^{vw} | i, j, k)
        log_acc = -np.inf * np.ones((n_states, n_states, n_states))
        for x, y, z in itertools.product(range(n_states), repeat=3):
            log_acc = np.logaddexp(alpha[m - 1, x, y, z] + np.log(trans_mat[x, y, z, ...]), log_acc)
        alpha[m, ...] = log_acc + log_emissions[m, np.newaxis, ...]
        # normalization step
        norm = logsumexp(alpha[m])
        alpha[m] -= norm
        # update log likelihood to save computation
        log_likelihood += norm

    assert np.allclose(logsumexp(alpha, axis=(1, 2, 3)), np.zeros(n_sites))
    return alpha, log_likelihood


def forward_pass(obs_vw, start_prob, trans_mat, log_emissions, eps=1e-5):
    """
    Compute the forward pass of the hidden markov model with three latent chains and return the forward probabilities.

    Parameters
    ----------
    obs_vw array of shape (n_sites, 2) with observations for pair of leaves
    start_prob array of shape (n_states, n_states, n_states) with start probabilities
    trans_mat array of shape (n_states, n_states, n_states, n_states, n_states, n_states) with transition probabilities
    log_emissions array of shape (n_sites, n_states, n_states) with log emissions
    eps float, small constant to avoid log(0)

    Returns
    -------
    array of shape (n_sites, n_states, n_states, n_states), forward probabilities

    """
    return _forward_pass_likelihood(obs_vw, start_prob, trans_mat, log_emissions, eps)[0]


def backward_pass(obs_vw, trans_mat, log_emissions):
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    beta = np.empty((n_sites, n_states, n_states, n_states))
    # initialize beta[M] = 1. (in log form) - normalized
    beta[-1, ...] = - 3 * np.log(n_states)

    # compute iteratively
    for m in reversed(range(n_sites - 1)):
        # -- ORIGINAL IMPLEMENTATION --
        # beta[m, ...] = logsumexp(beta[m + 1, ...] +
        #                          np.log(trans_mat)
        #                          + log_emissions[m + 1, np.newaxis, np.newaxis, np.newaxis, ...],
        #                          axis=(3, 4, 5))

        #  -- ALTERATIVE IMPLEMENTATION --
        log_acc = -np.inf * np.ones((n_states, n_states, n_states))
        for x, y, z in itertools.product(range(n_states), repeat=3):
            log_acc = np.logaddexp(beta[m + 1, x, y, z] +
                                   np.log(trans_mat[..., x, y, z]) +
                                   log_emissions[m + 1, y, z],  # ]
                                   log_acc)
        beta[m, ...] = log_acc

        # normalization step
        beta[m] -= logsumexp(beta[m])

    return beta


def two_slice_marginals(obs_vw, theta: np.ndarray, n_states: int, jcb: bool = False, alpha: float = 1.)\
        -> tuple[np.ndarray, float]:
    """
    Computes the two slice marginals of a hidden markov model with three latent chains.
    Specifically, for each point m of the chain (site), and each pair of triplet states
    (i,j,k)[m] -> (i'j'k')[m+1], it computes the
        $$ \log P(X_m = (i,j,k), X_{m+1} = (i',j',k') | Y, \theta) $$
    Also outputs the log likelihood of the observations which is computed in the forward pass.
    Parameters
    ----------
    obs_vw array of shape (n_sites, 2)
    theta array of shape (3,) with the triplet parameters
    n_states number of copy number states
    jcb if True, use Jukes-Cantor-Breakpoint model, otherwise use the CopyTree model
    alpha float, alpha parameter for the JCB model, length scaling factor

    Returns
    -------
    tuple of array of shape (n_sites - 1,) + (n_states,) * 6, log two slice marginals and float, log likelihood

    """
    n_sites = obs_vw.shape[0]
    # define start_probs and transitions depending on chosen model
    start_prob, trans_mat = get_start_transition_probs(alpha, jcb, n_states, theta)

    # define emission prob (for emission pair)
    # log emission shape (n_sites, n_states, n_states)
    log_emissions = compute_log_emissions(obs_vw, n_states)
    # compute forward: shape (n_sites, n_states, n_states, n_states)
    alpha_probs, loglik = _forward_pass_likelihood(obs_vw, start_prob, trans_mat, log_emissions)
    # compute backward: shape (n_sites, n_states, n_states, n_states)
    beta_probs = backward_pass(obs_vw, trans_mat, log_emissions)
    # tsm shape: (n_sites - 1,) + (n_states,) * 6
    # compute two slice: xi
    log_xi = alpha_probs[(np.arange(n_sites - 1), ...) + (np.newaxis,) * 3] + \
             np.log(trans_mat)[np.newaxis, ...] + \
             log_emissions[(np.arange(1, log_emissions.shape[0]),) + (np.newaxis,) * 4 + (...,)] + \
             beta_probs[(np.arange(1, beta_probs.shape[0]), ...) + (np.newaxis,) * 3]
    log_xi -= np.expand_dims(logsumexp(log_xi, axis=tuple(range(1, 7))), axis=tuple(range(1, 7)))

    assert np.allclose(logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6)), np.zeros(n_sites - 1))

    return log_xi, loglik


def get_start_transition_probs(alpha, jcb, n_states, theta):
    if jcb:
        # start prob shape: (n_states, n_states, n_states)
        start_prob = compute_l_start_prob(theta, n_states, alpha=alpha)
        # trans_mat shape: ((n_states,) * 6)
        trans_mat = compute_l_trans_mat(theta, n_states, alpha=alpha)
    else:
        start_prob = compute_eps_start_prob(theta, n_states)
        trans_mat = compute_eps_trans_mat(theta, n_states)
    return start_prob, trans_mat


def compute_exp_changes(theta, obs_vw, n_states, alpha=1., jcb=True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the sufficient statistics, i.e. the expected number of changes and no-changes for each pair
    of triplet states. Also returns the log likelihood of the observations.
    Parameters
    ----------
    theta array of shape (3,) with the triplet parameters
    obs_vw array of shape (n_sites, 2)
    n_states number of copy number states
    alpha float, alpha parameter for the JCB model, length scaling factor
    jcb if True, use Jukes-Cantor-Breakpoint model, otherwise use the CopyTree model

    Returns
    -------
    tuple of arrays of shape (3,), expected number of changes and no-changes, and float, log likelihood

    """
    d = np.empty(3)
    dp = np.empty_like(d)
    # compute two slice marginals
    # prob(Cm = ijk, Cm+1 = i'j'k' | Y)
    log_xi, loglik = two_slice_marginals(obs_vw, theta, n_states, jcb=jcb, alpha=alpha)

    comut_mask0 = get_zipping_mask0(n_states).transpose()
    comut_mask = get_zipping_mask(n_states).transpose(tuple(reversed(range(4))))
    # count expected changes/no-changes
    for e in range(3):
        if e == 0:
            # l_ru (sum over m, j, k, j', k')
            pair_tsm = logsumexp(log_xi, axis=(0, 2, 3, 5, 6))
            # use comut_mask0
            d[0] = np.exp(logsumexp(pair_tsm[~comut_mask0]))
            dp[0] = np.exp(logsumexp(pair_tsm[comut_mask0]))
        else:
            if e == 1:
                # eps_uv (sum over m, i, i')
                pair_tsm = logsumexp(log_xi, axis=(0, 3, 6))
            else:
                # eps_uw (sum over m, j, j'
                pair_tsm = logsumexp(log_xi, axis=(0, 2, 5))

            d[e] = np.exp(logsumexp(pair_tsm[~comut_mask]))
            dp[e] = np.exp(logsumexp(pair_tsm[comut_mask]))

    return d, dp, loglik


def likelihood(obs_vw, theta, n_states, alpha=1., jcb=True) -> float:
    start_prob, trans_mat = get_start_transition_probs(alpha, jcb, n_states, theta)
    log_emissions = compute_log_emissions(obs_vw, n_states)
    return _forward_pass_likelihood(obs_vw, start_prob, trans_mat, log_emissions)[1]


def update_eps(log_xi):
    n_states = log_xi.shape[1]
    eps_new = np.empty(3)
    # get set `A` and `\neg A` as masks on the quadruplet
    # i - i' == j - j' (indexing: `mask[j', j, i', i]`)
    # after reverse: `mask[i, i', j, j']`
    comut_mask0 = get_zipping_mask0(n_states).transpose()
    comut_mask = get_zipping_mask(n_states).transpose(tuple(reversed(range(4))))
    for e in range(3):
        if e == 0:
            # eps_ru
            pair_tsm = logsumexp(log_xi, axis=(0, 2, 3, 5, 6))
            # use comut_mask0
            deps = np.exp(logsumexp(pair_tsm[~comut_mask0]))
            deps_prime = np.exp(logsumexp(pair_tsm[comut_mask0]))
        else:
            if e == 1:
                # eps_uv
                pair_tsm = logsumexp(log_xi, axis=(0, 3, 6))
            else:
                # eps_uw
                pair_tsm = logsumexp(log_xi, axis=(0, 2, 5))

            deps = np.exp(logsumexp(pair_tsm[~comut_mask]))
            deps_prime = np.exp(logsumexp(pair_tsm[comut_mask]))

        eps_new[e] = deps / (deps + deps_prime)

    return eps_new


def em_alg(obs: np.ndarray) -> np.ndarray:
    """
Implementation of algorithm 6 in write-up
    Parameters
    ----------
    obs array of shape (n_sites, n_cells)
    Returns
    -------
    array of shape (n_cells, n_cells), centroid-to-root distance
    """
    # params
    n_states = 7
    eps_init = (1, 10)
    n_sites, n_cells = obs.shape
    epsilon_hat = - np.ones((n_cells, n_cells))

    # for each pair of cells
    for v, w in itertools.combinations(range(n_cells), r=2):
        # initialize eps = (eps_ru, eps_uv, eps_uw)
        epsilon_k = np.random.beta(*eps_init, size=3)
        # triplet state u, v, w
        convergence = False
        while not convergence:
            # compute two slice marginals
            # shape (n_sites, n_states x 3, n_states x 3)
            log_xi, loglik = two_slice_marginals(obs[:, [v, w]], epsilon_k, n_states)
            # update epsilon_k
            epsilon_kp1 = update_eps(log_xi)
            # check for convergence
            convergence = np.allclose(epsilon_k, epsilon_kp1, rtol=1e-2)
            # update current eps
            epsilon_k[...] = epsilon_kp1

        epsilon_hat[v, w] = epsilon_k[0]  # ctr distance is eps_ru (first of triplet)

    return epsilon_hat


def jcb_em_alg(obs: np.ndarray, alpha=1., l_init=None, max_iter: int = 200, rtol: float = 1e-6) -> np.ndarray:
    """
Implementation of JCB EM algorithm in write-up
    Parameters
    ----------
    obs array of shape (n_sites, n_cells)
    Returns
    -------
    array of shape (n_cells, n_cells, 3), estimated pairwise triplet distances (upper triangular, all other
    entries are -1)
    """
    # params
    n_states = 7
    n_sites, n_cells = obs.shape
    # init to an average of 5 changes over the whole length
    if l_init is None:
        l_init = np.array([l_from_p(5 / n_sites, n_states)] * 3)
    else:
        l_init = l_init
    l_hat = -np.ones((n_cells, n_cells, 3))
    zero_tol = 1e-10  # saturation level when dp << d (changes are much more prevalent)

    # for each pair of cells
    counter = 0
    logging.debug(f'pairwise EM started: {comb(n_cells, 2)} pairs')
    iterations = {}
    for v, w in itertools.combinations(range(n_cells), r=2):
        # initialize l = (l_ru, l_uv, l_uw)
        l_i = l_init
        d, dp, loglik = compute_exp_changes(l_i, obs[:, [v, w]], n_states, alpha=alpha)
        convergence = False
        it = 0
        logging.debug(f'pairwise EM: {v}, {w}, iteration {it}, loglik = {loglik}')
        while not convergence and it < max_iter:
            # update l according to formula
            if np.any((n_states - 1) * dp <= d):
                logging.error(f"too many changes detected: D = {d}, D' = {dp}\n"
                              f"...saturating l for cells {v},{w}")
            # if l -> +inf, pDeltaDelta == pDeltaDelta'
            num = np.clip((n_states - 1) * dp - d, a_min=zero_tol, a_max=None)
            l_i = - 1 / (alpha * n_states) * np.log(num / ((n_states - 1) * (dp + d)))

            # compute D and D'
            d, dp, new_loglik = compute_exp_changes(l_i, obs[:, [v, w]], n_states, alpha=alpha)
            logging.debug(f'pairwise EM: {v}, {w}, iteration {it}, loglik = {loglik}')

            if new_loglik < loglik:
                logging.error(f'log likelihood decreased: {new_loglik} < {loglik}')
            elif (new_loglik - loglik) / np.abs(loglik) < rtol:
                convergence = True
            loglik = new_loglik
            it += 1

        if it == max_iter and not convergence:
            logging.warning(f'pairwise EM: {v}, {w}, did not converge after {max_iter} iterations')
        else:
            logging.debug(f'pairwise EM: {v}, {w}, converged after {it} iterations')

        counter += 1
        if counter % 10 == 0:
            logging.debug(f'{comb(n_cells, 2) - counter} pairs remaining...')
        l_hat[v, w, :] = l_i
        iterations[(v, w)] = it
        # print(f'cell pair {v}, {w} done: l_trip = {l_i}, iterations = {it}')

    return l_hat


def _build_tree_rec(ctr: dict, ntc: dict, ntr: dict, otus: set, edges: set[tuple]) -> set[tuple]:
    if len(otus) == 2:
        for c in otus:
            # add edge with length
            edges.add(('r', c, ntr[c]))
    else:
        vw, l = max(ctr.items(), key=operator.itemgetter(1))
        # remove pair and add common ancestor with averaged distance

        # Computing edge lengths after merge
        # save node-to-root distance for edge computation later
        # removing the pair from the centroid to rood distances as they are merged
        v, w = vw

        # remove node-to-root for merged nodes
        ntr.pop(v)
        ntr.pop(w)

        # Update distances merging vw in one OTU
        vsw = v + '_' + w  # node with string showing merges v_w
        ntr[vsw] = ctr.pop(vw)  # save centroid to root as the new node-to-root distance (new OTU)
        new_otus = otus.difference({w, v})
        for c in new_otus:
            vc = frozenset({v, c})
            wc = frozenset({w, c})
            # new pairwise distances
            vsw_c = frozenset({vsw, c})
            # update ctr distance for new node
            ctr[vsw_c] = .5 * (ctr[vc] + ctr[wc])
            # update ntc distances for new node
            ntc[vsw, c] = ntr[vsw] - ctr[vsw_c]
            ntc[c, vsw] = .5 * (ntc[c, v] + ntc[c, w])
            # remove already merged nodes
            ctr.pop(vc)
            ctr.pop(wc)
            ntc.pop(c, v)
            ntc.pop(c, w)
            ntc.pop(v, c)
            ntc.pop(w, c)

        # add node/subtree as OTU
        new_otus.add(vsw)

        v_edge_length = ntc.pop((v, w))
        w_edge_length = ntc.pop((w, v))

        edges = _build_tree_rec(ctr, ntc, ntr, new_otus, edges)
        # find edge with merged node and add subtrees
        for x, v_, l in edges:
            if v_ == vsw:
                # add edge with length checking for negative values
                if v_edge_length < 0:
                    v_edge_length = 0
                    logging.warning(f'negative edge length for {v} <- {vsw}')
                if w_edge_length < 0:
                    w_edge_length = 0
                    logging.warning(f'negative edge length for {w} <- {vsw}')
                edges = edges.union([(v_, v, v_edge_length), (v_, w, w_edge_length)])
                break

    return edges


def build_tree(ctr_table: np.ndarray) -> nx.DiGraph:
    # operational taxonomic units, OTUs, init with cells
    otus = set(map(str, range(ctr_table.shape[0])))
    # at each iteration, contains the centroid to root distance for each pair of OTUs
    # OTU is a set of cells (frozenset) which consist of a non-modifiable subtree
    ctr = {}
    # node-to-centroid distances for each OTU (initially single-cells) wrt to each other (index order is important here
    # as opposed to ctr that is symmetric)
    ntc = {}  # dict (str,str) -> float
    # node-to-root distances for each OTU as average of node-to-centroid distances over all other OTUs
    ntr = {str(v): 0 for v in range(len(otus))}  # dict str -> float
    for v in range(len(otus)):
        v_str = str(v)
        for w in range(v + 1, len(otus)):
            w_str = str(w)
            vsw = frozenset({v_str, w_str})
            # init ctr distances
            ctr[vsw] = ctr_table[v, w, 0]

            # compute node to root distance of v wrt w
            ntc[v_str, w_str] = ctr_table[v, w, 1]
            # compute node to root distance of w wrt v
            ntc[w_str, v_str] = ctr_table[v, w, 2]

            # compute node to root distance of v
            ntr[v_str] += ntc[v_str, w_str] + ctr_table[v, w, 1]
            # compute node to root distance of w
            ntr[w_str] += ntc[w_str, v_str] + ctr_table[v, w, 2]

    # normalize node-to-root distances to get the average
    ntr = {str(v): ntr[str(v)] / (len(otus) - 1) for v in range(len(otus))}

    # build tree only using ctr distances
    edges = _build_tree_rec(ctr, ntc, ntr, otus, set())
    em_tree = nx.DiGraph()
    # add edges with lengths
    em_tree.add_weighted_edges_from(edges, weight='length')
    # add_lengths(em_tree, ctr_table)

    return em_tree


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # test EM algorithm
    n_cells = 10
    n_states = 7
    n_sites = 200
    print(f"Instance: {n_cells} cells, {n_states} states, {n_sites} sites")
    data = rand_dataset(n_cells, n_states, n_sites, obs_type='pois', p_change=0.05)
    print("True tree")
    data['tree'].print_plot(plot_metric='length')

    # true ctr_table
    # ctr_table = get_ctr_table(data['tree'])

    start_time = time.time()
    ctr_table = jcb_em_alg(data['obs'], max_iter=20)
    print(f"Total time: {time.time() - start_time}")

    nx_em_tree = build_tree(ctr_table)
    em_tree = convert_networkx_to_dendropy(nx_em_tree, taxon_namespace=data['tree'].taxon_namespace,
                                           edge_length='length')
    print("EM tree")
    em_tree.print_plot(plot_metric='length')
    print("EM tree (unweighted)")
    em_tree.print_plot()

    print(f"Symmetric unweighted difference: {symmetric_difference(data['tree'], em_tree)}")
    print(f"Unweighted Robinson-Foulds distance: {unweighted_robinson_foulds_distance(data['tree'], em_tree)}")
    print(f"Robinson-Foulds distance: {robinson_foulds_distance(data['tree'], em_tree, edge_weight_attr='length')}")
