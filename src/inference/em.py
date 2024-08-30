import itertools
import logging
import operator
from typing import Optional, Union

import numpy as np
import scipy.stats as stats
from scipy.special import comb
from scipy.special import logsumexp
import networkx as nx

from models.quadruplet import Quadruplet

from src.models.copy_tree import h_eps, get_zipping_mask, get_zipping_mask0, p_delta_trans_mat, p_delta_start_prob


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
                                                      np.clip(lam * np.array([i, j]), a_min=pois_mean_eps, a_max=None)).sum()

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


def compute_l_trans_mat(l_trip, n_states) -> np.ndarray:
    trans_mat = np.empty((n_states,) * 6)
    # transition from r -> u (fix r = 2)
    a_ru = p_delta_trans_mat(n_states, l_trip[0])[:, :, 2, 2]
    a_uv = p_delta_trans_mat(n_states, l_trip[1])
    a_uw = p_delta_trans_mat(n_states, l_trip[2])
    # results is state (m-1) i, j, k -> (m) x, y, z
    trans_mat[...] = np.einsum('xi,yjxi,zkxi->ijkxyz', a_ru, a_uv, a_uw)
    return trans_mat


def compute_eps_start_prob(eps_trip, n_states):
    # TODO: uniform prob, but not implemented properly yet
    return np.ones((n_states,) * 3) / n_states ** 3


def compute_l_start_prob(l_trip, n_states):
    trip_start = np.empty((n_states,) * 3)

    # transition from r -> u (fix r = 2)
    a_ru = p_delta_start_prob(n_states, l_trip[0])[:, 2]
    # transition from u -> v, w
    a_uv = p_delta_start_prob(n_states, l_trip[1])
    a_uw = p_delta_start_prob(n_states, l_trip[2])
    # results is state i, j, k
    trip_start[...] = np.einsum('i, ij, ik -> ijk', a_ru, a_uv, a_uw)
    return trip_start


def forward_pass(obs_vw, start_prob, trans_mat, log_emissions, eps=1e-5):
    # transmat idx = (i, j, k, x, y, z) = (m-1, m)
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    alpha = np.empty((n_sites, n_states, n_states, n_states))

    # init alpha[0] = start_prob * emission[0] (in log-form)
    alpha[0, ...] = np.log(start_prob.clip(eps)) + log_emissions[0, np.newaxis, ...]
    alpha[0] -= logsumexp(alpha[0])

    for m in range(1, n_sites):
        # [ \sum_{x,y,z} alpha_{m-1}(x, y, z) p(i, j, k | x, y, z) ] p(y_m^{vw} | i, j, k)
        alpha[m, ...] = logsumexp(alpha[m - 1] +
                                  np.log(trans_mat),
                                  axis=(0, 1, 2)) + log_emissions[m, np.newaxis, ...]
        # normalization step
        alpha[m] -= logsumexp(alpha[m])

    return alpha


def backward_pass(obs_vw, trans_mat, log_emissions):
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    beta = np.empty((n_sites, n_states, n_states, n_states))
    # initialize beta[M] = 1. (in log form) - normalized
    beta[-1, ...] = 3 * np.log(n_states)

    # compute iteratively
    for m in reversed(range(n_sites - 1)):
        beta[m, ...] = logsumexp(beta[m + 1, ...] +
                                 np.log(trans_mat)
                                 + log_emissions[m + 1, np.newaxis, np.newaxis, np.newaxis, ...],
                                 axis=(3, 4, 5))
        # normalization step
        beta[m] -= logsumexp(beta[m])

    return beta


def two_slice_marginals(obs_vw, theta: np.ndarray, n_states: int, jcb: bool = False):
    n_sites = obs_vw.shape[0]
    # define start_probs and transitions depending on chosen model
    if jcb:
        # start prob shape: (n_states, n_states, n_states)
        start_prob = compute_l_start_prob(theta, n_states)
        # trans_mat shape: ((n_states,) * 6)
        trans_mat = compute_l_trans_mat(theta, n_states)
    else:
        start_prob = compute_eps_start_prob(theta, n_states)
        trans_mat = compute_eps_trans_mat(theta, n_states)

    # define emission prob (for emission pair)
    # log emission shape (n_sites, n_states, n_states)
    log_emissions = compute_log_emissions(obs_vw, n_states)
    # compute forward: shape (n_sites, n_states, n_states, n_states)
    alpha = forward_pass(obs_vw, start_prob, trans_mat, log_emissions)
    # compute backward: shape (n_sites, n_states, n_states, n_states)
    beta = backward_pass(obs_vw, trans_mat, log_emissions)
    # tsm shape: (n_sites - 1,) + (n_states,) * 6
    # compute two slice: xi
    log_xi = alpha[(np.arange(n_sites - 1), ...) + (np.newaxis,) * 3] + np.log(trans_mat)[np.newaxis, ...] + \
        log_emissions[(np.arange(1, log_emissions.shape[0]),) + (np.newaxis,) * 4 + (...,)] + \
        beta[(np.arange(1, beta.shape[0]), ...) + (np.newaxis,) * 3]
    log_xi -= np.expand_dims(logsumexp(log_xi, axis=tuple(range(1, 7))), axis=tuple(range(1, 7)))

    assert np.allclose(logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6)), np.zeros(n_sites - 1))

    return log_xi


def compute_exp_changes(l_prev, obs_vw, n_states):
    d = np.empty(3)
    dp = np.empty_like(d)
    # compute two slice marginals
    # prob(Cm = ijk, Cm+1 = i'j'k' | Y)
    log_xi = two_slice_marginals(obs_vw, l_prev, n_states, jcb=True)

    comut_mask0 = get_zipping_mask0(n_states).transpose()
    comut_mask = get_zipping_mask(n_states).transpose(tuple(reversed(range(4))))
    # count expected changes/no-changes
    for e in range(3):
        if e == 0:
            # eps_ru (sum over m, j, k, j', k')
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

    return d, dp


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
            log_xi = two_slice_marginals(obs[:, [v, w]], epsilon_k, n_states)
            # update epsilon_k
            epsilon_kp1 = update_eps(log_xi)
            # check for convergence
            convergence = np.allclose(epsilon_k, epsilon_kp1, rtol=1.e-2)
            # update current eps
            epsilon_k[...] = epsilon_kp1

        epsilon_hat[v, w] = epsilon_k[0]  # ctr distance is eps_ru (first of triplet)

    return epsilon_hat


def jcb_em_alg(obs: np.ndarray) -> np.ndarray:
    """
Implementation of JCB EM algorithm in write-up
    Parameters
    ----------
    obs array of shape (n_sites, n_cells)
    Returns
    -------
    array of shape (n_cells, n_cells), centroid-to-root distance
    """
    # params
    n_states = 7
    l_init = 1.
    n_sites, n_cells = obs.shape
    l_hat = np.inf * np.ones((n_cells, n_cells))
    zero_tol = 1e-5  # saturation level when dp << d (changes are much more prevalent)

    # for each pair of cells
    counter = 0
    logging.debug(f'pairwise EM started: {comb(n_cells, 2)} pairs')
    for v, w in itertools.combinations(range(n_cells), r=2):
        # initialize l = (l_ru, l_uv, l_uw)
        l_i = np.array([l_init] * 3)
        # triplet state u, v, w
        convergence = False
        while not convergence:
            # compute D and D'
            d, dp = compute_exp_changes(l_i, obs[:, [v, w]], n_states)

            # update l according to formula
            if np.any((n_states - 1) * dp <= d):
                logging.warning(f"too many changes detected: D = {d}, D' = {dp}\n"
                                f"...saturating l for cells {v},{w}")
            # if l -> +inf, pDeltaDelta == pDeltaDelta'
            num = np.clip((n_states - 1) * dp - d, a_min=zero_tol, a_max=None)
            l_new = - 1 / n_states * np.log(num / ((n_states - 1) * (dp + d)))

            l_delta = l_new - l_i
            convergence = np.allclose(l_delta, np.zeros_like(l_delta), atol=0.001)
            l_i = l_new  # update current l

        counter += 1
        if counter % 10 == 0:
            logging.debug(f'{comb(n_cells, 2) - counter} pairs remaining...')
        l_hat[v, w] = l_i[0]  # ctr distance is l_ru (first of triplet)

    return l_hat


def _build_tree_rec(dist: dict, otus: set, edges: set[tuple]):

    if len(otus) == 2:
        for c in otus:
            edges.add(('r', c))
    else:
        vw, l = max(dist.items(), key=operator.itemgetter(1))
        # remove pair and add common ancestor with averaged distance
        dist.pop(vw)
        v, w = vw
        # update distances merging vw in one OTU
        vsw = v + '_' + w  # node with string showing merges v_w
        new_otus = otus.difference({w, v})
        for c in new_otus:
            vc = frozenset({v, c})
            wc = frozenset({w, c})
            # update distance for new node
            dist[frozenset({vsw, c})] = .5 * (dist[vc] + dist[wc])
            # remove already merged nodes
            dist.pop(vc)
            dist.pop(wc)

        # add node/subtree as OTU
        new_otus.add(vsw)

        edges = _build_tree_rec(dist, new_otus, edges)
        # find edge with merged node and add subtrees
        for x, v_ in edges:
            if v_ == vsw:
                edges = edges.union([(v_, v), (v_, w)])
                break

    return edges


def build_tree(ctr_table):

    # operational taxonomic units, OTUs, init with cells
    otus = set(map(str, range(ctr_table.shape[0])))
    dist = {frozenset({str(v), str(w)}): ctr_table[v, w] for v in range(len(otus)) for w in range(v + 1, len(otus))}

    edges = _build_tree_rec(dist, otus, set())
    em_tree = nx.DiGraph()
    em_tree.add_edges_from(edges)

    return em_tree