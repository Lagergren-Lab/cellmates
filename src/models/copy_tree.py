import logging
import math

import networkx as nx
import numpy as np
import scipy.stats as sp_stats
import itertools


# TODO: differentiate impossible cases to mask
def get_zipping_mask(n_states):
    """Build a mask on the i - i' == j - j' condition

    Parameters
    ----------
    n_states :
        Number of total copy number states (cn = 0, 1, ..., n_states - 1)
    Returns
    -------
    np boolean tensor of shape (n_states, n_states, n_states, n_states),
    true where the indices satisfy the condition.
    Idx order is `mask[j', j, i', i]`
    """
    ind_arr = np.indices((n_states, n_states))
    # i - j
    imj = ind_arr[0] - ind_arr[1]
    # i - j == k - l
    mask = imj == imj[:, :, np.newaxis, np.newaxis]
    return mask


def get_zipping_mask0(n_states):
    """Build a mask on the i == j condition

    Parameters
    ----------
    n_states :
        Number of total copy number states (cn = 0, 1, ..., n_states - 1)
    Returns
    -------
    np boolean tensor of shape (n_states, n_states),
    true where the indices satisfy the condition
    """
    ind_arr = np.indices((n_states, n_states))
    # i = j (diagonal)
    mask = ind_arr[0] == ind_arr[1]
    return mask


def p_delta(n_states, l, i, ii, j, jj, alpha=1.):
    """
    Returns the transition probability given the length and the
    copy number configurations.
    Parameters
    ----------
    n_states int, number of copy number states 0, ..., K (equals K + 1) with K max cn
    l float, length parameter
    i int, C_{m-1}^p
    ii int, C_m^p
    j int, C_{m-1}^v
    jj int, C_m^v

    Returns
    -------
    normalized transition probability
    """
    if ii - i + j < 0 or ii - i + j > n_states - 1:
        return 1 / n_states
    else:
        change = ii - i != jj - j
        return p_delta_change(n_states, l, change, alpha=alpha)


def p_delta_change(n_states, l, change: bool, alpha: float = 1.):
    if not change:
        p_out = 1 / n_states + (n_states - 1) / n_states * math.exp(- n_states * alpha * l)
        # p_out = 1 / n_states + (n_states - 1) / n_states * math.exp(- n_states / (n_states - 1) * alpha * l)
    else:
        p_out = 1 / n_states - math.exp(- n_states * alpha * l) / n_states
        # p_out = 1 / n_states - math.exp(- n_states / (n_states - 1) * alpha * l) / n_states
    return p_out


def p_delta_trans_mat(n_states, l, alpha: float = 1.):
    """
    Indexing order: [j', j, i', i]. Invariant: sum(dim=0) = 1.
    Args:
        n_states: total number of copy number states
        l: arc distance parameter

    Returns:
        tensor of shape (A x A x A x A) with A = n_states
    """
    mat = np.empty((n_states,) * 4)

    for (i, ii, j, jj) in itertools.product(range(n_states), repeat=4):
        mat[jj, j, ii, i] = p_delta(n_states, l, i, ii, j, jj, alpha=alpha)
    return mat


def p_delta_start_prob(n_states, l, alpha: float = 1.):
    """
    p_delta(C_1^v | C_1^p) initial probability
    Indexing order: [j, i]. Invariant: sum(dim=0) = 1.
    Args:
        n_states: total number of copy number states
        l: arc distance parameter

    Returns:
        tensor of shape (A x A) with A = n_states
    """
    mat = np.empty((n_states,) * 2)

    for (i, j) in itertools.product(range(n_states), repeat=2):
        mat[j, i] = p_delta_change(n_states, l, j != i, alpha=alpha)
    return mat


def h_eps(n_states: int, eps: float) -> np.ndarray:
    """
Zipping function tensor for given epsilon. In arc u->v, for each
combination, P(Cv_m=j'| Cv_{m-1}=j, Cu_m=i', Cu_{m-1}=i) = h(j'|j, i', i).
Indexing order: [j', j, i', i]. Invariant: sum(dim=0) = 1.
    Args:
        n_states: total number of copy number states
        eps: arc distance parameter

    Returns:
        tensor of shape (A x A x A x A) with A = n_states
    """
    # TODO: add zero-absorption P(Cu>0 | Cp=0) = 0
    mask_arr = get_zipping_mask(n_states=n_states)
    # put 1-eps where j'-j = i'-i
    a = mask_arr * (1 - eps)
    # put either 1 or 1-eps in j'-j != i'-i  and divide by the cases
    b = (1 - np.sum(a, axis=0)) / np.sum(~mask_arr, axis=0)
    # combine the two arrays
    out_arr = b * (~mask_arr) + a
    return out_arr


def h_eps0(n_states: int, eps0: float) -> np.ndarray:
    """
Simple zipping function tensor. P(Cv_1=j| Cu_1=i) = h0(j|i)
    Args:
        n_states: total number of copy number states
        eps: arc distance parameter

    Returns:
        tensor of shape (A x A) with A = n_states
    """
    heps0_arr = eps0 / (n_states - 1) * np.ones((n_states, n_states))
    diag_mask = get_zipping_mask0(n_states)
    heps0_arr[diag_mask] = 1 - eps0
    return heps0_arr


class CopyTree:
    def __init__(self, M, A, true_tree: nx.DiGraph):
        self.true_tree = true_tree
        self.M = M
        self.A = A
        self.K = len(true_tree.nodes)

    def simulate_data(self, eps_a, eps_b, eps_0):
        eps = np.zeros((self.K, self.K))
        logging.debug(f'Copy Tree data simulation - eps_a: {eps_a}, eps_b: {eps_b}, eps_0:{eps_0} ')
        eps_dist = sp_stats.beta(eps_a, eps_b)
        if eps_dist.std()**2 > 0.1 * eps_dist.mean():
            logging.warning(
                f'Large variance for epsilon: {eps_dist.std()**2} (mean: {eps_dist.mean()}. Consider increasing '
                f'eps_b param.')

        tree = self.true_tree
        for u, v in tree.edges:
            eps[u, v] = eps_dist.rvs()
            tree.edges[u, v]['weight'] = eps[u, v]

        # generate copy numbers
        c = np.empty((self.K, self.M), dtype=int)
        c[0, :] = 2 * np.ones(self.M, )
        h_eps0_cached = h_eps0(self.A, eps_0)
        for u, v in nx.bfs_edges(tree, source=0):
            t0 = h_eps0_cached[c[u, 0], :]
            c[v, 0] = np.argmax(sp_stats.multinomial(n=1, p=t0).rvs())
            h_eps_uv = h_eps(self.A, eps[u, v])
            for m in range(1, self.M):
                # j', j, i', i
                transition = h_eps_uv[:, c[v, m - 1], c[u, m], c[u, m - 1]]
                c[v, m] = np.argmax(sp_stats.multinomial(n=1, p=transition).rvs())

        return eps, c
