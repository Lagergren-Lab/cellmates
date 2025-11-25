"""
Utils for HMM functions (forward, backward, viterbi, etc.)
"""
import numpy as np
import scipy.special as sp

from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal


# helpers for pomegranate HMM
def pmg_make_normal_obs_models(n_states, mu, tau):
    obs_models = []
    for ijk in range(n_states ** 3):
        i = ijk // (n_states ** 2)
        j = (ijk // n_states) % n_states
        k = ijk % n_states
        mean_j = mu * j
        mean_k = mu * k
        # observation is a pair y_j, y_k ~ N(mu*j, tau) , N(mu*k, tau) independently
        obs_models.append(Normal((mean_j, mean_k), (tau, tau), covariance_type='diag'))
    return obs_models


def pmg_make_normal_emissions(X, n_states, mu, tau):
    n_samples = 1
    seq_length = X.shape[0]
    emissions = np.zeros((n_samples, seq_length, n_states ** 3))
    # for each m and each possible state ijk, compute the log prob of observing X[m] given state ijk
    for ijk in range(n_states ** 3):
        i = ijk // (n_states ** 2)
        j = (ijk // n_states) % n_states
        k = ijk % n_states
        mean_j = mu * j
        mean_k = mu * k
        # observation is a pair y_j, y_k ~ N(mu*j, tau) , N(mu*k, tau) independently
        obs_model = Normal((mean_j, mean_k), (tau, tau), covariance_type='diag')
        emissions[0, :, ijk] = obs_model.log_probability(X)
    return emissions

def pmg_convert_emissions(log_emissions):
    """
    From shape (n_sites, n_states, n_states) to (1, n_sites, n_states**3) to match pomegranate HMM input
    0th dim is batch size (1)
    """
    n_sites, n_states, _ = log_emissions.shape
    # add batch and median state dims, then reshape to (1, n_sites, n_states**3)
    emissions_3D = np.repeat(log_emissions[None, :, None, :, :], repeats=n_states, axis=2).reshape((1, n_sites, n_states ** 3))
    return emissions_3D

def _forward_likelihood_pomegranate(log_emissions, trans_mat, start_prob):
    """
    Compute forward variables and log likelihood using pomegranate HMM
    Parameters
    ----------
    log_emissions : np.ndarray
        Log emissions of shape (n_sites, n_states, n_states) as of the TripHMM model
    trans_mat : np.ndarray
        Transition matrix of shape (n_states, n_states, n_states, n_states, n_states, n_states)
    start_prob : np.ndarray
        Start probabilities of shape (n_states, n_states, n_states)
    Returns
    -------
    alpha : np.ndarray
        Forward variables of shape (n_sites, n_states, n_states, n_states)
    log_p : float
        Log likelihood of the observed sequence
    """
    n_sites, n_states, _ = log_emissions.shape
    log_emissions_3D = pmg_convert_emissions(log_emissions)
    trans_mat_2D = trans_mat.reshape((n_states ** 3, n_states ** 3))
    start_prob_1D = start_prob.flatten()
    distributions = [Normal() for _ in range(n_states ** 3)]  # dummy distributions, used only for shape
    model = DenseHMM(distributions=distributions, edges=trans_mat_2D, starts=start_prob_1D, ends=np.ones(n_states ** 3) / (n_states ** 3))
    f = model.forward(emissions=log_emissions_3D).numpy()
    alpha = f.reshape((n_sites, n_states, n_states, n_states))
    log_p = sp.logsumexp(alpha[-1], axis=(0, 1, 2))
    return alpha, log_p

def _forward_backward_pomegranate(log_emissions, trans_mat, start_prob, debug=False):
    """
    Compute expected counts, marginal probabilities, and log likelihood using pomegranate HMM
    Parameters
    ----------
    """
    n_sites, n_states, _ = log_emissions.shape
    log_emissions_3D = pmg_convert_emissions(log_emissions)
    trans_mat_2D = trans_mat.reshape((n_states ** 3, n_states ** 3))
    start_prob_1D = start_prob.flatten()
    distributions = [Normal() for _ in range(n_states ** 3)]  # dummy distributions, used only for shape
    model = DenseHMM(distributions=distributions, edges=trans_mat_2D, starts=start_prob_1D, ends=np.ones(n_states ** 3) / (n_states ** 3))
    expected_counts, marginal, _, _, log_p = model.forward_backward(emissions=log_emissions_3D)
    expected_counts = expected_counts.reshape((n_states,) * 6).numpy()  # from 2D* (1, n_states**3, n_states**3) to 6D (n_states,) * 6
    marginal = marginal.reshape((n_sites, n_states, n_states, n_states)).numpy()  # from 2D* (1, n_sites, n_states**3) to 4D (n_sites, n_states, n_states, n_states)
    if debug:
        # check expected counts sum to n_sites - 1
        assert np.isclose(expected_counts.sum(), n_sites - 1), f"Expected counts sum {expected_counts.sum()} != n_sites - 1 {n_sites - 1}"
        # check marginal sums to 1 at each site
        marginal_sums = sp.logsumexp(marginal, axis=(1, 2, 3))
        assert np.allclose(marginal_sums, np.zeros(n_sites)), f"Marginal sums not close to 1: {marginal_sums}"
    return expected_counts, marginal, log_p.item()

def _forward_likelihood_broadcast(log_emissions, trans_mat, start_prob, normalize=True):
    eps = 1e-10  # to avoid log(0)
    # transmat idx = (i, j, k, x, y, z) = (m-1, m)
    n_sites, n_states, _ = log_emissions.shape
    alpha = np.empty((n_sites, n_states, n_states, n_states))

    # init alpha[0] = start_prob * emission[0] (in log-form)
    alpha[0, ...] = np.log(start_prob.clip(eps)) + log_emissions[0, None, ...]
    log_trans_mat = np.log(trans_mat)
    norm = sp.logsumexp(alpha[0])
    alpha[0] -= norm if normalize else 0.0

    log_likelihood = norm
    for m in range(1, n_sites):
        log_acc = sp.logsumexp(alpha[m - 1, :, :, :, None, None, None] + log_trans_mat, axis=(0, 1, 2))
        alpha[m, ...] = log_acc + log_emissions[m, None, :, :]
        norm = sp.logsumexp(alpha[m])
        # normalization step
        if normalize:
            alpha[m] -= norm
        # update log likelihood to save computation
        log_likelihood += norm

    return alpha, log_likelihood

def _backward_pass_broadcast(log_emissions, trans_mat, normalize=True):
    n_sites, n_states, _ = log_emissions.shape

    log_trans_mat = np.log(trans_mat)
    # initialize beta[M] = 1. (in log form)
    beta = np.zeros((n_sites, n_states, n_states, n_states))
    if normalize:
        beta[-1, ...] = - 3 * np.log(n_states)

    # compute iteratively
    for m in reversed(range(n_sites - 1)):
        beta[m, ...] = sp.logsumexp(beta[m + 1, None, None, None, :, :, :] +
                                    log_trans_mat +
                                    log_emissions[m + 1, None, None, None, None, :, :], axis=(3, 4, 5))
        # normalization step
        if normalize:
            beta[m] -= sp.logsumexp(beta[m])
    return beta

def _backward_pass_pomegranate(log_emissions, trans_mat):
    n_sites, n_states, _ = log_emissions.shape
    log_emissions_3D = pmg_convert_emissions(log_emissions)
    trans_mat_2D = trans_mat.reshape((n_states ** 3, n_states ** 3))
    distributions = [Normal() for _ in range(n_states ** 3)]
    model = DenseHMM(distributions=distributions, edges=trans_mat_2D, starts=np.ones(n_states ** 3) / (n_states ** 3), ends=np.ones(n_states ** 3) / (n_states ** 3))
    b = model.backward(emissions=log_emissions_3D).numpy()
    beta = b.reshape((n_sites, n_states, n_states, n_states))
    return beta

def _forward_backward_broadcast(log_emissions, trans_mat, start_prob, debug=False):
    # define emission prob (for emission pair)
    # log emission shape (n_sites, n_states, n_states)
    n_sites, n_states, _ = log_emissions.shape
    # compute forward: shape (n_sites, n_states, n_states, n_states)
    alpha, log_p = _forward_likelihood_broadcast(log_emissions, trans_mat, start_prob)
    log_p = sp.logsumexp(alpha[-1], axis=(0, 1, 2))  # final log likelihood
    # compute backward: shape (n_sites, n_states, n_states, n_states)
    beta = _backward_pass_broadcast(log_emissions, trans_mat)
    # tsm shape: (n_sites - 1,) + (n_states,) * 6
    log_trans_mat = np.log(trans_mat)
    # compute two slice: xi
    alpha_ = alpha[:-1, :, :, :, np.newaxis, np.newaxis, np.newaxis]  # (n_sites - 1, n_states, n_states, n_states, 1, 1, 1)
    beta_ = beta[1:, np.newaxis, np.newaxis, np.newaxis, :, :, :] + log_emissions[1:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
    log_xi = alpha_ + beta_ + log_trans_mat[np.newaxis, ...]
    log_xi = log_xi - sp.logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6), keepdims=True)  # normalize
    expected_counts = np.exp(sp.logsumexp(log_xi, axis=0))
    # compute gamma
    log_gamma = alpha + beta
    log_gamma = log_gamma - sp.logsumexp(log_gamma, axis=(1, 2, 3), keepdims=True)  # normalize
    if debug:
        assert np.isclose(np.sum(expected_counts), n_sites - 1), f"Expected counts sum {np.sum(expected_counts)} != n_sites - 1 ({n_sites - 1})"
        assert np.allclose(sp.logsumexp(log_gamma, axis=(1, 2, 3)), np.zeros(n_sites)), f"Gamma sums not close to 1: {sp.logsumexp(log_gamma, axis=(1, 2, 3))}"
    return expected_counts, log_gamma, log_p

def viterbi_decode_pomegranate(log_emissions, trans_mat, start_prob) -> np.ndarray:
    """
    Viterbi decode using pomegranate HMM
    Parameters
    ----------
    log_emissions : np.ndarray
        Log emissions of shape (n_sites, n_states, n_states)
    trans_mat : np.ndarray
        Transition matrix of shape (n_states, n_states, n_states, n_states, n_states, n_states)
    start_prob : np.ndarray
        Start probabilities of shape (n_states, n_states, n_states)
    Returns
    -------
    path : np.ndarray
        Viterbi path of shape (n_sites, n_states, n_states, n_states)
    """
    n_sites, n_states, _ = log_emissions.shape
    log_emissions_3D = pmg_convert_emissions(log_emissions)
    start_prob_1D = start_prob.flatten()
    trans_mat_2D = trans_mat.reshape((n_states ** 3, n_states ** 3))
    distributions = [Normal() for _ in range(n_states ** 3)]  # dummy distributions, used only for shape
    model = DenseHMM(distributions=distributions, edges=trans_mat_2D, starts=start_prob_1D, ends=np.ones(n_states ** 3) / (n_states ** 3))
    path = model.viterbi(emissions=log_emissions_3D)
    path = np.array([((state // (n_states ** 2)) % n_states,
                      (state // n_states) % n_states,
                      state % n_states) for state in path[0]]).T  # extract states from (index, state)
    return path
