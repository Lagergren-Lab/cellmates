import logging
import math

import networkx as nx
import numpy as np
import scipy.stats as sp_stats
import itertools





class CopyNumberTransitionModel:
    def __init__(self, M, A):
        self.M = M
        self.A = A

    def simulate_data(self, eps_a, eps_b, eps_0):
        pass

    def get_zipping_mask(self, n_states):
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

    def get_zipping_mask0(self, n_states):
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

    # TODO: test that delta is normalized over jj for any l

    def h_eps(self, n_states: int, eps: float) -> np.ndarray:
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
        mask_arr = self.get_zipping_mask(n_states=n_states)
        # put 1-eps where j'-j = i'-i
        a = mask_arr * (1 - eps)
        # put either 1 or 1-eps in j'-j != i'-i  and divide by the cases
        b = (1 - np.sum(a, axis=0)) / np.sum(~mask_arr, axis=0)
        # combine the two arrays
        out_arr = b * (~mask_arr) + a
        return out_arr

    def h_eps0(self, n_states: int, eps0: float) -> np.ndarray:
        """
        Simple zipping function tensor. P(Cv_1=j| Cu_1=i) = h0(j|i)
        Args:
            n_states: total number of copy number states
            eps: arc distance parameter

        Returns:
            tensor of shape (A x A) with A = n_states
        """
        heps0_arr = eps0 / (n_states - 1) * np.ones((n_states, n_states))
        diag_mask = self.get_zipping_mask0(n_states)
        heps0_arr[diag_mask] = 1 - eps0
        return heps0_arr


class CopyNumberBranchLengthModel:
    def __init__(self, M, A):
        self.M = M
        self.A = A

    def simulate_data(self, eps_a, eps_b, eps_0):
        pass

    def p_delta_trans_mat(self, n_states, l):
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
            mat[jj, j, ii, i] = self.p_delta(n_states, l, i, ii, j, jj)
        return mat

    def p_delta_start_prob(self, n_states, l):
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
            mat[j, i] = self.p_delta_change(n_states, l, j != i)
        return mat

    def p_delta(self, n_states, l, i, ii, j, jj):
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
            return self.p_delta_change(n_states, l, change)

    def p_delta_change(self, n_states, l, change: bool):

        if not change:
            p_out = 1 / n_states + (n_states - 1) / n_states * math.exp(- (n_states - 1) * l)
        else:
            p_out = 1 / n_states - math.exp(- (n_states - 1) * l) / n_states
        return p_out






