import numpy as np


def l_from_p(p_change, n_states=4):
    """
    Inverse formula to determine length from p_change (probability of changing *to any state*).
    Parameters
    ----------
    p_change: float, probability of changing to any state
    n_states: int, number of copy number states

    Returns
    -------
    float, length parameter
    """
    return - 1 / n_states * np.log(1 - n_states / (n_states - 1) * p_change)

