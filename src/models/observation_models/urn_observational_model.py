import logging

import numpy as np
import scipy.stats as sp_stats


class UrnModel():

    def __init__(self, N, M, A):
        self.N = N
        self.M = M
        self.A = A

    def simulate_data(self, R_0, min_gc, max_gc, c):
        R = self.simulate_total_reads(R_0)
        gc = self.simulate_gc_site_corrections(min_gc, max_gc)
        x, phi = self.simulate_observations_Poisson(c, R, gc)
        return x, R, gc, phi

    def simulate_total_reads(self, R_0) -> np.array:
        a = int(R_0 - R_0 / 10.)
        b = int(R_0 + R_0 / 10.)
        logging.debug(f"Reads per cell simulation: R in  [{a},{b}] ")
        R = sp_stats.randint(a, b).rvs(self.N)
        return R

    def simulate_gc_site_corrections(self, min_gc=0.8, max_gc=1.2):
        logging.debug(f"GC correction per site simulation: g_m in  [{min_gc},{max_gc}] ")
        gc_dist = sp_stats.uniform(min_gc, max_gc)
        gc = gc_dist.rvs((self.M,))
        return gc

    def simulate_observations_Poisson(self, c, R, gc):
        x = np.zeros((self.N, self.M))
        phi = np.einsum("nm, m -> n", c, gc)
        for n in range(self.N):
            mu_n = c[n] * gc * R[n] / phi[n]
            x_n_dist = sp_stats.poisson(mu_n)
            x[n] = x_n_dist.rvs()

        return x, phi
