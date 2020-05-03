"""
Helper functions for computing common terms used in RNA-Sieve.
"""

import numpy as np

# Constants

CLIP_VALUE = 1e-10
ALPHA_EPS = 1e-4
ALPHA_MAX_ITER = 1


# General Functions

def compute_mixture_sigma(alpha, sigma, phi):
    return (phi**2 + sigma) @ alpha.T - (phi @ alpha.T)**2


def compute_weighted_norm(vec, weights, p=2):
    return np.sum(vec**p * weights)


def compute_full_likelihood(
        phi, observed_phi, observed_psis, alphas, sigma, ns, m):
    mixture_variance = np.clip(compute_mixture_sigma(
        alphas, sigma, phi), CLIP_VALUE, None)
    phi_likelihood = np.sum(np.tile(m, (len(phi), 1)) /
                            sigma * (phi - observed_phi)**2)
    psi_likelihood = np.sum(ns / mixture_variance *
                            (phi @ alphas.T - observed_psis / ns)**2)
    log_term = np.sum(np.log(ns * mixture_variance))
    return phi_likelihood + psi_likelihood + log_term


def compute_row_likelihood(phi_row, observed_phi_row,
                           observed_psi_scalars, alphas, sigma_row, ns, m):
    mixture_variance = np.clip(compute_mixture_sigma(
        alphas, sigma_row, phi_row), CLIP_VALUE, None)
    phi_likelihood = np.sum(m / sigma_row * (phi_row - observed_phi_row)**2)
    psi_likelihood = np.sum(ns / mixture_variance *
                            (phi_row @ alphas.T - observed_psi_scalars / ns)**2)
    log_term = np.sum(np.log(ns * mixture_variance))
    return phi_likelihood + psi_likelihood + log_term
