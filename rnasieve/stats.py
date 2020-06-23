import numpy as np
import scipy.linalg
from rnasieve.helper import compute_mixture_sigma

# Confidence Interval Computation

def partial_mixture_variance_alpha(phi, sigma, alpha):
    return sigma + phi**2 - 2 * phi * (phi @ alpha.T)

def partial_mixture_variance_phi(phi, alpha):
    return 2 * np.tile(alpha, (phi.shape[0], 1)) * (phi - phi @ alpha.T)

def partial_alpha_alpha(phi, sigma, alpha, n):
    mixture_variance = compute_mixture_sigma(alpha, sigma, phi)
    partial_mv_alpha = partial_mixture_variance_alpha(phi, sigma, alpha)
    return n * phi.T @ np.diag(1 / mixture_variance.reshape(-1)) @ phi + partial_mv_alpha.T @ np.diag(1 / mixture_variance.reshape(-1)**2) @ partial_mv_alpha / 2

def partial_phi_phi(phi, sigma, alpha, n, m):
    G, K = phi.shape
    mixture_variance = compute_mixture_sigma(alpha, sigma, phi)
    alpha_term = np.repeat(n * np.kron(alpha, alpha).reshape(K, K)[:, :, np.newaxis], G, axis=2) / mixture_variance.reshape(1, 1, -1)
    partial_mv_phi = partial_mixture_variance_phi(phi, alpha)
    phi_term = np.dstack([np.kron(partial_mv_phi[i], partial_mv_phi[i]).reshape(K, K) / (2 * mixture_variance[i]**2) for i in range(G)])
    return scipy.linalg.block_diag(*[np.squeeze(subarr) for subarr in np.dsplit(alpha_term + phi_term, G)]) + np.diag((m / sigma).reshape(-1))

def partial_alpha_phi(phi, sigma, alpha, n):
    G, K = phi.shape
    mixture_variance = compute_mixture_sigma(alpha, sigma, phi)
    partial_mv_alpha = partial_mixture_variance_alpha(phi, sigma, alpha)
    partial_mv_phi = partial_mixture_variance_phi(phi, alpha)

    @np.vectorize
    def partial_alpha_phi_helper(i, j):
        g, k = j // K, j % K
        return n * alpha[0, k] * phi[g, i] / mixture_variance[g, 0] + partial_mv_alpha[g, i] * partial_mv_phi[g, k] / (2 * mixture_variance[g, 0]**2)

    return partial_alpha_phi_helper(*np.mgrid[0:K, 0:G*K])

def partial_n_n(phi, sigma, alpha, n):
    mixture_variance = compute_mixture_sigma(alpha, sigma, phi)
    G = phi.shape[0]
    return (1 - G / 2) / n**2 + np.sum((phi @ alpha.T)**2 / mixture_variance) / n

def partial_alpha_n(phi, sigma, alpha, n):
    mixture_variance = compute_mixture_sigma(alpha, sigma, phi)
    partial_mv_alpha = partial_mixture_variance_alpha(phi, sigma, alpha)
    return (phi / mixture_variance).T @ (phi @ alpha.T) + np.sum(partial_mv_alpha / mixture_variance, axis=0).reshape(-1, 1) / (2 * n)

def partial_n_phi(phi, sigma, alpha, n):
    mixture_variance = compute_mixture_sigma(alpha, sigma, phi)
    partial_mv_phi = partial_mixture_variance_phi(phi, alpha)
    return (np.kron((phi @ alpha.T) / mixture_variance, alpha) + partial_mv_phi / (2 * n * mixture_variance)).reshape(1, -1)

def inverse_observed_fisher(phi, sigma, alpha, n, m):
    p_alpha_alpha = partial_alpha_alpha(phi, sigma, alpha, n)
    p_n_n = partial_n_n(phi, sigma, alpha, n)
    p_phi_phi = partial_phi_phi(phi, sigma, alpha, n, m)
    p_alpha_phi = partial_alpha_phi(phi, sigma, alpha, n)
    p_alpha_n = partial_alpha_n(phi, sigma, alpha, n)
    p_n_phi = partial_n_phi(phi, sigma, alpha, n)

    fisher_info = np.block([
        [p_alpha_alpha, p_alpha_n, p_alpha_phi],
        [p_alpha_n.T  , p_n_n    , p_n_phi    ],
        [p_alpha_phi.T, p_n_phi.T, p_phi_phi  ],
    ])

    inv_fisher = np.linalg.inv(fisher_info)
    return ((inv_fisher + inv_fisher.T) / 2), fisher_info
