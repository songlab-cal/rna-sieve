import numpy as np
import cvxpy as cp
import scipy.optimize as scopt
import scipy.special
import math
from multiprocessing import Pool
from multiprocessing import sharedctypes

CLIP_VALUE = 1e-8

# Parallelization Global shared arrays
shared_dict = {}

# General Functions


def compute_mixture_sigma(alpha, sigma, phi):
    return (phi**2 + sigma) @ alpha.T - (phi @ alpha.T)**2


def compute_weighted_norm(vec, weights, p=2):
    return np.sum(vec**p * weights)


def compute_full_likelihood(phi, observed_phi, observed_psis, alphas, sigma, ns, m):
    mixture_variance = np.clip(compute_mixture_sigma(
        alphas, sigma, phi), CLIP_VALUE, None)
    phi_likelihood = np.sum(np.tile(m, (len(phi), 1)) /
                            sigma * (phi - observed_phi)**2)
    psi_likelihood = np.sum(ns / mixture_variance *
                            (phi @ alphas.T - observed_psis / ns)**2)
    log_term = np.sum(np.log(mixture_variance / ns))
    return phi_likelihood + psi_likelihood + log_term


def compute_row_likelihood(phi_row, observed_phi_row, observed_psi_scalars, alphas, sigma_row, ns, m):
    mixture_variance = np.clip(compute_mixture_sigma(
        alphas, sigma_row, phi_row), CLIP_VALUE, None)
    phi_likelihood = np.sum(m / sigma_row * (phi_row - observed_phi_row)**2)
    psi_likelihood = np.sum(ns / mixture_variance *
                            (phi_row @ alphas.T - observed_psi_scalars / ns)**2)
    log_term = np.sum(np.log(mixture_variance / ns))
    return phi_likelihood + psi_likelihood + log_term

# Updates (1)


def update_n(phi, alpha, psi, sigma):
    mixture_var = np.clip(compute_mixture_sigma(
        alpha, sigma, phi), CLIP_VALUE, None)
    mix_var_norm = np.clip(compute_weighted_norm(
        phi @ alpha.T, 1 / mixture_var.T), CLIP_VALUE, None)
    frac_norm = psi.shape[0] / (2 * mix_var_norm)
    return frac_norm + np.sqrt(frac_norm**2 + compute_weighted_norm(psi, 1 / mixture_var) + mix_var_norm)


def minimize_phi_row(phi_prev, psi_scalars, observed_phi_row, sigma_row, alphas, ns, m, row_idx=None, parallelized=False, shared_array_id=None):
    phi_next = cp.Variable(phi_prev.shape)
    phi_final = np.copy(phi_prev)
    try:
        phi_prev = np.clip(phi_prev, CLIP_VALUE, None)
        sample_coef = m / np.clip(sigma_row, CLIP_VALUE, None)
        bulk_coef = ns / \
            np.clip(compute_mixture_sigma(
                alphas, sigma_row, phi_prev), CLIP_VALUE, None)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(sample_coef, (observed_phi_row - phi_next)**2)) +
                                      cp.sum(cp.multiply(bulk_coef, (phi_next@alphas.T - psi_scalars / ns)**2))), [phi_next >= 0])
        prob.solve()
        phi_final = np.clip(phi_next.value, 0, None)
    except (cp.SolverError, TypeError) as e:
        pass
    if phi_next.value is None:
        phi_final = np.clip(phi_prev, 0, None)

    if parallelized and shared_array_id:
        phi_tmp = np.ctypeslib.as_array(shared_dict[shared_array_id])
        phi_tmp[row_idx, :] = phi_final

    return phi_final


def minimize_phi(phi_prev, psis, observed_phi, sigma, alphas, ns, m, parallelized=False, num_process=1):
    if not parallelized:
        phi_next = np.empty((0, phi_prev.shape[1]), dtype=np.float64)
        for i in range(phi_prev.shape[0]):
            phi_next = np.vstack((phi_next, minimize_phi_row(phi_prev[i, :].reshape(
                1, -1), psis[i].reshape(1, -1), observed_phi[i, :].reshape(1, -1), sigma[i, :].reshape(1, -1), alphas, ns, m)))
        return phi_next
    else:
        phi_next = np.ctypeslib.as_ctypes(np.zeros(phi_prev.shape))
        phi_next_shared_array = sharedctypes.RawArray(
            phi_next._type_, phi_next)
        shared_array_id = "MINIMIZE_PHI"
        shared_dict[shared_array_id] = phi_next_shared_array
        p = Pool(processes=num_process)
        p.starmap(minimize_phi_row, [(phi_prev[i, :].reshape(1, -1), psis[i].reshape(1, -1), observed_phi[i, :].reshape(
            1, -1), sigma[i, :].reshape(1, -1), alphas, ns, m, i, True, shared_array_id) for i in range(phi_prev.shape[0])])
        p.close()
        return np.ctypeslib.as_array(phi_next_shared_array)


def minimize_alpha(alpha_prev, sigma, phi, psi, n):
    alpha_next = cp.Variable(alpha_prev.shape)
    alpha = np.copy(alpha_prev)
    try:
        coef = n / np.clip(compute_mixture_sigma(alpha_prev,
                                                 sigma, phi), CLIP_VALUE, None)
        prob = cp.Problem(cp.Minimize(
            cp.sum(cp.multiply(coef, (phi@alpha_next.T - psi)**2))), [alpha_next >= 0])
        prob.solve()
        alpha = np.clip(alpha_next.value, 0, None)
    except (cp.SolverError, TypeError) as e:
        pass
    if alpha_next.value is None:
        return alpha_prev
    return alpha / np.sum(alpha)


def minimize_alpha_LS_helper(alpha_prev, sigma, phi, psi, n, eps, max_iter):
    alpha_last = alpha_prev
    alpha_cur = minimize_alpha(alpha_prev, sigma, phi, psi, n)
    it = 0
    while np.linalg.norm(alpha_cur - alpha_last, ord=2) > eps and it < max_iter:
        alpha_cur, alpha_last = minimize_alpha(
            alpha_cur, sigma, phi, psi, n), alpha_cur
        it += 1
    return alpha_cur


def alternate_minimization(phi, sigma, m, psis, eps, delta, max_iter, uniform_init, parallelized=False, num_process=1):
    # note rcond parameter is set to silence a deprecation warning
    naive_LS = np.clip(np.linalg.lstsq(phi, psis, rcond=-1)[0], 0, None)
    if uniform_init:
        alpha_pasts = np.ones(naive_LS.T.shape) / naive_LS.T.shape[1]
        n_pasts = np.sum(phi @ alpha_pasts.T, axis=0) / np.sum(psis, axis=0)
    else:
        alpha_pasts = naive_LS.T / np.sum(naive_LS, axis=0).reshape(-1, 1)
        n_pasts = np.linalg.norm(naive_LS, ord=1, axis=0)
    phi_past = phi
    L_past = compute_full_likelihood(
        phi_past, phi, psis, alpha_pasts, sigma, n_pasts, m)

    phi_next = minimize_phi(phi_past, psis, phi, sigma,
                            alpha_pasts, n_pasts, m, parallelized, num_process)
    alpha_nexts = np.zeros(alpha_pasts.shape)
    for i in range(alpha_nexts.shape[0]):
        alpha_nexts[i] = minimize_alpha_LS_helper(alpha_pasts[i].reshape(
            1, -1), sigma, phi_next, psis[:, i].reshape(-1, 1), n_pasts[i], eps, max_iter)
    n_nexts = np.zeros(n_pasts.shape)
    for i in range(n_nexts.shape[0]):
        n_nexts[i] = update_n(phi_next, alpha_nexts[i].reshape(
            1, -1), psis[:, i].reshape(-1, 1), sigma)
    L_next = compute_full_likelihood(
        phi_next, phi, psis, alpha_nexts, sigma, n_nexts, m)

    if L_next + delta > L_past:
        alpha_nexts = alpha_pasts

    it = 0
    while np.max(np.linalg.norm(alpha_nexts - alpha_pasts, ord=np.inf, axis=1)) > eps and it < max_iter and L_next + delta < L_past:
        phi_past, alpha_pasts, n_pasts, L_past = phi_next, alpha_nexts, n_nexts, L_next

        phi_next = minimize_phi(
            phi_past, psis, phi, sigma, alpha_pasts, n_nexts, m, parallelized, num_process)
        alpha_nexts = np.zeros(alpha_pasts.shape)
        for i in range(alpha_nexts.shape[0]):
            alpha_nexts[i] = minimize_alpha_LS_helper(alpha_pasts[i].reshape(
                1, -1), sigma, phi_next, psis[:, i].reshape(-1, 1), n_pasts[i], eps, max_iter)
        n_nexts = np.zeros(n_pasts.shape)
        for i in range(n_nexts.shape[0]):
            n_nexts[i] = update_n(phi_next, alpha_nexts[i].reshape(
                1, -1), psis[:, i].reshape(-1, 1), sigma)
        L_next = compute_full_likelihood(
            phi_next, phi, psis, alpha_nexts, sigma, n_nexts, m)

        it += 1

    return alpha_nexts, n_nexts, phi_next

# Updates (2)


def minimize_phi_row_grad(observed_phi_row, observed_psi_scalars, alphas, sigma_row, phi_past, ns, m, row_idx=None, parallelized=False, shared_array_id=None):
    bnds = tuple((0, None) for _ in range(phi_past.shape[1]))
    res = scopt.minimize(lambda phi_row: compute_row_likelihood(phi_row, observed_phi_row,
                                                                observed_psi_scalars, alphas, sigma_row, ns, m), phi_past, method='SLSQP', bounds=bnds)
    phi_row_clipped = np.clip(res.x, 0., None)
    phi_row_clipped[phi_row_clipped <= CLIP_VALUE] = 0.

    if parallelized and shared_array_id:
        phi_tmp = np.ctypeslib.as_array(shared_dict[shared_array_id])
        phi_tmp[row_idx, :] = phi_row_clipped

    return phi_row_clipped


def minimize_phi_grad(observed_phi, observed_psis, alphas, sigma, phi_past, ns, m, parallelized=False, num_process=1):
    if not parallelized:
        phi_next = np.empty((0, phi_past.shape[1]), dtype=np.float64)
        for i in range(phi_past.shape[0]):
            phi_next = np.vstack((phi_next, minimize_phi_row_grad(phi_past[i, :].reshape(
                1, -1), observed_psis[i].reshape(1, -1), alphas, sigma[i, :].reshape(1, -1), observed_phi[i, :].reshape(1, -1), ns, m)))
        return phi_next
    else:
        phi_next = np.ctypeslib.as_ctypes(np.zeros(phi_past.shape))
        phi_next_shared_array = sharedctypes.RawArray(
            phi_next._type_, phi_next)
        shared_array_id = "PHI_GRAD"
        shared_dict[shared_array_id] = phi_next_shared_array
        p = Pool(processes=num_process)
        p.starmap(minimize_phi_row_grad, [(phi_past[i, :].reshape(1, -1), observed_psis[i].reshape(1, -1), alphas, sigma[i, :].reshape(
            1, -1), observed_phi[i, :].reshape(1, -1), ns, m, i, True, shared_array_id) for i in range(phi_past.shape[0])])
        p.close()
        return np.ctypeslib.as_array(phi_next_shared_array)


def alternate_gradient_descent(phi, phi_init, sigma, m, psis, alpha_inits, n_inits, eps, delta, max_iter, parallelized=False, num_process=1):
    phi_past = phi_init
    alpha_pasts = alpha_inits
    n_pasts = n_inits
    L_past = compute_full_likelihood(
        phi_past, phi, psis, alpha_pasts, sigma, n_pasts, m)
    phi_next = minimize_phi_grad(
        phi, psis, alpha_pasts, sigma, phi_past, n_pasts, m, parallelized, num_process)
    alpha_nexts = np.zeros(alpha_pasts.shape)
    for i in range(alpha_pasts.shape[0]):
        alpha_nexts[i] = minimize_alpha_LS_helper(alpha_pasts[i].reshape(
            1, -1), sigma, phi_next, psis[:, i].reshape(-1, 1), n_pasts[i], eps, max_iter)
    n_nexts = np.zeros(n_pasts.shape)
    for i in range(n_nexts.shape[0]):
        n_nexts[i] = update_n(phi_next, alpha_nexts[i].reshape(
            1, -1), psis[:, i].reshape(-1, 1), sigma)
    L_next = compute_full_likelihood(
        phi_next, phi, psis, alpha_nexts, sigma, n_nexts, m)

    if L_next + delta > L_past:
        alpha_nexts = alpha_pasts

    it = 0
    while np.max(np.linalg.norm(alpha_nexts - alpha_pasts, ord=np.inf, axis=1)) > eps and it < max_iter:
        phi_past, alpha_pasts, n_pasts, L_past = phi_next, alpha_nexts, n_nexts, L_next

        phi_next = minimize_phi_grad(
            phi, psis, alpha_pasts, sigma, phi_past, n_pasts, m, parallelized, num_process)
        alpha_nexts = np.zeros(alpha_pasts.shape)
        for i in range(alpha_pasts.shape[0]):
            alpha_nexts[i] = minimize_alpha_LS_helper(alpha_pasts[i].reshape(
                1, -1), sigma, phi_next, psis[:, i].reshape(-1, 1), n_pasts[i], eps, max_iter)
        n_nexts = np.zeros(n_pasts.shape)
        for i in range(n_nexts.shape[0]):
            n_nexts[i] = update_n(phi_next, alpha_nexts[i].reshape(
                1, -1), psis[:, i].reshape(-1, 1), sigma)
        L_next = compute_full_likelihood(
            phi_next, phi, psis, alpha_nexts, sigma, n_nexts, m)

        it += 1

    return alpha_nexts, n_nexts, phi_next


def find_mixtures(phi, sigma, m, psis, eps=1e-1, delta=1e-1, max_iter=10, uniform_init=False, parallelized=True, num_process=10):
    sigma += 1
    alpha_nexts, n_nexts, phi_hat = alternate_minimization(
        phi, sigma, m, psis, eps, delta, max_iter, uniform_init, parallelized, num_process)
    alpha_nexts, n_nexts, phi_hat = alternate_gradient_descent(
        phi, phi_hat, sigma, m, psis, alpha_nexts, n_nexts, eps, delta, max_iter, parallelized, num_process)

    alpha_LS = np.zeros(alpha_nexts.shape)
    for i in range(alpha_nexts.shape[0]):
        mixture_sigma_diag = np.diag(
            1 / np.sqrt(compute_mixture_sigma(alpha_nexts[i].reshape(1, -1), sigma, phi_hat)).ravel())
        # note rcond parameter is set to silence a deprecation warning
        alpha_LS_i = np.linalg.lstsq(
            mixture_sigma_diag @ phi, mixture_sigma_diag @ psis[:, i].reshape(-1, 1), rcond=-1)[0].T
        alpha_LS_i = np.clip(alpha_LS_i, 0, None)
        alpha_LS[i] = alpha_LS_i / np.sum(alpha_LS_i)

    return alpha_LS
