"""
The core algorithm of RNA-Sieve, which is defined as `find_mixtures(...)`.
"""

import numpy as np
import cvxpy as cp
import scipy.optimize
import math
from multiprocessing import Pool
from multiprocessing import sharedctypes
from rnasieve.helper import CLIP_VALUE, ALPHA_EPS, ALPHA_MAX_ITER, compute_mixture_sigma, compute_weighted_norm, compute_full_likelihood, compute_row_likelihood

# Parallelization Global shared arrays

shared_dict = {}
_MINIMIZE_PHI_KEY = 'MINIMIZE_PHI'
_PHI_GRAD_KEY = 'PHI_GRAD'
_PHI_SHAPE_KEY = 'PHI_SHAPE'


def initialize_shared_arr(shared_array, shared_array_id, phi_shape):
    shared_dict[shared_array_id] = shared_array
    shared_dict[_PHI_SHAPE_KEY] = phi_shape

# Updates (1)


def _update_n(phi, alpha, psi, sigma):
    mixture_variance = np.clip(compute_mixture_sigma(
        alpha, sigma, phi), CLIP_VALUE, None)
    G = psi.shape[0]
    psi_hat_term = compute_weighted_norm(phi @ alpha.T, 1 / mixture_variance)
    observed_psi_term = compute_weighted_norm(psi, 1 / mixture_variance)
    return np.max(np.roots([psi_hat_term, G, -observed_psi_term]))


def _minimize_phi_row(
        phi_prev,
        psi_scalars,
        observed_phi_row,
        sigma_row,
        alphas,
        ns,
        m,
        row_idx=None,
        parallelized=False,
        shared_array_id=None):
    phi_next = cp.Variable(phi_prev.shape)
    phi_final = np.copy(phi_prev)
    try:
        phi_prev = np.clip(phi_prev, CLIP_VALUE, None)
        sample_coef = m / np.clip(sigma_row, CLIP_VALUE, None)
        bulk_coef = ns / \
            np.clip(compute_mixture_sigma(
                alphas, sigma_row, phi_prev), CLIP_VALUE, None)
        prob = cp.Problem(
            cp.Minimize((observed_phi_row - phi_next)**2 @ sample_coef.T +
                        (phi_next @ alphas.T - psi_scalars / ns)**2 @ bulk_coef.T),
            [phi_next >= 0])
        prob.solve()
        phi_final = np.clip(phi_next.value, 0, None)
    except (cp.SolverError, TypeError) as e:
        pass
    if phi_next.value is None:
        phi_final = np.clip(phi_prev, 0, None)

    if parallelized and shared_array_id:
        phi_shape = shared_dict[_PHI_SHAPE_KEY]
        phi_tmp = np.frombuffer(
            shared_dict[shared_array_id]).reshape(phi_shape)
        phi_tmp[row_idx, :] = phi_final

    return phi_final


def _minimize_phi(
        phi_prev,
        psis,
        observed_phi,
        sigma,
        alphas,
        ns,
        m,
        parallelized=False,
        num_process=1):
    if not parallelized:
        phi_next = np.empty((0, phi_prev.shape[1]), dtype=np.float64)
        for i in range(phi_prev.shape[0]):
            phi_next = np.vstack((phi_next, _minimize_phi_row(
                phi_prev[i, :].reshape(1, -1),
                psis[i].reshape(1, -1),
                observed_phi[i, :].reshape(1, -1),
                sigma[i, :].reshape(1, -1),
                alphas, ns, m)))
        return phi_next
    else:
        phi_next_shared_array = sharedctypes.RawArray(
            'd', phi_prev.shape[0] * phi_prev.shape[1])
        shared_array_id = _MINIMIZE_PHI_KEY
        p = Pool(
            processes=num_process,
            initializer=initialize_shared_arr,
            initargs=(
                phi_next_shared_array,
                shared_array_id,
                phi_prev.shape))
        p.starmap(_minimize_phi_row, [(
            phi_prev[i, :].reshape(1, -1),
            psis[i].reshape(1, -1),
            observed_phi[i, :].reshape(1, -1),
            sigma[i, :].reshape(1, -1),
            alphas, ns, m, i, True, shared_array_id) for i in range(phi_prev.shape[0])])
        p.close()
        return np.frombuffer(phi_next_shared_array).reshape(phi_prev.shape)


def _minimize_alpha(alpha_prev, sigma, phi, psi, n):
    alpha_next = cp.Variable(alpha_prev.shape)
    alpha = np.copy(alpha_prev)
    try:
        coef = n / np.clip(compute_mixture_sigma(alpha_prev,
                                                 sigma, phi), CLIP_VALUE, None)
        prob = cp.Problem(cp.Minimize(
            coef.T @ (phi @ alpha_next.T - psi / n)**2,
            [alpha_next >= 0, cp.sum(alpha_next) == 1]))
        prob.solve()
        alpha = np.clip(alpha_next.value, 0, None)
    except (cp.SolverError, TypeError) as e:
        pass
    if alpha_next.value is None:
        return alpha_prev
    return alpha / np.sum(alpha)


def _minimize_alpha_LS_helper(alpha_prev, sigma, phi, psi, n, eps, max_iter):
    alpha_last = alpha_prev
    alpha_cur = _minimize_alpha(alpha_prev, sigma, phi, psi, n)
    it = 0
    while np.linalg.norm(alpha_cur - alpha_last,
                         ord=2) > eps and it < max_iter:
        alpha_cur, alpha_last = _minimize_alpha(
            alpha_cur, sigma, phi, psi, n), alpha_cur
        it += 1
    return alpha_cur


def _alternate_minimization(
        phi,
        sigma,
        m,
        psis,
        alpha_inits,
        n_inits,
        eps,
        delta,
        max_iter,
        parallelized=False,
        num_process=1):
    L_past = compute_full_likelihood(
        phi, phi, psis, alpha_inits, sigma, n_inits, m)
    phi_next = _minimize_phi(
        phi,
        psis,
        phi,
        sigma,
        alpha_inits,
        n_inits,
        m,
        parallelized,
        num_process)
    alpha_nexts = np.zeros(alpha_inits.shape)
    for i in range(alpha_nexts.shape[0]):
        alpha_nexts[i] = _minimize_alpha_LS_helper(alpha_inits[i].reshape(
            1, -1), sigma, phi_next, psis[:, i].reshape(-1, 1), n_inits[i],
            ALPHA_EPS, ALPHA_MAX_ITER)
    n_nexts = np.zeros(n_inits.shape)
    for i in range(n_nexts.shape[0]):
        n_nexts[i] = _update_n(phi_next, alpha_nexts[i].reshape(
            1, -1), psis[:, i].reshape(-1, 1), sigma)
    L_next = compute_full_likelihood(
        phi_next, phi, psis, alpha_nexts, sigma, n_nexts, m)

    if L_next + delta > L_past:
        alpha_nexts = alpha_inits

    phi_past = phi
    alpha_pasts = alpha_inits.copy()
    n_pasts = n_inits.copy()

    it = 0
    while (
            np.max(
                np.linalg.norm(
                    alpha_nexts - alpha_pasts,
                    ord=np.inf,
                    axis=1)) > eps and it < max_iter and L_next +
            delta < L_past):
        phi_past, alpha_pasts, n_pasts, L_past = phi_next, alpha_nexts, n_nexts, L_next

        phi_next = _minimize_phi(phi_past, psis, phi, sigma, alpha_pasts,
                                 n_nexts, m, parallelized, num_process)
        alpha_nexts = np.zeros(alpha_pasts.shape)
        for i in range(alpha_nexts.shape[0]):
            alpha_nexts[i] = _minimize_alpha_LS_helper(alpha_pasts[i].reshape(1, -1),
                                                       sigma, phi_next,
                                                       psis[:, i].reshape(-1, 1),
                                                       n_pasts[i],
                                                       ALPHA_EPS, ALPHA_MAX_ITER)
        n_nexts = np.zeros(n_pasts.shape)
        for i in range(n_nexts.shape[0]):
            n_nexts[i] = _update_n(phi_next, alpha_nexts[i].reshape(
                1, -1), psis[:, i].reshape(-1, 1), sigma)
        L_next = compute_full_likelihood(
            phi_next, phi, psis, alpha_nexts, sigma, n_nexts, m)

        it += 1

    return alpha_nexts, n_nexts, phi_next

# Updates (2)


def _minimize_phi_row_grad(
        phi_past,
        observed_phi_row,
        alphas,
        ns,
        observed_psi_scalars,
        sigma_row,
        m,
        row_idx=None,
        parallelized=False,
        shared_array_id=None):
    bnds = tuple((0, None) for _ in range(phi_past.shape[1]))
    res = scipy.optimize.minimize(
        lambda phi_row: compute_row_likelihood(
            phi_row.reshape(1, -1),
            observed_phi_row,
            observed_psi_scalars,
            alphas,
            sigma_row,
            ns,
            m),
        phi_past,
        method='SLSQP',
        bounds=bnds)
    phi_row_clipped = np.clip(res.x.reshape(1, -1), 0., None)
    phi_row_clipped[phi_row_clipped <= CLIP_VALUE] = 0.

    if parallelized and shared_array_id:
        phi_shape = shared_dict[_PHI_SHAPE_KEY]
        phi_tmp = np.frombuffer(
            shared_dict[shared_array_id]).reshape(phi_shape)
        phi_tmp[row_idx, :] = phi_row_clipped

    return phi_row_clipped


def _minimize_phi_grad(
        phi_past,
        observed_phi,
        alphas,
        ns,
        observed_psis,
        sigma,
        m,
        parallelized=False,
        num_process=1):
    if not parallelized:
        phi_next = np.empty((0, phi_past.shape[1]), dtype=np.float64)
        for i in range(phi_past.shape[0]):
            phi_next = np.vstack((phi_next, _minimize_phi_row_grad(
                phi_past[i, :].reshape(1, -1),
                observed_phi[i, :].reshape(1, -1),
                alphas,
                ns,
                observed_psis[i].reshape(1, -1),
                sigma[i, :].reshape(1, -1),
                m)))
        return phi_next
    else:
        phi_next_shared_array = sharedctypes.RawArray(
            'd', phi_past.shape[0] * phi_past.shape[1])
        shared_array_id = _PHI_GRAD_KEY
        p = Pool(
            processes=num_process,
            initializer=initialize_shared_arr,
            initargs=(
                phi_next_shared_array,
                shared_array_id,
                phi_past.shape))
        p.starmap(_minimize_phi_row_grad, [
            (phi_past[i, :].reshape(1, -1),
             observed_phi[i, :].reshape(1, -1),
             alphas,
             ns,
             observed_psis[i].reshape(1, -1),
             sigma[i, :].reshape(1, -1),
             m,
             i,
             True,
             shared_array_id) for i in range(phi_past.shape[0])])
        p.close()
        return np.frombuffer(phi_next_shared_array).reshape(phi_past.shape)


def _alternate_gradient_descent(
        phi,
        phi_init,
        sigma,
        m,
        psis,
        alpha_inits,
        n_inits,
        eps,
        delta,
        max_iter,
        parallelized=False,
        num_process=1):
    phi_past = phi_init
    alpha_pasts = alpha_inits
    n_pasts = n_inits
    L_past = compute_full_likelihood(
        phi_past, phi, psis, alpha_pasts, sigma, n_pasts, m)
    phi_next = _minimize_phi_grad(
        phi_past,
        phi,
        alpha_pasts,
        n_pasts,
        psis,
        sigma,
        m,
        parallelized=parallelized,
        num_process=num_process)
    alpha_nexts = np.zeros(alpha_pasts.shape)
    for i in range(alpha_pasts.shape[0]):
        alpha_nexts[i] = _minimize_alpha_LS_helper(
            alpha_pasts[i].reshape(1, -1),
            sigma,
            phi_next,
            psis[:, i].reshape(-1, 1),
            n_pasts[i],
            ALPHA_EPS,
            ALPHA_MAX_ITER)
    n_nexts = np.zeros(n_pasts.shape)
    for i in range(n_nexts.shape[0]):
        n_nexts[i] = _update_n(phi_next, alpha_nexts[i].reshape(
            1, -1), psis[:, i].reshape(-1, 1), sigma)
    L_next = compute_full_likelihood(
        phi_next, phi, psis, alpha_nexts, sigma, n_nexts, m)

    if L_next + delta > L_past:
        alpha_nexts = alpha_pasts

    it = 0
    while np.max(np.linalg.norm(alpha_nexts - alpha_pasts,
                                ord=np.inf, axis=1)) > eps and it < max_iter:
        phi_past, alpha_pasts, n_pasts, L_past = phi_next, alpha_nexts, n_nexts, L_next

        phi_next = _minimize_phi_grad(
            phi_past,
            phi,
            alpha_pasts,
            n_pasts,
            psis,
            sigma,
            m,
            parallelized,
            num_process)
        alpha_nexts = np.zeros(alpha_pasts.shape)
        for i in range(alpha_pasts.shape[0]):
            alpha_nexts[i] = _minimize_alpha_LS_helper(
                alpha_pasts[i].reshape(1, -1),
                sigma,
                phi_next,
                psis[:, i].reshape(-1, 1),
                n_pasts[i],
                ALPHA_EPS,
                ALPHA_MAX_ITER)
        n_nexts = np.zeros(n_pasts.shape)
        for i in range(n_nexts.shape[0]):
            n_nexts[i] = _update_n(phi_next, alpha_nexts[i].reshape(
                1, -1), psis[:, i].reshape(-1, 1), sigma)
        L_next = compute_full_likelihood(
            phi_next, phi, psis, alpha_nexts, sigma, n_nexts, m)

        it += 1

    return alpha_nexts, n_nexts, phi_next


def _compute_alpha_LS(alpha_hats, phi_hat, phi, sigma, psis):
    alpha_LS = np.zeros(alpha_hats.shape)
    for i in range(alpha_hats.shape[0]):
        mixture_sigma_diag = np.diag(
            1 / np.sqrt(compute_mixture_sigma(alpha_hats[i].reshape(1, -1),
                                              sigma, phi_hat)).ravel())
        # Note: rcond parameter is set to silence a deprecation warning
        alpha_LS_i = np.linalg.lstsq(
            mixture_sigma_diag @ phi, mixture_sigma_diag @ psis[:, i].reshape(-1, 1),
            rcond=-1)[0].T
        alpha_LS_i = np.clip(alpha_LS_i, 0, None)
        alpha_LS[i] = alpha_LS_i / np.sum(alpha_LS_i)
    return alpha_LS


def find_mixtures(phi, sigma, m, psis, eps=1e-1, delta=1e-1, max_iter=10,
                  uniform_init=False, parallelized=True, num_process=10):
    """Infers alpha (mixture proportion), n (bulk cell count), and phi (mean matrix)
    given observed phi (mean matrix), sigma (variance matrix), m from a reference set,
    and psis, the bulks being deconvolved."""
    # Increment sigma by one to reduce numerical instability and zero inflation
    sigma += 1
    if uniform_init:
        alpha_inits = np.ones(phi.shape[1], 1) / phi.shape[1]
        n_inits = np.sum(phi @ alpha_inits.T, axis=0) / np.sum(psis, axis=0)
    else:
        # Note: rcond parameter is set to silence a deprecation warning
        naive_LS = np.clip(np.linalg.lstsq(phi, psis, rcond=-1)[0], 0, None)
        alpha_inits = naive_LS.T / np.sum(naive_LS, axis=0).reshape(-1, 1)
        n_inits = np.linalg.norm(naive_LS, ord=1, axis=0)
    L_init = compute_full_likelihood(
        phi, phi, psis, alpha_inits, sigma, n_inits, m)

    alpha_opts = alpha_inits.copy()
    n_opts = n_inits.copy()
    phi_opt = phi.copy()
    L_opt = L_init

    alpha_proj_opts = alpha_inits.copy()
    n_proj_opts = n_inits.copy()
    phi_proj_opt = phi.copy()
    L_proj_opt = L_init

    alpha_nexts, n_nexts, phi_hat = _alternate_minimization(
        phi, sigma, m, psis, alpha_inits, n_inits,
        eps, delta, max_iter, parallelized=parallelized, num_process=num_process)
    alpha_nexts, n_nexts, phi_hat = _alternate_gradient_descent(
        phi, phi_hat, sigma, m, psis, alpha_nexts, n_nexts,
        eps, delta, max_iter, parallelized, num_process)
    L_pre = compute_full_likelihood(
        phi_hat, phi, psis, alpha_nexts, sigma, n_nexts, m)

    alpha_LS = _compute_alpha_LS(alpha_nexts, phi_hat, phi, sigma, psis)
    L_LS = compute_full_likelihood(
        phi_hat, phi, psis, alpha_LS, sigma, n_nexts, m)

    it = 0
    while L_LS < max(L_proj_opt, L_pre) and it < max_iter:
        if L_LS < L_proj_opt:
            alpha_proj_opts = alpha_LS.copy()
            n_proj_opts = n_nexts.copy()
            phi_proj_opt = phi_hat.copy()
            L_proj_opt = L_LS

        if L_pre < L_opt:
            alpha_opts = alpha_nexts.copy()
            n_opts = n_nexts.copy()
            phi_opt = phi_hat.copy()
            L_opt = L_pre

        alpha_nexts, n_nexts, phi_hat = _alternate_gradient_descent(
            phi, phi_hat, sigma, m, psis, alpha_LS, n_nexts,
            eps, delta, max_iter, parallelized, num_process)
        L_pre = compute_full_likelihood(
            phi_hat, phi, psis, alpha_nexts, sigma, n_nexts, m)

        alpha_LS = _compute_alpha_LS(alpha_nexts, phi_hat, phi, sigma, psis)
        L_LS = compute_full_likelihood(
            phi_hat, phi, psis, alpha_LS, sigma, n_nexts, m)

        it += 1

    if L_proj_opt < L_opt:
        alpha_opts = alpha_proj_opts.copy()
        n_opts = n_proj_opts.copy()
        phi_opt = phi_proj_opt.copy()
        L_opt = L_proj_opt

    return (alpha_proj_opts, n_proj_opts,
            phi_proj_opt), (alpha_opts, n_opts, phi_opt)
