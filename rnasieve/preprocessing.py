import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from collections import Counter
from functools import reduce
from rnasieve.model import RNASieveModel
from rnasieve.helper import CLIP_VALUE

# Filtering Functions


def _off_simplex_distances(means, variances, bulk):
    max_means = np.max(means, axis=1).reshape(-1, 1)
    min_means = np.min(means, axis=1).reshape(-1, 1)
    max_variances = np.max(variances, axis=1).reshape(-1, 1)
    below_min_distances = np.clip((min_means - bulk) / max_variances, 0, None)
    above_max_distances = np.clip((bulk - max_means) / max_variances, 0, None)
    return np.max(np.hstack(
        (below_min_distances, above_max_distances)), axis=1).reshape(-1, 1)


def off_simplex_filter(phi, sigma, psi, quantile):
    naive_LS = np.clip(np.linalg.lstsq(phi, psi, rcond=-1)[0], 0, None)
    n_init = np.linalg.norm(naive_LS, ord=1, axis=0)
    n_opt = scipy.optimize.minimize(lambda n: np.mean(_off_simplex_distances(
        phi, sigma, psi / n)), n_init, method='SLSQP', bounds=[(0, None)]).x
    dists = _off_simplex_distances(phi, sigma, psi / n_opt)
    filtered_idxs = np.where(dists <= np.quantile(dists, quantile))[0]
    return n_opt, filtered_idxs


def off_simplex_filter_absolute(phi, sigma, psi, threshold):
    naive_LS = np.clip(np.linalg.lstsq(phi, psi, rcond=-1)[0], 0, None)
    n_init = np.linalg.norm(naive_LS, ord=1, axis=0)
    n_opt = scipy.optimize.minimize(
        lambda n: np.mean(
            _off_simplex_distances(
                phi,
                sigma / n,
                psi / n)),
        n_init,
        method='SLSQP',
        bounds=[
            (0,
             None)]).x
    dists = _off_simplex_distances(phi, sigma / n_opt, psi / n_opt)[:, 0]
    filtered_idxs = np.where(dists <= threshold)
    return n_opt, filtered_idxs


def adjust_variances(phi, sigma, psi, threshold):
    naive_LS = np.clip(np.linalg.lstsq(phi, psi, rcond=-1)[0], 0, None)
    n_init = np.linalg.norm(naive_LS, ord=1, axis=0)
    n_opt = scipy.optimize.minimize(
        lambda n: np.mean(
            _off_simplex_distances(
                phi,
                sigma / n,
                psi / n)),
        n_init,
        method='SLSQP',
        bounds=[
            (0,
             None)]).x
    dists = _off_simplex_distances(phi, sigma / n_opt, psi / n_opt)
    dists = np.clip(dists, 1, None)
    return threshold * sigma * dists

# Empirical Protocol Filters
# Filter genes by their distance from the median in units of
# median deviations for various summary statistics.
# Individual thresholds are based on empirical trends in thirteen tissues
# of the Tabula Muris dataset.

# FACS/Droplet Filtering Helper Functions


def _fd_filter(phi, sigma, psi, md_plus=0, md_minus=np.inf):
    phi_max = np.max(phi, axis=1)
    sigma_max = np.max(sigma, axis=1)
    sigma_phi_ratios = np.max(sigma / np.clip(phi, CLIP_VALUE, None), axis=1)
    psi_phi_ratios = np.min(psi / np.clip(phi, CLIP_VALUE, None), axis=1)

    phi_max_median, phi_max_mad = np.median(
        phi_max), scipy.stats.median_absolute_deviation(phi_max, scale=1)
    sigma_max_median, sigma_max_mad = np.median(
        sigma_max), scipy.stats.median_absolute_deviation(sigma_max, scale=1)
    sp_ratios_median, sp_ratios_mad = np.median(
        sigma_phi_ratios), scipy.stats.median_absolute_deviation(sigma_phi_ratios, scale=1)
    pp_ratios_median, pp_ratios_mad = np.median(
        psi_phi_ratios), scipy.stats.median_absolute_deviation(psi_phi_ratios, scale=1)

    phi_max_non_zero_idxs = np.nonzero(phi_max)
    psi_non_zero_idxs = np.nonzero(psi[:, 0])
    phi_max_idxs = np.where(
        (phi_max_median -
         phi_max_mad *
         md_minus <= phi_max) & (
            phi_max <= phi_max_median +
            phi_max_mad *
            md_plus))
    sigma_max_idxs = np.where(
        (sigma_max_median -
         sigma_max_mad *
         md_minus <= sigma_max) & (
            sigma_max <= sigma_max_median +
            sigma_max_mad *
            md_plus))
    sp_ratios_idxs = np.where(
        (sp_ratios_median -
         sp_ratios_mad *
         md_minus <= sigma_phi_ratios) & (
            sigma_phi_ratios <= sp_ratios_median +
            sp_ratios_mad *
            md_plus))
    pp_ratios_idxs = np.where(
        psi_phi_ratios <= pp_ratios_median + pp_ratios_mad * md_plus)

    return reduce(
        np.intersect1d,
        (phi_max_non_zero_idxs,
         psi_non_zero_idxs,
         phi_max_idxs,
         sigma_max_idxs,
         sp_ratios_idxs,
         pp_ratios_idxs))


def _compute_fd_threshold(phi, sigma, psi):
    phi_max_non_zero_idxs = np.nonzero(np.max(phi, axis=1))
    psi_non_zero_idxs = np.nonzero(psi[:, 0])
    non_zero_idxs = np.intersect1d(phi_max_non_zero_idxs, psi_non_zero_idxs)
    psi_phi_ratios = np.min(
        psi[non_zero_idxs] /
        np.clip(
            phi[non_zero_idxs],
            CLIP_VALUE,
            None),
        axis=1)
    pp_ratios_median, pp_ratios_mad, pp_ratios_skew = np.median(
        psi_phi_ratios), scipy.stats.median_absolute_deviation(psi_phi_ratios, scale=1),
    scipy.stats.skew(psi_phi_ratios)

    if pp_ratios_mad / pp_ratios_median < 0.55 and pp_ratios_skew < 105:
        return 1
    return 5


def _compute_fd_max_iter(phi, sigma, psi):
    phi_max_non_zero_idxs = np.nonzero(np.max(phi, axis=1))
    psi_non_zero_idxs = np.nonzero(psi[:, 0])
    non_zero_idxs = np.intersect1d(phi_max_non_zero_idxs, psi_non_zero_idxs)
    psi_phi_ratios = np.min(
        psi[non_zero_idxs] /
        np.clip(
            phi[non_zero_idxs],
            CLIP_VALUE,
            None),
        axis=1)
    pp_ratios_median, pp_ratios_mad = np.median(
        psi_phi_ratios), scipy.stats.median_absolute_deviation(psi_phi_ratios, scale=1)

    if 0.52 < pp_ratios_mad / pp_ratios_median < 0.7 or .06 < np.quantile(
            psi_phi_ratios, 0.05) / pp_ratios_median < 0.18:
        return 1
    return 2


def filter_facs_to_droplet(phi, sigma, psi):
    """Given a FACS reference, filter genes suitably to deconvolve a droplet bulk."""
    md_plus = _compute_fd_threshold(phi, sigma, psi)
    max_iter = _compute_fd_max_iter(phi, sigma, psi)
    return _fd_filter(phi, sigma, psi, md_plus, 3), max_iter


def _df_filter(phi, sigma, psi, md_plus=0, md_minus=np.inf):
    phi_max = np.max(phi, axis=1)
    sigma_max = np.max(sigma, axis=1)
    sigma_phi_ratios = np.max(sigma / np.clip(phi, CLIP_VALUE, None), axis=1)
    psi_phi_ratios = np.min(psi / np.clip(phi, CLIP_VALUE, None), axis=1)

    phi_max_median, phi_max_mad = np.median(
        phi_max), scipy.stats.median_absolute_deviation(phi_max, scale=1)
    sigma_max_median, sigma_max_mad = np.median(
        sigma_max), scipy.stats.median_absolute_deviation(sigma_max, scale=1)
    sp_ratios_median, sp_ratios_mad = np.median(
        sigma_phi_ratios), scipy.stats.median_absolute_deviation(sigma_phi_ratios, scale=1)
    pp_ratios_median, pp_ratios_mad = np.median(
        psi_phi_ratios), scipy.stats.median_absolute_deviation(psi_phi_ratios, scale=1)

    phi_max_non_zero_idxs = np.nonzero(phi_max)
    psi_non_zero_idxs = np.nonzero(psi[:, 0])
    phi_max_idxs = np.where(
        (phi_max_median -
         phi_max_mad *
         md_minus <= phi_max) & (
            phi_max <= phi_max_median +
            phi_max_mad *
            md_plus))
    sigma_max_idxs = np.where(
        (sigma_max_median -
         sigma_max_mad *
         md_minus <= sigma_max) & (
            sigma_max <= sigma_max_median +
            sigma_max_mad *
            md_plus))
    sp_ratios_idxs = np.where(
        (sp_ratios_median -
         sp_ratios_mad *
         md_minus <= sigma_phi_ratios) & (
            sigma_phi_ratios <= sp_ratios_median +
            sp_ratios_mad *
            md_plus))
    pp_ratios_idxs = np.where(
        (pp_ratios_median -
         pp_ratios_mad *
         md_minus <= psi_phi_ratios) & (
            psi_phi_ratios <= pp_ratios_median +
            pp_ratios_mad *
            md_plus))

    return reduce(
        np.intersect1d,
        (phi_max_non_zero_idxs,
         psi_non_zero_idxs,
         phi_max_idxs,
         sigma_max_idxs,
         sp_ratios_idxs,
         pp_ratios_idxs))


def _compute_df_threshold(phi, sigma, psi):
    phi_max_non_zero_idxs = np.nonzero(np.max(phi, axis=1))
    psi_non_zero_idxs = np.nonzero(psi[:, 0])
    non_zero_idxs = np.intersect1d(phi_max_non_zero_idxs, psi_non_zero_idxs)
    sigma_phi_ratios = np.max(
        sigma[non_zero_idxs] /
        np.clip(
            phi[non_zero_idxs],
            CLIP_VALUE,
            None),
        axis=1)

    sp_ratios_skew = scipy.stats.skew(sigma_phi_ratios)

    if sp_ratios_skew <= 40:
        return 4
    elif sp_ratios_skew <= 60:
        return 1
    return 7


def _compute_df_max_iter(phi, sigma, psi):
    phi_max_non_zero_idxs = np.nonzero(np.max(phi, axis=1))
    psi_non_zero_idxs = np.nonzero(psi[:, 0])
    non_zero_idxs = np.intersect1d(phi_max_non_zero_idxs, psi_non_zero_idxs)
    psi_phi_ratios = np.min(
        psi[non_zero_idxs] /
        np.clip(
            phi[non_zero_idxs],
            CLIP_VALUE,
            None),
        axis=1)

    pp_ratios_left_tail = np.quantile(
        psi_phi_ratios, 0.05) / np.median(psi_phi_ratios)

    return 1 if pp_ratios_left_tail >= 0.1 else 3


def filter_droplet_to_facs(phi, sigma, psi):
    """Given a droplet reference, filter genes suitably to deconvolve a FACS bulk."""
    md_plus = _compute_df_threshold(phi, sigma, psi)
    max_iter = _compute_df_max_iter(phi, sigma, psi)
    return _df_filter(phi, sigma, psi, md_plus, np.inf), max_iter


def _trimmed_mean_mtx(M, frac):
    totals = M.sum(axis=0)
    sorted_idxs = np.argsort(totals)
    trim_idx = int(M.shape[1] * frac)
    if not trim_idx:
        return M
    else:
        return M[:, sorted_idxs[trim_idx:-trim_idx]]


def model_from_raw_counts(
        raw_counts,
        bulks,
        trim_percent=0.02,
        gene_thresh=0.2,
        normalization=True):
    """Given raw counts in the form of a dictionary { label : matrix } and a
    matrix of bulks, produce a RNASieveModel and corresponding psis.

    Matrices should be sorted by genes over the same set of genes.
    """
    g = bulks.shape[0]
    m = []
    labels = sorted(raw_counts.keys())

    phi = np.empty((g, 0), dtype=np.float32)
    sigma = np.empty((g, 0), dtype=np.float32)
    high_info_cts = Counter()

    for label in labels:
        data = _trimmed_mean_mtx(raw_counts[label], trim_percent)
        if normalization:
            data = data * 1e6 / np.sum(data, axis=0)
        high_info_cts.update(np.where(np.count_nonzero(
            data, axis=1) > gene_thresh * data.shape[1])[0])
        m.append(data.shape[1])
        phi = np.hstack((phi, np.mean(data, axis=1).reshape((-1, 1))))
        sigma = np.hstack((sigma, data.var(axis=1).reshape((-1, 1))))

    m = np.array(m).reshape(1, -1)

    # remove low info idxs
    high_info_idxs = sorted(
        [e for e in list(set(high_info_cts.elements())) if high_info_cts[e] > 0])
    phi = phi[high_info_idxs]
    sigma = sigma[high_info_idxs]
    psis = bulks[high_info_idxs]

    # remove all zero rows
    non_zero_idxs = np.where(phi.any(axis=1))
    phi = phi[non_zero_idxs]
    sigma = sigma[non_zero_idxs]
    psis = psis[non_zero_idxs]

    # convert to DataFrame
    phi_pd = pd.DataFrame(phi, columns=labels)
    sigma_pd = pd.DataFrame(sigma, columns=labels)
    m_pd = pd.DataFrame(m, columns=labels)
    psis_pd = pd.DataFrame(
        psis, columns=[
            f'Bulk {i}' for i in range(
                psis.shape[1])])

    return RNASieveModel(phi_pd, sigma_pd, m_pd), psis_pd
