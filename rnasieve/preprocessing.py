import numpy as np
from collections import Counter
from rnasieve.model import RNASieveModel


def trimmed_mean_mtx(M, frac):
    totals = M.sum(axis=0)
    sorted_idxs = np.argsort(totals)
    trim_idx = int(M.shape[1] * frac)
    return M[:, sorted_idxs[trim_idx:-trim_idx]]


# Takes in raw counts in the form of a dictionary { label : matrix } and a matrix of bulks
# All matrices should be sorted by genes over the same set of genes
def model_from_raw_counts(raw_counts, bulks,
                          trim_percent=0.02, gene_thresh=0.2, normalization=True):
    g = bulks.shape[0]
    m = []
    labels = sorted(raw_counts.keys())

    phi = np.empty((g, 0), dtype=np.float32)
    sigma = np.empty((g, 0), dtype=np.float32)
    high_info_cts = Counter()

    for label in labels:
        data = trimmed_mean_mtx(raw_counts[label], trim_percent)
        if normalization:
            data = data * 1e6 / np.sum(data, axis=0)
        high_info_cts.update(np.where(np.count_nonzero(
            data, axis=1) > gene_thresh * data.shape[1])[0])
        m.append(data.shape[1])
        phi = np.hstack((phi, np.mean(data, axis=1).reshape((-1, 1))))
        sigma = np.hstack((sigma, data.var(axis=1).reshape((-1, 1))))

    m = np.array(m).reshape(1, -1)

    # remove low info idxs
    high_info_idxs = sorted([e for e in list(set(high_info_cts.elements())) if high_info_cts[e] > 0])
    phi = phi[high_info_idxs]
    sigma = sigma[high_info_idxs]
    psis = bulks[high_info_idxs]

    # remove all zero rows
    non_zero_idxs = np.where(phi.any(axis=1))
    phi = phi[non_zero_idxs]
    sigma = sigma[non_zero_idxs]
    psis = psis[non_zero_idxs]

    return RNASieveModel(phi, sigma, m, labels), psis
