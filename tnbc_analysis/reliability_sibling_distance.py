from __future__ import annotations
import numpy as np


def trend_weights(k):
    k = np.asarray(k)
    (groups, inverse, counts) = np.unique(k, return_inverse=True, return_counts=True)
    if len(groups) < 2:
        return None
    centered = groups.astype(float) - groups.mean()
    return (centered / np.sum(centered**2) / counts)[inverse]


def permuted_slopes(distances, a, b, weights, permutations):
    (aa, bb) = (permutations[:, a], permutations[:, b])
    return np.column_stack([matrix[aa, bb] @ weights for matrix in distances])
