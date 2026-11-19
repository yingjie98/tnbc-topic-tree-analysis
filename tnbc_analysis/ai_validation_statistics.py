from __future__ import annotations
import numpy as np


def holm(values):
    p = np.asarray(values, float)
    out = np.full(len(p), np.nan)
    ix = np.flatnonzero(np.isfinite(p))
    order = ix[np.argsort(p[ix])]
    if len(order):
        out[order] = np.minimum(
            1, np.maximum.accumulate(p[order] * (len(order) - np.arange(len(order))))
        )
    return out
