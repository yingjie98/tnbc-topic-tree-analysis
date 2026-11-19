from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

BOOTSTRAPS = 49999
SEED = 20261030
MIN_TEST_PAPERS = 30


def holm(values):
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        result[order] = np.minimum(
            1, np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1))
        )
    return result


def significance(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paper_influence(incidence, vectors, chunk=4096):
    matrix = incidence.tocsr()
    counts = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    if np.any(counts <= 0):
        raise ValueError("Node has no papers")
    means = matrix @ vectors / counts[:, None]
    lengths = np.linalg.norm(means, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("Node centre is too close to zero to linearise")
    directions = means / lengths[:, None]
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    values = np.empty(matrix.nnz, dtype=float)
    for start in range(0, matrix.nnz, chunk):
        stop = min(start + chunk, matrix.nnz)
        rr = rows[start:stop]
        products = np.einsum(
            "ij,ij->i",
            directions[rr],
            vectors[matrix.indices[start:stop]],
            optimize=True,
        )
        values[start:stop] = (lengths[rr] - products) / counts[rr]
    influence = sparse.csr_matrix(
        (values, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape
    )
    return (1 - lengths, influence)


def statistic_weights(data):
    levels = np.array([data.tree[k]["level"] for k in data.eligible], dtype=int)
    (records, rows) = ([], [])
    for level in sorted(set(levels)):
        ix = np.flatnonzero(levels == level)
        rows.append(
            sparse.csr_matrix(
                (np.full(len(ix), 1 / len(ix)), (np.zeros(len(ix), int), ix)),
                shape=(1, len(levels)),
            )
        )
        records.append({"kind": "level_mean", "level": int(level), "n_items": len(ix)})
    for level in sorted(set(levels) - {int(levels.min())}):
        children = [
            i
            for (i, key) in enumerate(data.eligible)
            if levels[i] == level and data.parent.get(key) in data.ei
        ]
        if not children:
            continue
        parents = np.array([data.ei[data.parent[data.eligible[i]]] for i in children])
        if not np.all(levels[parents] == level - 1):
            raise ValueError("A parent-child edge skips a Level")
        n = len(children)
        row = np.zeros(len(levels))
        np.add.at(row, parents, 1 / n)
        np.add.at(row, children, -1 / n)
        rows.append(sparse.csr_matrix(row[None, :]))
        records.append(
            {
                "kind": "paired_difference",
                "level": int(level),
                "n_items": n,
                "source_parents": len(set(parents)),
            }
        )
    return (pd.DataFrame(records), sparse.vstack(rows, format="csr"))


def multiplier_draws(influence, reps=BOOTSTRAPS, seed=SEED):
    rng = np.random.default_rng(seed)
    matrix = np.ascontiguousarray(np.asarray(influence, float).T)
    matrix -= matrix.mean(axis=0, keepdims=True)
    matrix *= np.sqrt(len(matrix) / (len(matrix) - 1))
    draws = np.empty((reps, matrix.shape[1]))
    with threadpool_limits(limits=8):
        for start in range(0, reps, 256):
            size = min(256, reps - start)
            signs = rng.integers(0, 2, size=(size, len(matrix)), dtype=np.int8).astype(
                float
            )
            signs *= 2
            signs -= 1
            draws[start : start + size] = signs @ matrix
            if start and start % 10240 == 0:
                print(
                    f"S2 shared-paper multiplier bootstrap: {start}/{reps}", flush=True
                )
    return draws


def summarize(records, estimates, draws):
    result = records.copy()
    result["mean_w"] = estimates
    quantiles = np.quantile(draws, [0.025, 0.975], axis=0)
    result["CI_low"] = estimates - quantiles[1]
    result["CI_high"] = estimates - quantiles[0]
    result["bootstrap_SE"] = draws.std(axis=0, ddof=1)
    result["p_raw"] = np.nan
    result["test_status"] = "descriptive Level mean"
    for i, row in result.iterrows():
        if row.kind != "paired_difference":
            continue
        if row.support_papers < MIN_TEST_PAPERS:
            result.loc[i, "test_status"] = (
                "fewer than 30 supporting PMIDs; interval descriptive only"
            )
            continue
        if np.ptp(draws[:, i]) < 1e-12 and abs(estimates[i]) > 1e-12:
            result.loc[i, "test_status"] = "no estimable multiplier variation"
            continue
        p = (1 + np.count_nonzero(np.abs(draws[:, i]) >= abs(estimates[i]) - 1e-12)) / (
            len(draws) + 1
        )
        result.loc[i, ["p_raw", "test_status"]] = [
            p,
            "two-sided zero mean paired-difference test; first-order paper wild bootstrap",
        ]
    result["p_holm"] = holm(result.p_raw)
    result["significance"] = result.p_holm.map(significance)
    result["interval_method"] = (
        "pointwise 95% basic first-order shared-PMID Rademacher bootstrap; fixed tree and embeddings"
    )
    result["multiplicity_family"] = (
        "all testable adjacent-Level contrasts across both embeddings"
    )
    return result
