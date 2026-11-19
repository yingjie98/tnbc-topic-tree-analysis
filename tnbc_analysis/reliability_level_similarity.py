from __future__ import annotations
import numpy as np
import pandas as pd
from .reliability_vertical_refinement import BOOTSTRAPS

MIN_LEVEL_NODES = 5
GROUP_CI_REPS = 1999
GROUP_CI_SEED = 20260906
ADJACENT_SEED = 20261030


def original_mean_intervals(frame, label, reps=GROUP_CI_REPS, seed=GROUP_CI_SEED):
    rng = np.random.default_rng(seed)
    result = []
    for level, group in frame.groupby("level", sort=True):
        values = group.W.to_numpy(float)
        (_, inverse) = np.unique(
            group.parent_id.fillna("ROOT").to_numpy(), return_inverse=True
        )
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=values)
        cluster_low = cluster_high = node_low = node_high = np.nan
        primary = reference = None
        if len(counts) >= 2:
            indices = rng.integers(0, len(counts), size=(reps, len(counts)))
            primary = sums[indices].sum(1) / counts[indices].sum(1)
            (cluster_low, cluster_high) = np.quantile(primary, [0.025, 0.975])
            rng.integers(0, len(counts), size=(reps, len(counts)))
        if len(values) > 1:
            reference = values[
                rng.integers(0, len(values), size=(reps, len(values)))
            ].mean(1)
            (node_low, node_high) = np.quantile(reference, [0.025, 0.975])
        if primary is not None:
            (low, high, distribution) = (cluster_low, cluster_high, primary)
            method = (
                f"{reps} parent-group bootstrap percentile interval; original method"
            )
        elif reference is not None:
            (low, high, distribution) = (node_low, node_high, reference)
            method = f"{reps} node-bootstrap reference interval; only one parent group"
        else:
            low = high = np.nan
            distribution = None
            method = "not estimable: only one node/parent group"
        result.append(
            {
                "embedding": label,
                "kind": "mean_W",
                "level": int(level),
                "n_nodes": len(group),
                "parent_groups": int(group.parent_id.nunique()),
                "mean_W": values.mean(),
                "estimate": values.mean(),
                "CI_low": low,
                "CI_high": high,
                "cluster_CI_low": cluster_low,
                "cluster_CI_high": cluster_high,
                "node_CI_low": node_low,
                "node_CI_high": node_high,
                "bootstrap_SE": np.std(distribution, ddof=1)
                if distribution is not None
                else np.nan,
                "interval_method": method,
            }
        )
    return pd.DataFrame(result)


def joint_parent_draws(lower, upper, rng, reps=BOOTSTRAPS):
    parents = lower.parent_id.fillna("ROOT").to_numpy()
    (_, inverse) = np.unique(parents, return_inverse=True)
    mapping = dict(zip(lower.node_id, inverse))
    if not upper.parent_id.isin(mapping).all():
        raise ValueError("Next-Level nodes have a parent absent from the lower Level")
    child_groups = np.array([mapping[key] for key in upper.parent_id], dtype=int)
    groups = int(inverse.max()) + 1
    if groups < 2:
        return (None, groups)
    na = np.bincount(inverse, minlength=groups)
    nb = np.bincount(child_groups, minlength=groups)
    sa = np.bincount(inverse, weights=lower.W, minlength=groups)
    sb = np.bincount(child_groups, weights=upper.W, minlength=groups)
    result = np.full(reps, np.nan)
    for start in range(0, reps, 256):
        size = min(256, reps - start)
        indices = rng.integers(0, groups, size=(size, groups))
        (da, db) = (na[indices].sum(1), nb[indices].sum(1))
        valid = (da > 0) & (db > 0)
        block = np.full(size, np.nan)
        block[valid] = (
            sb[indices].sum(1)[valid] / db[valid]
            - sa[indices].sum(1)[valid] / da[valid]
        )
        result[start : start + size] = block
    return (result[np.isfinite(result)], groups)


def adjacent_statistics(lower, upper, label, reps=BOOTSTRAPS):
    (a, b) = (int(lower.level.iloc[0]), int(upper.level.iloc[0]))
    if b != a + 1:
        raise ValueError("Only adjacent Level contrasts are permitted")
    observed = float(upper.W.mean() - lower.W.mean())
    row = {
        "embedding": label,
        "kind": "adjacent_W_difference",
        "level_A": a,
        "level_B": b,
        "n_A": len(lower),
        "n_B": len(upper),
        "difference_W_B_minus_A": observed,
        "estimate": observed,
        "joint_parent_groups": int(lower.parent_id.fillna("ROOT").nunique()),
        "parent_group_level": a - 1,
        "CI_low": np.nan,
        "CI_high": np.nan,
        "bootstrap_SE": np.nan,
        "p_raw": np.nan,
        "valid_replicates": 0,
        "interval_method": "joint parent-group percentile bootstrap",
        "test_method": "two-sided centred parent-group bootstrap; approximate conditional inference",
    }
    if min(len(lower), len(upper)) < MIN_LEVEL_NODES:
        row["test_status"] = (
            "fewer than 5 eligible nodes in one Level; descriptive only"
        )
        return row
    if row["joint_parent_groups"] < 2:
        row["test_status"] = (
            "only one lower-Level parent group; grouped difference not estimable"
        )
        return row
    (draws, _) = joint_parent_draws(
        lower, upper, np.random.default_rng(ADJACENT_SEED + a), reps
    )
    row["valid_replicates"] = len(draws)
    if len(draws) < reps * 0.95:
        row["test_status"] = "too many empty resampled Levels; comparison unavailable"
        return row
    (row["CI_low"], row["CI_high"]) = np.quantile(draws, [0.025, 0.975])
    row["bootstrap_SE"] = draws.std(ddof=1)
    if np.ptp(draws) < 1e-12 and abs(observed) > 1e-12:
        row["test_status"] = "no estimable parent-group variation"
        return row
    deviations = draws - observed
    row["p_raw"] = (
        1 + np.count_nonzero(np.abs(deviations) >= abs(observed) - 1e-12)
    ) / (len(draws) + 1)
    row["test_status"] = "tested with joint parent-group bootstrap"
    return row
