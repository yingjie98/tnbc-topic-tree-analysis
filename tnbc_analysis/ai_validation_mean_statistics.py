from __future__ import annotations
import numpy as np

GROUPS = ("original", "random_mismatch")


def blocks_for(frame, task):
    blocks = []
    for _, g in frame[frame.condition.isin(GROUPS) & frame.score.notna()].groupby(
        "stratum"
    ):
        a = g[g.condition == "original"]
        b = g[g.condition == "random_mismatch"]
        if len(a) and len(b):
            blocks.append(
                (a, b, float(g.population_n.iloc[0]) if task == "paper" else 1.0)
            )
    return blocks


def compare_means(frame, task, B=19999, reps=4999, seed=20261031):
    rng = np.random.default_rng(seed)
    blocks = blocks_for(frame, task)
    if not blocks:
        return {
            "mean_difference": np.nan,
            "p_raw": np.nan,
            "n_original": 0,
            "n_random": 0,
            "strata": 0,
        }
    weights = np.array([w for (_, _, w) in blocks])
    weights /= weights.sum()
    observed = np.zeros(2)
    null = np.zeros(B)
    boot = np.zeros((reps, 2))
    parents = (
        sorted(
            {p for (a, b, _) in blocks for p in list(a.parent_id) + list(b.parent_id)}
        )
        if task == "edge"
        else []
    )
    parent_index = {p: i for (i, p) in enumerate(parents)}
    z = rng.exponential(size=(reps, len(parents))) if task == "edge" else None
    for w, (a, b, _) in zip(weights, blocks):
        x = a.score.to_numpy(float)
        y = b.score.to_numpy(float)
        n0 = len(x)
        n1 = len(y)
        observed += w * np.array([x.mean(), y.mean()])
        values = np.r_[x, y]
        total = values.sum()
        for start in range(0, B, 256):
            count = min(256, B - start)
            selected = np.argpartition(rng.random((count, n0 + n1)), n0 - 1, axis=1)[
                :, :n0
            ]
            sums = values[selected].sum(axis=1)
            null[start : start + count] += w * (sums / n0 - (total - sums) / n1)
        if task == "edge":
            for j, g in enumerate((a, b)):
                mult = z[:, [parent_index[p] for p in g.parent_id]]
                boot[:, j] += w * (mult @ g.score.to_numpy(float)) / mult.sum(axis=1)
        else:
            for start in range(0, reps, 256):
                count = min(256, reps - start)
                boot[start : start + count, 0] += w * x[
                    rng.integers(0, n0, (count, n0))
                ].mean(axis=1)
                boot[start : start + count, 1] += w * y[
                    rng.integers(0, n1, (count, n1))
                ].mean(axis=1)
    diff = observed[0] - observed[1]
    p = (1 + np.count_nonzero(np.abs(null) >= abs(diff) - 1e-12)) / (B + 1)
    intervals = np.quantile(boot, [0.025, 0.975], axis=0)
    (dlo, dhi) = np.quantile(boot[:, 0] - boot[:, 1], [0.025, 0.975])
    return dict(
        mean_original=observed[0],
        mean_random=observed[1],
        mean_difference=diff,
        original_CI_low=intervals[0, 0],
        original_CI_high=intervals[1, 0],
        random_CI_low=intervals[0, 1],
        random_CI_high=intervals[1, 1],
        difference_CI_low=dlo,
        difference_CI_high=dhi,
        p_raw=p,
        n_original=sum((len(a) for (a, b, w) in blocks)),
        n_random=sum((len(b) for (a, b, w) in blocks)),
        strata=len(blocks),
        included_strata=";".join((str(a.stratum.iloc[0]) for (a, b, w) in blocks)),
        ci_method="source-parent multiplier (conditional)"
        if task == "edge"
        else "stratified item bootstrap",
        weighting="equal common levels"
        if task == "edge"
        else "frozen paper-population strata",
        permutations=B,
        resamples=reps,
    )


def exact_mean_test(original, random):
    from itertools import combinations
    from math import comb

    x = np.asarray(original, float)
    y = np.asarray(random, float)
    if not len(x) or not len(y):
        return (np.nan, 0)
    values = np.r_[x, y]
    n0 = len(x)
    n1 = len(y)
    total = values.sum()
    assignments = comb(n0 + n1, n1)
    if assignments > 1000000:
        raise ValueError("Use within_level_test for large permutation spaces")
    selected = np.array(list(combinations(range(len(values)), n1)), dtype=int)
    sums = values[selected].sum(axis=1)
    null = (total - sums) / n0 - sums / n1
    observed = x.mean() - y.mean()
    return (
        float(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12) / assignments),
        assignments,
    )


def within_level_test(original, random, seed=20261102, B=19999):
    from math import comb

    x = np.asarray(original, float)
    y = np.asarray(random, float)
    if not len(x) or not len(y):
        return (np.nan, 0, 0, "not testable: missing comparison group")
    space = comb(len(x) + len(y), len(y))
    if space <= 1000000:
        (p, n) = exact_mean_test(x, y)
        return (p, n, space, "two-sided exact within-level label permutation")
    rng = np.random.default_rng(seed)
    values = np.r_[x, y]
    total = values.sum()
    n0 = len(x)
    n1 = len(y)
    observed = x.mean() - y.mean()
    extreme = 0
    for start in range(0, B, 256):
        count = min(256, B - start)
        selected = np.argpartition(rng.random((count, len(values))), n1 - 1, axis=1)[
            :, :n1
        ]
        sums = values[selected].sum(axis=1)
        null = (total - sums) / n0 - sums / n1
        extreme += np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12)
    return (
        float((1 + extreme) / (B + 1)),
        B,
        space,
        "two-sided Monte Carlo within-level label permutation (+1 correction)",
    )
