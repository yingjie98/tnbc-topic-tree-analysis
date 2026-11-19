from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .ai_validation_single_store import METRICS, EDGE_ID, NA, validate_rows
from .ai_validation_mean_statistics import within_level_test
from .ai_validation_statistics import holm

MARGIN = 0.25
EQ_FAMILY = 9


def frame_from_rows(rows):
    validate_rows(rows)
    records = []
    for row in rows:
        (major, qid, _, mismatch) = row[:4]
        match = EDGE_ID.fullmatch(qid) if major == 3 else None
        scores = [np.nan if v in (None, "", NA) else float(v) for v in row[4:9]]
        records.append(
            dict(
                major=major,
                metric=METRICS[major],
                item_id=qid,
                task_type="edge" if major == 3 else "paper",
                condition="random_mismatch" if mismatch else "original",
                level=int(match[1]) if match else 0,
                parent_id=match[2] if match else "",
                stratum=match[1] if match else "all",
                population_n=1,
                r1=scores[0],
                r2=scores[1],
                r3=scores[2],
                r4=scores[3],
                r5=scores[4],
                AI=np.mean(scores[:3]) if np.isfinite(scores[:3]).all() else np.nan,
                E1=scores[3],
                E2=scores[4],
                common=np.isfinite(scores).all(),
            )
        )
    return pd.DataFrame(records)


def weights_for(frame, task):
    if frame.empty:
        return np.array([])
    if task == "paper":
        return np.full(len(frame), 1 / len(frame))
    counts = frame.groupby("level").item_id.transform("count").to_numpy(float)
    return 1 / (frame.level.nunique() * counts)


def paired_estimate(frame, left, right, task):
    difference = frame[left].to_numpy(float) - frame[right].to_numpy(float)
    if not len(difference):
        return (np.nan, np.nan, np.nan, np.nan)
    weights = weights_for(frame, task)
    estimate = float(weights @ difference)
    mae = float(weights @ np.abs(difference))
    if task == "paper":
        if len(difference) < 2:
            return (estimate, np.nan, np.nan, mae)
        return (
            estimate,
            float(difference.std(ddof=1) / np.sqrt(len(difference))),
            len(difference) - 1,
            mae,
        )
    f = frame.assign(difference=difference)
    variances = []
    dfs = []
    strata = list(f.groupby("level", sort=True))
    for _, g in strata:
        groups = g.parent_id.nunique()
        if groups < 2:
            return (estimate, np.nan, np.nan, mae)
        residuals = (g.difference - g.difference.mean()) / len(g) / len(strata)
        influence = residuals.groupby(g.parent_id).sum().to_numpy()
        variances.append(groups / (groups - 1) * np.sum(influence**2))
        dfs.append(groups - 1)
    variance = float(sum(variances))
    denominator = sum((v * v / d for (v, d) in zip(variances, dfs)))
    df = variance**2 / denominator if denominator else min(dfs)
    return (estimate, np.sqrt(variance), df, mae)


def equivalence(frame, left, right, task, margin=MARGIN):
    (estimate, se, df, mae) = paired_estimate(frame, left, right, task)
    row = dict(
        estimate=estimate,
        SE=se,
        df=df,
        MAE=mae,
        n=len(frame),
        margin=margin,
        CI_low=np.nan,
        CI_high=np.nan,
        p_raw=np.nan,
        p_adjusted=np.nan,
        equivalent=False,
        status="n/a",
    )
    if not np.isfinite(se):
        return row
    if se < 1e-14:
        low = high = estimate
        p = 0.0 if -margin < estimate < margin else 1.0
        row["zero_variance"] = True
    else:
        critical = stats.t.ppf(1 - 0.05 / EQ_FAMILY, df)
        (low, high) = (estimate - critical * se, estimate + critical * se)
        p = max(
            stats.t.sf((estimate + margin) / se, df),
            stats.t.cdf((estimate - margin) / se, df),
        )
        row["zero_variance"] = False
    adjusted = min(1.0, EQ_FAMILY * p)
    row.update(
        CI_low=float(low),
        CI_high=float(high),
        p_raw=float(p),
        p_adjusted=float(adjusted),
        equivalent=bool(adjusted < 0.05 and low > -margin and (high < margin)),
        status="tested",
    )
    return row


def common_frame(frame, metric):
    f = frame[(frame.metric == metric) & frame.common].copy()
    if metric == "refinement_score":
        counts = frame[frame.metric == metric].groupby("level").condition.nunique()
        f = f[f.level.isin(counts[counts == 2].index)]
        counts = f.groupby("level").condition.nunique()
        f = f[f.level.isin(counts[counts == 2].index)]
    return f


def level_results(frame, reps=4999, permutations=19999):
    f = frame[frame.task_type == "edge"]
    (rows, tests) = ([], [])
    rng = np.random.default_rng(20261103)
    common = set(f.groupby("level").condition.nunique().loc[lambda x: x == 2].index)
    for (level, condition), g in f.groupby(["level", "condition"], sort=True):
        v = g[g.AI.notna()]
        parents = sorted(v.parent_id.unique())
        low = high = np.nan
        if len(parents) >= 2 and level in common and len(v):
            index = {p: i for (i, p) in enumerate(parents)}
            z = rng.exponential(size=(reps, len(parents)))[
                :, [index[p] for p in v.parent_id]
            ]
            boot = z @ v.AI.to_numpy() / z.sum(1)
            (low, high) = np.quantile(boot, [0.025, 0.975])
        rows.append(
            dict(
                level=level,
                condition=condition,
                n=len(v),
                mean_score=v.AI.mean(),
                CI_low=low,
                CI_high=high,
            )
        )
    for level, g in f.groupby("level", sort=True):
        a = g.loc[(g.condition == "original") & g.AI.notna(), "AI"].to_numpy()
        b = g.loc[(g.condition == "random_mismatch") & g.AI.notna(), "AI"].to_numpy()
        (p, *_) = within_level_test(a, b, B=permutations, seed=20261103 + int(level))
        tests.append(dict(level=level, p_raw=p))
    tests = pd.DataFrame(tests, columns=["level", "p_raw"])
    tests["p_holm"] = holm(tests.p_raw)
    return (pd.DataFrame(rows), tests)
