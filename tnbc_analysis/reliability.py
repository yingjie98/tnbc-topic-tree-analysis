from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from .topic_tree_analysis import normalize


def run(
    data,
    output,
    model_dir=None,
    model_id="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    reps=49999,
    mean_reps=1999,
    sibling_reps=49999,
):
    from .semantic_statistics import (
        node_statistics,
        coverage_statistics,
        histogram_statistics,
    )
    from .sibling_sampling import sibling_sample
    from .reliability_level_similarity import (
        original_mean_intervals,
        adjacent_statistics,
    )
    from .reliability_vertical_refinement import (
        statistic_weights,
        paper_influence,
        multiplier_draws,
        summarize,
        holm,
        significance,
    )
    from .reliability_sibling_distance import trend_weights, permuted_slopes
    import torch
    from transformers import AutoTokenizer, AutoModel

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    data.ix = data.node_papers
    data.eligible = [k for k in data.ids if data.counts[k] >= 3]
    data.ei = {k: i for (i, k) in enumerate(data.eligible)}
    if not data.eligible:
        raise ValueError("No node has at least three papers for semantic analysis")
    local = model_dir is not None
    source = str(model_dir or model_id)
    torch.manual_seed(20260906)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local)
    model = AutoModel.from_pretrained(source, local_files_only=local).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(data.texts), 32):
            batch = tokenizer(
                data.texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            vectors.append(model(**batch).last_hidden_state[:, 0].float().numpy())
    bert = normalize(np.vstack(vectors))
    del model
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        min_df=2,
        max_df=0.98,
        max_features=60000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=np.float32,
    ).fit_transform(data.texts)
    dim = min(768, tfidf.shape[1] - 1, len(data.pmids) - 1)
    if dim < 1:
        raise ValueError("Insufficient text features for SVD")
    lexical = normalize(
        TruncatedSVD(n_components=dim, n_iter=7, random_state=20260906).fit_transform(
            tfidf
        )
    )
    tracks = {"PubMedBERT": bert, "TFIDF_SVD": lexical}
    nodes = {}
    centres = {}
    means = []
    tests = []
    estimates = []
    influences = []
    records = []
    coverage_statistics(data).to_csv(
        out / "semantic_coverage.csv", index=False, encoding="utf-8-sig"
    )
    incidence = data.incidence(data.eligible)
    (base, coefficients) = statistic_weights(data)
    supports = [
        len(np.unique(incidence[coefficients.getrow(i).indices].indices))
        for i in range(len(base))
    ]
    for label, E in tracks.items():
        (frame, centres[label]) = node_statistics(data, E, label)
        frame["unique_papers"] = [
            len(set(data.tree[k].get("unique_pmids", [])) & set(data.pmids))
            for k in data.eligible
        ]
        nodes[label] = frame
        frame.to_csv(
            out / f"{label}_node_dispersion.csv", index=False, encoding="utf-8-sig"
        )
        means.append(original_mean_intervals(frame, label, reps=mean_reps))
        by = {int(k): v for (k, v) in frame.groupby("level")}
        for a, b in zip(sorted(by)[:-1], sorted(by)[1:]):
            if b == a + 1:
                tests.append(adjacent_statistics(by[a], by[b], label, reps=reps))
        (W, H) = paper_influence(incidence, E)
        r = base.copy()
        r.insert(0, "embedding", label)
        r["support_papers"] = supports
        records.append(r)
        estimates.append(np.asarray(coefficients @ W).ravel())
        influences.append((coefficients @ H).toarray())
    pd.concat(means, ignore_index=True).to_csv(
        out / "level_dispersion_intervals.csv", index=False, encoding="utf-8-sig"
    )
    adjacent = pd.DataFrame(tests)
    if len(adjacent):
        adjacent["p_holm"] = holm(adjacent.p_raw)
        adjacent["significance"] = adjacent.p_holm.map(significance)
    adjacent.to_csv(out / "adjacent_level_tests.csv", index=False, encoding="utf-8-sig")
    draws = multiplier_draws(np.vstack(influences), reps=reps)
    summarize(
        pd.concat(records, ignore_index=True), np.concatenate(estimates), draws
    ).to_csv(out / "vertical_refinement.csv", index=False, encoding="utf-8-sig")
    rho = stats.spearmanr(nodes["PubMedBERT"].W, nodes["TFIDF_SVD"].W).statistic
    pd.DataFrame([{"n_nodes": len(data.eligible), "spearman_rho": rho}]).to_csv(
        out / "embedding_concordance.csv", index=False, encoding="utf-8-sig"
    )
    histogram_statistics(pd.concat(nodes.values(), ignore_index=True)).to_csv(
        out / "dispersion_histogram.csv", index=False, encoding="utf-8-sig"
    )
    pairs = sibling_sample(data).reset_index(drop=True)
    summary = []
    trends = []
    for label, C in centres.items():
        pairs[label] = 1 - np.sum(
            C[[data.ei[k] for k in pairs.node_A]]
            * C[[data.ei[k] for k in pairs.node_B]],
            axis=1,
        )
    rng_ci = np.random.default_rng(20260908)
    for level, g in pairs.groupby("level", sort=True):
        ids = [k for k in data.eligible if data.tree[k]["level"] == level]
        index = {k: i for (i, k) in enumerate(ids)}
        a = np.array([index[k] for k in g.node_A])
        b = np.array([index[k] for k in g.node_B])
        K = g.k.to_numpy()
        weights = trend_weights(K)
        distances = [
            1
            - np.asarray(C[[data.ei[k] for k in ids]], float)
            @ np.asarray(C[[data.ei[k] for k in ids]], float).T
            for C in centres.values()
        ]
        observed = (
            np.array([D[a, b] @ weights for D in distances])
            if weights is not None
            else np.full(2, np.nan)
        )
        extreme = np.zeros(2, dtype=int)
        rng = np.random.default_rng(20261031 + int(level))
        if weights is not None:
            for start in range(0, sibling_reps, 128):
                mapping = np.array(
                    [
                        rng.permutation(len(ids))
                        for _ in range(min(128, sibling_reps - start))
                    ]
                )
                extreme += np.count_nonzero(
                    permuted_slopes(distances, a, b, weights, mapping)
                    >= observed[None, :] - 1e-12,
                    axis=0,
                )
        boot = {label: {k: [] for k in sorted(g.k.unique())} for label in centres}
        for _ in range(499):
            z = rng_ci.exponential(size=len(ids))
            dw = z[a] * z[b]
            for label in centres:
                for k in boot[label]:
                    boot[label][k].append(
                        np.average(g[label].to_numpy()[K == k], weights=dw[K == k])
                    )
        for j, label in enumerate(centres):
            trends.append(
                {
                    "embedding": label,
                    "level": level,
                    "k_groups": g.k.nunique(),
                    "n_pairs": len(g),
                    "n_nodes": len(ids),
                    "slope": observed[j],
                    "p_raw": (1 + extreme[j]) / (sibling_reps + 1)
                    if weights is not None
                    else np.nan,
                }
            )
            for k, h in g.groupby("k"):
                (low, high) = np.quantile(boot[label][k], [0.025, 0.975])
                summary.append(
                    {
                        "embedding": label,
                        "level": level,
                        "k": k,
                        "n": len(h),
                        "candidate_pairs": h.candidate_pairs.iloc[0],
                        "mean": h[label].mean(),
                        "CI_low": low,
                        "CI_high": high,
                    }
                )
    t = pd.DataFrame(trends)
    if len(t):
        t["p_holm"] = holm(t.p_raw)
        t["significance"] = t.p_holm.map(significance)
    t.to_csv(out / "sibling_trend_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(
        out / "sibling_distances.csv", index=False, encoding="utf-8-sig"
    )
    pairs.to_csv(out / "sibling_pairs.csv", index=False, encoding="utf-8-sig")
