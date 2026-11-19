from __future__ import annotations
import numpy as np
import pandas as pd

MODELS = ("PubMedBERT", "TFIDF_SVD")


def coverage_statistics(data):
    all_levels = pd.Series(
        [int(data.tree[k]["level"]) for k in data.ids]
    ).value_counts()
    included = pd.Series(
        [int(data.tree[k]["level"]) for k in data.eligible]
    ).value_counts()
    return pd.DataFrame(
        [
            {
                "level": int(level),
                "all_nodes": int(all_levels[level]),
                "included_nodes": int(included.get(level, 0)),
            }
            for level in sorted(all_levels.index)
        ]
    )


def node_statistics(data, vectors, label):
    counts = np.array([data.counts[k] for k in data.eligible], dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(data.pmids):
        raise ValueError(f"{label}: paper-vector order/shape does not match the cohort")
    means = data.incidence(data.eligible) @ vectors / counts[:, None]
    dispersion = 1 - np.linalg.norm(means, axis=1)
    frame = pd.DataFrame([data.meta(k) for k in data.eligible])
    frame.insert(0, "embedding", label)
    frame["W"] = dispersion
    centres = np.asarray(means, dtype=np.float32)
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
    return (frame, centres)


def histogram_statistics(nodes, bins=55):
    edges = np.histogram_bin_edges(nodes.W.to_numpy(), bins=bins)
    rows = []
    for label in MODELS:
        (counts, _) = np.histogram(nodes.loc[nodes.embedding == label, "W"], bins=edges)
        rows.extend(
            (
                {
                    "embedding": label,
                    "bin_left": left,
                    "bin_right": right,
                    "node_count": int(count),
                }
                for (left, right, count) in zip(edges[:-1], edges[1:], counts)
            )
        )
    return pd.DataFrame(rows)


def distance_statistics(data, centres, pairs):
    a = np.array([data.ei[k] for k in pairs.node_A])
    b = np.array([data.ei[k] for k in pairs.node_B])
    distances = 1 - np.sum(centres[a] * centres[b], axis=1)
    rows = []
    for (level, k), indices in pairs.groupby(["level", "k"], sort=True).groups.items():
        ix = np.array(list(indices), dtype=int)
        rows.append(
            {
                "level": int(level),
                "k": int(k),
                "node_pairs": len(ix),
                "mean_center_distance": np.mean(distances[ix]),
            }
        )
    return pd.DataFrame(rows)
