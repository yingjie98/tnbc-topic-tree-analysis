from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

SEED = 20260906


def sibling_sample(d):
    rng = np.random.default_rng(SEED)
    rows = []
    for lev in range(
        3, max((int(d.tree[k]["level"]) for k in d.eligible), default=2) + 1
    ):
        ids = [k for k in d.eligible if d.tree[k]["level"] == lev]
        for korder in range(1, lev):
            groups = {}
            for node in ids:
                path = d.paths[node]
                anc = path[lev - korder - 1]
                child = path[lev - korder]
                groups.setdefault(anc, {}).setdefault(child, []).append(node)
            blocks = []
            weights = []
            for sub in groups.values():
                for a, b in itertools.combinations(sub.values(), 2):
                    blocks.append((a, b))
                    weights.append(len(a) * len(b))
            total = sum(weights)
            if not total:
                continue
            cumulative = np.cumsum(weights, dtype=np.int64)
            ranks = np.sort(rng.choice(total, min(500, total), replace=False))
            for r in ranks:
                bi = np.searchsorted(cumulative, r, side="right")
                offset = int(r - (cumulative[bi - 1] if bi else 0))
                (a, b) = blocks[bi]
                (na, nb) = (a[offset // len(b)], b[offset % len(b)])
                rows.append(
                    dict(
                        level=lev,
                        k=korder,
                        node_A=na,
                        node_B=nb,
                        A_papers=d.counts[na],
                        B_papers=d.counts[nb],
                        candidate_pairs=total,
                        sampled_pairs=len(ranks),
                        sampling_fraction=len(ranks) / total,
                        paper_jaccard=len(np.intersect1d(d.ix[na], d.ix[nb]))
                        / len(np.union1d(d.ix[na], d.ix[nb])),
                    )
                )
    return pd.DataFrame(
        rows,
        columns=[
            "level",
            "k",
            "node_A",
            "node_B",
            "A_papers",
            "B_papers",
            "candidate_pairs",
            "sampled_pairs",
            "sampling_fraction",
            "paper_jaccard",
        ],
    )
