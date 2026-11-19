from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def maximal_memberships(
    positions: np.ndarray, entry: np.ndarray, exit_: np.ndarray
) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    if not len(positions):
        return positions
    ordered = positions[np.argsort(entry[positions], kind="stable")]
    keep = np.r_[entry[ordered[1:]] >= exit_[ordered[:-1]], True]
    return ordered[keep]


def frequency(values: Any, value_name: str) -> pd.DataFrame:
    values = np.asarray(values, dtype=int)
    (unique, counts) = np.unique(values, return_counts=True)
    frame = pd.DataFrame(
        {
            value_name: unique,
            "frequency": counts,
            "fraction": counts / max(len(values), 1),
            "denominator": len(values),
        }
    )
    return frame


def log_bins(values: Any) -> pd.DataFrame:
    values = np.asarray(values, dtype=int)
    positive = values[values > 0]
    maximum = int(positive.max()) if len(positive) else 1
    edges = np.unique(
        np.r_[1, np.floor(np.geomspace(1, maximum + 1, 31)).astype(int), maximum + 1]
    )
    (counts, edges) = np.histogram(positive, bins=edges)
    widths = np.diff(edges)
    frame = pd.DataFrame(
        {
            "left": edges[:-1],
            "right": edges[1:],
            "centre": np.sqrt(edges[:-1] * edges[1:]),
            "width": widths,
            "frequency": counts,
            "frequency_density": counts / widths,
            "probability_density": counts / (max(len(values), 1) * widths),
            "zero_nodes": int((values == 0).sum()),
            "node_total": len(values),
        }
    )
    return frame


def figure1a_statistics(data: Any) -> pd.DataFrame:
    groups: dict[int, list[str]] = {}
    for key in data.ids:
        groups.setdefault(int(data.tree[key]["level"]), []).append(key)
    rows = []
    previous = None
    for level, ids in sorted(groups.items()):
        participating = len({int(p) for key in ids for p in data.node_papers[key]})
        completed = 0 if previous is None else previous - participating
        if completed < 0:
            raise ValueError(
                f"Figure 1A Level {level}: participating papers increased from {previous} to {participating}; the requested completed-paper count is negative."
            )
        rows.append(
            {
                "level": level,
                "nodes": len(ids),
                "leaf_nodes": sum((not data.tree[key]["children_ids"] for key in ids)),
                "participating_papers": participating,
                "completed_papers": completed,
            }
        )
        previous = participating
    return pd.DataFrame(rows)


def figure1e_statistics(
    nodes: pd.DataFrame, papers: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if papers.pmid.astype(str).duplicated().any():
        raise ValueError("Figure 1E requires one row per unique PMID.")
    if not len(papers):
        raise ValueError("Figure 1E requires a nonempty paper cohort.")
    years = np.arange(int(papers.year.min()), int(papers.year.max()) + 1)
    starts = pd.to_numeric(nodes.earliest_year, errors="coerce").to_numpy(float)
    ends = pd.to_numeric(nodes.latest_year, errors="coerce").to_numpy(float)
    dated = np.isfinite(starts) & np.isfinite(ends)
    if np.any(starts[dated] > ends[dated]):
        raise ValueError("Figure 1E node intervals must start no later than they end.")
    leaf = (
        nodes.is_leaf.to_numpy()
        if nodes.is_leaf.dtype == bool
        else nodes.is_leaf.astype(str).str.lower().eq("true").to_numpy()
    )
    rows = []
    for year in years:
        covered = dated & (starts <= year) & (year <= ends)
        rows.append(
            {
                "year": int(year),
                "nodes": int(covered.sum()),
                "leaf_nodes": int((covered & leaf).sum()),
                "papers": int(papers.year.eq(year).sum()),
            }
        )
    annual = pd.DataFrame(rows)
    delta = np.zeros(len(years) + 1, dtype=int)
    for start, end in zip(starts[dated], ends[dated]):
        (lo, hi) = (max(int(start), int(years[0])), min(int(end), int(years[-1])))
        if lo <= hi:
            delta[lo - years[0]] += 1
            delta[hi - years[0] + 1] -= 1
    checks = {
        "annual_papers_sum_to_cohort": int(annual.papers.sum()) == len(papers),
        "leaf_nodes_are_subset_of_all_nodes": bool(
            annual.leaf_nodes.le(annual.nodes).all()
        ),
        "inclusive_interval_counts_match_event_sweep": bool(
            np.array_equal(np.cumsum(delta)[:-1], annual.nodes)
        ),
        "interval_node_years_conserved": int(annual.nodes.sum())
        == int(
            sum(
                (
                    max(0, min(int(e), int(years[-1])) - max(int(s), int(years[0])) + 1)
                    for (s, e) in zip(starts[dated], ends[dated])
                )
            )
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return (
        annual,
        {
            "checks": checks,
            "cohort_papers": len(papers),
            "nodes_with_year_interval": int(dated.sum()),
            "nodes_without_year_interval": int((~dated).sum()),
            "year_range": [int(years[0]), int(years[-1])],
            "node_counting": "A node is counted once in every year from earliest_year through latest_year, inclusive, including intervening years without papers. No minimum support threshold.",
            "paper_counting": "Each unique cohort PMID is counted once in its actual publication year.",
        },
    )
