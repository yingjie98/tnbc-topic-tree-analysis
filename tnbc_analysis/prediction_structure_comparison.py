from __future__ import annotations
from .project_config import TIME
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from .core_network_selection import MIN_NODES, MAX_NODES, select_coverage_nodes

OBSERVATION_START = TIME.observation_start
OBSERVATION_END = TIME.observation_end
FORECAST_START = TIME.forecast_start
FORECAST_END = TIME.forecast_end
FIG34_OBSERVED_START = TIME.recent_start
FIG34_OBSERVED_END = TIME.observation_end
FIG34_FORECAST_START = TIME.forecast_start
FIG34_FORECAST_END = TIME.forecast_end
FIG34_GROWTH_OR = 1.25
FIG34_DECLINE_OR = 0.8


@dataclass(frozen=True)
class Paths:
    output: Path
    tree: Path
    papers: Path

    @property
    def tables(self) -> Path:
        return self.output / "tables"

    @property
    def prediction_tables(self) -> Path:
        return self.output / "tables"

    @property
    def frozen_node_rows(self) -> Path:
        return self.prediction_tables / "08_未来概率网络模拟_结构块节点.csv"

    @property
    def frozen_edge_rows(self) -> Path:
        return self.prediction_tables / "08_未来概率网络模拟_结构块弦.csv"

    @property
    def frozen_merged_edges(self) -> Path:
        return (
            self.prediction_tables
            / "08_未来概率网络模拟_归并后结构块弦_筛选后正式结果.csv"
        )

    @property
    def frozen_core_nodes(self) -> Path:
        return self.prediction_tables / "08_未来概率网络模拟_归并后结构块核心节点.csv"

    @property
    def node_exposure_forecast(self) -> Path:
        return self.prediction_tables / "07_节点年度论文量基线预测.csv.gz"

    @property
    def unified_alignment(self) -> Path:
        return (
            self.prediction_tables / "21_所有年度与全量S_JCR_IF_未来结构块统一对齐.csv"
        )

    @property
    def formal_summary(self) -> Path:
        return (
            self.output.parent
            / "03_hypergraph"
            / "tables"
            / "16_S社群_完整清单_结构注释.csv"
        )

    @property
    def formal_membership(self) -> Path:
        return (
            self.output.parent
            / "03_hypergraph"
            / "tables"
            / "12_PMID逐层显著性_全部节点状态.csv"
        )

    @property
    def formal_core_nodes(self) -> Path:
        return (
            self.output.parent
            / "03_hypergraph"
            / "tables"
            / "16_S社群_核心节点清单.csv"
        )

    @property
    def formal_edges(self) -> Path:
        return (
            self.output.parent
            / "03_hypergraph"
            / "tables"
            / "16_S社群_显著性筛选弦列表.csv"
        )


def _read(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少预测结构比较所需输入文件：{path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def _normalise_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _as_bool(value: Any) -> bool:
    return _normalise_text(value).lower() in {"1", "true", "yes", "y", "是"}


def _natural_key(value: Any) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split("(\\d+)", _normalise_text(value))
    ]


def _numeric(frame: pd.DataFrame, column: str, fallback: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(fallback, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(fallback).astype(float)


def _edge_key(
    relation: Any, start: Any, end: Any, level: Any
) -> tuple[str, str, str, str]:
    kind = _normalise_text(relation)
    (a, b) = (_normalise_text(start), _normalise_text(end))
    if kind == "horizontal" and a > b:
        (a, b) = (b, a)
    return (kind, a, b, _normalise_text(level))


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} 缺少必要字段：{sorted(missing)}")


def _node_pmid_lookup(data) -> tuple[dict[str, set[str]], dict[str, dict[int, int]]]:
    pmid_sets: dict[str, set[str]] = {}
    annual: dict[str, dict[int, int]] = {}
    for node_id, indices in data.node_papers.items():
        pmids = {str(data.pmids[int(index)]) for index in indices}
        pmid_sets[str(node_id)] = pmids
        counts: dict[int, int] = {}
        for index in indices:
            year = int(data.years[int(index)])
            if OBSERVATION_START <= year <= OBSERVATION_END:
                counts[year] = counts.get(year, 0) + 1
        annual[str(node_id)] = counts
    return (pmid_sets, annual)


def _forecast_library_exposure(data, years: np.ndarray) -> dict[int, float]:
    observed_years = np.asarray(
        sorted(set((int(year) for year in data.years))), dtype=int
    )
    observed = np.asarray(
        [(np.asarray(data.years, dtype=int) == year).sum() for year in observed_years],
        dtype=float,
    )
    recent_n = min(3, len(observed))
    recent_mean = float(observed[-recent_n:].mean()) if recent_n else 1.0
    if len(observed) >= 4:
        coordinate = np.arange(len(observed), dtype=float)
        centered = coordinate - coordinate.mean()
        slope = float(
            (observed - observed.mean())
            @ centered
            / max(float(np.sum(centered**2)), 1.0)
        )
    else:
        slope = 0.0
    output: dict[int, float] = {}
    lookup = {
        int(year): float(value)
        for (year, value) in zip(observed_years, observed, strict=True)
    }
    for year in years:
        year = int(year)
        output[year] = (
            lookup[year]
            if year in lookup
            else max(1.0, recent_mean + slope * (year - OBSERVATION_END))
        )
    return output


def _historical_active_sources(data) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    from .topic_tree_analysis import build_pmid_hypergraph, levelwise_pmid_significance

    graph_data = build_pmid_hypergraph(data)
    calibration = levelwise_pmid_significance(data, graph_data)
    index = {
        str(node_id): position for (position, node_id) in enumerate(graph_data.ids)
    }
    node_pmids = {
        str(node_id): set(np.asarray(indices, dtype=int).tolist())
        for (node_id, indices) in data.node_papers.items()
    }
    active_sources = {
        str(node_id): set(calibration["active_sources"][index[str(node_id)]])
        for node_id in graph_data.ids
    }
    return (node_pmids, active_sources)


def _forecast_module_time_trajectory(
    paths: Paths, data, future_edges: pd.DataFrame, future_sets: dict[str, set[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_edges = {
        "future_module_id",
        "node_1_id",
        "node_2_id",
        "conditional_edge_evidence",
    }
    _validate_columns(future_edges, required_edges, "预测模块归并边")
    forecast = _read(paths.node_exposure_forecast, compression="gzip")
    _validate_columns(
        forecast,
        {
            "node_id",
            "forecast_year",
            "predicted_node_pmid_count_baseline",
            "recent_3_year_mean_pmid",
        },
        "节点年度论文量基线预测",
    )
    forecast["node_id"] = forecast.node_id.map(_normalise_text)
    forecast["forecast_year"] = pd.to_numeric(
        forecast.forecast_year, errors="raise"
    ).astype(int)
    forecast["predicted_node_pmid_count_baseline"] = _numeric(
        forecast, "predicted_node_pmid_count_baseline"
    )
    forecast["recent_3_year_mean_pmid"] = _numeric(forecast, "recent_3_year_mean_pmid")
    forecast = forecast.loc[
        forecast.forecast_year.between(FORECAST_START, FORECAST_END)
    ].copy()
    forecast_lookup = forecast.set_index(["node_id", "forecast_year"])[
        ["predicted_node_pmid_count_baseline", "recent_3_year_mean_pmid"]
    ].to_dict("index")
    annual_years = np.arange(TIME.history_start, FORECAST_END + 1, dtype=int)
    library_exposure = _forecast_library_exposure(data, annual_years)
    (paper_indices, active_sources) = _historical_active_sources(data)
    year_by_index = np.asarray(data.years, dtype=int)
    evidence_cache: dict[tuple[str, str, str], set[int]] = {}

    def _edge_evidence_indices(relation: str, left: str, right: str) -> set[int]:
        if relation == "horizontal":
            (first, second) = sorted((left, right))
            cache_key = (relation, first, second)
            if cache_key not in evidence_cache:
                if int(data.tree[first]["level"]) != int(data.tree[second]["level"]):
                    raise ValueError(f"预测横向弦跨层：{first}、{second}")
                evidence_cache[cache_key] = paper_indices.get(
                    first, set()
                ) & paper_indices.get(second, set())
            return evidence_cache[cache_key]
        if relation != "vertical_direct_parent_child":
            raise ValueError(f"未知预测弦类型：{relation}")
        if data.parent.get(right) == left:
            (parent, child) = (left, right)
        elif data.parent.get(left) == right:
            (parent, child) = (right, left)
        else:
            raise ValueError(f"预测纵向弦不是主题树直接父子边：{left}、{right}")
        cache_key = (relation, parent, child)
        if cache_key not in evidence_cache:
            evidence_cache[cache_key] = (
                active_sources.get(parent, set())
                & active_sources.get(child, set())
                & paper_indices.get(parent, set())
            )
        return evidence_cache[cache_key]

    annual_rows: list[dict[str, Any]] = []
    for module, module_nodes in future_sets.items():
        edge_part = future_edges.loc[
            future_edges.future_module_id.astype(str).eq(str(module))
        ].copy()
        evidence_indices: set[int] = set()
        node_weights = {str(node): 0.0 for node in module_nodes}
        for edge in edge_part.itertuples(index=False):
            (left, right) = (str(edge.node_1_id), str(edge.node_2_id))
            evidence_indices.update(
                _edge_evidence_indices(str(edge.relation_type), left, right)
            )
            weight = max(float(getattr(edge, "conditional_edge_evidence", 0.0)), 0.0)
            node_weights[left] = node_weights.get(left, 0.0) + weight
            node_weights[right] = node_weights.get(right, 0.0) + weight
        observed_counts = {
            int(year): int(np.sum(year_by_index[list(evidence_indices)] == year))
            if evidence_indices
            else 0
            for year in range(TIME.history_start, OBSERVATION_END + 1)
        }
        anchor_years = list(range(TIME.recent_start, OBSERVATION_END + 1))
        anchor = float(np.mean([observed_counts[year] for year in anchor_years]))
        if not node_weights or sum(node_weights.values()) <= 0:
            node_weights = {str(node): 1.0 for node in module_nodes}
        weight_total = max(sum(node_weights.values()), 1e-12)
        projected_counts: dict[int, float] = {}
        raw_projected_counts: dict[int, float] = {}
        feasibility_bound: dict[int, bool] = {}
        for year in range(FORECAST_START, FORECAST_END + 1):
            multiplier = 0.0
            for node, weight in node_weights.items():
                values = forecast_lookup.get((str(node), int(year)))
                if values is None:
                    continue
                denominator = max(float(values["recent_3_year_mean_pmid"]), 1.0)
                multiplier += (
                    float(weight)
                    / weight_total
                    * (
                        float(values["predicted_node_pmid_count_baseline"])
                        / denominator
                    )
                )
            raw_count = max(anchor * multiplier, 0.0)
            raw_projected_counts[year] = raw_count
            projected_counts[year] = min(raw_count, library_exposure[year])
            feasibility_bound[year] = raw_count > library_exposure[year]
        for year in annual_years:
            year = int(year)
            observed = year <= OBSERVATION_END
            count = (
                float(observed_counts[year])
                if observed
                else float(projected_counts[year])
            )
            annual_rows.append(
                {
                    "future_module_id": str(module),
                    "year": year,
                    "chord_evidence_paper_count": count,
                    "annual_library_paper_exposure": float(library_exposure[year]),
                    "raw_projected_chord_evidence_paper_equivalent": float(
                        raw_projected_counts[year]
                    )
                    if not observed
                    else np.nan,
                    "projected_feasibility_bound_applied": bool(
                        feasibility_bound.get(year, False)
                    ),
                    "data_status": "observed_chord_evidence_pmid"
                    if observed
                    else "projected_chord_evidence_paper_equivalent",
                    "forecast_construction": "observed, PMID-deduplicated horizontal or active-source-restricted vertical chord support"
                    if observed
                    else TIME.text(
                        "2021–2025 observed chord-evidence anchor × prediction-edge-weighted frozen node-exposure multiplier"
                    ),
                }
            )
    annual = pd.DataFrame(annual_rows)
    if annual.future_module_id.nunique() != len(future_sets):
        raise AssertionError("预测模块年度弦证据轨迹未覆盖全部冻结模块。")
    return (annual, _forecast_window_summary(annual))


def _forecast_window_summary(annual: pd.DataFrame) -> pd.DataFrame:
    required = {
        "future_module_id",
        "year",
        "chord_evidence_paper_count",
        "annual_library_paper_exposure",
        "raw_projected_chord_evidence_paper_equivalent",
        "projected_feasibility_bound_applied",
    }
    _validate_columns(annual, required, "预测模块年度弦证据论文趋势")
    from .topic_tree_analysis import fit_annual_binomial_trend

    rows: list[dict[str, Any]] = []
    required_years = set(range(FIG34_OBSERVED_START, FIG34_FORECAST_END + 1))
    for module_id, part in annual.groupby("future_module_id", sort=False):
        part = part.copy()
        part["year"] = pd.to_numeric(part.year, errors="raise").astype(int)
        part["chord_evidence_paper_count"] = _numeric(
            part, "chord_evidence_paper_count"
        )
        part["annual_library_paper_exposure"] = _numeric(
            part, "annual_library_paper_exposure"
        )
        window = part.loc[
            part.year.between(FIG34_OBSERVED_START, FIG34_FORECAST_END)
        ].sort_values("year")
        if set(window.year) != required_years:
            raise ValueError(
                f"{module_id}{TIME.text(' 缺少图34所需的2021–2030年度轨迹。')}"
            )
        observed = window.loc[
            window.year.between(FIG34_OBSERVED_START, FIG34_OBSERVED_END)
        ]
        forecast = window.loc[
            window.year.between(FIG34_FORECAST_START, FIG34_FORECAST_END)
        ]
        observed_count = float(observed.chord_evidence_paper_count.sum())
        forecast_count = float(forecast.chord_evidence_paper_count.sum())
        observed_total = float(observed.annual_library_paper_exposure.sum())
        forecast_total = float(forecast.annual_library_paper_exposure.sum())
        observed_share = (
            observed_count / observed_total if observed_total > 0 else np.nan
        )
        forecast_share = (
            forecast_count / forecast_total if forecast_total > 0 else np.nan
        )
        share_ratio = forecast_share / observed_share if observed_share > 0 else np.nan
        paper_ratio = forecast_count / observed_count if observed_count > 0 else np.nan
        counts = window.chord_evidence_paper_count.to_numpy(float)
        totals = window.annual_library_paper_exposure.to_numpy(float)
        boundary_mask = (counts <= 0.0) | (counts >= totals)
        adjusted_counts = np.minimum(
            np.maximum(counts, 0.5), np.maximum(totals - 0.5, 0.5)
        )
        trend = fit_annual_binomial_trend(
            window.year.to_numpy(float), adjusted_counts, totals
        )
        annual_or = float(trend.get("trend_odds_ratio_per_year", np.nan))
        five_year_or = (
            float(np.power(annual_or, 5))
            if np.isfinite(annual_or) and annual_or > 0
            else np.nan
        )
        if not np.isfinite(five_year_or) or five_year_or <= 0:
            change_class = "无法估计"
            estimate_status = "annual_binomial_logistic_not_estimable"
        elif five_year_or >= FIG34_GROWTH_OR:
            change_class = "预测增长"
            estimate_status = (
                "annual_binomial_logistic_boundary_corrected"
                if boundary_mask.any()
                else "annual_binomial_logistic"
            )
        elif five_year_or <= FIG34_DECLINE_OR:
            change_class = "预测消退"
            estimate_status = (
                "annual_binomial_logistic_boundary_corrected"
                if boundary_mask.any()
                else "annual_binomial_logistic"
            )
        else:
            change_class = "预测稳定"
            estimate_status = (
                "annual_binomial_logistic_boundary_corrected"
                if boundary_mask.any()
                else "annual_binomial_logistic"
            )
        rows.append(
            {
                "future_module_id": str(module_id),
                TIME.key(
                    "observed_chord_evidence_paper_count_2021_2025"
                ): observed_count,
                TIME.key(
                    "projected_chord_evidence_paper_equivalent_count_2026_2030"
                ): forecast_count,
                TIME.key("observed_library_paper_count_2021_2025"): observed_total,
                TIME.key("projected_library_paper_exposure_2026_2030"): forecast_total,
                TIME.key("observed_chord_evidence_share_2021_2025"): observed_share,
                TIME.key("projected_chord_evidence_share_2026_2030"): forecast_share,
                TIME.key(
                    "forecast_window_paper_ratio_2026_2030_vs_2021_2025"
                ): paper_ratio,
                TIME.key(
                    "forecast_window_paper_percent_change_2026_2030_vs_2021_2025"
                ): (paper_ratio - 1.0) * 100.0 if np.isfinite(paper_ratio) else np.nan,
                TIME.key(
                    "forecast_window_share_ratio_2026_2030_vs_2021_2025"
                ): share_ratio,
                "forecast_window_OR_per_5_years": five_year_or,
                "forecast_window_OR_per_5_years_definition": TIME.text(
                    "exp(5 * beta) from annual binomial-logistic share trend over 2021–2030; 2021–2025 observed chord evidence and 2026–2030 frozen predicted paper equivalents"
                ),
                "forecast_window_time_classification": change_class,
                "forecast_window_time_classification_rule": f"predicted increase OR≥{FIG34_GROWTH_OR:g}; projected stable {FIG34_DECLINE_OR:g}<OR<{FIG34_GROWTH_OR:g}; predicted decline OR≤{FIG34_DECLINE_OR:g}",
                "forecast_window_estimate_status": estimate_status,
                "forecast_window_boundary_corrected_year_count": int(
                    boundary_mask.sum()
                ),
                "forecast_window_prediction_bound_year_count": int(
                    forecast.projected_feasibility_bound_applied.map(_as_bool).sum()
                ),
                "forecast_window_is_point_forecast_description": TIME.text(
                    "2026–2030 values are frozen prediction-derived chord-evidence paper equivalents; the class is descriptive and is not a future hypothesis test"
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.future_module_id.duplicated().any():
        raise AssertionError("图34预测窗口摘要不能含重复模块。")
    return summary


def _formal_membership(path: Path) -> dict[str, set[str]]:
    frame = _read(path)
    _validate_columns(
        frame,
        {"node_id", "significance_block_id", "formal_community_confirmed"},
        "正式S节点状态",
    )
    frame = frame.loc[frame.formal_community_confirmed.map(_as_bool)].copy()
    frame["community_id"] = frame.significance_block_id.map(_normalise_text)
    frame["node_id"] = frame.node_id.map(_normalise_text)
    frame = frame.loc[frame.community_id.ne("") & frame.node_id.ne("")]
    return {
        str(community): set(part.node_id.astype(str))
        for (community, part) in frame.groupby("community_id", sort=False)
    }


def _load_formal_edges(path: Path, membership: dict[str, set[str]]) -> pd.DataFrame:
    edges = _read(path)
    _validate_columns(
        edges,
        {
            "edge_type",
            "start_node_id",
            "end_node_id",
            "formal_community_context_id",
            "effect_size_z",
        },
        "正式S显著弦",
    )
    work = edges.copy()
    work["community_id"] = work.formal_community_context_id.map(_normalise_text)
    work["start_node_id"] = work.start_node_id.map(_normalise_text)
    work["end_node_id"] = work.end_node_id.map(_normalise_text)
    work["relation_type"] = np.where(
        work.edge_type.map(_normalise_text).str.contains(
            "纵|vertical", case=False, regex=True
        ),
        "vertical_direct_parent_child",
        "horizontal",
    )
    work["edge_weight"] = _numeric(work, "corrected_excess_pmid_count").clip(lower=0.0)
    fallback = _numeric(work, "raw_independent_pmid_count").clip(lower=0.0)
    work.loc[work.edge_weight.le(0), "edge_weight"] = fallback.loc[
        work.edge_weight.le(0)
    ]
    work["effect_size_z"] = _numeric(work, "effect_size_z")
    work = work.loc[work.community_id.isin(membership)].copy()
    work = work.loc[
        work.apply(
            lambda row: row.start_node_id in membership[row.community_id]
            and row.end_node_id in membership[row.community_id],
            axis=1,
        )
    ].copy()
    return work


def _module_simulation_statistics(
    raw_nodes: pd.DataFrame, raw_edges: pd.DataFrame, frozen_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    _validate_columns(
        raw_nodes,
        {"simulation_id", "future_module_id", "node_id", "internal_evidence_score"},
        "预测模块节点明细",
    )
    _validate_columns(
        raw_edges,
        {
            "simulation_id",
            "future_module_id",
            "relation_type",
            "node_1_id",
            "node_2_id",
            "level",
            "evidence_z",
        },
        "预测模块弦明细",
    )
    nodes = raw_nodes.loc[
        raw_nodes.future_module_id.astype(str).isin(frozen_ids)
    ].copy()
    edges = raw_edges.loc[
        raw_edges.future_module_id.astype(str).isin(frozen_ids)
    ].copy()
    nodes["future_module_id"] = nodes.future_module_id.astype(str)
    edges["future_module_id"] = edges.future_module_id.astype(str)
    nodes["node_id"] = nodes.node_id.map(_normalise_text)
    edges["edge_key"] = [
        _edge_key(kind, left, right, level)
        for (kind, left, right, level) in zip(
            edges.relation_type, edges.node_1_id, edges.node_2_id, edges.level
        )
    ]
    module_simulations = {
        module: set(part.simulation_id.astype(int))
        for (module, part) in nodes.groupby("future_module_id", sort=False)
    }
    if set(module_simulations) != frozen_ids:
        missing = sorted(frozen_ids - set(module_simulations), key=_natural_key)
        raise ValueError(f"冻结预测模块缺少模拟节点明细：{missing[:8]}")
    node_rows: list[dict[str, Any]] = []
    for (module, node), part in nodes.groupby(
        ["future_module_id", "node_id"], sort=False
    ):
        denominator = len(module_simulations[str(module)])
        support = int(part.simulation_id.nunique())
        node_rows.append(
            {
                "future_module_id": str(module),
                "node_id": str(node),
                "node_name": _normalise_text(part.node_name.iloc[0]),
                "level": int(
                    pd.to_numeric(part.level, errors="coerce").dropna().iloc[0]
                ),
                "module_simulation_count": denominator,
                "node_simulation_count": support,
                "node_conditional_recurrence": support / max(denominator, 1),
                "mean_internal_evidence_score": float(
                    _numeric(part, "internal_evidence_score").mean()
                ),
                "mean_internal_horizontal_degree": float(
                    _numeric(part, "internal_horizontal_degree").mean()
                ),
                "mean_internal_vertical_degree": float(
                    _numeric(part, "internal_vertical_degree").mean()
                ),
            }
        )
    edge_rows: list[dict[str, Any]] = []
    for (module, key), part in edges.groupby(
        ["future_module_id", "edge_key"], sort=False
    ):
        denominator = len(module_simulations[str(module)])
        support = int(part.simulation_id.nunique())
        (relation, left, right, level) = key
        mean_z = float(_numeric(part, "evidence_z").mean())
        recurrence = support / max(denominator, 1)
        edge_rows.append(
            {
                "future_module_id": str(module),
                "relation_type": relation,
                "node_1_id": left,
                "node_2_id": right,
                "level": level,
                "module_simulation_count": denominator,
                "edge_simulation_count": support,
                "edge_conditional_recurrence": recurrence,
                "mean_predicted_evidence_z": mean_z,
                "conditional_edge_evidence": mean_z * recurrence,
            }
        )
    return (pd.DataFrame(node_rows), pd.DataFrame(edge_rows), module_simulations)


def _formal_core_table(path: Path) -> pd.DataFrame:
    core = _read(path)
    _validate_columns(
        core, {"community_id", "core_rank", "node_id", "node_name"}, "正式S核心节点"
    )
    core = core.copy()
    core["community_id"] = core.community_id.map(_normalise_text)
    core["node_id"] = core.node_id.map(_normalise_text)
    core["core_rank"] = _numeric(core, "core_rank", fallback=np.inf).astype(int)
    return core


def _future_core_table(path: Path) -> pd.DataFrame:
    core = _read(path)
    _validate_columns(
        core,
        {"future_module_id", "module_node_rank", "node_id", "node_name"},
        "预测模块核心节点",
    )
    core = core.copy()
    core["future_module_id"] = core.future_module_id.map(_normalise_text)
    core["node_id"] = core.node_id.map(_normalise_text)
    core["core_rank"] = _numeric(core, "module_node_rank", fallback=np.inf).astype(int)
    return core


def _build_figure36_core_network_tables(
    *,
    data: Any,
    node_pmids: dict[str, set[str]],
    selection: pd.DataFrame,
    formal_sets: dict[str, set[str]],
    future_sets: dict[str, set[str]],
    formal_core: pd.DataFrame,
    future_core: pd.DataFrame,
    formal_edges: pd.DataFrame,
    future_nodes: pd.DataFrame,
    future_edges: pd.DataFrame,
    formal_node_state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (all_target, evidence_target) = (0.75, 0.9)
    state = formal_node_state.copy()
    _validate_columns(
        state,
        {
            "node_id",
            "significance_block_id",
            "significant_horizontal_degree",
            "significant_vertical_degree",
        },
        "正式S节点状态（图36核心节点补充排序）",
    )
    state["node_id"] = state.node_id.map(_normalise_text)
    state["community_id"] = state.significance_block_id.map(_normalise_text)
    state["degree_score"] = _numeric(state, "significant_horizontal_degree").clip(
        lower=0
    ) + _numeric(state, "significant_vertical_degree").clip(lower=0)
    year_by_pmid = {
        str(pmid): int(year)
        for (pmid, year) in zip(data.pmids, data.years, strict=True)
    }
    (_, active_source_indices) = _historical_active_sources(data)
    active_source_pmids = {
        node: {str(data.pmids[int(index)]) for index in indices}
        for (node, indices) in active_source_indices.items()
    }
    chord_support_cache: dict[tuple[str, str, str], set[str]] = {}

    def chord_support_pmids(relation: str, left: str, right: str) -> set[str]:
        relation = str(relation)
        (first, second) = sorted((str(left), str(right)))
        key = (relation, first, second)
        if key in chord_support_cache:
            return chord_support_cache[key]
        if relation == "horizontal":
            support = node_pmids.get(first, set()) & node_pmids.get(second, set())
        elif relation == "vertical_direct_parent_child":
            if data.parent.get(first) == second:
                (parent, child) = (second, first)
            elif data.parent.get(second) == first:
                (parent, child) = (first, second)
            else:
                raise ValueError(f"图36遇到非直接父子纵向弦：{left}、{right}")
            support = (
                active_source_pmids.get(parent, set())
                & active_source_pmids.get(child, set())
                & node_pmids.get(parent, set())
            )
        else:
            raise ValueError(f"图36遇到未知弦类型：{relation}")
        chord_support_cache[key] = set(support)
        return chord_support_cache[key]

    def node_metadata(node_id: str) -> dict[str, Any]:
        node = data.tree[str(node_id)]
        pmids = node_pmids.get(str(node_id), set())
        valid_years = [year_by_pmid[p] for p in pmids if p in year_by_pmid]
        return {
            "node_name": str(node.get("name", node_id)),
            "level": int(node.get("level", 0)),
            "all_paper_pmid_count": len(pmids),
            "mean_publication_year": float(np.mean(valid_years))
            if valid_years
            else np.nan,
        }

    def formal_ranked(source_id: str) -> pd.DataFrame:
        members = set(formal_sets[source_id])
        saved = formal_core.loc[formal_core.community_id.eq(source_id)].copy()
        saved = saved.loc[saved.node_id.isin(members)].drop_duplicates(
            "node_id", keep="first"
        )
        saved_scores = saved.set_index("node_id").within_community_core_score.to_dict()
        saved_ranks = saved.set_index("node_id").core_rank.to_dict()
        fallback = state.loc[
            state.community_id.eq(source_id) & state.node_id.isin(members)
        ].copy()
        fallback = fallback.drop_duplicates("node_id", keep="first").set_index(
            "node_id"
        )
        records: list[dict[str, Any]] = []
        for node in members:
            degree = float(fallback.degree_score.get(node, 0.0))
            records.append(
                {
                    "node_id": str(node),
                    "source_core_rank": int(saved_ranks[node])
                    if node in saved_ranks
                    else np.inf,
                    "source_node_score": float(saved_scores.get(node, degree)),
                    "fallback_degree_score": degree,
                    "saved_core": node in saved_ranks,
                }
            )
        ranked = (
            pd.DataFrame(records)
            .sort_values(
                [
                    "saved_core",
                    "source_core_rank",
                    "fallback_degree_score",
                    "source_node_score",
                    "node_id",
                ],
                ascending=[False, True, False, False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        ranked["source_rank"] = np.arange(1, len(ranked) + 1)
        return ranked

    def future_ranked(source_id: str) -> pd.DataFrame:
        members = set(future_sets[source_id])
        saved = future_core.loc[future_core.future_module_id.eq(source_id)].copy()
        saved = saved.loc[saved.node_id.isin(members)].drop_duplicates(
            "node_id", keep="first"
        )
        stats = future_nodes.loc[future_nodes.future_module_id.eq(source_id)].copy()
        stats = stats.loc[stats.node_id.isin(members)].drop_duplicates(
            "node_id", keep="first"
        )
        merged = (
            pd.DataFrame({"node_id": sorted(members)})
            .merge(
                saved[["node_id", "core_rank", "internal_evidence_score"]],
                on="node_id",
                how="left",
            )
            .merge(
                stats[
                    [
                        "node_id",
                        "mean_internal_evidence_score",
                        "mean_internal_horizontal_degree",
                        "mean_internal_vertical_degree",
                    ]
                ],
                on="node_id",
                how="left",
            )
        )
        merged["source_node_score"] = _numeric(merged, "internal_evidence_score")
        merged.loc[merged.source_node_score.eq(0), "source_node_score"] = _numeric(
            merged, "mean_internal_evidence_score"
        ).loc[merged.source_node_score.eq(0)]
        merged["fallback_degree_score"] = _numeric(
            merged, "mean_internal_horizontal_degree"
        ).clip(lower=0) + _numeric(merged, "mean_internal_vertical_degree").clip(
            lower=0
        )
        merged["saved_core"] = merged.core_rank.notna()
        merged["source_core_rank"] = _numeric(merged, "core_rank", fallback=np.inf)
        ranked = merged.sort_values(
            [
                "saved_core",
                "source_core_rank",
                "source_node_score",
                "fallback_degree_score",
                "node_id",
            ],
            ascending=[False, True, False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        ranked["source_rank"] = np.arange(1, len(ranked) + 1)
        return ranked[
            [
                "node_id",
                "source_rank",
                "source_core_rank",
                "source_node_score",
                "fallback_degree_score",
                "saved_core",
            ]
        ]

    cases: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    source_audits: list[dict[str, Any]] = []
    chosen_ids = marked_ids(
        selection, "future_module_id", "selected_for_figure34", "figure34_display_rank"
    )
    chosen_order = {value: i for (i, value) in enumerate(chosen_ids, 1)}
    numeric = selection.future_module_id.str.extract("(\\d+)", expand=False).astype(int)
    for _, row in (
        selection.assign(_order=numeric).sort_values("_order", kind="stable").iterrows()
    ):
        future_id = str(row.future_module_id)
        formal_id = _normalise_text(row.formal_s_id)
        matched = bool(formal_id)
        case_kind = "matched_observed_to_prediction" if matched else "prediction_only"
        case_rank = chosen_order.get(future_id, np.nan)
        cases.append(
            {
                "future_module_id": future_id,
                "unified_id": str(row.unified_id),
                "formal_s_id": formal_id,
                "comparison_status": str(row.comparison_status),
                "structure_label": _normalise_text(row.structure_label) or future_id,
                "case_kind": case_kind,
                "figure36_recommended_for_main_text": bool(pd.notna(case_rank)),
                "figure36_selection_rank": int(case_rank)
                if pd.notna(case_rank)
                else np.nan,
            }
        )
        sources: list[tuple[str, str, set[str], pd.DataFrame, pd.DataFrame]] = []
        if matched:
            f_rank = formal_ranked(formal_id)
            f_edges = (
                formal_edges.loc[formal_edges.community_id.eq(formal_id)]
                .rename(
                    columns={
                        "start_node_id": "node_1_id",
                        "end_node_id": "node_2_id",
                        "edge_weight": "edge_evidence_weight",
                    }
                )[["node_1_id", "node_2_id", "relation_type", "edge_evidence_weight"]]
                .copy()
            )
            sources.append(
                (
                    TIME.text("Unweighted structure 2001–2025"),
                    formal_id,
                    formal_sets[formal_id],
                    f_rank,
                    f_edges,
                )
            )
        p_rank = future_ranked(future_id)
        p_edges = (
            future_edges.loc[future_edges.future_module_id.eq(future_id)][
                ["node_1_id", "node_2_id", "relation_type", "conditional_edge_evidence"]
            ]
            .rename(columns={"conditional_edge_evidence": "edge_evidence_weight"})
            .copy()
        )
        sources.append(
            (
                TIME.text("Predicted structure 2026–2030"),
                future_id,
                future_sets[future_id],
                p_rank,
                p_edges,
            )
        )
        source_papers = {
            panel: {node: node_pmids.get(node, set()) for node in members}
            for (panel, _, members, _, _) in sources
        }
        source_evidence = {
            panel: set().union(
                *(
                    chord_support_pmids(
                        str(edge.relation_type),
                        str(edge.node_1_id),
                        str(edge.node_2_id),
                    )
                    for edge in edge_part.itertuples(index=False)
                )
            )
            if len(edge_part)
            else set()
            for (panel, _, _, _, edge_part) in sources
        }
        for panel, values in source_papers.items():
            if not source_evidence[panel]:
                source_evidence[panel] = set().union(*values.values())
        result = select_coverage_nodes(
            {
                panel: ranked.node_id.astype(str).tolist()
                for (panel, _, _, ranked, _) in sources
            },
            source_papers,
            source_evidence,
        )
        cases[-1].update(
            displayed_node_count=len(result.nodes),
            minimum_display_nodes=MIN_NODES,
            maximum_display_nodes=MAX_NODES,
        )
        source_selected = {
            panel: sorted(
                (node for node in result.nodes if result.owners[node] == panel),
                key=ranked.set_index("node_id").source_rank.to_dict().get,
            )
            for (panel, _, _, ranked, _) in sources
        }
        source_reasons = {
            (result.owners[node], node): result.reasons[node] for node in result.nodes
        }
        source_ranks = {
            (panel, str(record.node_id)): record
            for (panel, _, _, ranked, _) in sources
            for record in ranked.itertuples(index=False)
        }
        for panel, source_id, _, _, _ in sources:
            audit = result.coverage[panel]
            source_audits.append(
                {
                    "future_module_id": future_id,
                    "unified_id": str(row.unified_id),
                    "source_panel": panel,
                    "source_structure_id": source_id,
                    "source_kind": "formal_unweighted_S"
                    if panel.startswith("Unweighted")
                    else "frozen_prediction",
                    "source_node_total": audit["source_node_count"],
                    "selected_node_count": audit["selected_node_count"],
                    "all_paper_pmid_denominator": audit["all_paper_denominator"],
                    "selected_all_paper_pmid_count": audit["selected_all_papers"],
                    "all_paper_coverage_percent": 100 * audit["all_paper_coverage"],
                    "all_paper_target_percent": 100 * all_target,
                    "all_paper_target_reached": audit["all_paper_target_reached"],
                    "chord_evidence_pmid_denominator": audit[
                        "evidence_paper_denominator"
                    ],
                    "selected_chord_evidence_pmid_count": audit[
                        "selected_evidence_papers"
                    ],
                    "chord_evidence_coverage_percent": 100
                    * audit["evidence_paper_coverage"],
                    "chord_evidence_target_percent": 100 * evidence_target,
                    "chord_evidence_target_reached": audit["evidence_target_reached"],
                    "requested_minimum_node_count": MIN_NODES,
                    "maximum_display_nodes": MAX_NODES,
                    "minimum_node_scope": "complete network",
                    "displayed_node_count": len(result.nodes),
                    "minimum_node_target_reached": len(result.nodes) >= MIN_NODES,
                }
            )
        ordered_union: list[tuple[str, str]] = []
        seen: set[str] = set()
        for panel, *_ in sources:
            for node in source_selected.get(panel, []):
                if node not in seen:
                    ordered_union.append((panel, node))
                    seen.add(node)
        panel_presence = {panel: set(members) for (panel, _, members, *_) in sources}
        panel_lookup = {
            panel: (source_id, ranked) for (panel, source_id, _, ranked, _) in sources
        }
        for number, (assigned_panel, node) in enumerate(ordered_union, start=1):
            (source_id, _) = panel_lookup[assigned_panel]
            ranked_record = source_ranks[assigned_panel, node]
            meta = node_metadata(node)
            node_rows.append(
                {
                    "future_module_id": future_id,
                    "unified_id": str(row.unified_id),
                    "formal_s_id": formal_id,
                    "comparison_status": str(row.comparison_status),
                    "case_kind": case_kind,
                    "figure36_selection_rank": case_rank,
                    "number": number,
                    "node_id": node,
                    "assigned_source_panel": assigned_panel,
                    "assigned_source_structure_id": source_id,
                    "node_name": meta["node_name"],
                    "level": meta["level"],
                    "all_paper_pmid_count": meta["all_paper_pmid_count"],
                    "mean_publication_year": meta["mean_publication_year"],
                    "source_rank": int(ranked_record.source_rank),
                    "source_core_rank": int(ranked_record.source_core_rank)
                    if np.isfinite(ranked_record.source_core_rank)
                    else np.nan,
                    "source_node_score": float(ranked_record.source_node_score),
                    "selection_reason": source_reasons[assigned_panel, node],
                    TIME.key("present_unweighted_2001_2025"): node
                    in panel_presence.get(
                        TIME.text("Unweighted structure 2001–2025"), set()
                    ),
                    TIME.key("present_predicted_2026_2030"): node
                    in panel_presence.get(
                        TIME.text("Predicted structure 2026–2030"), set()
                    ),
                }
            )
        selected_union = {node for (_, node) in ordered_union}
        for panel, source_id, _, _, edge_part in sources:
            subset = edge_part.loc[
                edge_part.node_1_id.astype(str).isin(selected_union)
                & edge_part.node_2_id.astype(str).isin(selected_union)
            ]
            for edge in subset.itertuples(index=False):
                support_years = [
                    year_by_pmid[pmid]
                    for pmid in chord_support_pmids(
                        str(edge.relation_type),
                        str(edge.node_1_id),
                        str(edge.node_2_id),
                    )
                    if pmid in year_by_pmid
                ]
                edge_rows.append(
                    {
                        "future_module_id": future_id,
                        "unified_id": str(row.unified_id),
                        "source_panel": panel,
                        "source_structure_id": source_id,
                        "relation_type": str(edge.relation_type),
                        "node_1_id": str(edge.node_1_id),
                        "node_2_id": str(edge.node_2_id),
                        "edge_evidence_weight": float(edge.edge_evidence_weight),
                        "mean_publication_year": float(np.mean(support_years))
                        if support_years
                        else np.nan,
                        "edge_support_description": "corrected excess PMID quantity from admitted significant chord"
                        if panel.startswith("Unweighted")
                        else "mean predicted evidence z × conditional edge recurrence",
                    }
                )
    case_table = pd.DataFrame(cases)
    node_table = pd.DataFrame(node_rows)
    edge_table = pd.DataFrame(edge_rows)
    audit_table = pd.DataFrame(source_audits)
    if set(case_table.future_module_id) != set(future_sets):
        raise AssertionError("图36必须覆盖全部全部冻结预测模块。")
    assert (
        node_table.groupby("future_module_id")
        .size()
        .between(MIN_NODES, MAX_NODES)
        .all()
    )
    if set(node_table.future_module_id) != set(future_sets):
        raise AssertionError("图36核心节点选择未覆盖全部冻结预测模块。")
    if set(
        audit_table.loc[
            audit_table.source_panel.eq(TIME.text("Predicted structure 2026–2030")),
            "future_module_id",
        ]
    ) != set(future_sets):
        raise AssertionError("图36预测来源审计未覆盖全部冻结预测模块。")
    return (
        case_table,
        node_table,
        edge_table.merge(
            audit_table[
                [
                    "future_module_id",
                    "source_panel",
                    "all_paper_coverage_percent",
                    "chord_evidence_coverage_percent",
                ]
            ],
            on=["future_module_id", "source_panel"],
            how="left",
            validate="many_to_one",
        ).copy(),
        audit_table,
    )


def marked_ids(frame, column, selected, rank):
    return frame[column].astype(str).tolist()
