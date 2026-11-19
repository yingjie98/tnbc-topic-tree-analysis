from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from .result_paths import artifact_directory
import re
from types import SimpleNamespace
from typing import Any
import numpy as np
import pandas as pd
from .topic_tree_analysis import TopicTreeDataset, _select_community_cores

STEMS = {
    "cases": "23_正式S及加权对照核心网络_案例清单",
    "nodes": "23_正式S及加权对照核心网络_节点与来源归属",
    "edges": "23_正式S及加权对照核心网络_显著边与来源归属",
    "coverage": "23_正式S及加权对照核心网络_来源覆盖审计",
    "details": "23_核心主题网络_详情组图来源清单",
}
SOURCE_ORDER = ("S", "JCR", "IF")
from .core_network_selection import (
    MIN_NODES,
    MAX_NODES,
    ALL_PAPER_TARGET,
    EVIDENCE_TARGET,
    select_coverage_nodes,
)

(ALL_TARGET, MAX_DISPLAY_NODES) = (ALL_PAPER_TARGET, MAX_NODES)


def natural(value: Any) -> tuple:
    return tuple(
        (
            int(part) if part.isdigit() else part
            for part in re.split("(\\d+)", str(value))
        )
    )


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"核心网络缺少冻结输入：{path}")
    return pd.read_csv(path, encoding="utf-8-sig").fillna("")


def union(values) -> set[int]:
    result: set[int] = set()
    for value in values:
        result.update(value)
    return result


@dataclass
class Source:
    name: str
    data: Any
    node_papers: dict[str, set[int]]
    state: pd.DataFrame
    chords: pd.DataFrame
    catalog: pd.DataFrame
    core: pd.DataFrame

    def __post_init__(self):
        self.members = {
            str(key): set(part.node_id.astype(str))
            for (key, part) in self.state.groupby("significance_block_id", sort=False)
        }
        self.active: dict[str, set[int]] = defaultdict(set)
        for edge in self.chords.loc[
            self.chords.edge_orientation.eq("horizontal")
        ].itertuples(index=False):
            (a, b) = (str(edge.start_node_id), str(edge.end_node_id))
            support = self.node_papers[a] & self.node_papers[b]
            self.active[a].update(support)
            self.active[b].update(support)
        admitted = self.chords.admitted_to_structure.map(
            truth
        ) & self.chords.community_chord_role.eq("core_chord")
        internal = self.chords.loc[admitted].copy()
        internal["community_id"] = internal.start_community_id.astype(str)
        self.internal = {
            str(key): part.copy()
            for (key, part) in internal.groupby("community_id", sort=False)
        }
        self._supports: dict[tuple, set[int]] = {}
        self._prepared: dict[str, dict] = {}

    def support(self, orientation: str, a: str, b: str) -> set[int]:
        key = (orientation, *sorted((a, b)))
        if key not in self._supports:
            if orientation == "horizontal":
                if self.data.tree[a]["level"] != self.data.tree[b]["level"]:
                    raise AssertionError("冻结横向边跨越了主题层级。")
                result = self.node_papers[a] & self.node_papers[b]
            elif orientation == "vertical":
                if self.data.parent.get(b) == a:
                    parent = a
                elif self.data.parent.get(a) == b:
                    parent = b
                else:
                    raise AssertionError("冻结纵向边不是直接父子边。")
                result = self.active[a] & self.active[b] & self.node_papers[parent]
            else:
                raise ValueError(f"未知边类型：{orientation}")
            self._supports[key] = result
        return self._supports[key]

    def prepare(self, raw_id: str) -> dict:
        if raw_id in self._prepared:
            return self._prepared[raw_id]
        (members, edges) = (self.members[raw_id], self.internal[raw_id])
        saved = self.core.loc[self.core.community_id.astype(str).eq(raw_id)].set_index(
            "node_id"
        )
        scores = {node: 0.0 for node in members}
        for edge in edges.itertuples(index=False):
            strength = max(float(edge.effect_size_z), 0.0)
            scores[str(edge.start_node_id)] += strength
            scores[str(edge.end_node_id)] += strength
        frozen_order = (
            saved.sort_values("core_rank", kind="stable").index.astype(str).tolist()
        )
        remaining = sorted(
            members - set(frozen_order),
            key=lambda node: (
                -scores[node],
                -self.data.counts[node],
                -int(self.data.tree[node]["level"]),
                node,
            ),
        )
        ordered = frozen_order + remaining
        evidence = union(
            (
                self.support(
                    str(e.edge_orientation), str(e.start_node_id), str(e.end_node_id)
                )
                for e in edges.itertuples(index=False)
            )
        )
        all_papers = union((self.node_papers[node] for node in ordered))
        if not all_papers or not evidence:
            raise ValueError(f"{self.name}/{raw_id} 的冻结论文或显著弦证据为空。")
        audit = {
            "source_node_count": len(ordered),
            "all_paper_denominator": len(all_papers),
            "evidence_paper_denominator": len(evidence),
        }
        catalog = self.catalog.loc[
            self.catalog.community_id.astype(str).eq(raw_id)
        ].iloc[0]
        for field, actual in [
            ("all_node_unique_all_paper_pmid_count", audit["all_paper_denominator"]),
            ("internal_chord_unique_evidence_pmid_count", len(evidence)),
        ]:
            if int(catalog[field]) != actual:
                raise AssertionError(
                    f"{self.name}/{raw_id} 的 {field} 与冻结结果不同：{actual} != {catalog[field]}"
                )
        result = {
            "members": members,
            "edges": edges,
            "ordered": ordered,
            "audit": audit,
            "scores": scores,
            "all_papers": all_papers,
            "evidence_papers": evidence,
            "ranks": {node: index + 1 for (index, node) in enumerate(ordered)},
            "catalog": catalog,
        }
        self._prepared[raw_id] = result
        return result


def load_sources(data: TopicTreeDataset, tables: Path) -> dict[str, Source]:
    node_papers = {
        node: set(values.tolist()) for (node, values) in data.node_papers.items()
    }
    graph_ids = [
        node
        for node in data.ids
        if data.counts[node] > 0 and data.tree[node]["level"] >= 2
    ]
    specs = {
        "S": (
            "12_PMID逐层显著性_全部节点状态",
            "16_S社群_显著性筛选弦列表",
            "16_S社群_完整清单",
        ),
        "JCR": (
            "22b_探索性质量加权_JCR_Q_正式候选社群节点",
            "22b_探索性质量加权_JCR_Q_显著弦清单",
            "22b_探索性质量加权_JCR_Q_正式候选社群清单",
        ),
        "IF": (
            "22b_探索性质量加权_IF_year_正式候选社群节点",
            "22b_探索性质量加权_IF_year_显著弦清单",
            "22b_探索性质量加权_IF_year_正式候选社群清单",
        ),
    }
    sources = {}
    for name, (state_stem, chord_stem, catalog_stem) in specs.items():
        state = read(tables / f"{state_stem}.csv")
        state = state.loc[state.significance_block_id.astype(str).ne("")].copy()
        chords = read(tables / f"{chord_stem}.csv")
        catalog = read(tables / f"{catalog_stem}.csv")
        for column in ("start", "end"):
            expected = chords[f"{column}_node_id"].map(data.counts).to_numpy()
            if not np.array_equal(
                expected, pd.to_numeric(chords[f"{column}_node_paper_count"]).to_numpy()
            ):
                raise AssertionError(f"{name} 的论文库与冻结显著边统计不一致。")
        if name == "S":
            core = read(tables / "16_S社群_核心节点清单.csv")
        else:
            (core, _, _) = _select_community_cores(
                data,
                SimpleNamespace(ids=graph_ids),
                state,
                "significance_block_id",
                chords,
            )
        annotated = tables / f"{catalog_stem}_结构注释.csv"
        if annotated.is_file():
            names = read(annotated).set_index("community_id")["structure_label"]
            catalog["structure_label"] = catalog.community_id.map(names)
        sources[name] = Source(name, data, node_papers, state, chords, catalog, core)
    return sources


def make_cases(
    alignment: pd.DataFrame, sources: dict[str, Source]
) -> tuple[pd.DataFrame, dict]:
    slots = {"S": "正式s结构块编号", "JCR": "JCR加权结构编号", "IF": "IF加权结构编号"}
    maps = {"S": {key: key for key in sources["S"].members}}
    for source, prefix in [("JCR", "J"), ("IF", "I")]:
        maps[source] = {
            f"{prefix}{i}": key
            for (i, key) in enumerate(sorted(sources[source].members, key=natural), 1)
        }
    work = alignment.fillna("").copy()
    for source, column in slots.items():
        work[column] = work[column].astype(str).str.strip()
        present = work.loc[work[column].ne(""), column]
        if present.duplicated().any() or set(present) != set(maps[source]):
            raise AssertionError(f"{source} 来源编号不是冻结目录的严格一对一全集。")
    records = []
    formal_rows = work.loc[work[slots["S"]].ne("")].copy()
    ordered = sorted(
        formal_rows.to_dict("records"), key=lambda row: natural(row[slots["S"]])
    )
    for row in ordered:
        sid = row[slots["S"]]
        common = {
            "formal_s_id": sid,
            "unified_id": str(row["整合结构块编号"]),
            "structure_name": str(row.get("U主名称", ""))
            or str(row.get("正式s结构块编号structure_label", sid))
            or sid,
        }
        records.append(
            {
                **common,
                "case_id": sid,
                "case_kind": "formal",
                "weighted_type": "",
                "S": sid,
                "JCR": "",
                "IF": "",
                "source_order": "S",
            }
        )
    for row in ordered:
        (sid, jcr, imp) = (row[slots[source]] for source in SOURCE_ORDER)
        if not jcr and (not imp):
            continue
        records.append(
            {
                "case_id": f"W_{sid}",
                "case_kind": "weighted",
                "formal_s_id": sid,
                "unified_id": str(row["整合结构块编号"]),
                "structure_name": str(row.get("U主名称", ""))
                or str(row.get("正式s结构块编号structure_label", sid))
                or sid,
                "weighted_type": "both"
                if jcr and imp
                else "JCR_only"
                if jcr
                else "IF_only",
                "S": sid,
                "JCR": jcr,
                "IF": imp,
                "source_order": "|".join(
                    (
                        source
                        for (source, value) in zip(SOURCE_ORDER, (sid, jcr, imp))
                        if value
                    )
                ),
            }
        )
    extra = work.loc[
        work[slots["S"]].eq("") & (work[slots["JCR"]].ne("") | work[slots["IF"]].ne(""))
    ]
    for row in sorted(
        extra.to_dict("records"), key=lambda row: natural(row["整合结构块编号"])
    ):
        (jcr, imp) = (row[slots["JCR"]], row[slots["IF"]])
        names = [
            str(row.get(f"{slots[source]}structure_label", ""))
            for (source, value) in [("JCR", jcr), ("IF", imp)]
            if value
        ]
        label = str(row.get("U主名称", "")) or " / ".join(
            dict.fromkeys((name for name in names if name))
        )
        records.append(
            {
                "case_id": f"Q_{row['整合结构块编号']}",
                "case_kind": "weighted_only",
                "formal_s_id": "",
                "unified_id": str(row["整合结构块编号"]),
                "structure_name": label
                or "/".join((value for value in [jcr, imp] if value)),
                "weighted_type": "JCR_IF"
                if jcr and imp
                else "JCR_only"
                if jcr
                else "IF_only",
                "S": "",
                "JCR": jcr,
                "IF": imp,
                "source_order": "|".join(
                    (source for (source, value) in [("JCR", jcr), ("IF", imp)] if value)
                ),
            }
        )
    cases = pd.DataFrame(records)
    cases["display_id"] = cases.apply(
        lambda row: str(row.formal_s_id)
        or "/".join((str(row[source]) for source in ["JCR", "IF"] if str(row[source]))),
        axis=1,
    )
    cases["case_order"] = np.arange(1, len(cases) + 1)
    return (cases, maps)


def build_tables(
    tree: Path, papers: Path, output: Path, alignment: Path
) -> dict[str, pd.DataFrame]:
    data = TopicTreeDataset(tree, papers, None, None)
    sources = load_sources(data, artifact_directory(output / "tables"))
    (cases, maps) = make_cases(read(alignment), sources)
    cases["observed_start_year"] = int(data.years.min())
    cases["observed_end_year"] = int(data.years.max())
    cases["minimum_display_nodes"] = MIN_NODES
    cases["maximum_display_nodes"] = MAX_DISPLAY_NODES
    (nodes, edges, coverage) = ([], [], [])
    for ordinal, case in enumerate(cases.to_dict("records"), 1):
        source_order = case["source_order"].split("|")
        prepared = {
            source: sources[source].prepare(maps[source][case[source]])
            for source in source_order
        }
        result = select_coverage_nodes(
            {source: part["ordered"] for (source, part) in prepared.items()},
            {
                source: {
                    node: sources[source].node_papers[node] for node in part["ordered"]
                }
                for (source, part) in prepared.items()
            },
            {source: part["evidence_papers"] for (source, part) in prepared.items()},
        )
        assignment = result.owners
        selected_union = [
            node
            for source in source_order
            for node in sorted(
                (node for node in result.nodes if assignment[node] == source),
                key=prepared[source]["ranks"].get,
            )
        ]
        selected_set = set(selected_union)
        cases.loc[cases.case_id.eq(case["case_id"]), "displayed_node_count"] = len(
            selected_union
        )
        for source, part in prepared.items():
            visible = selected_set & part["members"]
            horizontal = part["edges"].loc[
                part["edges"].edge_orientation.eq("horizontal")
            ]
            coverage.append(
                {
                    "case_id": case["case_id"],
                    "source": source,
                    "source_display_id": case[source],
                    "source_internal_id": maps[source][case[source]],
                    **result.coverage[source],
                    "all_paper_target": ALL_TARGET,
                    "evidence_paper_target": EVIDENCE_TARGET,
                    "selected_horizontal_edge_count": sum(
                        (
                            str(e.start_node_id) in visible
                            and str(e.end_node_id) in visible
                            for e in horizontal.itertuples(index=False)
                        )
                    ),
                }
            )
        for number, node in enumerate(selected_union, 1):
            pmids = sources["S"].node_papers[node]
            meta = {
                "case_id": case["case_id"],
                "node_id": node,
                "number": number,
                "node_name": str(data.tree[node]["name"]),
                "level": int(data.tree[node]["level"]),
                "assigned_source_panel": assignment[node],
                "paper_count": len(pmids),
                "mean_year": float(np.mean(data.years[list(pmids)])),
                "selection_reason": result.reasons[node],
            }
            for source in SOURCE_ORDER:
                meta[f"present_{source}"] = (
                    source in prepared and node in prepared[source]["members"]
                )
                meta[f"selected_{source}"] = assignment[node] == source
                meta[f"core_rank_{source}"] = (
                    prepared[source]["ranks"].get(node, np.nan)
                    if source in prepared
                    else np.nan
                )
                meta[f"core_score_{source}"] = (
                    prepared[source]["scores"].get(node, np.nan)
                    if source in prepared
                    else np.nan
                )
            nodes.append(meta)
        combined = {}
        for source, part in prepared.items():
            for edge in part["edges"].itertuples(index=False):
                (a, b) = (str(edge.start_node_id), str(edge.end_node_id))
                if a not in selected_set or b not in selected_set:
                    continue
                orientation = str(edge.edge_orientation)
                (a, b) = sorted((a, b))
                key = (orientation, a, b)
                if key not in combined:
                    combined[key] = {
                        "case_id": case["case_id"],
                        "orientation": orientation,
                        "node_1_id": a,
                        "node_2_id": b,
                        "_support": set(),
                        **{f"present_{s}": False for s in SOURCE_ORDER},
                        **{f"evidence_z_{s}": np.nan for s in SOURCE_ORDER},
                    }
                record = combined[key]
                if record[f"present_{source}"]:
                    raise AssertionError("同一来源的同一显著边重复。")
                record[f"present_{source}"] = True
                record[f"evidence_z_{source}"] = float(edge.effect_size_z)
                record["_support"].update(sources[source].support(orientation, a, b))
        for edge in combined.values():
            support = edge.pop("_support")
            edge["evidence_paper_count"] = len(support)
            edge["mean_year"] = (
                float(np.mean(data.years[list(support)])) if support else np.nan
            )
            edge["edge_evidence_weight"] = max(
                (
                    float(edge[f"evidence_z_{s}"])
                    for s in source_order
                    if edge[f"present_{s}"]
                )
            )
            edges.append(edge)
        if ordinal % 25 == 0 or ordinal == len(cases):
            print(f"[核心网络分析] {ordinal}/{len(cases)}", flush=True)
    frames = {
        "cases": cases,
        "nodes": pd.DataFrame(nodes),
        "edges": pd.DataFrame(edges),
        "coverage": pd.DataFrame(coverage),
    }
    output.mkdir(parents=True, exist_ok=True)
    artifact_directory(output / "tables").mkdir(exist_ok=True)
    for key, frame in frames.items():
        frame.to_csv(
            artifact_directory(output / "tables") / f"{STEMS[key]}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return frames
