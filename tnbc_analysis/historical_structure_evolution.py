from __future__ import annotations
from .project_config import TIME
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from .core_network_selection import (
    MIN_NODES,
    MAX_NODES,
    ALL_PAPER_TARGET,
    EVIDENCE_TARGET,
    select_coverage_nodes,
)
import numpy as np
import pandas as pd

SNAPSHOT_ENDS = TIME.snapshot_ends
SNAPSHOT_LABELS = TIME.snapshot_labels
HISTORY_START_YEAR = TIME.history_start


@dataclass(frozen=True)
class Paths:
    project: Path
    history: Path
    main: Path
    output: Path
    tree: Path
    papers: Path

    @property
    def source_tables(self) -> Path:
        return self.history / "tables"

    @property
    def main_tables(self) -> Path:
        return self.main / "tables"

    @property
    def tables(self) -> Path:
        return self.output / "tables"


@dataclass
class Inputs:
    blocks: pd.DataFrame
    members: pd.DataFrame
    annotations: pd.DataFrame
    events: pd.DataFrame
    unified_alignment: pd.DataFrame
    tree: dict[str, dict[str, Any]]
    papers: dict[str, dict[str, Any]]


def natural_key(value: Any) -> tuple[Any, ...]:
    return tuple(
        (
            int(part) if part.isdigit() else part.lower()
            for part in re.split("(\\d+)", str(value))
        )
    )


def parse_year(record: dict[str, Any]) -> int | None:
    for key in ("year", "Year", "publication_year", "pub_year", "PubYear"):
        if (value := record.get(key)) is not None:
            match = re.search("(?:18|19|20)\\d{2}", str(value))
            if match:
                return int(match.group())
    for key in ("PubDate", "publication_date", "date"):
        match = re.search("(?:18|19|20)\\d{2}", str(record.get(key, "")))
        if match:
            return int(match.group())
    return None


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False if pd.isna(value) else bool(value)


def as_number(value: Any, default: float = 0.0) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(number) else float(number)


def write_table(paths: Paths, frame: pd.DataFrame, stem: str) -> Path:
    path = paths.tables / f"{stem}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def structure_uid(snapshot_end: Any, community_id: Any) -> str:
    return f"H{int(float(snapshot_end))}_{str(community_id).strip()}"


def snapshot_end_from_label(value: Any) -> int:
    match = re.search("(20\\d{2})\\s*$", str(value))
    if not match:
        raise ValueError(f"无法从历史快照标签解析截止年份：{value!r}")
    return int(match.group(1))


def pmids_for_nodes(
    node_ids: Iterable[str],
    tree: dict[str, dict[str, Any]],
    paper_year: dict[str, int],
    end_year: int,
) -> set[str]:
    pmids: set[str] = set()
    for node_id in node_ids:
        node = tree.get(str(node_id), {})
        for pmid in node.get("all_paper_pmid", []) or []:
            key = str(pmid).strip()
            year = paper_year.get(key)
            if year is not None and HISTORY_START_YEAR <= year <= int(end_year):
                pmids.add(key)
    return pmids


def prepare_block_state(
    inputs: Inputs,
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    blocks = inputs.blocks.copy()
    blocks["snapshot_end"] = pd.to_numeric(blocks.snapshot_end, errors="raise").astype(
        int
    )
    blocks["snapshot_start"] = pd.to_numeric(
        blocks.snapshot_start, errors="raise"
    ).astype(int)
    blocks["community_id"] = blocks.community_id.astype(str)
    blocks["structure_uid"] = [
        structure_uid(end, cid)
        for (end, cid) in zip(blocks.snapshot_end, blocks.community_id)
    ]
    annotations = inputs.annotations.copy()
    annotations["snapshot_end"] = pd.to_numeric(
        annotations.snapshot_end, errors="raise"
    ).astype(int)
    annotations["community_id"] = annotations.community_id.astype(str)
    annotations = annotations.sort_values(
        ["snapshot_end", "community_id"], kind="stable"
    ).drop_duplicates(["snapshot_end", "community_id"], keep="last")
    annotation_columns = [
        name
        for name in [
            "snapshot_end",
            "community_id",
            "annotation_label",
            "structure_label",
            "display_identity_id",
            "structure_display_name",
            "aligned_identity_id",
            "is_framework_structure",
        ]
        if name in annotations.columns
    ]
    blocks = blocks.merge(
        annotations[annotation_columns],
        on=["snapshot_end", "community_id"],
        how="left",
        suffixes=("", "_annotation"),
    )
    blocks["annotation_label"] = (
        blocks.get("annotation_label", pd.Series(index=blocks.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    blocks["structure_label"] = (
        blocks["structure_label"].fillna("").astype(str).str.strip()
    )
    canonical = blocks.structure_display_name.fillna("").astype(str).str.strip()
    blocks["display_label"] = canonical.where(
        canonical.ne(""),
        blocks.annotation_label.where(
            blocks.annotation_label.ne(""), blocks["structure_label"]
        ),
    )
    blocks["display_label"] = blocks.display_label.where(
        blocks.display_label.ne(""), blocks.community_id
    )
    if (
        blocks.aligned_identity_id.isna().any()
        or blocks.is_framework_structure.isna().any()
    ):
        raise ValueError("历史结构缺少既有身份或独立框架属性。")
    blocks["display_identity_id"] = blocks.aligned_identity_id.astype(str)
    blocks["structure_display_name"] = blocks.display_label
    blocks["is_framework_structure"] = blocks.is_framework_structure.map(as_bool)
    blocks["snapshot_label"] = blocks.snapshot_end.map(SNAPSHOT_LABELS).fillna(
        blocks.snapshot_end.astype(str)
    )
    members = inputs.members.copy()
    members["snapshot_end"] = pd.to_numeric(
        members.snapshot_end, errors="raise"
    ).astype(int)
    members["significance_block_id"] = members.significance_block_id.astype(str)
    members["node_id"] = members.node_id.astype(str)
    node_sets = {
        structure_uid(end, cid): set(part.node_id.astype(str))
        for ((end, cid), part) in members.groupby(
            ["snapshot_end", "significance_block_id"], sort=False
        )
    }
    paper_year = {
        str(pmid).strip(): year
        for (pmid, paper) in inputs.papers.items()
        if (year := parse_year(paper)) is not None
    }
    pmid_sets = {
        str(row.structure_uid): pmids_for_nodes(
            node_sets.get(str(row.structure_uid), set()),
            inputs.tree,
            paper_year,
            int(row.snapshot_end),
        )
        for row in blocks.itertuples(index=False)
    }
    blocks["unique_pmid_count"] = blocks.structure_uid.map(
        lambda uid: len(pmid_sets.get(str(uid), set()))
    ).astype(int)
    blocks["mean_pmid_publication_year"] = blocks.structure_uid.map(
        lambda uid: float(
            np.mean([paper_year[pmid] for pmid in pmid_sets.get(str(uid), set())])
        )
        if pmid_sets.get(str(uid), set())
        else np.nan
    )
    blocks["node_count_from_member_table"] = blocks.structure_uid.map(
        lambda uid: len(node_sets.get(str(uid), set()))
    ).astype(int)
    mismatch = blocks.node_count_from_member_table.ne(
        pd.to_numeric(blocks.node_count, errors="coerce").fillna(-1).astype(int)
    )
    if mismatch.any():
        sample = blocks.loc[
            mismatch, ["structure_uid", "node_count", "node_count_from_member_table"]
        ].head(5)
        raise ValueError(
            "历史结构成员表与结构表的节点数不一致：\n" + sample.to_string(index=False)
        )
    return (blocks, node_sets, pmid_sets)


HISTORICAL_ALIGNMENT_SLOTS = {
    end: f"{slot}结构编号" for (end, slot) in TIME.history_slots.items()
}


def build_unified_history_identity_map(unified_alignment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, values in unified_alignment.iterrows():
        unified_id = str(values["整合结构块编号"]).strip()
        if not unified_id:
            continue
        for end, slot in HISTORICAL_ALIGNMENT_SLOTS.items():
            community_id = str(values.get(slot, "")).strip()
            if community_id:
                rows.append(
                    {
                        "snapshot_end": int(end),
                        "community_id": community_id,
                        "structure_uid": structure_uid(end, community_id),
                        "unified_id": unified_id,
                        "alignment_slot": slot,
                    }
                )
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise ValueError("统一对齐表没有任何历史快照结构，不能建立 U 身份。")
    if mapping.duplicated(["snapshot_end", "community_id"]).any():
        duplicate = mapping.loc[
            mapping.duplicated(["snapshot_end", "community_id"], keep=False)
        ].head(8)
        raise ValueError(
            "统一对齐表违反历史来源内一对一约束：\n" + duplicate.to_string(index=False)
        )
    if mapping.unified_id.eq("").any():
        raise ValueError("历史结构存在空 U 身份。")
    return mapping.sort_values(
        ["snapshot_end", "community_id"],
        key=lambda series: series.map(natural_key)
        if series.name == "community_id"
        else series,
        kind="stable",
    ).reset_index(drop=True)


def _node_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def build_u_lineages(
    blocks: pd.DataFrame,
    node_sets: dict[str, set[str]],
    unified_alignment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    map_frame = build_unified_history_identity_map(unified_alignment)
    mapping = map_frame.set_index("structure_uid").unified_id.to_dict()
    work = blocks.copy()
    work["unified_id"] = work.structure_uid.astype(str).map(mapping)
    missing = work.loc[
        work.unified_id.isna(), ["structure_uid", "snapshot_end", "community_id"]
    ]
    if not missing.empty:
        raise ValueError(
            "历史结构未映射到全局 U 身份：\n" + missing.head(10).to_string(index=False)
        )
    work["unified_id"] = work.unified_id.astype(str)
    if work.duplicated(["snapshot_end", "unified_id"]).any():
        duplicate = work.loc[
            work.duplicated(["snapshot_end", "unified_id"], keep=False),
            ["snapshot_end", "community_id", "unified_id"],
        ]
        raise ValueError(
            "同一历史快照中有多个结构映射到同一 U 身份：\n"
            + duplicate.head(12).to_string(index=False)
        )
    end_order = {end: index for (index, end) in enumerate(SNAPSHOT_ENDS)}
    positions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for unified_id, group in work.groupby("unified_id", sort=False):
        group = group.sort_values("snapshot_end", kind="stable")
        (first, last) = (group.iloc[0], group.iloc[-1])
        ends = group.snapshot_end.astype(int).tolist()
        adjacent_flags = [False] + [
            end_order[right] - end_order[left] == 1
            for (left, right) in zip(ends[:-1], ends[1:])
        ]
        has_gap = any((not flag for flag in adjacent_flags[1:]))
        jaccards: list[float] = []
        changes: list[float] = []
        cumulative = 0.0
        previous = None
        for position, row in enumerate(group.itertuples(index=False), start=1):
            adjacent = bool(adjacent_flags[position - 1])
            if previous is None:
                (jaccard, step_change) = (np.nan, 0.0)
            elif adjacent:
                jaccard = _node_jaccard(
                    node_sets[str(previous.structure_uid)],
                    node_sets[str(row.structure_uid)],
                )
                step_node = abs(
                    math.log1p(float(row.node_count))
                    - math.log1p(float(previous.node_count))
                )
                step_horizontal = abs(
                    math.log1p(float(row.significant_horizontal_relations))
                    - math.log1p(float(previous.significant_horizontal_relations))
                )
                step_pmid = abs(
                    math.log1p(float(row.unique_pmid_count))
                    - math.log1p(float(previous.unique_pmid_count))
                )
                step_change = float(
                    np.mean([step_node, step_horizontal, step_pmid, 1.0 - jaccard])
                )
                cumulative += step_change
                jaccards.append(jaccard)
                changes.append(step_change)
            else:
                (jaccard, step_change) = (np.nan, 0.0)
            positions.append(
                {
                    "structure_uid": str(row.structure_uid),
                    "unified_id": str(unified_id),
                    "lineage_id": str(unified_id),
                    "lineage_position": int(position),
                    "lineage_length": int(len(group)),
                    "first_snapshot_end": int(first.snapshot_end),
                    "last_snapshot_end": int(last.snapshot_end),
                    "adjacent_snapshot_continuation": adjacent,
                    "snapshot_gap_before": bool(
                        previous is not None and (not adjacent)
                    ),
                    "adjacent_node_jaccard_from_previous": jaccard,
                    "step_structural_change_from_previous_snapshot": step_change,
                    "cumulative_structural_change_to_snapshot": cumulative,
                }
            )
            previous = row
        reaches_current = int(last.snapshot_end) == max(SNAPSHOT_ENDS)
        has_reconfiguration = any((value < 0.75 for value in jaccards))
        if reaches_current and has_gap:
            state = "间断持续结构"
        elif reaches_current and len(group) >= 2 and has_reconfiguration:
            state = "持续并重组"
        elif reaches_current and len(group) >= 2:
            state = "持续扩展"
        elif reaches_current:
            state = "后期新近可识别"
        elif has_reconfiguration:
            state = "重组后未延续"
        else:
            state = "未延续结构"
        summaries.append(
            {
                "lineage_id": str(unified_id),
                "unified_id": str(unified_id),
                "lineage_state": state,
                "lineage_length": int(len(group)),
                "first_snapshot_end": int(first.snapshot_end),
                "last_snapshot_end": int(last.snapshot_end),
                "first_snapshot_label": SNAPSHOT_LABELS[int(first.snapshot_end)],
                "last_snapshot_label": SNAPSHOT_LABELS[int(last.snapshot_end)],
                TIME.key("reaches_current_2025"): reaches_current,
                "has_snapshot_gap": has_gap,
                "has_split_or_merge_signal": has_reconfiguration,
                "mean_adjacent_node_jaccard": float(np.mean(jaccards))
                if jaccards
                else np.nan,
                "structural_change_index": float(np.mean(changes)) if changes else 0.0,
                "max_unique_pmid_count": int(group.unique_pmid_count.max()),
                "latest_unique_pmid_count": int(last.unique_pmid_count),
                "max_node_count": int(group.node_count.max()),
                "latest_node_count": int(last.node_count),
                "max_horizontal_relation_count": int(
                    group.significant_horizontal_relations.max()
                ),
                "latest_horizontal_relation_count": int(
                    last.significant_horizontal_relations
                ),
                "representative_structure_uid": str(last.structure_uid),
                "representative_annotation_label": str(last.display_label),
                "display_identity_id": str(last.display_identity_id),
                "structure_display_name": str(last.structure_display_name),
                "is_framework_structure": bool(last.is_framework_structure),
                "top15_node_names_at_latest": str(last.top15_node_names),
                "all_structure_uids": " | ".join(group.structure_uid.astype(str)),
            }
        )
    positions_frame = pd.DataFrame(positions)
    summary_frame = pd.DataFrame(summaries)
    enriched = work.merge(
        positions_frame,
        on=["structure_uid", "unified_id"],
        how="left",
        validate="one_to_one",
    )
    enriched = enriched.merge(
        summary_frame[
            [
                "lineage_id",
                "lineage_state",
                TIME.key("reaches_current_2025"),
                "has_split_or_merge_signal",
            ]
        ],
        on="lineage_id",
        how="left",
        validate="many_to_one",
    )
    if enriched.lineage_id.isna().any():
        raise ValueError("有历史结构未获得 U 谱系身份。")
    return (enriched, summary_frame, map_frame)


def build_lineages(
    blocks: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = events.copy()
    work["from_end"] = work.from_snapshot.map(snapshot_end_from_label)
    work["to_end"] = work.to_snapshot.map(snapshot_end_from_label)
    work["from_uid"] = [
        structure_uid(end, cid) if pd.notna(cid) else ""
        for (end, cid) in zip(work.from_end, work.from_community_id)
    ]
    work["to_uid"] = [
        structure_uid(end, cid) if pd.notna(cid) else ""
        for (end, cid) in zip(work.to_end, work.to_community_id)
    ]
    claimed = work.loc[
        work.event_type.astype(str).eq("best_match")
        & work.from_uid.ne("")
        & work.to_uid.ne("")
    ].copy()
    if claimed.from_uid.duplicated().any() or claimed.to_uid.duplicated().any():
        raise ValueError("历史对齐表违反来源内一对一认领：同一结构被重复认领。")
    successor = claimed.set_index("from_uid").to_uid.to_dict()
    predecessor = claimed.set_index("to_uid").from_uid.to_dict()
    all_uids = set(blocks.structure_uid.astype(str))
    unknown = (set(successor) | set(successor.values())) - all_uids
    if unknown:
        raise ValueError(f"历史对齐表引用了不存在的结构：{sorted(unknown)[:5]}")
    roots = sorted((uid for uid in all_uids if uid not in predecessor), key=natural_key)
    chains: list[list[str]] = []
    visited: set[str] = set()
    for root in roots:
        chain: list[str] = []
        current = root
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            current = successor.get(current, "")
        chains.append(chain)
    for orphan in sorted(all_uids - visited, key=natural_key):
        chains.append([orphan])
    block_lookup = blocks.set_index("structure_uid", drop=False)
    chains.sort(
        key=lambda chain: (
            -max((int(block_lookup.loc[uid, "unique_pmid_count"]) for uid in chain)),
            min((int(block_lookup.loc[uid, "snapshot_end"]) for uid in chain)),
            natural_key(chain[0]),
        )
    )
    edge_lookup = {
        (str(row.from_uid), str(row.to_uid)): row
        for row in claimed.itertuples(index=False)
    }
    positions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for number, chain in enumerate(chains, start=1):
        lineage_id = f"HL{number:03d}"
        chain_rows = block_lookup.loc[chain].sort_values("snapshot_end", kind="stable")
        edges = [
            edge_lookup[left, right] for (left, right) in zip(chain[:-1], chain[1:])
        ]
        first = chain_rows.iloc[0]
        last = chain_rows.iloc[-1]
        reaches_current = int(last.snapshot_end) == max(SNAPSHOT_ENDS)
        has_reconfiguration = any(
            (
                as_bool(getattr(edge, "split_signal", False))
                or as_bool(getattr(edge, "merge_signal", False))
                for edge in edges
            )
        )
        mean_jaccard = (
            float(
                np.mean(
                    [as_number(getattr(edge, "node_jaccard", 0.0)) for edge in edges]
                )
            )
            if edges
            else np.nan
        )
        node_change = abs(
            math.log1p(float(last.node_count)) - math.log1p(float(first.node_count))
        )
        horizontal_change = abs(
            math.log1p(float(last.significant_horizontal_relations))
            - math.log1p(float(first.significant_horizontal_relations))
        )
        pmid_change = abs(
            math.log1p(float(last.unique_pmid_count))
            - math.log1p(float(first.unique_pmid_count))
        )
        member_reconfiguration = (
            0.0
            if not edges
            else max(0.0, 1.0 - float(np.nan_to_num(mean_jaccard, nan=0.0)))
        )
        change_index = float(
            np.mean(
                [node_change, horizontal_change, pmid_change, member_reconfiguration]
            )
        )
        if reaches_current and has_reconfiguration:
            state = "持续并重组"
        elif reaches_current and len(chain_rows) >= 2:
            state = "持续扩展"
        elif reaches_current:
            state = "后期新近可识别"
        elif has_reconfiguration:
            state = "重组后未延续"
        else:
            state = "未延续结构"
        cumulative_change = 0.0
        previous_row = None
        for order, row in enumerate(chain_rows.itertuples(index=False), start=1):
            if previous_row is None:
                step_change = 0.0
            else:
                edge = edge_lookup[
                    str(previous_row.structure_uid), str(row.structure_uid)
                ]
                step_node_change = abs(
                    math.log1p(float(row.node_count))
                    - math.log1p(float(previous_row.node_count))
                )
                step_horizontal_change = abs(
                    math.log1p(float(row.significant_horizontal_relations))
                    - math.log1p(float(previous_row.significant_horizontal_relations))
                )
                step_pmid_change = abs(
                    math.log1p(float(row.unique_pmid_count))
                    - math.log1p(float(previous_row.unique_pmid_count))
                )
                step_member_reconfiguration = max(
                    0.0, 1.0 - as_number(getattr(edge, "node_jaccard", 0.0))
                )
                step_change = float(
                    np.mean(
                        [
                            step_node_change,
                            step_horizontal_change,
                            step_pmid_change,
                            step_member_reconfiguration,
                        ]
                    )
                )
                cumulative_change += step_change
            positions.append(
                {
                    "lineage_id": lineage_id,
                    "lineage_position": order,
                    "lineage_length": len(chain_rows),
                    "lineage_state": state,
                    TIME.key("reaches_current_2025"): reaches_current,
                    "has_split_or_merge_signal": has_reconfiguration,
                    "mean_adjacent_node_jaccard": mean_jaccard,
                    "structural_change_index": change_index,
                    "step_structural_change_from_previous_snapshot": step_change,
                    "cumulative_structural_change_to_snapshot": cumulative_change,
                    "first_snapshot_end": int(first.snapshot_end),
                    "last_snapshot_end": int(last.snapshot_end),
                    "structure_uid": str(row.structure_uid),
                }
            )
            previous_row = row
        summaries.append(
            {
                "lineage_id": lineage_id,
                "lineage_state": state,
                "lineage_length": int(len(chain_rows)),
                "first_snapshot_end": int(first.snapshot_end),
                "last_snapshot_end": int(last.snapshot_end),
                "first_snapshot_label": SNAPSHOT_LABELS[int(first.snapshot_end)],
                "last_snapshot_label": SNAPSHOT_LABELS[int(last.snapshot_end)],
                TIME.key("reaches_current_2025"): reaches_current,
                "has_split_or_merge_signal": has_reconfiguration,
                "mean_adjacent_node_jaccard": mean_jaccard,
                "structural_change_index": change_index,
                "max_unique_pmid_count": int(chain_rows.unique_pmid_count.max()),
                "latest_unique_pmid_count": int(last.unique_pmid_count),
                "max_node_count": int(chain_rows.node_count.max()),
                "latest_node_count": int(last.node_count),
                "max_horizontal_relation_count": int(
                    chain_rows.significant_horizontal_relations.max()
                ),
                "latest_horizontal_relation_count": int(
                    last.significant_horizontal_relations
                ),
                "representative_structure_uid": str(last.structure_uid),
                "representative_annotation_label": str(last.display_label),
                "display_identity_id": str(last.display_identity_id),
                "structure_display_name": str(last.structure_display_name),
                "is_framework_structure": bool(last.is_framework_structure),
                "top15_node_names_at_latest": str(last.top15_node_names),
                "all_structure_uids": " | ".join(chain),
            }
        )
    position_frame = pd.DataFrame(positions)
    summary_frame = pd.DataFrame(summaries)
    enriched_blocks = blocks.merge(
        position_frame, on="structure_uid", how="left", validate="one_to_one"
    )
    if enriched_blocks.lineage_id.isna().any():
        raise ValueError("有历史结构未被纳入谱系。")
    enhanced_events = work.merge(
        position_frame[["structure_uid", "lineage_id"]].rename(
            columns={"structure_uid": "from_uid", "lineage_id": "from_lineage_id"}
        ),
        on="from_uid",
        how="left",
    ).merge(
        position_frame[["structure_uid", "lineage_id"]].rename(
            columns={"structure_uid": "to_uid", "lineage_id": "to_lineage_id"}
        ),
        on="to_uid",
        how="left",
    )
    enhanced_events["same_lineage_continuation"] = (
        enhanced_events.from_lineage_id.eq(enhanced_events.to_lineage_id)
        & enhanced_events.from_lineage_id.notna()
    )
    return (enriched_blocks, summary_frame, enhanced_events)


def make_inventory(
    blocks: pd.DataFrame, pmid_sets: dict[str, set[str]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for end, group in blocks.groupby("snapshot_end", sort=True):
        ids = group.structure_uid.astype(str).tolist()
        union = (
            set().union(*(pmid_sets.get(uid, set()) for uid in ids)) if ids else set()
        )
        nonframework = group.loc[~group.is_framework_structure]
        rows.append(
            {
                "snapshot_end": int(end),
                "snapshot_label": SNAPSHOT_LABELS[int(end)],
                "formal_structure_count": int(len(group)),
                "nonframework_structure_count": int(len(nonframework)),
                "lineage_count_present": int(group.lineage_id.nunique()),
                "unique_union_pmid_count": int(len(union)),
                "total_node_memberships": int(group.node_count.sum()),
                "total_horizontal_relations": int(
                    group.significant_horizontal_relations.sum()
                ),
                "total_vertical_relations": int(
                    group.significant_direct_parent_child_relations.sum()
                ),
                "newly_detectable_structure_count": int(
                    (
                        group.lineage_position.eq(1)
                        & group.snapshot_end.ne(min(SNAPSHOT_ENDS))
                    ).sum()
                ),
                "continued_from_previous_count": int(
                    group.lineage_position.gt(1).sum()
                ),
                TIME.key("reaches_2025_count"): int(
                    getattr(group, TIME.key("reaches_current_2025")).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("snapshot_end", kind="stable")


def transition_summary(events: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "best_match": "继承",
        "birth_or_unmatched": "新近可识别或未匹配",
        "death_or_unmatched": "未延续或未匹配",
    }
    work = events.copy()
    work["event_label"] = (
        work.event_type.astype(str).map(labels).fillna(work.event_type.astype(str))
    )
    result = (
        work.groupby(
            ["from_snapshot", "to_snapshot", "event_type", "event_label"], dropna=False
        )
        .size()
        .rename("event_count")
        .reset_index()
    )
    (split_counts, merge_counts) = ([], [])
    for row in result.itertuples(index=False):
        part = work.loc[
            work.from_snapshot.eq(row.from_snapshot)
            & work.to_snapshot.eq(row.to_snapshot)
            & work.event_type.eq(row.event_type)
        ]
        split_counts.append(int(part.split_signal.map(as_bool).sum()))
        merge_counts.append(int(part.merge_signal.map(as_bool).sum()))
    result["with_split_signal_count"] = split_counts
    result["with_merge_signal_count"] = merge_counts
    return result


FIGURE30_STEMS = {
    "figure30_candidates": "17_图30核心网络_U身份筛选审计",
    "figure30_nodes": "18_图30核心网络_节点名单与跨期存在",
    "figure30_quotas": "19_图30核心网络_快照名额分配",
    "figure30_edges": "20_图30核心网络_去重显著弦",
    "figure30_evidence": "21_图30核心网络_逐快照弦证据",
    "figure30_bridges": "22_图30核心网络_相邻快照交集审计",
    "figure30_core_turnover": "23_图30相邻快照前5核心节点更替审计",
}
FIGURE30_CORE_COMPARISON_K = 5
FIGURE30_ALL_PAPER_COVERAGE_TARGET = ALL_PAPER_TARGET
FIGURE30_CHORD_EVIDENCE_COVERAGE_TARGET = EVIDENCE_TARGET


def _figure30_snapshot_coverage_inputs(
    paths: Paths, inputs: Inputs, blocks: pd.DataFrame, node_sets: dict[str, set[str]]
) -> tuple[
    dict[tuple[int, str], set[str]],
    dict[tuple[int, str], set[str]],
    dict[tuple[int, str], set[str]],
]:
    from .topic_tree_analysis import (
        TopicTreeDataset,
        build_pmid_hypergraph,
        levelwise_pmid_significance,
        significant_vertical_evidence,
    )
    import gc

    members_by_snapshot: dict[tuple[int, str], set[str]] = {}
    for row in blocks.itertuples(index=False):
        key = (int(row.snapshot_end), str(row.community_id))
        members_by_snapshot[key] = set(node_sets[str(row.structure_uid)])
    node_pmids: dict[tuple[int, str], set[str]] = {}
    evidence_pmids: dict[tuple[int, str], set[str]] = {
        key: set() for key in members_by_snapshot
    }
    for end in SNAPSHOT_ENDS:
        print(
            f"[图30节点选择] 截至{end}年：重建历史结构的节点和显著弦 PMID 覆盖",
            flush=True,
        )
        data = TopicTreeDataset(paths.tree, paths.papers, None, end)
        graph = build_pmid_hypergraph(data)
        calibration = levelwise_pmid_significance(data, graph)
        vertical = significant_vertical_evidence(
            data, graph, calibration["active_sources"]
        )
        index_by_node = {node: index for (index, node) in enumerate(graph.ids)}
        owner_by_index: dict[int, str] = {}
        for (snapshot_end, community_id), members in members_by_snapshot.items():
            if snapshot_end != end:
                continue
            for node in members:
                if node not in index_by_node:
                    node_pmids[end, node] = set()
                    continue
                position = index_by_node[node]
                if position in owner_by_index:
                    raise AssertionError(f"{end}年历史结构成员重叠：{node}")
                owner_by_index[position] = community_id
                node_pmids[end, node] = {
                    str(data.pmids[index])
                    for index in data.node_papers[node]
                    if HISTORY_START_YEAR <= int(data.years[index]) <= end
                }
        cached_papers: dict[int, set[int]] = {}

        def paper_indices(
            position: int, cache=cached_papers, state=data, nodes=graph.ids
        ) -> set[int]:
            if position not in cache:
                cache[position] = set(state.node_papers[nodes[position]].tolist())
            return cache[position]

        specifications = [
            (
                calibration["horizontal"],
                "significant_horizontal_relation",
                "node_1_index",
                "node_2_index",
                "horizontal",
            ),
            (
                vertical,
                "significant_vertical_relation",
                "parent_index",
                "child_index",
                "vertical",
            ),
        ]
        for (
            table,
            significant_column,
            left_column,
            right_column,
            orientation,
        ) in specifications:
            for row in table.loc[table[significant_column].astype(bool)].itertuples(
                index=False
            ):
                (left, right) = (
                    int(getattr(row, left_column)),
                    int(getattr(row, right_column)),
                )
                community_id = owner_by_index.get(left)
                if not community_id or community_id != owner_by_index.get(right):
                    continue
                if orientation == "horizontal":
                    support = paper_indices(left) & paper_indices(right)
                    expected = int(row.independent_pmid_support)
                else:
                    support = (
                        calibration["active_sources"][left]
                        & calibration["active_sources"][right]
                        & paper_indices(left)
                    )
                    expected = int(row.effective_pmid_support)
                if len(support) != expected:
                    raise AssertionError(
                        "图30覆盖选择：重建弦 PMID 与冻结显著性检验不一致"
                    )
                evidence_pmids[end, community_id].update(
                    (
                        str(data.pmids[index])
                        for index in support
                        if HISTORY_START_YEAR <= int(data.years[index]) <= end
                    )
                )
        del (
            paper_indices,
            data,
            graph,
            calibration,
            vertical,
            index_by_node,
            owner_by_index,
            cached_papers,
            specifications,
            table,
        )
        gc.collect()
    return (members_by_snapshot, node_pmids, evidence_pmids)


def _figure30_candidate_nodes(
    paths: Paths,
    inputs: Inputs,
    blocks: pd.DataFrame,
    lineages: pd.DataFrame,
    node_sets: dict[str, set[str]],
) -> dict[str, pd.DataFrame]:
    (snapshot_members, snapshot_node_pmids, snapshot_evidence_pmids) = (
        _figure30_snapshot_coverage_inputs(paths, inputs, blocks, node_sets)
    )
    ranks: dict[tuple[int, str], list[str]] = {}
    for (end, cid), part in inputs.members.groupby(
        ["snapshot_end", "significance_block_id"], sort=False
    ):
        ranked_part = part.copy()
        ranked_part["structural_centrality"] = pd.to_numeric(
            ranked_part.significant_horizontal_degree, errors="raise"
        ) + pd.to_numeric(ranked_part.significant_vertical_degree, errors="raise")
        ranks[int(end), str(cid)] = (
            ranked_part.sort_values(
                [
                    "structural_centrality",
                    "significant_horizontal_degree",
                    "significant_vertical_degree",
                    "level",
                    "node_id",
                ],
                ascending=[False, False, False, False, True],
                kind="stable",
            )
            .node_id.astype(str)
            .tolist()
        )
    years = {
        str(p): y
        for (p, record) in inputs.papers.items()
        if (y := parse_year(record)) is not None
    }
    (node_rows, quota_rows, bridge_rows, core_turnover_rows, candidates) = (
        [],
        [],
        [],
        [],
        [],
    )
    for lineage in lineages.itertuples(index=False):
        uid = str(lineage.unified_id)
        part = blocks.loc[blocks.unified_id.astype(str).eq(uid)].sort_values(
            "snapshot_end"
        )
        candidate = {
            "unified_id": uid,
            "display_label": str(part.display_label.iloc[-1]),
            "display_identity_id": str(part.display_identity_id.iloc[-1]),
            "structure_display_name": str(part.structure_display_name.iloc[-1]),
            "snapshot_count": len(part),
            "max_unique_pmid_count": int(part.unique_pmid_count.max()),
            "cumulative_structural_change": float(
                part.cumulative_structural_change_to_snapshot.iloc[-1]
            ),
            "adjacent_core_transition_count": 0,
            "top5_core_cumulative_turnover": np.nan,
            "top5_core_min_turnover": np.nan,
            "top5_core_mean_turnover": np.nan,
            "top5_core_max_turnover": np.nan,
            "top5_core_mean_jaccard": np.nan,
            "latest_evidence_snapshot_end": np.nan,
            "latest_all_unique_pmid_count": np.nan,
            "latest_corrected_chord_evidence_pmid_count": np.nan,
            "latest_evidence_pmid_gap": np.nan,
            "latest_evidence_pmid_coverage": np.nan,
            "latest_chord_evidence_strength": np.nan,
            "latest_horizontal_relation_count": np.nan,
            "latest_vertical_relation_count": np.nan,
            "latest_horizontal_relation_share": np.nan,
            "selected": False,
            "selection_rank": np.nan,
            "selection_note": "",
        }
        if len(part) < 2:
            candidate.update(eligible=False, selection_note="仅在一个历史快照中出现。")
            candidates.append(candidate)
            continue
        records = part.to_dict("records")
        ranked = [
            ranks[int(r["snapshot_end"]), str(r["community_id"])] for r in records
        ]
        turnovers: list[float] = []
        for index, (left, right) in enumerate(zip(records[:-1], records[1:])):
            (left_end, right_end) = (
                int(left["snapshot_end"]),
                int(right["snapshot_end"]),
            )
            if SNAPSHOT_ENDS.index(right_end) - SNAPSHOT_ENDS.index(left_end) != 1:
                continue
            left_core = ranked[index][:FIGURE30_CORE_COMPARISON_K]
            right_core = ranked[index + 1][:FIGURE30_CORE_COMPARISON_K]
            (left_set, right_set) = (set(left_core), set(right_core))
            jaccard = _node_jaccard(left_set, right_set)
            turnover = 1.0 - jaccard
            turnovers.append(turnover)
            core_turnover_rows.append(
                {
                    "unified_id": uid,
                    "from_snapshot_end": left_end,
                    "to_snapshot_end": right_end,
                    "from_snapshot_label": left["snapshot_label"],
                    "to_snapshot_label": right["snapshot_label"],
                    "core_rank_cutoff": FIGURE30_CORE_COMPARISON_K,
                    "from_top_core_node_ids": " | ".join(left_core),
                    "from_top_core_node_names": " | ".join(
                        (str(inputs.tree[node]["name"]) for node in left_core)
                    ),
                    "to_top_core_node_ids": " | ".join(right_core),
                    "to_top_core_node_names": " | ".join(
                        (str(inputs.tree[node]["name"]) for node in right_core)
                    ),
                    "shared_top_core_node_count": len(left_set & right_set),
                    "top5_core_jaccard": jaccard,
                    "top5_core_turnover": turnover,
                }
            )
        if turnovers:
            candidate.update(
                adjacent_core_transition_count=len(turnovers),
                top5_core_cumulative_turnover=float(np.sum(turnovers)),
                top5_core_min_turnover=float(np.min(turnovers)),
                top5_core_mean_turnover=float(np.mean(turnovers)),
                top5_core_max_turnover=float(np.max(turnovers)),
                top5_core_mean_jaccard=float(1.0 - np.mean(turnovers)),
            )
        member_sets = [
            snapshot_members[int(r["snapshot_end"]), str(r["community_id"])]
            for r in records
        ]
        paper_denominators = [
            set().union(
                *(
                    snapshot_node_pmids.get((int(r["snapshot_end"]), node), set())
                    for node in members
                )
            )
            if members
            else set()
            for (r, members) in zip(records, member_sets)
        ]
        evidence_denominators = [
            snapshot_evidence_pmids[int(r["snapshot_end"]), str(r["community_id"])]
            for r in records
        ]
        latest_record = records[-1]
        latest_all_pmids = paper_denominators[-1]
        latest_evidence_pmids = evidence_denominators[-1]
        latest_horizontal = int(latest_record["significant_horizontal_relations"])
        latest_vertical = int(
            latest_record["significant_direct_parent_child_relations"]
        )
        latest_total_relations = latest_horizontal + latest_vertical
        candidate.update(
            latest_evidence_snapshot_end=int(latest_record["snapshot_end"]),
            latest_all_unique_pmid_count=len(latest_all_pmids),
            latest_corrected_chord_evidence_pmid_count=len(latest_evidence_pmids),
            latest_evidence_pmid_gap=max(
                len(latest_all_pmids) - len(latest_evidence_pmids), 0
            ),
            latest_evidence_pmid_coverage=len(latest_evidence_pmids)
            / max(len(latest_all_pmids), 1),
            latest_chord_evidence_strength=len(latest_evidence_pmids) ** 2
            / max(len(latest_all_pmids), 1),
            latest_horizontal_relation_count=latest_horizontal,
            latest_vertical_relation_count=latest_vertical,
            latest_horizontal_relation_share=latest_horizontal
            / max(latest_total_relations, 1),
        )
        selection = select_coverage_nodes(
            {index: order for (index, order) in enumerate(ranked)},
            {
                index: {
                    node: snapshot_node_pmids.get(
                        (int(record["snapshot_end"]), node), set()
                    )
                    for node in ranked[index]
                }
                for (index, record) in enumerate(records)
            },
            {index: values for (index, values) in enumerate(evidence_denominators)},
        )
        chosen = {
            node: (selection.owners[node], selection.reasons[node])
            for node in selection.nodes
        }
        assigned = [
            [node for node in selection.nodes if selection.owners[node] == index]
            for index in range(len(records))
        ]
        covered_papers = [
            set().union(
                *(
                    snapshot_node_pmids.get((int(record["snapshot_end"]), node), set())
                    for node in selection.nodes
                    if node in member_sets[index]
                )
            )
            for (index, record) in enumerate(records)
        ]
        covered_evidence = [
            covered_papers[index] & evidence_denominators[index]
            for index in range(len(records))
        ]

        def coverage(index: int) -> tuple[float, float]:
            audit = selection.coverage[index]
            return (audit["all_paper_coverage"], audit["evidence_paper_coverage"])

        def is_complete(index: int) -> bool:
            audit = selection.coverage[index]
            return (
                audit["all_paper_target_reached"] and audit["evidence_target_reached"]
            )

        for index, record in enumerate(records):
            assigned[index].sort(key=ranked[index].index)
            (all_coverage, evidence_coverage) = coverage(index)
            quota_rows.append(
                {
                    "unified_id": uid,
                    "snapshot_end": int(record["snapshot_end"]),
                    "community_id": record["community_id"],
                    "snapshot_label": record["snapshot_label"],
                    "independent_chord_pmid_count": len(evidence_denominators[index]),
                    "requested_quota": np.nan,
                    "assigned_quota": len(assigned[index]),
                    "quota_weight_definition": "共用覆盖率选择器；整图最少3、最多15个不同节点",
                    "minimum_display_nodes": MIN_NODES,
                    "maximum_display_nodes": MAX_NODES,
                    "displayed_node_count": len(chosen),
                    "all_paper_pmid_denominator": len(paper_denominators[index]),
                    "selected_all_paper_pmid_count": len(covered_papers[index]),
                    "all_paper_coverage_percent": 100 * all_coverage,
                    "all_paper_target_percent": 100
                    * FIGURE30_ALL_PAPER_COVERAGE_TARGET,
                    "all_paper_target_reached": all_coverage
                    >= FIGURE30_ALL_PAPER_COVERAGE_TARGET,
                    "chord_evidence_pmid_denominator": len(
                        evidence_denominators[index]
                    ),
                    "selected_chord_evidence_pmid_count": len(covered_evidence[index]),
                    "chord_evidence_coverage_percent": 100 * evidence_coverage,
                    "chord_evidence_target_percent": 100
                    * FIGURE30_CHORD_EVIDENCE_COVERAGE_TARGET,
                    "chord_evidence_target_reached": evidence_coverage
                    >= FIGURE30_CHORD_EVIDENCE_COVERAGE_TARGET,
                }
            )
        number = 0
        for index, group in enumerate(assigned):
            for node in group:
                number += 1
                pmids = {
                    str(p).strip()
                    for p in inputs.tree[node].get("all_paper_pmid", [])
                    if HISTORY_START_YEAR
                    <= years.get(str(p).strip(), 0)
                    <= max(SNAPSHOT_ENDS)
                }
                record = {
                    "unified_id": uid,
                    "number": number,
                    "node_id": node,
                    "node_name": inputs.tree[node]["name"],
                    "level": int(inputs.tree[node]["level"]),
                    "selected_snapshot_end": int(records[index]["snapshot_end"]),
                    "source_community_id": records[index]["community_id"],
                    "saved_core_rank": ranked[index].index(node) + 1,
                    "selection_reason": chosen[node][1],
                    TIME.key("paper_count_2007_2025"): len(pmids),
                    "mean_publication_year": float(np.mean([years[p] for p in pmids]))
                    if pmids
                    else np.nan,
                    "paper_pmids": " | ".join(sorted(pmids)),
                }
                for end in SNAPSHOT_ENDS:
                    source = next(
                        (r for r in records if int(r["snapshot_end"]) == end), None
                    )
                    record[f"present_{end}"] = (
                        source is not None
                        and node in node_sets[str(source["structure_uid"])]
                    )
                node_rows.append(record)
        for left, right in zip(records[:-1], records[1:]):
            if (
                SNAPSHOT_ENDS.index(int(right["snapshot_end"]))
                - SNAPSHOT_ENDS.index(int(left["snapshot_end"]))
                != 1
            ):
                continue
            shared = (
                node_sets[str(left["structure_uid"])]
                & node_sets[str(right["structure_uid"])]
            )
            retained = shared.intersection(chosen)
            bridge_rows.append(
                {
                    "unified_id": uid,
                    "from_end": int(left["snapshot_end"]),
                    "to_end": int(right["snapshot_end"]),
                    "intersection_count": len(shared),
                    "selected_intersection_count": len(retained),
                    "selected_intersection_node_ids": " | ".join(sorted(retained)),
                    "anchor_constraint_satisfied": not shared or bool(retained),
                }
            )
        candidate.update(
            eligible=True,
            core_node_selection_complete=bool(
                all((is_complete(index) for index in range(len(records))))
            ),
            coverage_targets_all_satisfied=bool(
                all((is_complete(index) for index in range(len(records))))
            ),
            continuity_anchor_all_satisfied=bool(
                all(
                    (
                        r["anchor_constraint_satisfied"]
                        for r in bridge_rows
                        if r["unified_id"] == uid
                    )
                )
            ),
            selected_node_count=len(chosen),
            minimum_display_nodes=MIN_NODES,
            maximum_display_nodes=MAX_NODES,
            represented_snapshot_count=sum((bool(a) for a in assigned)),
            selection_note="所有历史快照均达到全部论文75%与弦证据论文90%的覆盖目标。"
            if all((is_complete(index) for index in range(len(records))))
            else "实际覆盖率见逐快照审计表。",
        )
        candidates.append(candidate)
    return {
        "figure30_candidates": pd.DataFrame(candidates),
        "figure30_nodes": pd.DataFrame(node_rows),
        "figure30_quotas": pd.DataFrame(quota_rows),
        "figure30_bridges": pd.DataFrame(bridge_rows),
        "figure30_core_turnover": pd.DataFrame(core_turnover_rows),
    }


def _figure30_edge_evidence(
    paths: Paths,
    blocks: pd.DataFrame,
    node_sets: dict[str, set[str]],
    candidate_nodes: pd.DataFrame,
) -> pd.DataFrame:
    from .topic_tree_analysis import (
        TopicTreeDataset,
        build_pmid_hypergraph,
        levelwise_pmid_significance,
        significant_vertical_evidence,
    )
    import gc

    selected = {
        str(uid): set(part.node_id.astype(str))
        for (uid, part) in candidate_nodes.groupby("unified_id")
    }
    evidence = []
    for end in SNAPSHOT_ENDS:
        print(
            f"[图30] 截至{end}年：复用全快照显著边检验，固定历史结构和核心排名",
            flush=True,
        )
        data = TopicTreeDataset(paths.tree, paths.papers, None, end)
        graph = build_pmid_hypergraph(data)
        calibration = levelwise_pmid_significance(data, graph)
        vertical = significant_vertical_evidence(
            data, graph, calibration["active_sources"]
        )
        node_index = {node: i for (i, node) in enumerate(graph.ids)}
        ownership: dict[tuple[int, int], list[str]] = {}
        for row in blocks.loc[blocks.snapshot_end.eq(end)].itertuples(index=False):
            uid = str(row.unified_id)
            members = selected.get(uid, set()) & node_sets[str(row.structure_uid)]
            indices = sorted((node_index[n] for n in members if n in node_index))
            for i, a in enumerate(indices):
                for b in indices[i + 1 :]:
                    ownership.setdefault((a, b), []).append(uid)
        node_pmids: dict[int, set[int]] = {}

        def paper_set(
            i: int, cache=node_pmids, state=data, nodes=graph.ids
        ) -> set[int]:
            if i not in cache:
                cache[i] = set(state.node_papers[nodes[i]].tolist())
            return cache[i]

        specs = [
            (
                calibration["horizontal"],
                "horizontal",
                "significant_horizontal_relation",
                "node_1_index",
                "node_2_index",
                "q_BY_within_level",
                "p_configuration_approx",
            ),
            (
                vertical,
                "vertical",
                "significant_vertical_relation",
                "parent_index",
                "child_index",
                "q_BY_all_direct_parent_child_tests",
                "p_tree_constrained_hypergeom",
            ),
        ]
        for (
            table,
            orientation,
            significant,
            a_field,
            b_field,
            q_field,
            p_field,
        ) in specs:
            for row in table.loc[table[significant].astype(bool)].itertuples(
                index=False
            ):
                (a, b) = (int(getattr(row, a_field)), int(getattr(row, b_field)))
                owners = ownership.get(tuple(sorted((a, b))), [])
                if not owners:
                    continue
                if orientation == "horizontal":
                    support = paper_set(a) & paper_set(b)
                    corrected = float(row.observed_fractional_pair_mass)
                    expected_count = int(row.independent_pmid_support)
                else:
                    support = (
                        calibration["active_sources"][a]
                        & calibration["active_sources"][b]
                        & paper_set(a)
                    )
                    corrected = float(row.effective_pmid_support)
                    expected_count = int(row.effective_pmid_support)
                if len(support) != expected_count:
                    raise AssertionError("重建弦的支持 PMID 与原检验计数不一致")
                pmids = sorted(
                    (
                        str(data.pmids[i])
                        for i in support
                        if HISTORY_START_YEAR
                        <= int(data.years[i])
                        <= max(SNAPSHOT_ENDS)
                    )
                )
                for uid in owners:
                    evidence.append(
                        {
                            "unified_id": uid,
                            "snapshot_end": end,
                            "orientation": orientation,
                            "start_node_id": graph.ids[a],
                            "end_node_id": graph.ids[b],
                            "independent_support_count_original": len(support),
                            TIME.key("support_count_2007_2025"): len(pmids),
                            "corrected_fractional_pmid_mass": corrected,
                            "effect_size_z": float(row.evidence_z),
                            "q_value": float(getattr(row, q_field)),
                            "p_value": float(getattr(row, p_field)),
                            "support_pmids": " | ".join(pmids),
                        }
                    )
        del (
            paper_set,
            data,
            graph,
            calibration,
            vertical,
            node_pmids,
            ownership,
            specs,
            table,
        )
        gc.collect()
    return pd.DataFrame(evidence)


def _figure30_union_edges(evidence: pd.DataFrame, inputs: Inputs) -> pd.DataFrame:
    rows = []
    years = {
        str(p): y
        for (p, r) in inputs.papers.items()
        if (y := parse_year(r)) is not None
    }
    for key, part in evidence.groupby(
        ["unified_id", "orientation", "start_node_id", "end_node_id"], sort=False
    ):
        pmids = set().union(
            *(set(str(s).split(" | ")) - {""} for s in part.support_pmids)
        )
        rows.append(
            dict(
                zip(["unified_id", "orientation", "start_node_id", "end_node_id"], key),
                support_pmid_count=len(pmids),
                support_pmids=" | ".join(sorted(pmids)),
                mean_publication_year=float(np.mean([years[p] for p in pmids])),
                admitted_snapshot_ends=" | ".join(
                    map(str, sorted(part.snapshot_end.unique()))
                ),
                best_q_value=float(part.q_value.min()),
                max_effect_size_z=float(part.effect_size_z.max()),
            )
        )
    return pd.DataFrame(rows)


def configure_figure30_details(paths, frames, blocks):
    f = frames["figure30_candidates"].copy()
    f["selection_rank"] = f.unified_id.map(
        {u: i + 1 for (i, u) in enumerate(sorted(f.unified_id, key=natural_key))}
    )
    frames["figure30_candidates"] = f
    return frames


def build_figure30_network_tables(
    paths: Paths,
    inputs: Inputs,
    blocks: pd.DataFrame,
    lineages: pd.DataFrame,
    nesting: pd.DataFrame,
    node_sets: dict[str, set[str]],
) -> dict[str, pd.DataFrame]:
    frames = _figure30_candidate_nodes(paths, inputs, blocks, lineages, node_sets)
    candidates = frames["figure30_candidates"]
    candidate_ids = set(candidates.loc[candidates.snapshot_count.ge(2), "unified_id"])
    evidence = _figure30_edge_evidence(
        paths,
        blocks,
        node_sets,
        frames["figure30_nodes"].loc[
            frames["figure30_nodes"].unified_id.isin(candidate_ids)
        ],
    )
    edges = _figure30_union_edges(evidence, inputs)
    candidates["display_edge_count"] = (
        candidates.unified_id.map(edges.groupby("unified_id").size())
        .fillna(0)
        .astype(int)
    )
    candidates["display_horizontal_count"] = (
        candidates.unified_id.map(
            edges.loc[edges.orientation.eq("horizontal")].groupby("unified_id").size()
        )
        .fillna(0)
        .astype(int)
    )
    candidates["display_edge_unique_pmid_count"] = (
        candidates.unified_id.map(
            {
                str(u): len(
                    set().union(*(set(s.split(" | ")) for s in p.support_pmids))
                )
                for (u, p) in edges.groupby("unified_id")
            }
        )
        .fillna(0)
        .astype(int)
    )
    candidates["selected"] = candidates.snapshot_count.ge(2)
    candidates["selection_rank"] = np.nan
    candidates["recommended_for_main_text"] = False
    candidates["output_stem"] = ""
    frames["figure30_candidates"] = candidates
    frames = configure_figure30_details(paths, frames, blocks)
    candidates = frames["figure30_candidates"]
    ordered = candidates.loc[candidates.selected].sort_values("selection_rank")
    frames["figure30_candidates"] = candidates
    picked = ordered.unified_id.astype(str).tolist()
    for key in (
        "figure30_nodes",
        "figure30_quotas",
        "figure30_bridges",
        "figure30_core_turnover",
    ):
        frames[key] = frames[key].loc[frames[key].unified_id.isin(picked)].copy()
    frames["figure30_edges"] = edges.loc[edges.unified_id.isin(picked)].copy()
    frames["figure30_evidence"] = evidence.loc[evidence.unified_id.isin(picked)].copy()
    for index, q in frames["figure30_quotas"].iterrows():
        (uid, end) = (str(q.unified_id), int(q.snapshot_end))
        e = frames["figure30_evidence"].loc[
            frames["figure30_evidence"].unified_id.eq(uid)
            & frames["figure30_evidence"].snapshot_end.eq(end)
        ]
        p = (
            set().union(*(set(s.split(" | ")) - {""} for s in e.support_pmids))
            if len(e)
            else set()
        )
        block = blocks.loc[
            blocks.unified_id.eq(uid) & blocks.snapshot_end.eq(end)
        ].iloc[0]
        ns = frames["figure30_nodes"].loc[frames["figure30_nodes"].unified_id.eq(uid)]
        displayed_members = set(ns.loc[ns[f"present_{end}"].map(as_bool), "node_id"])
        frames["figure30_quotas"].loc[index, "display_horizontal_edge_fraction"] = int(
            e.orientation.eq("horizontal").sum()
        ) / max(int(block.significant_horizontal_relations), 1)
        frames["figure30_quotas"].loc[index, "display_edge_evidence_pmid_fraction"] = (
            len(p) / max(int(q.independent_chord_pmid_count), 1)
        )
        frames["figure30_quotas"].loc[index, "display_member_fraction"] = len(
            displayed_members
        ) / max(len(node_sets[str(block.structure_uid)]), 1)
    for key, stem in FIGURE30_STEMS.items():
        write_table(paths, frames[key], stem)
    return frames
