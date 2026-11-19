from __future__ import annotations
from .result_paths import artifact_directory
from .project_config import TIME

ENGLISH_NAME_COLUMN = "structure_label_en"
from typing import Any, Mapping


def _split_pipe(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _node_pmids(data: TopicTreeDataset, nodes: set[str]) -> set[int]:
    return (
        set().union(
            *(
                set(map(int, data.node_papers.get(node, np.asarray([], dtype=int))))
                for node in nodes
            )
        )
        if nodes
        else set()
    )


def _direct_path_count(data: TopicTreeDataset, nodes: set[str]) -> int:
    children = {node: 0 for node in nodes}
    for node in nodes:
        parent = data.parent.get(node)
        if parent in children:
            children[parent] += 1
    return sum((value == 0 for value in children.values()))


def _read_table_if_present(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是"}


def _natural_structure_key(value: Any) -> tuple[int, str]:
    text = str(value)
    digits = ""
    for char in reversed(text):
        if not char.isdigit():
            break
        digits = char + digits
    return (int(digits) if digits else 10**9, text)


def _unique_ids(value: Any, limit: int | None = None) -> list[str]:
    values = _split_pipe(value)
    values = list(dict.fromkeys(values))
    return values[:limit] if limit is not None else values


def _compute_block_paths(data: TopicTreeDataset, nodes: set[str]) -> list[str]:
    nodes = {node for node in nodes if node in data.tree}
    if not nodes:
        return []
    parent = data.parent
    children: dict[str, list[str]] = {}
    for node in nodes:
        p = parent.get(node)
        if p in nodes:
            children.setdefault(p, []).append(node)
    leaves = sorted(
        (node for node in nodes if node not in children),
        key=lambda node: (-int(data.tree[node]["level"]), node),
    )
    rows: list[tuple[int, str]] = []
    for leaf in leaves:
        path_nodes = list(data.paths[leaf])
        pmids: set[int] = set()
        for node in path_nodes:
            if node in nodes:
                pmids.update(data.node_papers[node].tolist())
        rows.append(
            (len(pmids), "/".join((data.tree[node]["name"] for node in path_nodes)))
        )
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [text for (_, text) in rows]


def _unified_record(
    data: TopicTreeDataset,
    *,
    structure_uid: str,
    display_id: str,
    source_type: str,
    source_label: str,
    original_id: str,
    interval_label: str,
    nodes: set[str],
    top15: list[str],
    summary_row: dict[str, Any] | None = None,
    snapshot_start: int | None = None,
    snapshot_end: int | None = None,
    source_definition: str = "",
) -> dict[str, Any]:
    summary_row = summary_row or {}
    node_names = [
        str(data.tree[node].get("name", node)) for node in top15 if node in data.tree
    ]
    summary_path_count = summary_row.get("classification_path_count")
    if summary_path_count is None or (
        isinstance(summary_path_count, float) and np.isnan(summary_path_count)
    ):
        summary_path_count = summary_row.get("all_predicted_path_count")
    if summary_path_count is None or (
        isinstance(summary_path_count, float) and np.isnan(summary_path_count)
    ):
        summary_path_count = _direct_path_count(
            data, set((str(node) for node in nodes if str(node) in data.tree))
        )
    return {
        "structure_uid": structure_uid,
        "display_structure_id": display_id,
        "source_type": source_type,
        "source_label": source_label,
        "source_community_id": original_id,
        "interval_label": interval_label,
        "snapshot_start": snapshot_start if snapshot_start is not None else "",
        "snapshot_end": snapshot_end if snapshot_end is not None else "",
        "nodes": set((str(node) for node in nodes if str(node) in data.tree)),
        "top15_node_ids": list(
            dict.fromkeys((str(node) for node in top15 if str(node) in data.tree))
        )[:15],
        "top15_node_names": node_names[:15],
        "node_count": len(nodes),
        "all_node_unique_all_paper_pmid_count": len(_node_pmids(data, nodes)),
        "summary_node_count": _safe_int(summary_row.get("node_count"), len(nodes)),
        "summary_all_node_unique_all_paper_pmid_count": _safe_int(
            summary_row.get("all_node_unique_all_paper_pmid_count"),
            len(_node_pmids(data, nodes)),
        ),
        "internal_chord_count": _safe_int(
            summary_row.get(
                "internal_chord_count",
                summary_row.get(
                    "all_predicted_edge_count",
                    _safe_int(summary_row.get("significant_horizontal_relations"))
                    + _safe_int(
                        summary_row.get("significant_direct_parent_child_relations")
                    ),
                ),
            )
        ),
        "internal_horizontal_chord_count": _safe_int(
            summary_row.get(
                "internal_horizontal_chord_count",
                summary_row.get(
                    "all_predicted_horizontal_edge_count",
                    summary_row.get("significant_horizontal_relations"),
                ),
            )
        ),
        "internal_vertical_chord_count": _safe_int(
            summary_row.get(
                "internal_vertical_chord_count",
                summary_row.get(
                    "all_predicted_vertical_edge_count",
                    summary_row.get("significant_direct_parent_child_relations"),
                ),
            )
        ),
        "classification_path_count": _safe_int(summary_path_count),
        "source_definition": source_definition,
    }


def _build_unified_structure_records(
    out: Output, data: TopicTreeDataset
) -> list[dict[str, Any]]:
    hyper_tables = artifact_directory(out.root.parent / "03_hypergraph" / "tables")
    records: list[dict[str, Any]] = []
    s_summary = _read_table_if_present(hyper_tables / "16_S社群_完整清单.csv")
    s_core = _read_table_if_present(hyper_tables / "16_S社群_核心节点清单.csv")
    s_members = _read_table_if_present(
        hyper_tables / "12_PMID逐层显著性_全部节点状态.csv"
    )
    if not s_summary.empty:
        s_summary = (
            s_summary[
                s_summary.get("formal_community_confirmed", True).map(_bool_value)
            ]
            if "formal_community_confirmed" in s_summary
            else s_summary
        )
    if not s_members.empty and "formal_community_confirmed" in s_members:
        s_members = s_members[
            s_members.formal_community_confirmed.map(_bool_value)
        ].copy()
    s_top15: dict[str, list[str]] = {}
    if not s_core.empty and {"community_id", "node_id"}.issubset(s_core.columns):
        s_core = s_core.copy()
        s_core["_rank"] = pd.to_numeric(
            s_core.get("core_rank", 10**9), errors="coerce"
        ).fillna(10**9)
        for community, group in s_core.sort_values(
            ["community_id", "_rank"], kind="stable"
        ).groupby("community_id", sort=False):
            s_top15[str(community)] = _unique_ids(
                " | ".join(group.node_id.astype(str).tolist()), 15
            )
    s_summary_by_id = (
        {
            str(row.community_id): row._asdict()
            for row in s_summary.itertuples(index=False)
        }
        if not s_summary.empty
        else {}
    )
    if not s_members.empty and "significance_block_id" in s_members.columns:
        s_member_sets = {
            str(key): set(group.node_id.astype(str))
            for (key, group) in s_members.groupby("significance_block_id", sort=False)
        }
    else:
        s_member_sets = {}
    for community in sorted(s_summary_by_id, key=_natural_structure_key):
        nodes = s_member_sets.get(community, set())
        records.append(
            _unified_record(
                data,
                structure_uid=f"full_S:{community}",
                display_id=community,
                source_type="full_S",
                source_label="全量正式S",
                original_id=community,
                interval_label=TIME.text("全量2001–2025"),
                nodes=nodes,
                top15=s_top15.get(community, []),
                summary_row=s_summary_by_id.get(community),
                source_definition="全量数据正式 S 结构块；作为统一对齐的主原始结构空间。",
            )
        )

    def add_weighted(
        prefix: str,
        source_label: str,
        summary_name: str,
        node_name: str,
        source_type: str,
    ) -> None:
        summary = _read_table_if_present(hyper_tables / summary_name)
        members = _read_table_if_present(hyper_tables / node_name)
        if summary.empty:
            return
        if "formal_community_confirmed" in summary:
            summary = summary[
                summary.formal_community_confirmed.map(_bool_value)
            ].copy()
        if "formal_community_confirmed" in members:
            members = members[
                members.formal_community_confirmed.map(_bool_value)
            ].copy()
        summary_by_id = {
            str(row.community_id): row._asdict()
            for row in summary.itertuples(index=False)
        }
        member_sets = (
            {
                str(key): set(group.node_id.astype(str))
                for (key, group) in members.groupby("significance_block_id", sort=False)
            }
            if "significance_block_id" in members.columns
            else {}
        )
        display_map = {
            community: f"{prefix}{index}"
            for (index, community) in enumerate(
                sorted(summary_by_id, key=_natural_structure_key), start=1
            )
        }
        for community in sorted(summary_by_id, key=_natural_structure_key):
            row = summary_by_id[community]
            records.append(
                _unified_record(
                    data,
                    structure_uid=f"{source_type}:{community}",
                    display_id=display_map[community],
                    source_type=source_type,
                    source_label=source_label,
                    original_id=community,
                    interval_label=TIME.text("全量2001–2025"),
                    nodes=member_sets.get(community, set()),
                    top15=_unique_ids(row.get("core_node_ids", ""), 15),
                    summary_row=row,
                    source_definition=f"全量数据{source_label}结构块；作为 S 无法直接对齐时的第二层原始结构空间。",
                )
            )

    add_weighted(
        "J",
        "JCR加权",
        "22b_探索性质量加权_JCR_Q_正式候选社群清单.csv",
        "22b_探索性质量加权_JCR_Q_正式候选社群节点.csv",
        "full_JCR",
    )
    add_weighted(
        "I",
        "IF加权",
        "22b_探索性质量加权_IF_year_正式候选社群清单.csv",
        "22b_探索性质量加权_IF_year_正式候选社群节点.csv",
        "full_IF",
    )
    hist_summary = _read_table_if_present(out.tables / "09_历史累积S结构块.csv")
    hist_members = _read_table_if_present(out.tables / "09_历史累积S结构块节点.csv")
    if not hist_summary.empty:
        for row in hist_summary.itertuples(index=False):
            end = _safe_int(getattr(row, "snapshot_end", 0))
            start = _safe_int(getattr(row, "snapshot_start", 0))
            community = str(row.community_id)
            if (
                not hist_members.empty
                and "snapshot_end" in hist_members
                and ("significance_block_id" in hist_members)
            ):
                part = hist_members[
                    (hist_members.snapshot_end.astype(str) == str(end))
                    & (hist_members.significance_block_id.astype(str) == community)
                ]
                nodes = set(part.node_id.astype(str))
            else:
                nodes = set()
            records.append(
                _unified_record(
                    data,
                    structure_uid=f"historical:{end}:{community}",
                    display_id=f"H{end}_{community}",
                    source_type="historical",
                    source_label=f"历史累积S {start}–{end}",
                    original_id=community,
                    interval_label=f"{start}–{end}",
                    nodes=nodes,
                    top15=_unique_ids(getattr(row, "top15_node_ids", ""), 15),
                    summary_row=row._asdict(),
                    snapshot_start=start,
                    snapshot_end=end,
                    source_definition="历史累积 S 快照；已使用与正式 S 一致的发现后结构门槛。",
                )
            )
    future_summary = _read_table_if_present(
        out.tables / "08_未来概率网络模拟_重复率25%结构块_筛选后正式结果.csv"
    )
    future_members = _read_table_if_present(
        out.tables / "08_未来概率网络模拟_结构块节点.csv"
    )
    if not future_summary.empty:
        future_sets = (
            {
                str(key): set(group.node_id.astype(str))
                for (key, group) in future_members.groupby(
                    "future_module_id", sort=False
                )
            }
            if not future_members.empty and "future_module_id" in future_members.columns
            else {}
        )
        for row in future_summary.itertuples(index=False):
            future_id = str(row.future_module_id)
            records.append(
                _unified_record(
                    data,
                    structure_uid=f"future:{future_id}",
                    display_id=future_id,
                    source_type="future_prediction",
                    source_label=TIME.text("2026–2030预测"),
                    original_id=future_id,
                    interval_label=TIME.text("2026–2030"),
                    nodes=future_sets.get(future_id, set()),
                    top15=_unique_ids(getattr(row, "predicted_top15_node_ids", ""), 15),
                    summary_row=row._asdict(),
                    snapshot_start=FORECAST_START,
                    snapshot_end=FORECAST_END,
                    source_definition="未来概率网络模拟中重复率≥25%且通过归并后节点、边、横向边和连续路径门槛的正式预测结构；节点和边使用重复样本并集。",
                )
            )
    for record in records:
        record["classification_paths"] = _compute_block_paths(data, record["nodes"])
    return records


def _reference_led_one_to_one_claims(
    references: list[dict[str, Any]],
    candidate_pool: dict[str, dict[str, Any]],
    *,
    find: Any,
    members_by_root: dict[str, set[str]],
    slot_by_uid: dict[str, str],
    assignments: dict[str, dict[str, Any]],
    stage: str,
    status: str,
    order_start: int = 0,
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any], int]], int]:
    available = {
        uid: record
        for (uid, record) in candidate_pool.items()
        if uid not in assignments
    }
    accepted: dict[str, tuple[dict[str, Any], dict[str, Any], int]] = {}
    ordered_references = sorted(
        references,
        key=lambda record: (
            -int(record.get("node_count", 0)),
            _natural_structure_key(record.get("display_structure_id", "")),
        ),
    )
    order = order_start
    for reference in ordered_references:
        root = find(reference["structure_uid"])
        occupied_slots = {
            slot_by_uid[member_uid] for member_uid in members_by_root[root]
        }
        ranked = list(dict.fromkeys(reference.get("top15_node_ids", [])[:15]))
        best: (
            tuple[tuple[int, int, tuple[int, str]], dict[str, Any], list[str]] | None
        ) = None
        for cand_uid, candidate in available.items():
            if cand_uid == reference["structure_uid"]:
                continue
            if slot_by_uid[cand_uid] in occupied_slots:
                continue
            hits = [node for node in ranked if node in candidate["nodes"]]
            score = len(hits)
            if score < 1:
                continue
            key = (
                -score,
                -int(candidate.get("node_count", 0)),
                _natural_structure_key(candidate.get("display_structure_id", "")),
            )
            if best is None or key < best[0]:
                best = (key, candidate, hits)
        if best is None:
            continue
        (_, candidate, hits) = best
        metrics = {
            "score": int(len(hits)),
            "source_top15_count": int(len(ranked)),
            "fraction": float(len(hits) / max(len(ranked), 1)),
            "hits": hits,
            "candidate_count": int(len(available)),
            "claim_reference_structure_id": str(
                reference.get("display_structure_id", "")
            ),
            "claim_reference_source_type": str(reference.get("source_type", "")),
        }
        order += 1
        accepted[candidate["structure_uid"]] = (reference, metrics, order)
        assignments[candidate["structure_uid"]] = _apply_alignment_result(
            candidate, reference, metrics, stage, order, status
        )
        del available[candidate["structure_uid"]]
    return (accepted, order)


def _cross_model_weighted_pairs(
    unmatched_jcr: list[dict[str, Any]], unmatched_if: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    records = list(unmatched_jcr) + list(unmatched_if)
    others = {"full_JCR": unmatched_if, "full_IF": unmatched_jcr}
    claimed: set[str] = set()
    pairs: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    ordered = sorted(
        records,
        key=lambda record: (
            -int(record.get("node_count", 0)),
            _natural_structure_key(record.get("display_structure_id", "")),
        ),
    )
    for record in ordered:
        uid = record["structure_uid"]
        if uid in claimed:
            continue
        ranked = list(dict.fromkeys(record.get("top15_node_ids", [])[:15]))
        best: (
            tuple[
                tuple[int, int, tuple[int, str]], dict[str, Any], list[str], list[str]
            ]
            | None
        ) = None
        for candidate in others[record["source_type"]]:
            if candidate["structure_uid"] in claimed:
                continue
            forward = [node for node in ranked if node in candidate["nodes"]]
            backward_ranked = list(
                dict.fromkeys(candidate.get("top15_node_ids", [])[:15])
            )
            backward = [node for node in backward_ranked if node in record["nodes"]]
            score = len(forward) + len(backward)
            if score < 1:
                continue
            key = (
                -score,
                -int(candidate.get("node_count", 0)),
                _natural_structure_key(candidate.get("display_structure_id", "")),
            )
            if best is None or key < best[0]:
                best = (key, candidate, forward, backward)
        if best is None:
            continue
        (_, candidate, forward, backward) = best
        claimed.add(uid)
        claimed.add(candidate["structure_uid"])
        if record["source_type"] == "full_JCR":
            (jcr_record, if_record) = (record, candidate)
            (jcr_hits, if_hits) = (forward, backward)
        else:
            (if_record, jcr_record) = (record, candidate)
            (if_hits, jcr_hits) = (forward, backward)
        jcr_metrics = {
            "score": int(len(jcr_hits)),
            "source_top15_count": int(len(jcr_record.get("top15_node_ids", []))),
            "fraction": float(
                len(jcr_hits) / max(len(jcr_record.get("top15_node_ids", [])), 1)
            ),
            "hits": jcr_hits,
            "candidate_count": len(others["full_JCR"]),
        }
        if_metrics = {
            "score": int(len(if_hits)),
            "source_top15_count": int(len(if_record.get("top15_node_ids", []))),
            "fraction": float(
                len(if_hits) / max(len(if_record.get("top15_node_ids", [])), 1)
            ),
            "hits": if_hits,
            "candidate_count": len(others["full_IF"]),
        }
        pairs.append((jcr_record, if_record, jcr_metrics, if_metrics))
    return pairs


def _apply_alignment_result(
    source: dict[str, Any],
    base: dict[str, Any] | None,
    metrics: dict[str, Any],
    stage: str,
    order: int,
    status: str,
) -> dict[str, Any]:
    source_nodes = set(source.get("nodes", set()))
    if base is None:
        overlap = 0
        jaccard = 0.0
        base_id = ""
        base_type = ""
        base_count = 0
        base_pmids = 0
        base_top15 = []
    else:
        base_nodes = set(base.get("nodes", set()))
        overlap = len(source_nodes & base_nodes)
        jaccard = overlap / max(len(source_nodes | base_nodes), 1)
        base_id = str(base.get("display_structure_id", ""))
        base_type = str(base.get("source_type", ""))
        base_count = int(base.get("node_count", 0))
        base_pmids = int(base.get("all_node_unique_all_paper_pmid_count", 0))
        base_top15 = list(base.get("top15_node_names", []))
    return {
        "matched_base_structure_id": base_id,
        "matched_base_source_type": base_type,
        "matched_base_node_count": base_count,
        "matched_base_unique_pmid_count": base_pmids,
        "matched_base_top15_node_names": " | ".join(base_top15),
        "alignment_stage": stage,
        "alignment_status": status,
        "matching_order_by_source_node_count": order,
        "top15_presence_score": int(metrics.get("score", 0)),
        "top15_source_node_count": int(metrics.get("source_top15_count", 0)),
        "top15_presence_fraction": float(metrics.get("fraction", 0.0)),
        "matched_top15_node_ids": " | ".join(metrics.get("hits", [])),
        "candidate_count_at_stage": int(metrics.get("candidate_count", 0)),
        "all_node_overlap": int(overlap),
        "all_node_jaccard": float(jaccard),
    }


def _canonical_edge(
    row: Any,
    first: str = "start_node_id",
    second: str = "end_node_id",
    relation_column: str = "edge_orientation",
) -> tuple[str, str, str] | None:
    left = getattr(row, first, None)
    right = getattr(row, second, None)
    relation = str(getattr(row, relation_column, "")).strip().lower()
    if left is None or right is None or str(left) == "nan" or (str(right) == "nan"):
        return None
    if relation in {"horizontal", "horizontal_same_level_pmid_chord"}:
        return ("horizontal",) + tuple(sorted((str(left), str(right))))
    if (
        relation in {"vertical", "vertical_direct_parent_child", "direct_parent_child"}
        or "vertical" in relation
    ):
        return ("vertical",) + tuple(sorted((str(left), str(right))))
    return None


def _record_edge_sets_from_rows(
    records: list[dict[str, Any]],
    edge_rows: pd.DataFrame,
    first: str = "start_node_id",
    second: str = "end_node_id",
    relation_column: str = "edge_orientation",
    group_column: str | None = None,
) -> dict[str, dict[str, set[tuple[str, str, str]]]]:
    result = {
        record["structure_uid"]: {"horizontal": set(), "vertical": set()}
        for record in records
    }
    if edge_rows.empty:
        return result
    edge_list: list[tuple[tuple[str, str, str], str | None]] = []
    for row in edge_rows.itertuples(index=False):
        edge = _canonical_edge(row, first, second, relation_column)
        if edge is None:
            continue
        group = (
            str(getattr(row, group_column))
            if group_column and hasattr(row, group_column)
            else None
        )
        edge_list.append((edge, group))
    for record in records:
        nodes = record["nodes"]
        group_id = str(record.get("source_community_id", ""))
        for edge, row_group in edge_list:
            if (
                row_group is not None
                and record["source_type"] in {"full_S", "full_JCR", "full_IF"}
                and (row_group not in {group_id, record["display_structure_id"]})
            ):
                continue
            (_, left, right) = edge
            if left in nodes and right in nodes:
                result[record["structure_uid"]][edge[0]].add(edge)
    return result


def _historical_edge_sets(
    data: TopicTreeDataset, records: list[dict[str, Any]]
) -> dict[str, dict[str, set[tuple[str, str, str]]]]:
    result = {
        record["structure_uid"]: {"horizontal": set(), "vertical": set()}
        for record in records
        if record["source_type"] == "historical"
    }
    if not result:
        return result
    ends = sorted(
        {
            _safe_int(record.get("snapshot_end"))
            for record in records
            if record["source_type"] == "historical"
        }
    )
    set(data.tree)
    for end in ends:
        snapshot_records = [
            record
            for record in records
            if record["source_type"] == "historical"
            and _safe_int(record.get("snapshot_end")) == end
        ]
        if not snapshot_records:
            continue
        snapshot = TopicTreeDataset(
            data.tree_path, data.papers_path, TIME.history_start, end
        )
        graph = build_pmid_hypergraph(snapshot)
        calibration = levelwise_pmid_significance(snapshot, graph)
        vertical = significant_vertical_evidence(
            snapshot, graph, calibration["active_sources"]
        )
        horizontal_rows: list[tuple[str, str, str]] = []
        horizontal = calibration.get("horizontal", pd.DataFrame())
        if not horizontal.empty and "significant_horizontal_relation" in horizontal:
            for row in horizontal[
                horizontal.significant_horizontal_relation.astype(bool)
            ].itertuples(index=False):
                left = graph.ids[int(row.node_1_index)]
                right = graph.ids[int(row.node_2_index)]
                horizontal_rows.append(
                    ("horizontal",) + tuple(sorted((str(left), str(right))))
                )
        vertical_rows: list[tuple[str, str, str]] = []
        if not vertical.empty and "significant_vertical_relation" in vertical:
            for row in vertical[
                vertical.significant_vertical_relation.astype(bool)
            ].itertuples(index=False):
                left = graph.ids[int(row.parent_index)]
                right = graph.ids[int(row.child_index)]
                vertical_rows.append(
                    ("vertical",) + tuple(sorted((str(left), str(right))))
                )
        for record in snapshot_records:
            nodes = record["nodes"]
            result[record["structure_uid"]]["horizontal"].update(
                (
                    edge
                    for edge in horizontal_rows
                    if edge[1] in nodes and edge[2] in nodes
                )
            )
            result[record["structure_uid"]]["vertical"].update(
                (
                    edge
                    for edge in vertical_rows
                    if edge[1] in nodes and edge[2] in nodes
                )
            )
    return result


def _build_source_edge_sets(
    out: Output, data: TopicTreeDataset, records: list[dict[str, Any]]
) -> dict[str, dict[str, set[tuple[str, str, str]]]]:
    result = {
        record["structure_uid"]: {"horizontal": set(), "vertical": set()}
        for record in records
    }
    hyper_tables = artifact_directory(out.root.parent / "03_hypergraph" / "tables")
    groups = {
        "full_S": (
            "16_S社群_显著性筛选弦列表.csv",
            "start_node_id",
            "end_node_id",
            "edge_orientation",
        ),
        "full_JCR": (
            "22b_探索性质量加权_JCR_Q_显著弦清单.csv",
            "start_node_id",
            "end_node_id",
            "edge_orientation",
        ),
        "full_IF": (
            "22b_探索性质量加权_IF_year_显著弦清单.csv",
            "start_node_id",
            "end_node_id",
            "edge_orientation",
        ),
    }
    for source_type, (filename, first, second, relation) in groups.items():
        source_records = [
            record for record in records if record["source_type"] == source_type
        ]
        rows = _read_table_if_present(hyper_tables / filename)
        assigned = _record_edge_sets_from_rows(
            source_records, rows, first, second, relation
        )
        for key, value in assigned.items():
            result[key]["horizontal"].update(value["horizontal"])
            result[key]["vertical"].update(value["vertical"])
    future_records = [
        record for record in records if record["source_type"] == "future_prediction"
    ]
    future_rows = _read_table_if_present(
        out.tables / "08_未来概率网络模拟_归并后结构块弦_筛选后正式结果.csv"
    )
    assigned = _record_edge_sets_from_rows(
        future_records,
        future_rows,
        "node_1_id",
        "node_2_id",
        "relation_type",
        "future_module_id",
    )
    for key, value in assigned.items():
        result[key]["horizontal"].update(value["horizontal"])
        result[key]["vertical"].update(value["vertical"])
    historical = _historical_edge_sets(data, records)
    for key, value in historical.items():
        result[key]["horizontal"].update(value["horizontal"])
        result[key]["vertical"].update(value["vertical"])
    return result


def _compact_interval_label(record: dict[str, Any]) -> str:
    if record["source_type"] == "full_S":
        return "正式s"
    if record["source_type"] == "full_JCR":
        return "JCR加权"
    if record["source_type"] == "full_IF":
        return "IF加权"
    if record["source_type"] == "future_prediction":
        return TIME.key("26-30预测")
    end = _safe_int(record.get("snapshot_end"))
    labels = TIME.history_slots
    return labels.get(end, str(record.get("interval_label", f"历史{end}")))


def _compact_source_id(record: dict[str, Any]) -> str:
    if record["source_type"] == "historical":
        return str(record["source_community_id"])
    return str(record["display_structure_id"])


SLOT_ORDER = [
    "正式s",
    "JCR加权",
    "IF加权",
    *(TIME.history_slots[end] for end in reversed(TIME.snapshot_ends)),
    TIME.forecast_slot,
]
SLOT_COLUMN_NAMES = {
    slot: (f"{slot}结构编号", f"{slot}前15名节点名称") for slot in SLOT_ORDER
}
SLOT_COLUMN_NAMES["正式s"] = ("正式s结构块编号", "正式s结构块前15名节点名称")


def _compact_structure_columns(
    records: list[dict[str, Any]],
    edge_sets: dict[str, dict[str, set[tuple[str, str, str]]]],
    find: Any,
    preserved_labels: Mapping[str, str] | None = None,
    preserved_english_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    slot_order = {slot: index for (index, slot) in enumerate(SLOT_ORDER)}
    components: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        components.setdefault(find(record["structure_uid"]), []).append(record)

    def component_sort_key(
        members: list[dict[str, Any]],
    ) -> tuple[int, tuple[int, str]]:
        present = [
            slot
            for slot in SLOT_ORDER
            if any((_compact_interval_label(record) == slot for record in members))
        ]
        first_slot = slot_order.get(present[0], 99) if present else 99
        natural = _natural_structure_key(
            min(
                (str(record.get("display_structure_id", "")) for record in members),
                default="",
            )
        )
        return (first_slot, natural)

    preserved_labels = preserved_labels or {}
    preserved_english_names = preserved_english_names or {}
    include_label_columns = bool(preserved_labels)
    component_rows: list[dict[str, Any]] = []
    for members in sorted(components.values(), key=component_sort_key):
        by_slot: dict[str, dict[str, Any] | None] = {
            slot: next(
                (
                    record
                    for record in members
                    if _compact_interval_label(record) == slot
                ),
                None,
            )
            for slot in SLOT_ORDER
        }
        row: dict[str, Any] = {"整合结构块编号": f"U{len(component_rows) + 1}"}
        for slot in SLOT_ORDER:
            (id_column, name_column) = SLOT_COLUMN_NAMES[slot]
            label_column = f"{id_column}structure_label"
            english_column = f"{id_column}structure_label_en"
            member = by_slot[slot]
            if member is None:
                row[id_column] = ""
                row[name_column] = ""
                if include_label_columns:
                    row[label_column] = ""
                    row[english_column] = ""
            else:
                row[id_column] = _compact_source_id(member)
                row[name_column] = " | ".join(member["top15_node_names"])
                if include_label_columns:
                    row[label_column] = str(
                        preserved_labels.get(member["structure_uid"], "")
                    )
                    row[english_column] = str(
                        preserved_english_names.get(member["structure_uid"], "")
                    )
        for left_index, left in enumerate(SLOT_ORDER):
            left_member = by_slot[left]
            for right in SLOT_ORDER[left_index + 1 :]:
                right_member = by_slot[right]
                prefix = f"{left}vs{right}"
                if left_member is None or right_member is None:
                    continue
                left_nodes = left_member["nodes"]
                right_nodes = right_member["nodes"]
                common_nodes = len(left_nodes & right_nodes)
                node_jaccard = common_nodes / max(len(left_nodes | right_nodes), 1)
                left_edges = edge_sets.get(
                    left_member["structure_uid"],
                    {"horizontal": set(), "vertical": set()},
                )
                right_edges = edge_sets.get(
                    right_member["structure_uid"],
                    {"horizontal": set(), "vertical": set()},
                )
                row[f"{prefix}Jaccard"] = float(node_jaccard)
                row[f"{prefix}相同节点数"] = int(common_nodes)
                row[f"{prefix}相同纵向边数"] = int(
                    len(left_edges["vertical"] & right_edges["vertical"])
                )
                row[f"{prefix}相同横向边数"] = int(
                    len(left_edges["horizontal"] & right_edges["horizontal"])
                )
        component_rows.append(row)
    compact = pd.DataFrame(component_rows)
    if compact.empty:
        return compact
    ordered_columns = ["整合结构块编号"]
    for slot in SLOT_ORDER:
        (id_column, name_column) = SLOT_COLUMN_NAMES[slot]
        ordered_columns.append(id_column)
        if include_label_columns:
            ordered_columns.append(f"{id_column}structure_label")
            ordered_columns.append(f"{id_column}structure_label_en")
        ordered_columns.append(name_column)
    pair_columns: list[str] = []
    for left_index, left in enumerate(SLOT_ORDER):
        for right in SLOT_ORDER[left_index + 1 :]:
            prefix = f"{left}vs{right}"
            pair_columns.extend(
                [
                    f"{prefix}Jaccard",
                    f"{prefix}相同节点数",
                    f"{prefix}相同纵向边数",
                    f"{prefix}相同横向边数",
                ]
            )
    missing_columns = [
        column for column in ordered_columns + pair_columns if column not in compact
    ]
    if missing_columns:
        compact = pd.concat(
            [compact, pd.DataFrame("", index=compact.index, columns=missing_columns)],
            axis=1,
        )
    return compact[ordered_columns + pair_columns]


def build_unified_structure_alignment(
    out: Output, data: TopicTreeDataset
) -> dict[str, Any]:
    prior_audit = _read_table_if_present(out.tables / "21_统一对齐明细审计.csv")
    preserved_labels: dict[str, str] = {}
    preserved_english_names: dict[str, str] = {}
    preserved_name_versions: dict[str, str] = {}
    preserved_label_sources: dict[str, str] = {}
    if {"structure_uid", "structure_label"}.issubset(prior_audit.columns):
        columns = ["structure_uid", "structure_label"]
        if "structure_label_source" in prior_audit.columns:
            columns.append("structure_label_source")
        for row in prior_audit[columns].itertuples(index=False):
            name = str(row[1]).strip() if pd.notna(row[1]) else ""
            if name:
                preserved_labels[str(row[0])] = name
                provenance = (
                    str(row[2]).strip()
                    if len(row) > 2 and pd.notna(row[2])
                    else "缓存或既有AI注释"
                )
                preserved_label_sources[str(row[0])] = provenance or "缓存或既有AI注释"
        for row in prior_audit.fillna("").to_dict("records"):
            preserved_english_names[str(row["structure_uid"])] = str(
                row.get(ENGLISH_NAME_COLUMN, "")
            ).strip()
            preserved_name_versions[str(row["structure_uid"])] = str(
                row.get("label_version", "")
            ).strip()
    records = _build_unified_structure_records(out, data)
    if not records:
        empty = pd.DataFrame()
        out.table(empty, "21_所有年度与全量S_JCR_IF_未来结构块统一对齐")
        out.table(empty, "21_统一对齐汇总")
        return {"record_count": 0, "matched_count": 0}
    uid_map = {record["structure_uid"]: record for record in records}
    slot_by_uid = {
        uid: _compact_interval_label(record) for (uid, record) in uid_map.items()
    }
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(record["source_type"], []).append(record)
    s_records = by_type.get("full_S", [])
    jcr_records = by_type.get("full_JCR", [])
    if_records = by_type.get("full_IF", [])
    historical_records = by_type.get("historical", [])
    future_records = by_type.get("future_prediction", [])
    assignments: dict[str, dict[str, Any]] = {}
    parent: dict[str, str] = {uid: uid for uid in uid_map}
    members_by_root: dict[str, set[str]] = {uid: {uid} for uid in uid_map}

    def find(uid: str) -> str:
        parent.setdefault(uid, uid)
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    def union(left: str, right: str) -> None:
        (root_left, root_right) = (find(left), find(right))
        if root_left == root_right:
            return
        parent[root_right] = root_left
        members_by_root[root_left] |= members_by_root[root_right]
        del members_by_root[root_right]

    order = 0
    source_pools: list[list[dict[str, Any]]] = [jcr_records, if_records, future_records]
    for end in sorted(
        {_safe_int(record.get("snapshot_end")) for record in historical_records}
    ):
        source_pools.append(
            [
                record
                for record in historical_records
                if _safe_int(record.get("snapshot_end")) == end
            ]
        )
    for source_pool in source_pools:
        if not source_pool:
            continue
        (accepted, order) = _reference_led_one_to_one_claims(
            s_records,
            {record["structure_uid"]: record for record in source_pool},
            find=find,
            members_by_root=members_by_root,
            slot_by_uid=slot_by_uid,
            assignments=assignments,
            stage="full_data_S_primary",
            status="aligned_to_full_S",
            order_start=order,
        )
        for cand_uid, (base, _metrics, _position) in accepted.items():
            union(cand_uid, base["structure_uid"])
    for record in s_records:
        metrics = {
            "score": len(record["top15_node_ids"]),
            "source_top15_count": len(record["top15_node_ids"]),
            "fraction": 1.0,
            "hits": record["top15_node_ids"],
            "candidate_count": 1,
        }
        assignments[record["structure_uid"]] = _apply_alignment_result(
            record, record, metrics, "primary_S_identity", 0, "primary_full_S"
        )
    weighted_unmatched = [
        record
        for record in jcr_records + if_records
        if record["structure_uid"] not in assignments
    ]
    unmatched_jcr = [
        record for record in weighted_unmatched if record["source_type"] == "full_JCR"
    ]
    unmatched_if = [
        record for record in weighted_unmatched if record["source_type"] == "full_IF"
    ]
    for jcr_record, if_record, jcr_metrics, if_metrics in _cross_model_weighted_pairs(
        unmatched_jcr, unmatched_if
    ):
        order += 1
        assignments[jcr_record["structure_uid"]] = _apply_alignment_result(
            jcr_record,
            if_record,
            jcr_metrics,
            "weighted_cross_model_fallback",
            order,
            "aligned_to_weighted_fallback",
        )
        order += 1
        assignments[if_record["structure_uid"]] = _apply_alignment_result(
            if_record,
            jcr_record,
            if_metrics,
            "weighted_cross_model_fallback",
            order,
            "aligned_to_weighted_fallback",
        )
        union(jcr_record["structure_uid"], if_record["structure_uid"])
    for record in sorted(
        weighted_unmatched,
        key=lambda item: (-int(item.get("node_count", 0)), item["structure_uid"]),
    ):
        if record["structure_uid"] in assignments:
            continue
        order += 1
        assignments[record["structure_uid"]] = _apply_alignment_result(
            record,
            record,
            {
                "score": 0,
                "source_top15_count": len(record["top15_node_ids"]),
                "fraction": 0.0,
                "hits": [],
                "candidate_count": 0,
            },
            "weighted_self_anchor",
            order,
            "weighted_unmatched_anchor",
        )
    weighted_anchors = jcr_records + if_records
    fallback_pools: list[list[dict[str, Any]]] = [future_records]
    for end in sorted(
        {_safe_int(record.get("snapshot_end")) for record in historical_records},
        reverse=True,
    ):
        fallback_pools.append(
            [
                record
                for record in historical_records
                if _safe_int(record.get("snapshot_end")) == end
            ]
        )
    for source_pool in fallback_pools:
        residual = {
            record["structure_uid"]: record
            for record in source_pool
            if record["structure_uid"] not in assignments
        }
        if not residual:
            continue
        (accepted, order) = _reference_led_one_to_one_claims(
            weighted_anchors,
            residual,
            find=find,
            members_by_root=members_by_root,
            slot_by_uid=slot_by_uid,
            assignments=assignments,
            stage="weighted_fallback_after_S",
            status="aligned_to_weighted_fallback",
            order_start=order,
        )
        for cand_uid, (base, _metrics, _position) in accepted.items():
            union(cand_uid, base["structure_uid"])
    residual = {
        uid: record
        for (uid, record) in uid_map.items()
        if uid not in assignments
        and record["source_type"] in {"future_prediction", "historical"}
    }
    for end in sorted(
        {int(record["snapshot_end"]) for record in historical_records}, reverse=True
    ):
        if not residual:
            break
        references = [
            record
            for record in historical_records
            if int(record["snapshot_end"]) == end
        ]
        if not references:
            continue
        (accepted, order) = _reference_led_one_to_one_claims(
            references,
            residual,
            find=find,
            members_by_root=members_by_root,
            slot_by_uid=slot_by_uid,
            assignments=assignments,
            stage="historical_fallback_recent_to_old",
            status="aligned_to_historical_fallback",
            order_start=order,
        )
        for cand_uid, (base, _metrics, _position) in accepted.items():
            union(cand_uid, base["structure_uid"])
        residual = {
            uid: record for (uid, record) in residual.items() if uid not in assignments
        }
    residual = {
        uid: record for (uid, record) in uid_map.items() if uid not in assignments
    }
    for record in sorted(
        residual.values(),
        key=lambda item: _natural_structure_key(item["display_structure_id"]),
    ):
        order += 1
        assignments[record["structure_uid"]] = _apply_alignment_result(
            record,
            None,
            {
                "score": 0,
                "source_top15_count": len(record["top15_node_ids"]),
                "fraction": 0.0,
                "hits": [],
                "candidate_count": 0,
            },
            "unmatched_zero_score",
            order,
            "unmatched_no_positive_claim",
        )
    source_order = {
        "full_S": 0,
        "full_JCR": 1,
        "full_IF": 2,
        "historical": 3,
        "future_prediction": 4,
    }
    rows: list[dict[str, Any]] = []
    for position, record in enumerate(
        sorted(
            records,
            key=lambda item: (
                source_order.get(item["source_type"], 99),
                _safe_int(item.get("snapshot_end"), 9999),
                _natural_structure_key(item["display_structure_id"]),
            ),
        ),
        start=1,
    ):
        alignment = assignments[record["structure_uid"]]
        base_type = alignment.get("matched_base_source_type", "")
        base_id = alignment.get("matched_base_structure_id", "")
        lineage_id = (
            f"{base_type}:{base_id}"
            if base_id
            else f"unmatched:{record['display_structure_id']}"
        )
        rows.append(
            {
                "全表唯一行号": int(position),
                "分类路径": " || ".join(record["classification_paths"]),
                "structure_uid": record["structure_uid"],
                "source_type": record["source_type"],
                "source_label": record["source_label"],
                "source_structure_id": record["display_structure_id"],
                "source_community_id": record["source_community_id"],
                "interval_label": record["interval_label"],
                "snapshot_start": record["snapshot_start"],
                "snapshot_end": record["snapshot_end"],
                "source_node_count": record["node_count"],
                "source_summary_node_count": record["summary_node_count"],
                "source_unique_pmid_count": record[
                    "all_node_unique_all_paper_pmid_count"
                ],
                "source_summary_unique_pmid_count": record[
                    "summary_all_node_unique_all_paper_pmid_count"
                ],
                "source_internal_chord_count": record["internal_chord_count"],
                "source_internal_horizontal_chord_count": record[
                    "internal_horizontal_chord_count"
                ],
                "source_internal_vertical_chord_count": record[
                    "internal_vertical_chord_count"
                ],
                "source_classification_path_count": record["classification_path_count"],
                "source_top15_node_ids": " | ".join(record["top15_node_ids"]),
                "source_top15_node_names": " | ".join(record["top15_node_names"]),
                "matched_base_identity": lineage_id,
                **alignment,
                "source_definition": record["source_definition"],
            }
        )
    aligned_table = pd.DataFrame(rows)
    if not aligned_table.empty:
        aligned_table["structure_label"] = aligned_table.structure_uid.map(
            preserved_labels
        ).fillna("")
        aligned_table["structure_label_source"] = aligned_table.structure_uid.map(
            preserved_label_sources
        ).fillna("")
        aligned_table[ENGLISH_NAME_COLUMN] = aligned_table.structure_uid.map(
            preserved_english_names
        ).fillna("")
        aligned_table["label_version"] = aligned_table.structure_uid.map(
            preserved_name_versions
        ).fillna("")
        aligned_table["is_primary_full_S_identity"] = (
            aligned_table.matched_base_source_type.eq("full_S")
        )
        aligned_table["is_positive_claim"] = aligned_table.top15_presence_score.ge(1)
        aligned_table["_source_sort_order"] = aligned_table.source_type.map(
            source_order
        ).fillna(99)
        aligned_table["_end_sort_order"] = pd.to_numeric(
            aligned_table.snapshot_end, errors="coerce"
        ).fillna(9999)
        aligned_table = (
            aligned_table.sort_values(
                [
                    "_source_sort_order",
                    "_end_sort_order",
                    "source_node_count",
                    "source_structure_id",
                ],
                ascending=[True, True, False, True],
                kind="stable",
            )
            .drop(columns=["_source_sort_order", "_end_sort_order"])
            .reset_index(drop=True)
        )
    out.table(aligned_table, "21_统一对齐明细审计")
    edge_sets = _build_source_edge_sets(out, data, records)
    compact_table = _compact_structure_columns(
        records, edge_sets, find, preserved_labels, preserved_english_names
    )
    out.table(compact_table, "21_所有年度与全量S_JCR_IF_未来结构块统一对齐")
    if aligned_table.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            aligned_table.groupby(
                ["source_type", "alignment_status", "alignment_stage"], dropna=False
            )
            .agg(
                structure_count=("structure_uid", "size"),
                positive_claim_count=("is_positive_claim", "sum"),
                median_top15_presence_score=("top15_presence_score", "median"),
                median_all_node_jaccard=("all_node_jaccard", "median"),
                median_source_node_count=("source_node_count", "median"),
            )
            .reset_index()
        )
        summary = summary.sort_values(["source_type", "alignment_stage"], kind="stable")
    out.table(summary, "21_统一对齐汇总")
    return {
        "record_count": int(len(aligned_table)),
        "compact_identity_count": int(len(compact_table)),
        "positive_claim_count": int(aligned_table.is_positive_claim.sum())
        if not aligned_table.empty
        else 0,
        "direct_full_S_identity_count": int(
            aligned_table.is_primary_full_S_identity.sum()
        )
        if not aligned_table.empty
        else 0,
        "unmatched_count": int((~aligned_table.is_positive_claim).sum())
        if not aligned_table.empty
        else 0,
        "source_counts": aligned_table.source_type.value_counts().to_dict()
        if not aligned_table.empty
        else {},
    }


import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .topic_tree_analysis import (
    apply_posthoc_community_screen,
    build_community_catalog,
    build_readable_pmid_chord_list,
    HypergraphData,
    Output,
    TopicTreeDataset,
    build_pmid_hypergraph,
    factor_pair_layer,
    levelwise_pmid_significance,
    significant_pmid_blocks,
    significant_vertical_evidence,
)

try:
    import torch
except ImportError:
    torch = None
SEED = 20260913
OBSERVATION_END = TIME.observation_end
FORECAST_START = TIME.forecast_start
FORECAST_END = TIME.forecast_end
DEFAULT_OPEN_PER_NODE = 60
DEFAULT_SIMULATIONS = 24


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _direct_path_count_for_nodes(
    data: TopicTreeDataset, node_ids: Iterable[Any]
) -> int:
    nodes = {str(node) for node in node_ids}
    child_count = {node: 0 for node in nodes}
    for node in nodes:
        parent = data.parent.get(node)
        if parent in child_count:
            child_count[parent] += 1
    return int(sum((count == 0 for count in child_count.values())))


def _rank_nodes_by_internal_forecast_evidence(
    data: TopicTreeDataset,
    node_ids: Iterable[Any],
    edge_rows: pd.DataFrame,
    *,
    left_column: str = "node_1_id",
    right_column: str = "node_2_id",
    relation_column: str = "relation_type",
    strength_column: str = "evidence_z",
    recurrence_column: str | None = None,
) -> pd.DataFrame:
    ordered_nodes = sorted({str(node) for node in node_ids if str(node).strip()})
    node_set = set(ordered_nodes)
    score = {node: 0.0 for node in ordered_nodes}
    horizontal_degree = {node: 0 for node in ordered_nodes}
    vertical_degree = {node: 0 for node in ordered_nodes}
    if not edge_rows.empty:
        required = {left_column, right_column, relation_column, strength_column}
        missing = required - set(edge_rows.columns)
        if missing:
            raise ValueError(f"预测模块边表缺少核心节点排序所需列：{sorted(missing)}")
        for row in edge_rows.itertuples(index=False):
            left = str(getattr(row, left_column))
            right = str(getattr(row, right_column))
            if left not in node_set or right not in node_set:
                continue
            strength = max(_finite(getattr(row, strength_column)), 0.0)
            if recurrence_column is not None and recurrence_column in edge_rows.columns:
                strength *= float(
                    np.clip(_finite(getattr(row, recurrence_column), 1.0), 0.0, 1.0)
                )
            score[left] += strength
            score[right] += strength
            relation = str(getattr(row, relation_column)).strip().lower()
            degree = horizontal_degree if relation == "horizontal" else vertical_degree
            degree[left] += 1
            degree[right] += 1
    rows = []
    for node in ordered_nodes:
        tree_node = data.tree[node]
        rows.append(
            {
                "node_id": node,
                "node_name": str(tree_node["name"]),
                "level": int(tree_node["level"]),
                "internal_evidence_score": float(score[node]),
                "internal_horizontal_degree": int(horizontal_degree[node]),
                "internal_vertical_degree": int(vertical_degree[node]),
            }
        )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(
        [
            "internal_evidence_score",
            "internal_horizontal_degree",
            "internal_vertical_degree",
            "level",
            "node_id",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "module_node_rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def _simulation_module_edge_table(
    graph_ids: list[str], horizontal: pd.DataFrame, vertical: pd.DataFrame
) -> pd.DataFrame:
    horizontal_rows = horizontal[["node_1_index", "node_2_index", "evidence_z"]].copy()
    horizontal_rows["node_1_id"] = horizontal_rows.node_1_index.map(
        lambda index: str(graph_ids[int(index)])
    )
    horizontal_rows["node_2_id"] = horizontal_rows.node_2_index.map(
        lambda index: str(graph_ids[int(index)])
    )
    horizontal_rows["relation_type"] = "horizontal"
    vertical_rows = vertical[["parent_index", "child_index", "evidence_z"]].copy()
    vertical_rows["node_1_id"] = vertical_rows.parent_index.map(
        lambda index: str(graph_ids[int(index)])
    )
    vertical_rows["node_2_id"] = vertical_rows.child_index.map(
        lambda index: str(graph_ids[int(index)])
    )
    vertical_rows["relation_type"] = "vertical_direct_parent_child"
    return pd.concat(
        [
            horizontal_rows[["node_1_id", "node_2_id", "relation_type", "evidence_z"]],
            vertical_rows[["node_1_id", "node_2_id", "relation_type", "evidence_z"]],
        ],
        ignore_index=True,
    )


def _summarize_merged_future_module_cores(
    data: TopicTreeDataset, module_nodes: pd.DataFrame, merged_edges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if module_nodes.empty:
        return (pd.DataFrame(), pd.DataFrame())
    summary_rows: list[dict[str, Any]] = []
    rank_frames: list[pd.DataFrame] = []
    for module_id, group in module_nodes.groupby("future_module_id", sort=False):
        node_ids = group.node_id.astype(str).unique().tolist()
        edge_group = (
            merged_edges[merged_edges.future_module_id.astype(str).eq(str(module_id))]
            if not merged_edges.empty
            else pd.DataFrame()
        )
        ranked = _rank_nodes_by_internal_forecast_evidence(
            data,
            node_ids,
            edge_group,
            strength_column="mean_predicted_evidence_z",
            recurrence_column="simulation_edge_recurrence_rate",
        )
        if ranked.empty:
            continue
        ranked.insert(0, "future_module_id", str(module_id))
        rank_frames.append(ranked)
        top = ranked.head(15)
        summary_rows.append(
            {
                "future_module_id": str(module_id),
                "predicted_lowest_level": int(ranked.level.min()),
                "predicted_highest_level": int(ranked.level.max()),
                "predicted_top15_node_ids": " | ".join(top.node_id.astype(str)),
                "predicted_top15_node_names": " | ".join(top.node_name.astype(str)),
                "predicted_top15_ranking_definition": "归并后模块内部弦的平均预测证据强度×该弦模拟重复率之和；横向度、纵向度、层级和节点ID依次打破并列",
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame(),
    )


@dataclass
class CandidateBatch:
    cutoff: int
    horizon: int
    data: TopicTreeDataset
    graph: HypergraphData
    candidates: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    future_strength_z: np.ndarray
    future_horizontal_mass: np.ndarray
    future_end: int
    future_significant_edge_count: int = 0
    future_candidate_coverage: float = np.nan


@dataclass
class UnifiedTemporalEdgeModel:
    activation: Any | None = None
    strength: Any | None = None
    fallback_prevalence: float = 0.01
    fallback_strength: float = 0.0
    backend: str = "cpu_sklearn_reference"
    device: str = "cpu"
    requested_accelerator: str = "auto"
    cuda_available: bool = False
    accelerator_note: str = ""

    def fit(
        self,
        batches: list[CandidateBatch],
        *,
        accelerator: str = "auto",
        seed: int = SEED,
    ) -> "UnifiedTemporalEdgeModel":
        x_parts = [batch.features for batch in batches if len(batch.features)]
        y_parts = [batch.labels for batch in batches if len(batch.labels)]
        s_parts = [
            batch.future_strength_z for batch in batches if len(batch.future_strength_z)
        ]
        if not x_parts:
            raise ValueError("时间预测没有可用于训练的候选关系。")
        x = np.vstack(x_parts)
        y = np.concatenate(y_parts).astype(int)
        strength = np.concatenate(s_parts).astype(float)
        self.fallback_prevalence = float(np.mean(y)) if len(y) else 0.01
        self.fallback_strength = (
            float(np.mean(strength[y == 1])) if np.any(y == 1) else 0.0
        )
        self.requested_accelerator = str(accelerator)
        self.cuda_available = bool(torch is not None and torch.cuda.is_available())
        if accelerator not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"未知 accelerator：{accelerator}")
        if accelerator == "cuda" and (not self.cuda_available):
            raise RuntimeError("请求了 CUDA，但当前 Python 环境未检测到可用 CUDA。")
        (self.backend, self.device) = ("cpu_sklearn_reference", "cpu")
        self.accelerator_note = (
            "CUDA 可用；为保证与 CPU 完全同模型、同概率语义，预测拟合与 CPM Leiden 均使用 CPU 参考实现。"
            if self.cuda_available and accelerator in {"auto", "cuda"}
            else "使用 CPU 参考实现；CPM Leiden 与时间模型均保持固定算法。"
        )
        if len(np.unique(y)) >= 2:
            self.activation = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=600, C=0.5, class_weight=None, random_state=seed, n_jobs=1
                ),
            )
            self.activation.fit(x, y)
        positive = y == 1
        if positive.sum() >= 4:
            self.strength = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
            self.strength.fit(
                x[positive], np.log1p(np.maximum(strength[positive], 0.0))
            )
        return self

    def predict_activation(self, features: np.ndarray) -> np.ndarray:
        if self.activation is None:
            return np.full(len(features), self.fallback_prevalence, dtype=float)
        return np.clip(self.activation.predict_proba(features)[:, 1], 1e-06, 1 - 1e-06)

    def predict_strength_if_active(self, features: np.ndarray) -> np.ndarray:
        if self.strength is None:
            return np.full(len(features), max(self.fallback_strength, 0.0), dtype=float)
        return np.maximum(np.expm1(self.strength.predict(features)), 0.0)

    def predict(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probability = self.predict_activation(features)
        conditional = self.predict_strength_if_active(features)
        return (probability, conditional, probability * conditional)


def _node_history_features(
    data: TopicTreeDataset, ids: list[str]
) -> dict[str, np.ndarray]:
    incidence = data.incidence(ids).tocsr()
    years = np.asarray(data.year_values, dtype=int)
    annual = (
        np.column_stack(
            [
                np.asarray(incidence @ (data.years == year), dtype=float).ravel()
                for year in years
            ]
        )
        if len(years)
        else np.zeros((len(ids), 0), dtype=float)
    )
    total = annual.sum(axis=1) if annual.size else np.zeros(len(ids), dtype=float)
    recent_n = min(3, annual.shape[1])
    previous_n = min(3, max(0, annual.shape[1] - recent_n))
    recent = annual[:, -recent_n:].sum(axis=1) if recent_n else np.zeros(len(ids))
    previous = (
        annual[:, -recent_n - previous_n : -recent_n].sum(axis=1)
        if previous_n
        else np.zeros(len(ids))
    )
    recent_mean = recent / max(recent_n, 1)
    previous_mean = previous / max(previous_n, 1)
    growth = (recent_mean - previous_mean) / np.maximum(previous_mean, 1.0)
    if annual.shape[1] >= 4:
        x = np.arange(annual.shape[1], dtype=float)
        x_centered = x - x.mean()
        denominator = float(np.sum(x_centered**2))
        slope = (
            (annual - annual.mean(axis=1, keepdims=True))
            @ x_centered
            / max(denominator, 1.0)
        )
        slope = slope / np.maximum(annual.mean(axis=1), 1.0)
    else:
        slope = np.zeros(len(ids), dtype=float)
    return {
        "annual": annual,
        "total": total,
        "recent": recent,
        "previous": previous,
        "growth": np.clip(growth, -5, 5),
        "slope": np.clip(slope, -5, 5),
        "years": years,
    }


def _angle_features(
    data: TopicTreeDataset, ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    angles = data.compute_hierarchy_intervals()
    midpoint = np.asarray(
        [(angles[node][0] + angles[node][1]) / 2.0 for node in ids], dtype=float
    )
    span = np.asarray(
        [max(angles[node][1] - angles[node][0], 0.0) for node in ids], dtype=float
    )
    return (midpoint, span)


def _open_pair_candidates(
    pair_layer: sparse.csr_matrix, graph: HypergraphData, max_per_node: int
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    binary = pair_layer.copy().tocsr()
    binary.data = np.ones_like(binary.data)
    binary.eliminate_zeros()
    common = (binary @ binary).tocsr()
    common.setdiag(0)
    common = common - common.multiply(binary)
    common.eliminate_zeros()
    selected: dict[tuple[int, int], float] = {}
    observed: dict[tuple[int, int], float] = {}
    upper = sparse.triu(pair_layer, k=1).tocoo()
    for left, right, value in zip(
        upper.row.tolist(), upper.col.tolist(), upper.data.tolist()
    ):
        observed[int(left), int(right)] = float(value)
    per_node: dict[int, list[tuple[float, int]]] = collections.defaultdict(list)
    coo = sparse.triu(common, k=1).tocoo()
    for left, right, value in zip(
        coo.row.tolist(), coo.col.tolist(), coo.data.tolist()
    ):
        per_node[int(left)].append((float(value), int(right)))
        per_node[int(right)].append((float(value), int(left)))
    for node, values in per_node.items():
        values.sort(key=lambda item: (-item[0], item[1]))
        for score, other in values[:max_per_node]:
            key = (node, other) if node < other else (other, node)
            selected[key] = max(selected.get(key, 0.0), score)
    return (observed, selected)


def _candidate_table(
    data: TopicTreeDataset,
    graph: HypergraphData,
    max_open_per_node: int = DEFAULT_OPEN_PER_NODE,
) -> pd.DataFrame:
    pair_layer = factor_pair_layer(graph, np.ones(len(graph.members), dtype=bool))
    (observed, open_pairs) = _open_pair_candidates(pair_layer, graph, max_open_per_node)
    index_to_node = dict(enumerate(graph.ids))
    node_to_index = {node: index for (index, node) in index_to_node.items()}
    (midpoint, span) = _angle_features(data, graph.ids)
    history = _node_history_features(data, graph.ids)
    rows: list[dict[str, Any]] = []
    for (left, right), mass in observed.items():
        (node_left, node_right) = (index_to_node[left], index_to_node[right])
        (level_left, level_right) = (
            int(data.tree[node_left]["level"]),
            int(data.tree[node_right]["level"]),
        )
        if level_left != level_right:
            continue
        (parent_left, parent_right) = (
            data.parent.get(node_left, ""),
            data.parent.get(node_right, ""),
        )
        rows.append(
            {
                "node_1_index": left,
                "node_2_index": right,
                "node_1_id": node_left,
                "node_2_id": node_right,
                "node_1_name": data.tree[node_left]["name"],
                "node_2_name": data.tree[node_right]["name"],
                "relation_type": "horizontal",
                "level": level_left,
                "candidate_origin": "historically_observed",
                "historical_pair_mass": float(mass),
                "historical_common_neighbors": 0.0,
                "same_parent": bool(parent_left and parent_left == parent_right),
            }
        )
    for (left, right), common in open_pairs.items():
        if (left, right) in observed:
            continue
        (node_left, node_right) = (index_to_node[left], index_to_node[right])
        (level_left, level_right) = (
            int(data.tree[node_left]["level"]),
            int(data.tree[node_right]["level"]),
        )
        if level_left != level_right:
            continue
        (parent_left, parent_right) = (
            data.parent.get(node_left, ""),
            data.parent.get(node_right, ""),
        )
        rows.append(
            {
                "node_1_index": left,
                "node_2_index": right,
                "node_1_id": node_left,
                "node_2_id": node_right,
                "node_1_name": data.tree[node_left]["name"],
                "node_2_name": data.tree[node_right]["name"],
                "relation_type": "horizontal",
                "level": level_left,
                "candidate_origin": "historical_open_pair_common_neighbor",
                "historical_pair_mass": 0.0,
                "historical_common_neighbors": float(common),
                "same_parent": bool(parent_left and parent_left == parent_right),
            }
        )
    for child, parent in data.parent.items():
        if child not in node_to_index or parent not in node_to_index:
            continue
        (child_index, parent_index) = (node_to_index[child], node_to_index[parent])
        rows.append(
            {
                "node_1_index": parent_index,
                "node_2_index": child_index,
                "node_1_id": parent,
                "node_2_id": child,
                "node_1_name": data.tree[parent]["name"],
                "node_2_name": data.tree[child]["name"],
                "relation_type": "vertical_direct_parent_child",
                "level": int(data.tree[child]["level"]),
                "candidate_origin": "real_direct_tree_edge",
                "historical_pair_mass": 0.0,
                "historical_common_neighbors": 0.0,
                "same_parent": False,
            }
        )
    table = pd.DataFrame(rows).drop_duplicates(
        ["node_1_index", "node_2_index", "relation_type"]
    )
    if table.empty:
        raise ValueError(f"截至 {data.end_year} 年没有可生成的合法 PMID 弦候选。")
    left = table.node_1_index.to_numpy(int)
    right = table.node_2_index.to_numpy(int)
    table["node_1_total_pmid"] = history["total"][left]
    table["node_2_total_pmid"] = history["total"][right]
    table["node_1_recent_3y_pmid"] = history["recent"][left]
    table["node_2_recent_3y_pmid"] = history["recent"][right]
    table["node_1_previous_3y_pmid"] = history["previous"][left]
    table["node_2_previous_3y_pmid"] = history["previous"][right]
    table["node_1_growth"] = history["growth"][left]
    table["node_2_growth"] = history["growth"][right]
    table["node_1_temporal_slope"] = history["slope"][left]
    table["node_2_temporal_slope"] = history["slope"][right]
    table["angle_distance"] = np.minimum(
        np.abs(midpoint[left] - midpoint[right]),
        360 - np.abs(midpoint[left] - midpoint[right]),
    )
    table["mean_node_sector_span"] = (span[left] + span[right]) / 2.0
    table["child_to_parent_pmid_ratio"] = np.where(
        table.relation_type.eq("vertical_direct_parent_child"),
        history["total"][right] / np.maximum(history["total"][left], 1.0),
        0.0,
    )
    max_level = max(int(max((data.tree[node]["level"] for node in graph.ids))), 1)
    table["level_scaled"] = table.level / max_level
    table["same_parent_numeric"] = table.same_parent.astype(float)
    table["is_vertical"] = table.relation_type.eq(
        "vertical_direct_parent_child"
    ).astype(float)
    return table.reset_index(drop=True)


FEATURE_COLUMNS = [
    "log_node_1_total",
    "log_node_2_total",
    "log_min_node_total",
    "log_max_node_total",
    "node_1_recent_log",
    "node_2_recent_log",
    "node_1_previous_log",
    "node_2_previous_log",
    "node_1_growth",
    "node_2_growth",
    "node_1_temporal_slope",
    "node_2_temporal_slope",
    "historical_pair_mass",
    "historical_common_neighbors_log",
    "angle_distance_scaled",
    "mean_node_sector_span_scaled",
    "child_to_parent_pmid_ratio_log",
    "level_scaled",
    "same_parent_numeric",
    "is_vertical",
]


def make_features(table: pd.DataFrame) -> np.ndarray:
    work = table.copy()
    work["log_node_1_total"] = np.log1p(work.node_1_total_pmid)
    work["log_node_2_total"] = np.log1p(work.node_2_total_pmid)
    work["log_min_node_total"] = np.log1p(
        np.minimum(work.node_1_total_pmid, work.node_2_total_pmid)
    )
    work["log_max_node_total"] = np.log1p(
        np.maximum(work.node_1_total_pmid, work.node_2_total_pmid)
    )
    work["node_1_recent_log"] = np.log1p(work.node_1_recent_3y_pmid)
    work["node_2_recent_log"] = np.log1p(work.node_2_recent_3y_pmid)
    work["node_1_previous_log"] = np.log1p(work.node_1_previous_3y_pmid)
    work["node_2_previous_log"] = np.log1p(work.node_2_previous_3y_pmid)
    work["historical_common_neighbors_log"] = np.log1p(work.historical_common_neighbors)
    work["angle_distance_scaled"] = work.angle_distance / 180.0
    work["mean_node_sector_span_scaled"] = work.mean_node_sector_span / 360.0
    work["child_to_parent_pmid_ratio_log"] = np.log1p(
        np.clip(work.child_to_parent_pmid_ratio, 0, 1000000.0)
    )
    return (
        work[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _future_truth(
    data: TopicTreeDataset, candidates: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    graph = build_pmid_hypergraph(data)
    calibration = levelwise_pmid_significance(data, graph)
    vertical = significant_vertical_evidence(data, graph, calibration["active_sources"])
    {node: index for (index, node) in enumerate(graph.ids)}
    horizontal_sig: dict[tuple[str, str], float] = {}
    horizontal_mass: dict[tuple[str, str], float] = {}
    if not calibration["horizontal"].empty:
        frame = calibration["horizontal"]
        for row in frame[
            frame.significant_horizontal_relation.astype(bool)
        ].itertuples():
            key = _pair_key(
                graph.ids[int(row.node_1_index)], graph.ids[int(row.node_2_index)]
            )
            horizontal_sig[key] = float(row.evidence_z)
            horizontal_mass[key] = float(row.observed_fractional_pair_mass)
    vertical_sig: dict[tuple[str, str], float] = {}
    if not vertical.empty:
        for row in vertical[
            vertical.significant_vertical_relation.astype(bool)
        ].itertuples():
            key = (graph.ids[int(row.parent_index)], graph.ids[int(row.child_index)])
            vertical_sig[key] = float(row.evidence_z)
    future_keys = {("horizontal",) + key for key in horizontal_sig} | {
        ("vertical_direct_parent_child",) + key for key in vertical_sig
    }
    candidate_keys = {
        ("horizontal",) + _pair_key(str(row.node_1_id), str(row.node_2_id))
        if row.relation_type == "horizontal"
        else ("vertical_direct_parent_child", str(row.node_1_id), str(row.node_2_id))
        for row in candidates.itertuples()
    }
    labels = np.zeros(len(candidates), dtype=np.int8)
    z_values = np.zeros(len(candidates), dtype=float)
    masses = np.zeros(len(candidates), dtype=float)
    for position, row in enumerate(candidates.itertuples()):
        if row.relation_type == "horizontal":
            key = _pair_key(str(row.node_1_id), str(row.node_2_id))
            if key in horizontal_sig:
                labels[position] = 1
                z_values[position] = horizontal_sig[key]
                masses[position] = horizontal_mass.get(key, 0.0)
        else:
            key = (str(row.node_1_id), str(row.node_2_id))
            if key in vertical_sig:
                labels[position] = 1
                z_values[position] = vertical_sig[key]
    return (
        labels,
        z_values,
        masses,
        len(future_keys),
        len(future_keys & candidate_keys) / max(len(future_keys), 1),
    )


def build_candidate_batch(
    tree_path: Path,
    papers_path: Path,
    cutoff: int,
    horizon: int,
    max_open_per_node: int,
) -> CandidateBatch:
    history = TopicTreeDataset(tree_path, papers_path, TIME.observation_start, cutoff)
    graph = build_pmid_hypergraph(history)
    candidates = _candidate_table(history, graph, max_open_per_node=max_open_per_node)
    features = make_features(candidates)
    future_end = min(cutoff + horizon, OBSERVATION_END)
    if future_end <= cutoff:
        labels = np.zeros(len(candidates), dtype=np.int8)
        strength = np.zeros(len(candidates), dtype=float)
        mass = np.zeros(len(candidates), dtype=float)
    else:
        future = TopicTreeDataset(tree_path, papers_path, cutoff + 1, future_end)
        (labels, strength, mass, future_count, coverage) = _future_truth(
            future, candidates
        )
    if future_end <= cutoff:
        (future_count, coverage) = (0, np.nan)
    candidates["cutoff"] = cutoff
    candidates["future_window_start"] = cutoff + 1
    candidates["future_window_end"] = future_end
    candidates["future_screen_activated"] = labels.astype(bool)
    candidates["future_evidence_z"] = strength
    candidates["future_horizontal_corrected_fractional_mass"] = mass
    candidates["historical_candidate_node_count"] = len(graph.ids)
    candidates["candidate_coverage_scope"] = (
        "历史有PMID节点；已观测同层对+共同邻居开放对+真实直接父子边"
    )
    candidates["future_significant_edge_count"] = future_count
    candidates["future_candidate_coverage"] = coverage
    return CandidateBatch(
        cutoff,
        horizon,
        history,
        graph,
        candidates,
        features,
        labels,
        strength,
        mass,
        future_end,
        future_count,
        coverage,
    )


def _snapshot_blocks(
    tree_path: Path, papers_path: Path, start: int | None, end: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]], pd.DataFrame]:
    data = TopicTreeDataset(tree_path, papers_path, start, end)
    graph = build_pmid_hypergraph(data)
    calibration = levelwise_pmid_significance(data, graph)
    vertical = significant_vertical_evidence(data, graph, calibration["active_sources"])
    (members, blocks, _, resolution) = significant_pmid_blocks(
        data, graph, calibration["horizontal"], vertical, seed, trial_count=2
    )
    chords = build_readable_pmid_chord_list(
        data,
        graph,
        calibration["horizontal"],
        vertical,
        members,
        structure="S_snapshot",
        membership_column="significance_block_id",
        status_column="node_status",
    )
    (summary, _, _, _) = build_community_catalog(
        data,
        graph,
        members,
        "significance_block_id",
        chords,
        calibration["active_sources"],
        structure="S_snapshot",
    )
    (screened_summary, confirmed) = apply_posthoc_community_screen(summary)
    screened_summary = screened_summary.copy()
    screened_summary["criterion_node_count_ge_3"] = screened_summary.node_count.ge(3)
    screened_summary["formal_community_confirmed"] = (
        screened_summary.formal_community_confirmed.astype(bool)
        & screened_summary.criterion_node_count_ge_3
    )
    screened_summary["posthoc_screening_result"] = np.where(
        screened_summary.criterion_node_count_ge_3,
        screened_summary.posthoc_screening_result,
        screened_summary.posthoc_screening_result.astype(str) + "；节点数<3",
    )
    screened_summary["posthoc_structure_class"] = np.where(
        screened_summary.formal_community_confirmed,
        "confirmed_multibranch_community",
        np.where(
            screened_summary.node_count.lt(3),
            "singleton_or_partition_fragment",
            screened_summary.posthoc_structure_class,
        ),
    )
    confirmed = set(
        screened_summary.loc[
            screened_summary.formal_community_confirmed, "community_id"
        ].astype(str)
    )
    screened_summary.insert(
        0, "snapshot_start", int(start if start is not None else data.year_values.min())
    )
    screened_summary.insert(1, "snapshot_end", int(end))
    screened_summary.insert(
        2, "snapshot_type", "cumulative" if start is None else "period_only"
    )
    confirmed = {str(value) for value in confirmed}
    snapshot = "cumulative" if start is None else "period_only"
    block_rows: list[dict[str, Any]] = []
    sets: dict[str, set[str]] = {}
    for row in blocks[
        blocks.significance_block_id.astype(str).isin(confirmed)
    ].itertuples(index=False):
        community = str(row.significance_block_id)
        node_part = members[
            members.significance_block_id.astype(str).eq(community)
        ].copy()
        ordered = node_part.sort_values(
            [
                "significant_horizontal_degree",
                "significant_vertical_degree",
                "level",
                "node_id",
            ],
            ascending=[False, False, True, True],
            kind="stable",
        )
        node_ids = ordered.node_id.astype(str).tolist()
        sets[community] = set(node_ids)
        block_rows.append(
            {
                "snapshot_type": snapshot,
                "snapshot_start": int(
                    start if start is not None else data.year_values.min()
                ),
                "snapshot_end": int(end),
                "community_id": community,
                "node_count": int(len(node_ids)),
                "lowest_level": int(row.lowest_level),
                "highest_level": int(row.highest_level),
                "level_distribution": row.levels_present,
                "significant_horizontal_relations": int(
                    row.significant_horizontal_relations
                ),
                "significant_direct_parent_child_relations": int(
                    row.significant_direct_parent_child_relations
                ),
                "total_evidence_z": float(row.total_evidence_z),
                "top15_node_ids": " | ".join(node_ids[:15]),
                "top15_node_names": " | ".join(
                    (data.tree[node]["name"] for node in node_ids[:15])
                ),
                "definition": "按当前 PMID 逐层校正、直接父子约束和 CPM S 结构块流程独立重建。",
            }
        )
    block_table = pd.DataFrame(block_rows)
    members = members.copy()
    members.insert(0, "snapshot_type", snapshot)
    members.insert(
        1, "snapshot_start", int(start if start is not None else data.year_values.min())
    )
    members.insert(2, "snapshot_end", int(end))
    members = members[members.significance_block_id.astype(str).isin(confirmed)].copy()
    resolution.insert(0, "snapshot_type", snapshot)
    resolution.insert(
        1, "snapshot_start", int(start if start is not None else data.year_values.min())
    )
    resolution.insert(2, "snapshot_end", int(end))
    block_table.attrs["resolution"] = resolution
    return (block_table, members, sets, screened_summary)


def _snapshot_top15_ids(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item.strip() for item in str(value).split(" | ") if item.strip()]


def align_snapshot_blocks(
    snapshot_blocks: list[pd.DataFrame], snapshot_sets: list[dict[str, set[str]]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(len(snapshot_blocks) - 1):
        (old_table, new_table) = (snapshot_blocks[index], snapshot_blocks[index + 1])
        (old_sets, new_sets) = (snapshot_sets[index], snapshot_sets[index + 1])
        old_meta = old_table.set_index("community_id").to_dict("index")
        new_meta = new_table.set_index("community_id").to_dict("index")
        remaining_targets = set(new_sets)
        old_order = sorted(
            old_sets,
            key=lambda community: (
                -int(old_meta[community].get("node_count", len(old_sets[community]))),
                str(community),
            ),
        )
        assigned_targets: set[str] = set()
        for match_order, old_id in enumerate(old_order, start=1):
            old_nodes = old_sets[old_id]
            ranked_top15 = _snapshot_top15_ids(
                old_meta[old_id].get("top15_node_ids", "")
            )[:15]
            if not ranked_top15:
                ranked_top15 = sorted(old_nodes)[:15]
            candidates: list[tuple[int, int, str, list[str]]] = []
            for new_id in sorted(remaining_targets):
                hits = [node for node in ranked_top15 if node in new_sets[new_id]]
                candidates.append((len(hits), len(new_sets[new_id]), new_id, hits))
            candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
            if not candidates or candidates[0][0] < 1:
                rows.append(
                    {
                        "from_snapshot": f"{old_meta[old_id]['snapshot_start']}-{old_meta[old_id]['snapshot_end']}",
                        "to_snapshot": f"{new_table.snapshot_start.iloc[0]}-{new_table.snapshot_end.iloc[0]}",
                        "from_community_id": old_id,
                        "to_community_id": "",
                        "event_type": "dissolution_or_unmatched",
                        "matching_method": "top15_presence_greedy_one_to_one",
                        "matching_order_by_source_node_count": match_order,
                        "top15_presence_score": 0,
                        "top15_source_node_count": len(ranked_top15),
                        "top15_presence_fraction": 0.0,
                        "matched_top15_node_ids": "",
                        "node_overlap": 0,
                        "node_jaccard": 0.0,
                        "from_node_count": int(len(old_nodes)),
                        "to_node_count": 0,
                        "jaccard_role": "report_only_not_used_for_matching",
                    }
                )
                continue
            (score, _, new_id, hits) = candidates[0]
            remaining_targets.remove(new_id)
            assigned_targets.add(new_id)
            new_nodes = new_sets[new_id]
            overlap = len(old_nodes & new_nodes)
            union = len(old_nodes | new_nodes)
            jaccard = overlap / max(union, 1)
            rows.append(
                {
                    "from_snapshot": f"{old_meta[old_id]['snapshot_start']}-{old_meta[old_id]['snapshot_end']}",
                    "to_snapshot": f"{new_meta[new_id]['snapshot_start']}-{new_meta[new_id]['snapshot_end']}",
                    "from_community_id": old_id,
                    "to_community_id": new_id,
                    "event_type": "best_match",
                    "matching_method": "top15_presence_greedy_one_to_one",
                    "matching_order_by_source_node_count": match_order,
                    "top15_presence_score": int(score),
                    "top15_source_node_count": int(len(ranked_top15)),
                    "top15_presence_fraction": float(score / max(len(ranked_top15), 1)),
                    "matched_top15_node_ids": " | ".join(hits),
                    "node_overlap": int(overlap),
                    "node_jaccard": float(jaccard),
                    "from_node_count": int(len(old_nodes)),
                    "to_node_count": int(len(new_nodes)),
                    "jaccard_role": "report_only_not_used_for_matching",
                }
            )
        for new_id in sorted(remaining_targets):
            new_nodes = new_sets[new_id]
            rows.append(
                {
                    "from_snapshot": f"{old_table.snapshot_start.iloc[0]}-{old_table.snapshot_end.iloc[0]}",
                    "to_snapshot": f"{new_meta[new_id]['snapshot_start']}-{new_meta[new_id]['snapshot_end']}",
                    "from_community_id": "",
                    "to_community_id": new_id,
                    "event_type": "birth_or_unmatched",
                    "matching_method": "top15_presence_greedy_one_to_one",
                    "matching_order_by_source_node_count": "",
                    "top15_presence_score": 0,
                    "top15_source_node_count": 0,
                    "top15_presence_fraction": 0.0,
                    "matched_top15_node_ids": "",
                    "node_overlap": 0,
                    "node_jaccard": 0.0,
                    "from_node_count": 0,
                    "to_node_count": int(len(new_nodes)),
                    "jaccard_role": "report_only_not_used_for_matching",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["split_signal"] = False
    result["merge_signal"] = False
    result["evolution_interpretation"] = np.select(
        [
            result.event_type.eq("birth_or_unmatched"),
            result.event_type.eq("dissolution_or_unmatched"),
        ],
        ["possible_birth", "possible_dissolution"],
        default="continuity_by_top15_presence",
    )
    return result


def _read_frozen_gamma(
    output_root: Path,
    *,
    gamma_csv: Path | None = None,
    project_root: Path | None = None,
) -> float:
    if gamma_csv is not None:
        candidates: list[Path] = [Path(gamma_csv)]
    else:
        candidates = []
        if output_root is not None:
            candidates.append(
                output_root.parent
                / "03_hypergraph"
                / "tables"
                / "12_PMID逐层显著性_结构块尺度校准.csv"
            )
    for path in candidates:
        if path.is_file():
            try:
                table = pd.read_csv(path)
                if "selected" not in table or "resolution_gamma" not in table:
                    continue
                selected = table["selected"]
                if pd.api.types.is_bool_dtype(selected):
                    mask = selected.fillna(False)
                else:
                    mask = (
                        selected.astype(str)
                        .str.strip()
                        .str.lower()
                        .isin({"1", "true", "yes", "y"})
                    )
                values = table.loc[mask, "resolution_gamma"]
                if len(values):
                    return float(values.iloc[0])
            except Exception:
                continue
    raise FileNotFoundError(
        "未找到冻结 CPM 分辨率 γ 的来源表（12_PMID逐层显著性_结构块尺度校准.csv）。请先运行主管线生成该表，或通过 --gamma-csv 显式指定。已尝试："
        + "；".join((str(p) for p in candidates))
    )


def simulate_forecast_blocks(
    final_batch: CandidateBatch,
    model: UnifiedTemporalEdgeModel,
    output: Output,
    seed: int,
    simulations: int,
    gamma: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (probability, conditional, expected) = model.predict(final_batch.features)
    table = final_batch.candidates.copy()
    table["predicted_activation_probability"] = probability
    table["predicted_strength_if_active"] = conditional
    table["predicted_expected_strength"] = expected
    horizontal = table[table.relation_type.eq("horizontal")].copy()
    vertical = table[table.relation_type.eq("vertical_direct_parent_child")].copy()
    horizontal = horizontal.sort_values(
        "predicted_expected_strength", ascending=False
    ).head(5000)
    vertical = vertical.sort_values(
        "predicted_expected_strength", ascending=False
    ).head(5000)
    rng = np.random.default_rng(seed)
    module_rows: list[dict[str, Any]] = []
    block_node_rows: list[dict[str, Any]] = []
    block_edge_rows: list[dict[str, Any]] = []
    graph_ids = list(final_batch.graph.ids)
    for simulation in range(simulations):
        h_keep = rng.random(
            len(horizontal)
        ) < horizontal.predicted_activation_probability.to_numpy(float)
        v_keep = rng.random(
            len(vertical)
        ) < vertical.predicted_activation_probability.to_numpy(float)
        if not h_keep.any() and len(horizontal):
            h_keep[
                np.argmax(horizontal.predicted_expected_strength.to_numpy(float))
            ] = True
        if not v_keep.any() and len(vertical):
            v_keep[np.argmax(vertical.predicted_expected_strength.to_numpy(float))] = (
                True
            )
        h = horizontal.loc[
            h_keep, ["node_1_index", "node_2_index", "predicted_strength_if_active"]
        ].copy()
        h = h.rename(columns={"predicted_strength_if_active": "evidence_z"})
        h["significant_horizontal_relation"] = True
        v = vertical.loc[
            v_keep, ["node_1_index", "node_2_index", "predicted_strength_if_active"]
        ].copy()
        v = v.rename(
            columns={
                "node_1_index": "parent_index",
                "node_2_index": "child_index",
                "predicted_strength_if_active": "evidence_z",
            }
        )
        v["significant_vertical_relation"] = True
        if h.empty and v.empty:
            continue
        (members, blocks, _, _) = significant_pmid_blocks(
            final_batch.data,
            final_batch.graph,
            h,
            v,
            seed + simulation,
            resolution_grid=(gamma,),
            trial_count=1,
        )
        sampled_edges = _simulation_module_edge_table(graph_ids, h, v)
        for block in blocks.itertuples(index=False):
            block_id = str(block.significance_block_id)
            part = members[members.significance_block_id.astype(str).eq(block_id)]
            ranked_nodes = _rank_nodes_by_internal_forecast_evidence(
                final_batch.data, part.node_id.astype(str).tolist(), sampled_edges
            )
            node_ids = ranked_nodes.node_id.astype(str).tolist()
            if not node_ids:
                continue
            module_rows.append(
                {
                    "simulation_id": simulation + 1,
                    "simulation_community_id": block_id,
                    "predicted_node_count": int(len(node_ids)),
                    "predicted_lowest_level": int(block.lowest_level),
                    "predicted_highest_level": int(block.highest_level),
                    "predicted_horizontal_edge_count": int(
                        block.significant_horizontal_relations
                    ),
                    "predicted_vertical_edge_count": int(
                        block.significant_direct_parent_child_relations
                    ),
                    "predicted_total_evidence_z": float(block.total_evidence_z),
                    "predicted_top15_node_ids": " | ".join(node_ids[:15]),
                    "predicted_top15_node_names": " | ".join(
                        (final_batch.data.tree[node]["name"] for node in node_ids[:15])
                    ),
                    "predicted_top15_ranking_definition": "模块内部保留预测弦的正向证据强度之和；横向度、纵向度、层级和节点ID依次打破并列",
                }
            )
            for ranked in ranked_nodes.itertuples(index=False):
                block_node_rows.append(
                    {
                        "simulation_id": simulation + 1,
                        "simulation_community_id": block_id,
                        "node_id": ranked.node_id,
                        "node_name": ranked.node_name,
                        "level": int(ranked.level),
                        "simulation_module_node_rank": int(ranked.module_node_rank),
                        "internal_evidence_score": float(
                            ranked.internal_evidence_score
                        ),
                        "internal_horizontal_degree": int(
                            ranked.internal_horizontal_degree
                        ),
                        "internal_vertical_degree": int(
                            ranked.internal_vertical_degree
                        ),
                    }
                )
            block_node_set = set(node_ids)
            for edge in h.itertuples(index=False):
                node_1 = str(graph_ids[int(edge.node_1_index)])
                node_2 = str(graph_ids[int(edge.node_2_index)])
                if node_1 not in block_node_set or node_2 not in block_node_set:
                    continue
                block_edge_rows.append(
                    {
                        "simulation_id": simulation + 1,
                        "simulation_community_id": block_id,
                        "relation_type": "horizontal",
                        "node_1_id": node_1,
                        "node_2_id": node_2,
                        "node_1_name": final_batch.data.tree[node_1]["name"],
                        "node_2_name": final_batch.data.tree[node_2]["name"],
                        "level": int(final_batch.data.tree[node_1]["level"]),
                        "evidence_z": float(edge.evidence_z),
                    }
                )
            for edge in v.itertuples(index=False):
                parent = str(graph_ids[int(edge.parent_index)])
                child = str(graph_ids[int(edge.child_index)])
                if parent not in block_node_set or child not in block_node_set:
                    continue
                block_edge_rows.append(
                    {
                        "simulation_id": simulation + 1,
                        "simulation_community_id": block_id,
                        "relation_type": "vertical_direct_parent_child",
                        "node_1_id": parent,
                        "node_2_id": child,
                        "node_1_name": final_batch.data.tree[parent]["name"],
                        "node_2_name": final_batch.data.tree[child]["name"],
                        "level": int(final_batch.data.tree[child]["level"]),
                        "evidence_z": float(edge.evidence_z),
                    }
                )
    modules = pd.DataFrame(module_rows)
    nodes = pd.DataFrame(block_node_rows)
    edges = pd.DataFrame(block_edge_rows)
    if modules.empty:
        return (modules, nodes, edges)
    signatures: list[set[str]] = []
    canonical: list[dict[str, Any]] = []
    for row in modules.itertuples(index=False):
        signature = set(str(row.predicted_top15_node_ids).split(" | ")) - {""}
        (best_index, best_score) = (None, 0.0)
        for index, existing in enumerate(signatures):
            score = len(signature & existing) / max(len(signature | existing), 1)
            if score > best_score:
                (best_index, best_score) = (index, score)
        if best_index is None or best_score < 0.4:
            signatures.append(signature)
            canonical.append(
                {
                    "future_module_id": f"F{len(signatures)}",
                    "signature": signature,
                    "simulation_ids": set(),
                }
            )
            best_index = len(canonical) - 1
        canonical[best_index]["simulation_ids"].add(int(row.simulation_id))
    assignments: list[str] = []
    for row in modules.itertuples(index=False):
        signature = set(str(row.predicted_top15_node_ids).split(" | ")) - {""}
        best = max(
            canonical,
            key=lambda item: len(signature & item["signature"])
            / max(len(signature | item["signature"]), 1),
        )
        assignments.append(best["future_module_id"])
    modules["future_module_id"] = assignments
    module_mapping = modules[
        ["simulation_id", "simulation_community_id", "future_module_id"]
    ]
    nodes = nodes.merge(
        module_mapping, on=["simulation_id", "simulation_community_id"], how="left"
    )
    edges = edges.merge(
        module_mapping, on=["simulation_id", "simulation_community_id"], how="left"
    )
    return (modules, nodes, edges)


def forecast_node_counts(
    final_batch: CandidateBatch, end_year: int = FORECAST_END
) -> pd.DataFrame:
    ids = final_batch.graph.ids
    history = _node_history_features(final_batch.data, ids)
    years = history["years"]
    rows: list[dict[str, Any]] = []
    for position, node in enumerate(ids):
        recent = history["recent"][position] / max(min(3, len(years)), 1)
        slope = history["slope"][position] * max(recent, 1.0)
        for year in range(FORECAST_START, end_year + 1):
            steps = year - OBSERVATION_END
            predicted = max(0.0, recent + slope * steps)
            rows.append(
                {
                    "node_id": node,
                    "node_name": final_batch.data.tree[node]["name"],
                    "level": int(final_batch.data.tree[node]["level"]),
                    TIME.key("historical_pmid_count_to_2025"): int(
                        history["total"][position]
                    ),
                    "recent_3_year_mean_pmid": float(recent),
                    "historical_normalized_slope": float(history["slope"][position]),
                    "forecast_year": year,
                    "predicted_node_pmid_count_baseline": float(predicted),
                    "forecast_note": "节点论文量为暴露量基线预测，用于解释弦强度；未来结构结论以弦预测为主。",
                }
            )
    return pd.DataFrame(rows)


def write_gzip_table(output: Output, frame: pd.DataFrame, stem: str) -> Path:
    path = output.tables / f"{stem}.csv.gz"
    frame.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
    return path


def run_temporal_prediction(
    tree_path: Path,
    papers_path: Path,
    output_root: Path,
    seed: int = SEED,
    max_open_per_node: int = DEFAULT_OPEN_PER_NODE,
    simulations: int = DEFAULT_SIMULATIONS,
    skip_snapshots: bool = False,
    accelerator: str = "auto",
    gamma_csv: Path | None = None,
) -> None:
    output = Output(output_root)
    full_data = TopicTreeDataset(
        tree_path, papers_path, TIME.observation_start, OBSERVATION_END
    )
    (_available_start, _available_end) = (
        int(full_data.year_values.min()),
        int(full_data.year_values.max()),
    )
    fold_specs_5 = [(year, TIME.forecast_years) for year in TIME.training_cutoffs]
    fold_specs_3 = [(year, TIME.emerging_years) for year in TIME.training_cutoffs]
    fold_cache: dict[tuple[int, int], CandidateBatch] = {}

    def get_batch(cutoff: int, horizon: int) -> CandidateBatch:
        key = (cutoff, horizon)
        if key not in fold_cache:
            print(
                f"[预测数据] 构建 {cutoff} -> {cutoff + horizon} 候选与未来标签",
                flush=True,
            )
            fold_cache[key] = build_candidate_batch(
                tree_path, papers_path, cutoff, horizon, max_open_per_node
            )
        return fold_cache[key]

    def run_horizon(
        specs: list[tuple[int, int]], label: str
    ) -> tuple[list[CandidateBatch], UnifiedTemporalEdgeModel]:
        batches = [get_batch(cutoff, horizon) for (cutoff, horizon) in specs]
        final_model = UnifiedTemporalEdgeModel().fit(
            batches, accelerator=accelerator, seed=seed + 100
        )
        final_batch = build_candidate_batch(
            tree_path, papers_path, OBSERVATION_END, specs[0][1], max_open_per_node
        )
        (probability, conditional, expected) = final_model.predict(final_batch.features)
        forecast = final_batch.candidates.copy()
        forecast["forecast_window_start"] = FORECAST_START
        forecast["forecast_window_end"] = FORECAST_START + specs[0][1] - 1
        forecast["horizon_label"] = label
        forecast["predicted_activation_probability"] = probability
        forecast["predicted_strength_if_active"] = conditional
        forecast["predicted_expected_strength"] = expected
        forecast["historical_status"] = np.where(
            forecast.historical_pair_mass > 0,
            "historically_observed_relation",
            "historically_unobserved_candidate",
        )
        forecast["predicted_relation_status"] = np.select(
            [
                forecast.predicted_activation_probability >= 0.5,
                forecast.predicted_activation_probability >= 0.2,
            ],
            ["high_probability_candidate", "intermediate_probability_candidate"],
            default="exploratory_low_probability_candidate",
        )
        out_name = (
            (
                "05_五年弦预测"
                if TIME.forecast_years == 5
                else f"05_{TIME.forecast_years}年弦预测"
            )
            if label == TIME.primary_label
            else "06_三年新兴弦预测"
            if TIME.emerging_years == 3
            else f"06_{TIME.emerging_years}年新兴弦预测"
        )
        keep_columns = [
            "node_1_id",
            "node_2_id",
            "node_1_name",
            "node_2_name",
            "relation_type",
            "level",
            "candidate_origin",
            "historical_pair_mass",
            "historical_common_neighbors",
            "node_1_total_pmid",
            "node_2_total_pmid",
            "predicted_activation_probability",
            "predicted_strength_if_active",
            "predicted_expected_strength",
            "historical_status",
            "predicted_relation_status",
            "forecast_window_start",
            "forecast_window_end",
            "horizon_label",
        ]
        compact_columns = [
            "node_1_id",
            "node_2_id",
            "relation_type",
            "level",
            "candidate_origin",
            "historical_pair_mass",
            "historical_common_neighbors",
            "predicted_activation_probability",
            "predicted_strength_if_active",
            "predicted_expected_strength",
            "historical_status",
            "predicted_relation_status",
            "forecast_window_start",
            "forecast_window_end",
            "horizon_label",
        ]
        compact_path = output.tables / f"{out_name}_全部合法候选弦.csv.gz"
        forecast.sort_values(
            ["predicted_expected_strength", "predicted_activation_probability"],
            ascending=False,
        )[compact_columns].to_csv(
            compact_path, index=False, encoding="utf-8-sig", compression="gzip"
        )
        output.table(
            forecast.sort_values("predicted_expected_strength", ascending=False).head(
                1000
            )[keep_columns],
            f"{out_name}_Top1000",
        )
        return ([*batches, final_batch], final_model)

    (batches_5, model_5) = run_horizon(fold_specs_5, TIME.primary_label)
    (batches_3, model_3) = run_horizon(fold_specs_3, TIME.emerging_label)
    final_batch = batches_5[-1]
    final_forecast_table = final_batch.candidates.copy()
    (p5, c5, e5) = model_5.predict(final_batch.features)
    final_forecast_table["predicted_activation_probability"] = p5
    final_forecast_table["predicted_strength_if_active"] = c5
    final_forecast_table["predicted_expected_strength"] = e5
    model_table = pd.DataFrame(
        {
            "horizon_label": [TIME.primary_label, TIME.emerging_label],
            "training_fold_count": [len(fold_specs_5), len(fold_specs_3)],
            "model_type": [
                "共享节点历史特征 + Logistic 激活模型 + 正向证据强度 Ridge 回归（固定两阶段参考模型）"
            ]
            * 2,
            "fit_backend": [model_5.backend, model_3.backend],
            "fit_device": [model_5.device, model_3.device],
            "requested_accelerator": [
                model_5.requested_accelerator,
                model_3.requested_accelerator,
            ],
            "cuda_available": [model_5.cuda_available, model_3.cuda_available],
            "accelerator_policy": [model_5.accelerator_note, model_3.accelerator_note],
            "activation_target": ["future exact existing edge screen retained"] * 2,
            "strength_target": ["future retained edge evidence_z"] * 2,
            "post_prediction_filter": [
                "same-level horizontal or real direct parent-child only"
            ]
            * 2,
        }
    )
    output.table(model_table, "04_统一时间弦模型定义")
    write_gzip_table(
        output, forecast_node_counts(final_batch), "07_节点年度论文量基线预测"
    )
    print("[未来结构] 生成概率未来网络并运行冻结 CPM S 结构块发现", flush=True)
    gamma = _read_frozen_gamma(output_root, gamma_csv=gamma_csv)
    (modules, module_nodes, module_edges) = simulate_forecast_blocks(
        final_batch, model_5, output, seed, simulations, gamma
    )
    output.table(modules, "08_未来概率网络模拟_结构块明细")
    output.table(module_nodes, "08_未来概率网络模拟_结构块节点")
    output.table(module_edges, "08_未来概率网络模拟_结构块弦")
    merged_edges_all = pd.DataFrame()
    merged_core_nodes = pd.DataFrame()
    if not modules.empty:
        recurrent = (
            modules.groupby("future_module_id")
            .agg(
                simulation_count=("simulation_id", "nunique"),
                predicted_block_count=("simulation_id", "size"),
                median_node_count=("predicted_node_count", "median"),
                median_horizontal_edges=("predicted_horizontal_edge_count", "median"),
                median_vertical_edges=("predicted_vertical_edge_count", "median"),
                median_total_evidence_z=("predicted_total_evidence_z", "median"),
            )
            .reset_index()
        )
        recurrent["simulation_recurrence_rate"] = (
            recurrent.simulation_count / simulations
        )
        if not module_edges.empty:
            merged_edges_all = module_edges.groupby(
                [
                    "future_module_id",
                    "relation_type",
                    "node_1_id",
                    "node_2_id",
                    "node_1_name",
                    "node_2_name",
                    "level",
                ],
                as_index=False,
                sort=False,
            ).agg(
                simulation_count=("simulation_id", "nunique"),
                observed_simulation_edge_count=("simulation_id", "size"),
                mean_predicted_evidence_z=("evidence_z", "mean"),
                max_predicted_evidence_z=("evidence_z", "max"),
            )
            merged_edges_all["simulation_edge_recurrence_rate"] = (
                merged_edges_all.simulation_count / max(simulations, 1)
            )
            edge_counts = (
                merged_edges_all.assign(
                    horizontal_edge=merged_edges_all.relation_type.eq(
                        "horizontal"
                    ).astype(int),
                    vertical_edge=merged_edges_all.relation_type.eq(
                        "vertical_direct_parent_child"
                    ).astype(int),
                )
                .groupby("future_module_id", as_index=False)
                .agg(
                    all_predicted_edge_count=("relation_type", "size"),
                    all_predicted_horizontal_edge_count=("horizontal_edge", "sum"),
                    all_predicted_vertical_edge_count=("vertical_edge", "sum"),
                )
            )
            recurrent = recurrent.merge(edge_counts, on="future_module_id", how="left")
        else:
            merged_edges_all = pd.DataFrame()
            recurrent["all_predicted_edge_count"] = 0
            recurrent["all_predicted_horizontal_edge_count"] = 0
            recurrent["all_predicted_vertical_edge_count"] = 0

        def _join_unique(values: Iterable[Any]) -> str:
            return " | ".join(
                sorted({str(value) for value in values if str(value).strip()})
            )

        union_nodes = (
            module_nodes.groupby("future_module_id", sort=False)
            .agg(
                all_predicted_node_count=("node_id", "nunique"),
                all_predicted_node_ids=("node_id", _join_unique),
                all_predicted_node_names=("node_name", _join_unique),
            )
            .reset_index()
        )
        union_path_counts = (
            module_nodes.groupby("future_module_id", sort=False)["node_id"]
            .apply(
                lambda values: _direct_path_count_for_nodes(final_batch.data, values)
            )
            .rename("all_predicted_path_count")
            .reset_index()
        )
        union_nodes = union_nodes.merge(
            union_path_counts, on="future_module_id", how="left"
        )
        recurrent = recurrent.merge(union_nodes, on="future_module_id", how="left")
        (merged_core_summary, merged_core_nodes) = (
            _summarize_merged_future_module_cores(
                full_data, module_nodes, merged_edges_all
            )
        )
        recurrent = recurrent.merge(
            merged_core_summary, on="future_module_id", how="left"
        )
    else:
        recurrent = pd.DataFrame()
    recurrent_all = recurrent.copy()
    output.table(
        merged_edges_all.sort_values(
            ["future_module_id", "relation_type", "node_1_id", "node_2_id"]
        )
        if not merged_edges_all.empty
        else merged_edges_all,
        "08_未来概率网络模拟_归并后结构块弦_未筛选全体审计",
    )
    output.table(
        merged_core_nodes.sort_values(["future_module_id", "module_node_rank"])
        if not merged_core_nodes.empty
        else merged_core_nodes,
        "08_未来概率网络模拟_归并后结构块核心节点",
    )
    output.table(
        recurrent_all.sort_values("simulation_recurrence_rate", ascending=False)
        if not recurrent_all.empty
        else recurrent_all,
        "08_未来概率网络模拟_重复出现结构块_未筛选全体审计",
    )
    recurrence_pass = (
        recurrent_all.simulation_recurrence_rate.ge(0.25)
        if not recurrent_all.empty
        else pd.Series(dtype=bool)
    )
    recurrent_pre_screen = (
        recurrent_all.loc[recurrence_pass].copy()
        if not recurrent_all.empty
        else recurrent_all.copy()
    )
    recurrent_pre_screen["passes_merged_structure_screen"] = True
    recurrent_pre_screen["merged_structure_screen_exclusion_reason"] = ""
    pre_module_ids = set(recurrent_pre_screen.future_module_id.astype(str))
    pre_screen_edges = (
        merged_edges_all[
            merged_edges_all.future_module_id.astype(str).isin(pre_module_ids)
        ].copy()
        if not merged_edges_all.empty
        else merged_edges_all.copy()
    )
    output.table(
        pre_screen_edges.sort_values(
            ["future_module_id", "relation_type", "node_1_id", "node_2_id"]
        )
        if not pre_screen_edges.empty
        else pre_screen_edges,
        "08_未来概率网络模拟_归并后结构块弦_重复率25%筛选前全体",
    )
    if not recurrent_pre_screen.empty:
        gate_node = recurrent_pre_screen.all_predicted_node_count.ge(3)
        gate_edge = recurrent_pre_screen.all_predicted_edge_count.fillna(0).ge(3)
        gate_horizontal = (
            recurrent_pre_screen.all_predicted_horizontal_edge_count.fillna(0).ge(2)
        )
        gate_path = recurrent_pre_screen.all_predicted_path_count.fillna(0).ge(3)
        recurrent_pre_screen["passes_merged_structure_screen"] = (
            gate_node & gate_edge & gate_horizontal & gate_path
        )
        reasons = []
        for row in recurrent_pre_screen.itertuples():
            failed = []
            if not gate_node.loc[row.Index]:
                failed.append("归并节点<3")
            if not gate_edge.loc[row.Index]:
                failed.append("归并全部边<3")
            if not gate_horizontal.loc[row.Index]:
                failed.append("归并横向边<2")
            if not gate_path.loc[row.Index]:
                failed.append("归并连续路径<3")
            reasons.append("；".join(failed))
        recurrent_pre_screen["merged_structure_screen_exclusion_reason"] = reasons
    recurrent_post_screen = recurrent_pre_screen[
        recurrent_pre_screen.passes_merged_structure_screen
    ].copy()
    output.table(
        recurrent_pre_screen.sort_values("simulation_recurrence_rate", ascending=False),
        "08_未来概率网络模拟_重复率25%结构块_筛选前全体",
    )
    output.table(
        recurrent_post_screen.sort_values(
            "simulation_recurrence_rate", ascending=False
        ),
        "08_未来概率网络模拟_重复率25%结构块_筛选后正式结果",
    )
    output.table(
        recurrent_pre_screen.sort_values("simulation_recurrence_rate", ascending=False),
        "08_未来概率网络模拟_重复率25%结构块_筛选审计",
    )
    recurrent = recurrent_post_screen.copy()
    if not merged_edges_all.empty:
        selected_module_ids = set(recurrent.future_module_id.astype(str))
        merged_edges = merged_edges_all[
            merged_edges_all.future_module_id.astype(str).isin(selected_module_ids)
        ].copy()
    else:
        merged_edges = merged_edges_all
    output.table(
        merged_edges.sort_values(
            ["future_module_id", "relation_type", "node_1_id", "node_2_id"]
        )
        if not merged_edges.empty
        else merged_edges,
        "08_未来概率网络模拟_归并后结构块弦",
    )
    output.table(
        merged_edges.sort_values(
            ["future_module_id", "relation_type", "node_1_id", "node_2_id"]
        )
        if not merged_edges.empty
        else merged_edges,
        "08_未来概率网络模拟_归并后结构块弦_筛选后正式结果",
    )
    output.table(
        recurrent.sort_values("simulation_recurrence_rate", ascending=False)
        if not recurrent.empty
        else recurrent,
        "08_未来概率网络模拟_重复出现结构块",
    )
    if not skip_snapshots:
        snapshot_ends = list(TIME.snapshot_ends)
        snapshot_blocks: list[pd.DataFrame] = []
        snapshot_members: list[pd.DataFrame] = []
        snapshot_sets: list[dict[str, set[str]]] = []
        snapshot_resolution: list[pd.DataFrame] = []
        snapshot_screening: list[pd.DataFrame] = []
        for end in snapshot_ends:
            print(f"[S结构快照] 累积截至 {end} 年重建", flush=True)
            (blocks, members, sets, screening) = _snapshot_blocks(
                tree_path, papers_path, TIME.history_start, end, seed + end
            )
            snapshot_blocks.append(blocks)
            snapshot_members.append(members)
            snapshot_sets.append(sets)
            snapshot_resolution.append(blocks.attrs.get("resolution", pd.DataFrame()))
            snapshot_screening.append(screening)
        block_frames = []
        for frame in snapshot_blocks:
            clean = frame.copy()
            clean.attrs = {}
            block_frames.append(clean)
        member_frames = []
        for frame in snapshot_members:
            clean = frame.copy()
            clean.attrs = {}
            member_frames.append(clean)
        resolution_frames = []
        for frame in snapshot_resolution:
            clean = frame.copy()
            clean.attrs = {}
            resolution_frames.append(clean)
        output.table(
            pd.concat(block_frames, ignore_index=True)
            if block_frames
            else pd.DataFrame(),
            "09_历史累积S结构块",
        )
        output.table(
            pd.concat(member_frames, ignore_index=True)
            if member_frames
            else pd.DataFrame(),
            "09_历史累积S结构块节点",
        )
        output.table(
            pd.concat(resolution_frames, ignore_index=True)
            if resolution_frames
            else pd.DataFrame(),
            "09_历史累积S结构块分辨率审计",
        )
        output.table(
            pd.concat(snapshot_screening, ignore_index=True)
            if snapshot_screening
            else pd.DataFrame(),
            "09_历史累积S结构块发现后筛选审计",
        )
        aligned_history = align_snapshot_blocks(snapshot_blocks, snapshot_sets)
        output.table(aligned_history, "10_历史S结构块跨时间对齐与演化事件")
    build_unified_structure_alignment(output, full_data)
