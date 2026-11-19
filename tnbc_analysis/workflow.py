from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from .project_config import TIME
from .topic_tree_analysis import TopicTreeDataset, Output


def read(path):
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def descriptive(data, root):
    from .tree_structure_descriptive import (
        frequency,
        log_bins,
        figure1a_statistics,
        figure1e_statistics,
    )

    out = Output(Path(root) / "01_tree_structure")
    inverse = {i: [] for i in range(len(data.pmids))}
    nodes = []
    for k in data.ids:
        years = data.years[data.node_papers[k]].astype(float)
        for p in data.node_papers[k]:
            inverse[int(p)].append(k)
        nodes.append(
            {
                "node_id": k,
                "name": data.tree[k]["name"],
                "level": data.tree[k]["level"],
                "direct_children": len(data.tree[k]["children_ids"]),
                "is_leaf": not data.tree[k]["children_ids"],
                "cohort_papers": len(years),
                "median_year": np.median(years) if len(years) else np.nan,
                "mean_year": years.mean() if len(years) else np.nan,
                "earliest_year": years.min() if len(years) else np.nan,
                "latest_year": years.max() if len(years) else np.nan,
            }
        )
    rows = []
    for p, ids in inverse.items():
        ancestors = {a for k in ids for a in data.paths[k][:-1]}
        rows.append(
            {
                "pmid": data.pmids[p],
                "year": int(data.years[p]),
                "path_count": len(set(ids) - ancestors),
            }
        )
    n = pd.DataFrame(nodes)
    papers = pd.DataFrame(rows)
    (annual, _) = figure1e_statistics(n, papers)
    for key, frame in {
        "levels": figure1a_statistics(data),
        "nodes": n,
        "papers": papers,
        "direct_children": frequency(n.direct_children, "direct_children"),
        "classification_paths": frequency(papers.path_count, "path_count"),
        "paper_count_bins": log_bins(n.cohort_papers),
        "annual": annual,
    }.items():
        out.table(frame, key)


def observed(data, root, seed):
    from .topic_tree_analysis import (
        build_pmid_hypergraph,
        run_significance_calibrated_pmid_layer,
        formal_community_bibliometrics,
        formal_community_nesting,
        formal_community_discussion_screen,
    )

    out = Output(Path(root) / "03_hypergraph")
    graph = build_pmid_hypergraph(data)
    run_significance_calibrated_pmid_layer(data, out, graph, seed)
    formal_community_bibliometrics(data, out)
    formal_community_nesting(data, out)
    formal_community_discussion_screen(data, out)


def weighted(data, root, seed):
    from .topic_tree_analysis import (
        run_quality_weighted_exploration,
        _profile_weighted_structure_time_trends,
    )

    out = Output(Path(root) / "03_hypergraph")
    run_quality_weighted_exploration(data, out, seed)
    for model, prefix in [
        ("JCR_Q", "26_JCR加权正式结构"),
        ("IF_year", "27_影响因子加权正式结构"),
    ]:
        catalogue = read(
            out.tables / f"22b_探索性质量加权_{model}_正式候选社群清单.csv"
        )
        members = read(out.tables / f"22b_探索性质量加权_{model}_正式候选社群节点.csv")
        if catalogue.empty:
            continue
        (annual, trends) = _profile_weighted_structure_time_trends(
            data, catalogue, members
        )
        out.table(annual, prefix + "_年度去重PMID覆盖")
        out.table(trends, prefix + "_重算时间趋势汇总")


def annotate_identifiers(root, tree, papers):
    from . import pmid_hypergraph_pipeline as p

    out = Output(Path(root) / "06_predict")
    data = TopicTreeDataset(tree, papers, None, None)
    records = p._build_unified_structure_records(out, data)
    compact = read(
        out.tables / "21_所有年度与全量S_JCR_IF_未来结构块统一对齐.csv"
    ).fillna("")
    rows = []
    for r in records:
        slot = p._compact_interval_label(r)
        column = p.SLOT_COLUMN_NAMES[slot][0]
        matches = compact[compact[column].astype(str).eq(p._compact_source_id(r))]
        if len(matches) != 1:
            raise ValueError(f"Identity mapping is not unique: {r['structure_uid']}")
        uid = matches.iloc[0]["整合结构块编号"]
        rows.append(
            {
                "structure_uid": r["structure_uid"],
                "source_type": r["source_type"],
                "community_id": r["source_community_id"],
                "source_community_id": r["source_community_id"],
                "source_structure_id": r["display_structure_id"],
                "snapshot_end": r["snapshot_end"],
                "snapshot_start": r["snapshot_start"],
                "annotation_label": uid,
                "structure_label": uid,
                "structure_label_en": uid,
                "display_identity_id": uid,
                "structure_display_name": uid,
                "aligned_identity_id": uid,
                "is_framework_structure": False,
            }
        )
    annotations = pd.DataFrame(rows)
    direct = read(Path(root) / "03_hypergraph/tables/20_S正式社群_直接多层嵌套关系.csv")
    formal = annotations.loc[annotations.source_type.eq("full_S")]
    children = {}
    if len(direct):
        for n in direct.itertuples():
            children.setdefault(str(n.parent_community_id), []).append(
                str(n.child_community_id)
            )
    frameworks = set()
    for n in formal.itertuples():
        seen = set()
        stack = list(children.get(str(n.source_community_id), []))
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(children.get(current, []))
        if len(seen) / max(len(formal) - 1, 1) >= 0.45:
            frameworks.add(n.aligned_identity_id)
    annotations["is_framework_structure"] = annotations.aligned_identity_id.isin(
        frameworks
    )
    out.table(annotations, "structure_identifiers")
    for typ, source, target in [
        (
            "full_S",
            "03_hypergraph/tables/16_S社群_完整清单.csv",
            "03_hypergraph/tables/16_S社群_完整清单_结构注释.csv",
        ),
        (
            "full_JCR",
            "04_weighted/tables/22b_探索性质量加权_JCR_Q_正式候选社群清单.csv",
            "04_weighted/tables/22b_探索性质量加权_JCR_Q_正式候选社群清单_结构注释.csv",
        ),
        (
            "full_IF",
            "04_weighted/tables/22b_探索性质量加权_IF_year_正式候选社群清单.csv",
            "04_weighted/tables/22b_探索性质量加权_IF_year_正式候选社群清单_结构注释.csv",
        ),
    ]:
        f = read(Path(root) / source)
        if f.empty:
            continue
        q = annotations[annotations.source_type.eq(typ)].set_index("community_id")
        for c in [
            "structure_label",
            "structure_display_name",
            "annotation_label",
            "aligned_identity_id",
            "is_framework_structure",
        ]:
            f[c] = f.community_id.map(q[c])
        target = Path(root) / target
        target.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(target, index=False, encoding="utf-8-sig")
    return annotations


def temporal(tree, papers, root, seed, simulations, open_candidates):
    from .pmid_hypergraph_pipeline import run_temporal_prediction

    run_temporal_prediction(
        tree,
        papers,
        Path(root) / "06_predict",
        seed=seed,
        max_open_per_node=open_candidates,
        simulations=simulations,
        skip_snapshots=False,
        accelerator="cpu",
        gamma_csv=Path(root)
        / "03_hypergraph/tables/12_PMID逐层显著性_结构块尺度校准.csv",
    )
    annotate_identifiers(root, tree, papers)


def history(tree, papers, root, core_details=True):
    from . import historical_structure_evolution as h

    root = Path(root)
    src = root / "06_predict/tables"
    out = Output(root / "05_historical")
    ann = read(src / "structure_identifiers.csv")
    ann = ann[ann.source_type.eq("historical")]
    inputs = h.Inputs(
        read(src / "09_历史累积S结构块.csv"),
        read(src / "09_历史累积S结构块节点.csv"),
        ann,
        read(src / "10_历史S结构块跨时间对齐与演化事件.csv"),
        read(src / "21_所有年度与全量S_JCR_IF_未来结构块统一对齐.csv").fillna(""),
        json.loads(Path(tree).read_text(encoding="utf-8-sig")),
        json.loads(Path(papers).read_text(encoding="utf-8-sig")),
    )
    (blocks, sets, pmids) = h.prepare_block_state(inputs)
    if blocks.empty:
        return
    (_, _, events) = h.build_lineages(blocks.copy(), inputs.events)
    (blocks, lineages, mapping) = h.build_u_lineages(
        blocks, sets, inputs.unified_alignment
    )
    for stem, frame in {
        "historical_states": blocks,
        "historical_lineages": lineages,
        "historical_identity_map": mapping,
        "historical_inventory": h.make_inventory(blocks, pmids),
        "historical_transitions": h.transition_summary(events),
        "historical_events": events,
    }.items():
        out.table(frame, stem)
    if core_details:
        paths = SimpleNamespace(
            tree=Path(tree),
            papers=Path(papers),
            output=out.root,
            tables=out.tables,
            source_tables=src,
            main_tables=root / "03_hypergraph/tables",
        )
        h.build_figure30_network_tables(
            paths, inputs, blocks, lineages, pd.DataFrame(), sets
        )


def comparison(tree, papers, root):
    from . import prediction_structure_comparison as c

    root = Path(root)
    out = Output(root / "06_predict")
    paths = c.Paths(out.root, Path(tree), Path(papers))
    summary = read(
        paths.prediction_tables
        / "08_未来概率网络模拟_重复率25%结构块_筛选后正式结果.csv"
    )
    if summary.empty:
        return
    ids = set(summary.future_module_id)
    (nodes, edges, sims) = c._module_simulation_statistics(
        read(paths.frozen_node_rows), read(paths.frozen_edge_rows), ids
    )
    sets = {
        str(k): set(v.node_id.astype(str))
        for (k, v) in nodes.groupby("future_module_id")
    }
    data = TopicTreeDataset(tree, papers, TIME.observation_start, TIME.observation_end)
    (annual, trends) = c._forecast_module_time_trajectory(paths, data, edges, sets)
    out.table(annual, "forecast_annual_evidence")
    out.table(trends, "forecast_window_trends")
    out.table(nodes, "forecast_members")
    out.table(edges, "forecast_relations")
    compact = read(
        out.tables / "21_所有年度与全量S_JCR_IF_未来结构块统一对齐.csv"
    ).fillna("")
    future_column = f"{TIME.forecast_slot}结构编号"
    rows = []
    for n in compact.to_dict("records"):
        s = n.get("正式s结构块编号", "")
        f = n.get(future_column, "")
        if not s and (not f):
            continue
        rows.append(
            {
                "unified_id": n["整合结构块编号"],
                "formal_s_id": s,
                "future_module_id": f,
                "comparison_status": "prediction-matched"
                if s and f
                else "reference-only"
                if s
                else "prediction-only",
            }
        )
    identities = pd.DataFrame(rows)
    out.table(identities, "forecast_correspondence")
    formal_sets = c._formal_membership(paths.formal_membership)
    fcore = c._formal_core_table(paths.formal_core_nodes)
    future_core = c._future_core_table(paths.frozen_core_nodes)
    (node_pmids, _) = c._node_pmid_lookup(data)
    selection = summary.merge(
        identities, on="future_module_id", how="left", validate="one_to_one"
    )
    selection["structure_label"] = selection.unified_id
    selection["selected_for_figure34"] = True
    selection["figure34_display_rank"] = np.arange(1, len(selection) + 1)
    (cases, core_nodes, core_edges, coverage) = c._build_figure36_core_network_tables(
        data=data,
        node_pmids=node_pmids,
        selection=selection,
        formal_sets=formal_sets,
        future_sets=sets,
        formal_core=fcore,
        future_core=future_core,
        formal_edges=c._load_formal_edges(paths.formal_edges, formal_sets),
        future_nodes=nodes,
        future_edges=edges,
        formal_node_state=read(paths.formal_membership),
    )
    for stem, frame in [
        ("forecast_core_cases", cases),
        ("forecast_core_nodes", core_nodes),
        ("forecast_core_edges", core_edges),
        ("forecast_core_coverage", coverage),
    ]:
        out.table(frame, stem)


def core_networks(tree, papers, root):
    from .formal_structure_networks import build_tables

    build_tables(
        Path(tree),
        Path(papers),
        Path(root) / "03_hypergraph",
        Path(root)
        / "06_predict/tables/21_所有年度与全量S_JCR_IF_未来结构块统一对齐.csv",
    )
