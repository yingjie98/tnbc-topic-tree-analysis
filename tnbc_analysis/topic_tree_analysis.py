from __future__ import annotations
from .result_paths import artifact_directory
from .project_config import TIME
import collections
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import fisher_exact
from sklearn.metrics import normalized_mutual_info_score

try:
    import igraph as ig
    import leidenalg as la
except ImportError as exc:
    ig = la = None
    LEIDEN_IMPORT_ERROR = exc
else:
    LEIDEN_IMPORT_ERROR = None


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def norm_name(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def adjust_pvalues(p_values: Iterable[float], method: str = "BY") -> np.ndarray:
    p = np.asarray(list(p_values), float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return q
    ordered = valid[np.argsort(p[valid])]
    factor = len(ordered) / np.arange(1, len(ordered) + 1)
    if method.upper() == "BY":
        factor *= np.sum(1 / np.arange(1, len(ordered) + 1))
    q[ordered] = np.minimum(1, np.minimum.accumulate((p[ordered] * factor)[::-1])[::-1])
    return q


def _require_table_files(folder: Path, stems: Iterable[str], stage: str) -> None:
    missing = [stem for stem in stems if not (folder / f"{stem}.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{stage} 缺少前置结果表（目录：{folder}）："
            + "；".join(missing)
            + "。请先完整运行主管线生成这些表，或恢复对应的文件命名。"
        )


@dataclass
class Output:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.tables = artifact_directory(self.root / "tables")
        self.tables.mkdir(parents=True, exist_ok=True)

    def table(self, frame, stem):
        path = self.tables / (stem + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path


def parse_year(record: dict[str, Any]) -> int | None:
    for key in ("year", "Year", "publication_year", "pub_year", "PubYear"):
        value = record.get(key)
        if value is not None:
            match = re.search("(?:18|19|20)\\d{2}", str(value))
            if match:
                return int(match.group())
    for key in ("PubDate", "publication_date", "date"):
        match = re.search("(?:18|19|20)\\d{2}", str(record.get(key, "")))
        if match:
            return int(match.group())
    return None


def paper_text(record: dict[str, Any]) -> str:
    title = str(record.get("Title") or record.get("title") or "").strip()
    abstract = str(record.get("Abstract") or record.get("abstract") or "").strip()
    tandem = str(record.get("TandA") or record.get("text") or "").strip()
    return (f"{title}. {abstract}" if abstract else title) or tandem


class TopicTreeDataset:
    def __init__(
        self,
        tree_path: Path,
        papers_path: Path,
        start_year: int | None,
        end_year: int | None,
    ) -> None:
        self.tree_path = Path(tree_path)
        self.papers_path = Path(papers_path)
        self.tree = self._load_tree(self.tree_path)
        self.papers = self._load_papers(self.papers_path)
        (self.start_year, self.end_year) = (start_year, end_year)
        self.audit: dict[str, Any] = {}
        self._validate_and_index()

    @staticmethod
    def _load_tree(path: Path) -> dict[str, dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("主题树 JSON 必须是 {node_id: node_object} 的对象。")
        result: dict[str, dict[str, Any]] = {}
        for raw_id, raw in data.items():
            if not isinstance(raw, dict):
                raise ValueError(f"主题节点 {raw_id!r} 不是对象。")
            node = dict(raw)
            node.setdefault("name", str(raw_id))
            node.setdefault("children_ids", [])
            node.setdefault("all_paper_pmid", [])
            try:
                node["level"] = int(node["level"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"主题节点 {raw_id!r} 缺少可转换为整数的 level。"
                ) from exc
            if not isinstance(node["children_ids"], list):
                raise ValueError(f"主题节点 {raw_id!r} 的 children_ids 必须是列表。")
            if not isinstance(node["all_paper_pmid"], list):
                raise ValueError(f"主题节点 {raw_id!r} 的 all_paper_pmid 必须是列表。")
            result[str(raw_id)] = node
        if not result:
            raise ValueError("主题树为空。")
        return result

    @staticmethod
    def _load_papers(path: Path) -> dict[str, dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("论文库 JSON 必须是 {pmid: paper_object} 的对象。")
        result: dict[str, dict[str, Any]] = {}
        for pmid, paper in data.items():
            if isinstance(paper, dict):
                result[str(pmid).strip()] = dict(paper)
        if not result:
            raise ValueError("论文库为空，或其中没有论文对象。")
        return result

    def _validate_and_index(self) -> None:
        unknown_children: list[tuple[str, str]] = []
        self.parent: dict[str, str] = {}
        for node_id, node in self.tree.items():
            cleaned: list[str] = []
            for child in node["children_ids"]:
                child = str(child)
                if child not in self.tree:
                    unknown_children.append((node_id, child))
                    continue
                if child in self.parent and self.parent[child] != node_id:
                    raise ValueError(
                        f"节点 {child!r} 被多个父节点引用：{self.parent[child]!r} 与 {node_id!r}。"
                    )
                self.parent[child] = node_id
                cleaned.append(child)
            node["children_ids"] = cleaned
        roots = [node_id for node_id in self.tree if node_id not in self.parent]
        if not roots:
            raise ValueError("主题树没有根节点；请检查 children_ids 是否形成循环。")
        self.roots = sorted(roots, key=lambda key: (self.tree[key]["level"], key))
        self.ids = sorted(self.tree, key=lambda key: (self.tree[key]["level"], key))
        self.paths: dict[str, list[str]] = {}
        visiting: set[str] = set()

        def visit(node_id: str, prefix: list[str]) -> None:
            if node_id in visiting:
                raise ValueError(f"主题树在 {node_id!r} 出现循环。")
            if node_id in self.paths:
                return
            visiting.add(node_id)
            self.paths[node_id] = prefix + [node_id]
            for child in self.tree[node_id]["children_ids"]:
                visit(child, self.paths[node_id])
            visiting.remove(node_id)

        for root in self.roots:
            visit(root, [])
        disconnected = [key for key in self.ids if key not in self.paths]
        if disconnected:
            raise ValueError(
                f"主题树存在 {len(disconnected)} 个无法从根访问的节点，例如 {disconnected[:3]}。"
            )
        tree_pmids = {
            str(p).strip()
            for node in self.tree.values()
            for p in node["all_paper_pmid"]
        }
        paper_rows: list[tuple[str, int, str]] = []
        for pmid, record in self.papers.items():
            if pmid not in tree_pmids:
                continue
            year = parse_year(record)
            if year is None:
                continue
            if self.start_year is not None and year < self.start_year:
                continue
            if self.end_year is not None and year > self.end_year:
                continue
            paper_rows.append((pmid, year, paper_text(record)))
        paper_rows.sort(key=lambda row: (row[1], row[0]))
        if not paper_rows:
            raise ValueError(
                "年份筛选后没有与主题树匹配的论文；请检查 PMID 和年份字段。"
            )
        self.pmids = np.asarray([row[0] for row in paper_rows], dtype=object)
        self.years = np.asarray([row[1] for row in paper_rows], dtype=np.int16)
        self.texts = [row[2] for row in paper_rows]
        self.pmid_index = {pmid: index for (index, pmid) in enumerate(self.pmids)}
        self.node_papers: dict[str, np.ndarray] = {}
        raw_unknown = 0
        for node_id, node in self.tree.items():
            values = []
            for pmid in node["all_paper_pmid"]:
                key = str(pmid).strip()
                if key in self.pmid_index:
                    values.append(self.pmid_index[key])
                elif key:
                    raw_unknown += 1
            self.node_papers[node_id] = np.asarray(sorted(set(values)), dtype=np.int32)
        self.counts = {
            node_id: len(indices) for (node_id, indices) in self.node_papers.items()
        }
        self.node_index = {node_id: index for (index, node_id) in enumerate(self.ids)}
        self.year_values = np.arange(
            int(self.years.min()), int(self.years.max()) + 1, dtype=int
        )
        fingerprint = hashlib.sha256(
            "\n".join(
                (f"{pmid}\t{year}\t{text}" for (pmid, year, text) in paper_rows)
            ).encode("utf-8")
        ).hexdigest()
        self.audit = {
            "tree_file": str(self.tree_path.resolve()),
            "papers_file": str(self.papers_path.resolve()),
            "tree_nodes": len(self.ids),
            "tree_roots": len(self.roots),
            "cohort_papers": len(self.pmids),
            "year_range_observed": [int(self.years.min()), int(self.years.max())],
            "requested_year_range": [self.start_year, self.end_year],
            "unknown_child_references_removed_from_analysis": len(unknown_children),
            "tree_pmid_occurrences_not_in_cohort": raw_unknown,
            "nodes_with_no_cohort_papers": int(
                sum((count == 0 for count in self.counts.values()))
            ),
            "cohort_fingerprint_sha256": fingerprint,
        }

    def incidence(self, ids: list[str]) -> sparse.csr_matrix:
        lengths = np.asarray(
            [len(self.node_papers[node]) for node in ids], dtype=np.int64
        )
        if not lengths.sum():
            return sparse.csr_matrix((len(ids), len(self.pmids)), dtype=np.float32)
        columns = np.concatenate([self.node_papers[node] for node in ids])
        rows = np.repeat(np.arange(len(ids), dtype=np.int32), lengths)
        return sparse.csr_matrix(
            (np.ones(len(columns), dtype=np.float32), (rows, columns)),
            shape=(len(ids), len(self.pmids)),
        )

    def meta(self, node_id: str) -> dict[str, Any]:
        parent = self.parent.get(node_id, "")
        return {
            "node_id": node_id,
            "name": str(self.tree[node_id]["name"]),
            "level": int(self.tree[node_id]["level"]),
            "total_papers": int(self.counts[node_id]),
            "parent_id": parent,
            "parent_name": str(self.tree[parent]["name"]) if parent else "",
        }

    def compute_hierarchy_intervals(self) -> dict[str, tuple[float, float]]:
        exclusive: dict[str, int] = {}
        descendants: dict[str, set[int]] = {}
        for node_id in reversed(self.ids):
            child_union: set[int] = set()
            for child in self.tree[node_id]["children_ids"]:
                child_union.update(descendants[child])
            own = set(self.node_papers[node_id].tolist())
            exclusive[node_id] = len(own - child_union)
            descendants[node_id] = own | child_union
        fake_total: dict[str, int] = {}
        for node_id in reversed(self.ids):
            fake_total[node_id] = exclusive[node_id] + sum(
                (fake_total[child] for child in self.tree[node_id]["children_ids"])
            )
        angles: dict[str, tuple[float, float]] = {}
        root_weights = sum((max(fake_total[root], 0) for root in self.roots))
        cursor = 0.0
        stack: list[tuple[str, float, float]] = []
        for root in self.roots:
            span = (
                360.0 * fake_total[root] / root_weights
                if root_weights
                else 360.0 / len(self.roots)
            )
            stack.append((root, cursor, cursor + span))
            cursor += span
        while stack:
            (node_id, start, end) = stack.pop()
            angles[node_id] = (start, end)
            children = sorted(
                self.tree[node_id]["children_ids"], key=lambda x: (-fake_total[x], x)
            )
            if not children:
                continue
            total = sum((fake_total[child] for child in children))
            point = start
            for position, child in enumerate(children):
                span = (end - start) * fake_total[child] / total if total else 0.0
                child_end = end if position == len(children) - 1 else point + span
                stack.append((child, point, child_end))
                point = child_end
        return angles


def require_leiden() -> None:
    if la is None or ig is None:
        raise RuntimeError(
            "Community discovery requires igraph and leidenalg; install the declared package dependencies."
        ) from LEIDEN_IMPORT_ERROR


def graph_from_sparse(matrix: sparse.csr_matrix) -> Any:
    require_leiden()
    upper = sparse.triu(matrix, k=1).tocoo()
    graph = ig.Graph(
        n=matrix.shape[0],
        edges=list(zip(upper.row.tolist(), upper.col.tolist())),
        directed=False,
    )
    graph.es["weight"] = upper.data.astype(float).tolist()
    return graph


def cpm_partition(
    horizontal: sparse.csr_matrix,
    vertical: sparse.csr_matrix,
    vertical_weight: float,
    gamma: float,
    seed: int,
) -> np.ndarray:
    horizontal_graph = graph_from_sparse(horizontal)
    if horizontal_graph.ecount() == 0:
        return np.arange(horizontal.shape[0], dtype=int)
    if vertical_weight <= 0 or vertical.nnz == 0:
        part = la.find_partition(
            horizontal_graph,
            la.CPMVertexPartition,
            weights="weight",
            resolution_parameter=float(gamma),
            n_iterations=4,
            seed=int(seed),
        )
        return np.asarray(part.membership, dtype=int)
    vertical_graph = graph_from_sparse(vertical)
    (membership, _) = la.find_partition_multiplex(
        [horizontal_graph, vertical_graph],
        la.CPMVertexPartition,
        layer_weights=[1.0, float(vertical_weight)],
        weights="weight",
        resolution_parameter=float(gamma),
        n_iterations=4,
        seed=int(seed),
    )
    return np.asarray(membership, dtype=int)


@dataclass
class HypergraphData:
    kind: str
    ids: list[str]
    members: list[np.ndarray]
    factor_source: np.ndarray
    factor_weight: np.ndarray
    hyperedges: pd.DataFrame
    source_factor_count: np.ndarray | None = None


@dataclass
class CommunityResult:
    kind: str
    data: HypergraphData
    members: pd.DataFrame
    communities: pd.DataFrame
    model_selection: pd.DataFrame
    diagnostics: pd.DataFrame
    labels: np.ndarray
    horizontal_pairs: sparse.csr_matrix
    stability_nmi: float


def build_pmid_hypergraph(data: TopicTreeDataset) -> HypergraphData:
    ids = [
        node
        for node in data.ids
        if data.counts[node] > 0 and data.tree[node]["level"] >= 2
    ]
    levels = np.asarray([data.tree[node]["level"] for node in ids], dtype=int)
    incidence = data.incidence(ids).tocsc()
    members: list[np.ndarray] = []
    sources: list[int] = []
    weights: list[float] = []
    records: list[dict[str, Any]] = []
    source_counts = np.zeros(len(data.pmids), dtype=np.int32)
    for paper_index in range(incidence.shape[1]):
        nodes = incidence.indices[
            incidence.indptr[paper_index] : incidence.indptr[paper_index + 1]
        ]
        groups: dict[int, list[int]] = collections.defaultdict(list)
        for node_index in nodes:
            groups[int(levels[node_index])].append(int(node_index))
        valid = [
            (level, np.asarray(values, dtype=np.int32))
            for (level, values) in groups.items()
            if len(values) >= 2
        ]
        if not valid:
            continue
        source_counts[paper_index] = len(valid)
        for level, nodes_at_level in valid:
            factor_id = len(members)
            mass = 1.0 / len(valid)
            members.append(nodes_at_level)
            sources.append(paper_index)
            weights.append(mass)
            records.append(
                {
                    "factor_id": factor_id,
                    "pmid": data.pmids[paper_index],
                    "pmid_index": paper_index,
                    "level": int(level),
                    "hyperedge_size": int(len(nodes_at_level)),
                    "horizontal_weight": mass,
                    "incidence_weight_per_member": mass / len(nodes_at_level),
                    "definition": "同一 PMID 在同一 level 的主题节点；仅此层内连接。",
                }
            )
    if not members:
        raise ValueError(
            "没有 PMID 在任何 level 同时连接两个主题节点，无法构建 PMID 超图。"
        )
    return HypergraphData(
        "pmid",
        ids,
        members,
        np.asarray(sources, dtype=np.int32),
        np.asarray(weights, dtype=float),
        pd.DataFrame(records),
        source_counts,
    )


def poisson_excess_z(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    expected = np.maximum(np.asarray(expected, dtype=float), 1e-12)
    deviance = 2.0 * (
        observed * np.log(np.maximum(observed, 1e-12) / expected)
        - (observed - expected)
    )
    return np.sign(observed - expected) * np.sqrt(np.maximum(deviance, 0.0))


_GPU_STATE: dict[str, Any] = {
    "checked": False,
    "available": False,
    "validation_enabled": False,
    "diverged": False,
    "validation_calls": 0,
}


def _gpu_state() -> dict[str, Any]:
    if _GPU_STATE["checked"]:
        return _GPU_STATE
    _GPU_STATE["checked"] = True
    import os

    setting = os.environ.get("TNBC_USE_CUDA", "auto").strip().lower()
    validate = os.environ.get("TNBC_VALIDATE_CUDA_NUMERICS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    if setting == "1":
        if not available:
            raise RuntimeError("TNBC_USE_CUDA=1 但当前环境没有可用的 CUDA/torch。")
        _GPU_STATE["available"] = True
    elif setting == "0":
        _GPU_STATE["available"] = False
    else:
        _GPU_STATE["available"] = available
    _GPU_STATE["validation_enabled"] = bool(_GPU_STATE["available"] and validate)
    return _GPU_STATE


def _poisson_sf(observed: np.ndarray, expected: np.ndarray, alpha: float) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.maximum(np.asarray(expected, dtype=np.float64), 1e-300)
    reference = stats.poisson.sf(observed - 1.0, expected)
    state = _gpu_state()
    if state["validation_enabled"]:
        import torch

        k = torch.as_tensor(observed - 1.0, dtype=torch.float64, device="cuda")
        mu = torch.as_tensor(expected, dtype=torch.float64, device="cuda")
        with torch.no_grad():
            candidate = torch.special.gammainc(k + 1.0, mu).cpu().numpy()
        if not np.allclose(
            candidate, reference, rtol=1e-12, atol=1e-15, equal_nan=True
        ):
            _GPU_STATE["diverged"] = True
        _GPU_STATE["validation_calls"] += 1
    return reference


def _normal_sf(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = stats.norm.sf(values)
    state = _gpu_state()
    if state["validation_enabled"]:
        import torch

        tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
        with torch.no_grad():
            candidate = 0.5 * torch.special.erfc(tensor / math.sqrt(2.0)).cpu().numpy()
        if not np.allclose(
            candidate, reference, rtol=1e-12, atol=1e-15, equal_nan=True
        ):
            _GPU_STATE["diverged"] = True
        _GPU_STATE["validation_calls"] += 1
    return reference


def levelwise_pmid_significance(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    alpha: float = 0.05,
    min_independent_pmids: int = 2,
) -> dict[str, Any]:
    n_nodes = len(graph_data.ids)
    levels = graph_data.hyperedges["level"].to_numpy(dtype=int)
    active_sources: list[set[int]] = [set() for _ in range(n_nodes)]
    horizontal_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for level in sorted(np.unique(levels)):
        factor_ids = np.flatnonzero(levels == level)
        factor_nodes = [graph_data.members[int(factor_id)] for factor_id in factor_ids]
        sizes = np.asarray([len(nodes) for nodes in factor_nodes], dtype=np.int32)
        stubs = np.concatenate(factor_nodes)
        degree = np.bincount(stubs, minlength=n_nodes).astype(float)
        total_stubs = float(len(stubs))
        ordered_slot_pairs = float(np.sum(sizes * (sizes - 1)))
        pair_values: dict[tuple[int, int], list[float]] = {}
        for factor_id, nodes in zip(factor_ids, factor_nodes):
            size = len(nodes)
            if size < 2:
                continue
            pair_mass = (
                float(graph_data.factor_weight[int(factor_id)])
                * 2.0
                / (size * (size - 1))
            )
            for left, right in combinations(nodes.tolist(), 2):
                if left > right:
                    (left, right) = (right, left)
                item = pair_values.setdefault((left, right), [0.0, 0.0])
                item[0] += 1.0
                item[1] += pair_mass
        keys = list(pair_values)
        if not keys or total_stubs < 2:
            summaries.append(
                {
                    "level": int(level),
                    "factors": int(len(factor_ids)),
                    "tested_pairs": 0,
                    "significant_pairs": 0,
                    "active_nodes": 0,
                    "status": "没有可检验的同层 PMID 节点对",
                }
            )
            continue
        left = np.asarray([key[0] for key in keys], dtype=np.int32)
        right = np.asarray([key[1] for key in keys], dtype=np.int32)
        observed = np.asarray([pair_values[key][0] for key in keys], dtype=float)
        observed_mass = np.asarray([pair_values[key][1] for key in keys], dtype=float)
        expected = (
            degree[left]
            * degree[right]
            * ordered_slot_pairs
            / (total_stubs * (total_stubs - 1.0))
        )
        p_values = _poisson_sf(observed, expected, alpha)
        q_values = adjust_pvalues(p_values, "BY")
        z_values = poisson_excess_z(observed, expected)
        expected_mass = (
            2.0
            * float(graph_data.factor_weight[factor_ids].sum())
            * degree[left]
            * degree[right]
            / (total_stubs * (total_stubs - 1.0))
        )
        supported = (
            (q_values < alpha) & (z_values > 0) & (observed >= min_independent_pmids)
        )
        significant_keys = {keys[position] for position in np.flatnonzero(supported)}
        for factor_id, nodes in zip(factor_ids, factor_nodes):
            source = int(graph_data.factor_source[int(factor_id)])
            for left_node, right_node in combinations(nodes.tolist(), 2):
                key = (
                    (left_node, right_node)
                    if left_node < right_node
                    else (right_node, left_node)
                )
                if key in significant_keys:
                    active_sources[left_node].add(source)
                    active_sources[right_node].add(source)
        for position, (node_left, node_right) in enumerate(keys):
            horizontal_rows.append(
                {
                    "level": int(level),
                    "node_1_index": int(node_left),
                    "node_2_index": int(node_right),
                    "independent_pmid_support": int(observed[position]),
                    "observed_fractional_pair_mass": float(observed_mass[position]),
                    "null_expected_support": float(expected[position]),
                    "null_expected_fractional_pair_mass": float(
                        expected_mass[position]
                    ),
                    "excess_support": float(observed[position] - expected[position]),
                    "fold_over_null": float(
                        observed[position] / max(expected[position], 1e-12)
                    ),
                    "evidence_z": float(z_values[position]),
                    "p_configuration_approx": float(p_values[position]),
                    "q_BY_within_level": float(q_values[position]),
                    "significant_horizontal_relation": bool(supported[position]),
                }
            )
        summaries.append(
            {
                "level": int(level),
                "factors": int(len(factor_ids)),
                "tested_pairs": int(len(keys)),
                "significant_pairs": int(supported.sum()),
                "active_nodes": int(
                    len(set(left[supported].tolist() + right[supported].tolist()))
                ),
                "median_null_expected_support": float(np.median(expected)),
                "status": "已完成逐层度数与超边大小保持的配置零模型检验",
            }
        )
    horizontal = pd.DataFrame(horizontal_rows)
    backend = _gpu_state()
    return {
        "horizontal": horizontal,
        "level_summary": pd.DataFrame(summaries),
        "active_sources": active_sources,
        "compute_backend": {
            "p_value_kernel": "scipy_cpu_reference",
            "cuda_numeric_validation_enabled": bool(backend["validation_enabled"]),
            "cuda_numeric_divergence": bool(backend["diverged"]),
        },
    }


def significant_vertical_evidence(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    active_sources: list[set[int]],
    alpha: float = 0.05,
    min_independent_pmids: int = 2,
) -> pd.DataFrame:
    index = {node: position for (position, node) in enumerate(graph_data.ids)}
    rows: list[dict[str, Any]] = []
    for child, parent in data.parent.items():
        if child not in index or parent not in index:
            continue
        (child_index, parent_index) = (index[child], index[parent])
        parent_pool = set(data.node_papers[parent].tolist())
        parent_active = active_sources[parent_index] & parent_pool
        child_active = active_sources[child_index] & parent_pool
        if not parent_active or not child_active:
            continue
        (population, successes, draws) = (
            len(parent_pool),
            len(parent_active),
            len(child_active),
        )
        observed = len(parent_active & child_active)
        if population <= 1 or successes == 0 or draws == 0:
            continue
        expected = draws * successes / population
        variance = (
            draws
            * (successes / population)
            * (1.0 - successes / population)
            * ((population - draws) / max(population - 1, 1))
        )
        z_value = (observed - expected) / math.sqrt(max(variance, 1e-12))
        p_value = stats.hypergeom.sf(observed - 1, population, successes, draws)
        rows.append(
            {
                "parent_index": int(parent_index),
                "child_index": int(child_index),
                "parent_active_pmids": int(successes),
                "child_active_pmids": int(draws),
                "effective_pmid_support": int(observed),
                "null_expected_support": float(expected),
                "null_variance": float(variance),
                "evidence_z": float(z_value),
                "p_tree_constrained_hypergeom": float(p_value),
            }
        )
    vertical = pd.DataFrame(rows)
    if len(vertical):
        vertical["q_BY_all_direct_parent_child_tests"] = adjust_pvalues(
            vertical.p_tree_constrained_hypergeom, "BY"
        )
        vertical["significant_vertical_relation"] = (
            (vertical.q_BY_all_direct_parent_child_tests < alpha)
            & (vertical.evidence_z > 0)
            & (vertical.effective_pmid_support >= min_independent_pmids)
        )
    else:
        vertical = pd.DataFrame(
            columns=[
                "parent_index",
                "child_index",
                "parent_active_pmids",
                "child_active_pmids",
                "effective_pmid_support",
                "null_expected_support",
                "null_variance",
                "evidence_z",
                "p_tree_constrained_hypergeom",
                "q_BY_all_direct_parent_child_tests",
                "significant_vertical_relation",
            ]
        )
    return vertical


def significant_pmid_blocks(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    horizontal: pd.DataFrame,
    vertical: pd.DataFrame,
    seed: int,
    *,
    resolution_grid: Iterable[float] | None = None,
    trial_count: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_nodes = len(graph_data.ids)
    horizontal_sig = horizontal[horizontal.significant_horizontal_relation].copy()
    if vertical.empty:
        vertical = vertical.reindex(
            columns=list(
                dict.fromkeys(
                    [
                        *vertical.columns,
                        "significant_vertical_relation",
                        "parent_index",
                        "child_index",
                        "evidence_z",
                    ]
                )
            )
        )
    vertical_sig = vertical.loc[
        vertical.significant_vertical_relation.fillna(False).astype(bool)
    ].copy()
    rows = np.concatenate(
        [
            horizontal_sig.node_1_index.to_numpy(int),
            horizontal_sig.node_2_index.to_numpy(int),
            vertical_sig.parent_index.to_numpy(int),
            vertical_sig.child_index.to_numpy(int),
        ]
    )
    cols = np.concatenate(
        [
            horizontal_sig.node_2_index.to_numpy(int),
            horizontal_sig.node_1_index.to_numpy(int),
            vertical_sig.child_index.to_numpy(int),
            vertical_sig.parent_index.to_numpy(int),
        ]
    )
    weights = np.concatenate(
        [
            horizontal_sig.evidence_z.to_numpy(float),
            horizontal_sig.evidence_z.to_numpy(float),
            vertical_sig.evidence_z.to_numpy(float),
            vertical_sig.evidence_z.to_numpy(float),
        ]
    )
    combined = (
        sparse.coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        if len(rows)
        else sparse.csr_matrix((n_nodes, n_nodes))
    )
    active = np.flatnonzero(np.asarray(combined.sum(axis=1)).ravel() > 0)
    zero_vertical = sparse.csr_matrix(combined.shape)
    calibration_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    gamma_grid = (
        tuple(resolution_grid)
        if resolution_grid is not None
        else (0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032)
    )
    for gamma in gamma_grid:
        trials = (
            [
                cpm_partition(
                    combined,
                    zero_vertical,
                    0.0,
                    gamma,
                    seed + int(gamma * 10000000.0) + run,
                )
                for run in range(trial_count)
            ]
            if len(active)
            else [np.arange(n_nodes)]
        )
        labels = trials[0]
        nmis = [
            normalized_mutual_info_score(left[active], right[active])
            for (left, right) in combinations(trials, 2)
        ]
        sizes = (
            np.unique(labels[active], return_counts=True)[1]
            if len(active)
            else np.asarray([], dtype=int)
        )
        in_target = sizes[(sizes >= 20) & (sizes <= 200)]
        metrics = {
            "community_count": float(len(sizes)),
            "max_nodes": float(sizes.max()) if len(sizes) else 0.0,
            "share_active_nodes_in_20_200": float(
                in_target.sum() / max(len(active), 1)
            ),
            "share_blocks_in_20_200": float(len(in_target) / max(len(sizes), 1)),
            "median_nodes": float(np.median(sizes)) if len(sizes) else 0.0,
            "seed_NMI": float(np.mean(nmis)) if nmis else 1.0,
        }
        candidates.append((gamma, labels, metrics))
        calibration_rows.append(
            {
                "resolution_gamma": gamma,
                **metrics,
                "selection_rule": "先保留随机种子稳定的候选；随后将20–200节点覆盖率作为软偏好排序。不会合并、删除或按规模强制切分结构块。",
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    stable_threshold = float(calibration.seed_NMI.median())
    valid = [
        (gamma, labels, metrics)
        for (gamma, labels, metrics) in candidates
        if metrics["seed_NMI"] >= stable_threshold
    ]
    choice = max(
        valid or candidates,
        key=lambda item: (
            item[2]["share_active_nodes_in_20_200"],
            item[2]["share_blocks_in_20_200"],
            item[2]["seed_NMI"],
            -item[2]["max_nodes"],
        ),
    )
    (selected_gamma, labels, _) = choice
    calibration["selected"] = calibration.resolution_gamma == selected_gamma
    order = (
        sorted(
            np.unique(labels[active]),
            key=lambda value: (-int((labels[active] == value).sum()), int(value)),
        )
        if len(active)
        else []
    )
    mapping = {int(value): f"S{position + 1}" for (position, value) in enumerate(order)}
    block_lookup = {
        int(position): mapping[int(labels[position])] for position in active
    }
    node_rows = []
    level_by_index = np.asarray(
        [int(data.tree[node]["level"]) for node in graph_data.ids]
    )
    horizontal_degree = np.zeros(n_nodes, dtype=int)
    if len(horizontal_sig):
        np.add.at(horizontal_degree, horizontal_sig.node_1_index.to_numpy(int), 1)
        np.add.at(horizontal_degree, horizontal_sig.node_2_index.to_numpy(int), 1)
    vertical_degree = np.zeros(n_nodes, dtype=int)
    if len(vertical_sig):
        np.add.at(vertical_degree, vertical_sig.parent_index.to_numpy(int), 1)
        np.add.at(vertical_degree, vertical_sig.child_index.to_numpy(int), 1)
    for position, node in enumerate(graph_data.ids):
        node_rows.append(
            {
                "node_id": node,
                "name": data.tree[node]["name"],
                "level": int(level_by_index[position]),
                "significant_horizontal_degree": int(horizontal_degree[position]),
                "significant_vertical_degree": int(vertical_degree[position]),
                "significance_block_id": block_lookup.get(position, ""),
                "node_status": "进入显著性校正结构块"
                if position in block_lookup
                else "保留在原始 PMID 层；未进入高置信结构块",
            }
        )
    members = pd.DataFrame(node_rows)
    blocks: list[dict[str, Any]] = []
    for block in sorted(set(block_lookup.values()), key=lambda value: int(value[1:])):
        part = members[members.significance_block_id == block]
        positions = np.asarray(
            [index for (index, value) in block_lookup.items() if value == block],
            dtype=int,
        )
        internal_horizontal = horizontal_sig[
            horizontal_sig.node_1_index.isin(positions)
            & horizontal_sig.node_2_index.isin(positions)
        ]
        internal_vertical = vertical_sig[
            vertical_sig.parent_index.isin(positions)
            & vertical_sig.child_index.isin(positions)
        ]
        blocks.append(
            {
                "significance_block_id": block,
                "topic_nodes": int(len(part)),
                "lowest_level": int(part.level.min()),
                "highest_level": int(part.level.max()),
                "levels_present": " | ".join(
                    (
                        f"L{level}:{count}"
                        for (level, count) in part.level.value_counts()
                        .sort_index()
                        .items()
                    )
                ),
                "significant_horizontal_relations": int(len(internal_horizontal)),
                "significant_direct_parent_child_relations": int(
                    len(internal_vertical)
                ),
                "total_evidence_z": float(
                    internal_horizontal.evidence_z.sum()
                    + internal_vertical.evidence_z.sum()
                ),
                "representative_topics": " | ".join(
                    part.sort_values(
                        [
                            "significant_horizontal_degree",
                            "significant_vertical_degree",
                        ],
                        ascending=False,
                    ).name.head(15)
                ),
                "definition": "逐层显著横向关系，经显著直接父子边连续扩展；不允许跨越未支持中间节点。",
            }
        )
    return (members, pd.DataFrame(blocks), combined, calibration)


def _clean_community_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _node_pmid_sets(
    data: TopicTreeDataset, graph_data: HypergraphData
) -> dict[str, set[int]]:
    return {node: set(data.node_papers[node].tolist()) for node in graph_data.ids}


def build_readable_pmid_chord_list(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    horizontal: pd.DataFrame,
    vertical: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    structure: str,
    membership_column: str,
    status_column: str,
) -> pd.DataFrame:
    ids = graph_data.ids
    index_to_id = dict(enumerate(ids))
    node_name = {
        index: data.tree[node]["name"] for (index, node) in index_to_id.items()
    }
    node_level = {
        index: int(data.tree[node]["level"]) for (index, node) in index_to_id.items()
    }
    node_papers = {
        index: int(data.counts[node]) for (index, node) in index_to_id.items()
    }
    membership_map = {
        str(row.node_id): _clean_community_id(getattr(row, membership_column))
        for row in membership.itertuples()
    }
    status_map = {
        str(row.node_id): str(getattr(row, status_column, ""))
        for row in membership.itertuples()
    }

    def attach_node_details(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for side in ("start", "end"):
            index_column = f"{side}_node_graph_index"
            frame[f"{side}_node_id"] = frame[index_column].map(index_to_id)
            frame[f"{side}_node_level"] = (
                frame[index_column].map(node_level).astype("Int64")
            )
            frame[f"{side}_node_name"] = frame[index_column].map(node_name)
            frame[f"{side}_node_paper_count"] = (
                frame[index_column].map(node_papers).astype("Int64")
            )
            frame[f"{side}_community_id"] = (
                frame[f"{side}_node_id"].map(membership_map).fillna("")
            )
            frame[f"{side}_membership_status"] = (
                frame[f"{side}_node_id"].map(status_map).fillna("")
            )
        same = (frame.start_community_id != "") & (
            frame.start_community_id == frame.end_community_id
        )
        both = (frame.start_community_id != "") & (frame.end_community_id != "")
        frame["community_context_id"] = np.where(
            same,
            frame.start_community_id,
            np.where(
                both, frame.start_community_id + " ↔ " + frame.end_community_id, ""
            ),
        )
        frame["community_chord_role"] = np.select(
            [same, both],
            ["core_chord", "intercommunity_chord"],
            default="peripheral_or_unassigned_chord",
        )
        frame["admitted_to_structure"] = frame.statistically_significant
        return frame

    h = horizontal.copy()
    for column, default in {
        "observed_fractional_pair_mass": np.nan,
        "null_expected_fractional_pair_mass": np.nan,
        "excess_support": np.nan,
        "fold_over_null": np.nan,
        "p_configuration_approx": np.nan,
        "q_BY_within_level": np.nan,
        "significant_horizontal_relation": False,
    }.items():
        if column not in h:
            h[column] = default
    h = h.assign(
        edge_type="horizontal_same_level_pmid_chord",
        edge_orientation="horizontal",
        start_node_graph_index=h.node_1_index.astype(int),
        end_node_graph_index=h.node_2_index.astype(int),
        raw_independent_pmid_count=h.independent_pmid_support.astype(float),
        corrected_fractional_pmid_mass=h.observed_fractional_pair_mass.astype(float),
        null_expected_pmid_count=h.null_expected_support.astype(float),
        corrected_excess_pmid_count=h.excess_support.astype(float),
        effect_size_fold_over_null=h.fold_over_null.astype(float),
        effect_size_z=h.evidence_z.astype(float),
        p_value=h.p_configuration_approx.astype(float),
        q_value=h.q_BY_within_level.astype(float),
        statistically_significant=h.significant_horizontal_relation.astype(bool),
        corrected_pmid_quantity_definition="同一 PMID 在各有效 level 间按总质量1分配，并按同层超边大小分摊至节点对的观察质量",
    )
    v = vertical.copy()
    p_column = (
        "p_tree_constrained_hypergeom"
        if "p_tree_constrained_hypergeom" in v
        else "p_tree_constrained_weighted_normal"
        if "p_tree_constrained_weighted_normal" in v
        else "p_direct_tree_null"
    )
    q_column = (
        "q_BY_all_direct_parent_child_tests"
        if "q_BY_all_direct_parent_child_tests" in v
        else None
    )
    for column, default in {
        "effective_pmid_support": 0.0,
        "null_expected_support": np.nan,
        "evidence_z": np.nan,
        p_column: np.nan,
        "significant_vertical_relation": False,
    }.items():
        if column not in v:
            v[column] = default
    if q_column is None:
        v["_q_value"] = np.nan
    else:
        v["_q_value"] = v[q_column].astype(float)
    v = v.assign(
        edge_type="vertical_direct_parent_child_chord",
        edge_orientation="vertical",
        start_node_graph_index=v.parent_index.astype(int),
        end_node_graph_index=v.child_index.astype(int),
        raw_independent_pmid_count=v.effective_pmid_support.astype(float),
        corrected_fractional_pmid_mass=v.effective_pmid_support.astype(float),
        null_expected_pmid_count=v.null_expected_support.astype(float),
        corrected_excess_pmid_count=(
            v.effective_pmid_support - v.null_expected_support
        ).astype(float),
        effect_size_fold_over_null=(
            v.effective_pmid_support / v.null_expected_support.clip(lower=1e-12)
        ).astype(float),
        effect_size_z=v.evidence_z.astype(float),
        p_value=v[p_column].astype(float),
        q_value=v._q_value.astype(float),
        statistically_significant=v.significant_vertical_relation.astype(bool),
        corrected_pmid_quantity_definition="在受限父节点论文池中，对贡献横向证据的 PMID 进行直接父子继承校正后的有效支持量",
    )
    columns = [
        "edge_type",
        "edge_orientation",
        "start_node_graph_index",
        "start_node_id",
        "start_node_level",
        "start_node_name",
        "start_node_paper_count",
        "end_node_graph_index",
        "end_node_id",
        "end_node_level",
        "end_node_name",
        "end_node_paper_count",
        "raw_independent_pmid_count",
        "corrected_fractional_pmid_mass",
        "corrected_pmid_quantity_definition",
        "null_expected_pmid_count",
        "corrected_excess_pmid_count",
        "effect_size_fold_over_null",
        "effect_size_z",
        "p_value",
        "q_value",
        "statistically_significant",
        "start_community_id",
        "end_community_id",
        "community_context_id",
        "community_chord_role",
        "admitted_to_structure",
    ]
    result = pd.concat(
        [attach_node_details(h)[columns], attach_node_details(v)[columns]],
        ignore_index=True,
    )
    result.insert(0, "structure", structure)
    result.insert(
        1,
        "chord_id",
        [f"{structure}_E{position:07d}" for position in range(1, len(result) + 1)],
    )
    return result.sort_values(
        ["edge_orientation", "start_node_level", "start_node_id", "end_node_id"],
        kind="stable",
    ).reset_index(drop=True)


def _edge_key(row: Any) -> tuple[str, int, int]:
    (left, right) = (int(row.start_node_graph_index), int(row.end_node_graph_index))
    if row.edge_orientation == "horizontal":
        (left, right) = (min(left, right), max(left, right))
    return (str(row.edge_orientation), left, right)


def _edge_pmid_indices(
    row: Any,
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    node_pmids: dict[str, set[int]],
    active_sources: list[set[int]] | None,
) -> set[int]:
    left_node = graph_data.ids[int(row.start_node_graph_index)]
    right_node = graph_data.ids[int(row.end_node_graph_index)]
    if row.edge_orientation == "horizontal":
        return node_pmids[left_node] & node_pmids[right_node]
    if active_sources is None:
        return node_pmids[left_node] & node_pmids[right_node]
    return (
        active_sources[int(row.start_node_graph_index)]
        & active_sources[int(row.end_node_graph_index)]
        & node_pmids[left_node]
    )


def _select_community_cores(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    membership: pd.DataFrame,
    membership_column: str,
    chords: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, list[str]]]:
    membership_map = {
        str(row.node_id): _clean_community_id(getattr(row, membership_column))
        for row in membership.itertuples()
    }
    scores = {node: 0.0 for node in graph_data.ids}
    admitted = chords[
        chords.admitted_to_structure.astype(bool)
        & chords.community_chord_role.eq("core_chord")
    ]
    for row in admitted.itertuples():
        strength = max(float(row.effect_size_z), 0.0)
        scores[str(row.start_node_id)] += strength
        scores[str(row.end_node_id)] += strength
    cores: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    groups = sorted({group for group in membership_map.values() if group})
    for group in groups:
        nodes = [
            node for node in graph_data.ids if membership_map.get(node, "") == group
        ]
        target = min(15, max(1, int(math.ceil(len(nodes) * 0.15))))
        ranked = sorted(
            nodes,
            key=lambda node: (
                -scores[node],
                -data.counts[node],
                -int(data.tree[node]["level"]),
                node,
            ),
        )
        selected = ranked[:target]
        cores[group] = selected
        for rank, node in enumerate(selected, start=1):
            rows.append(
                {
                    "community_id": group,
                    "core_rank": rank,
                    "node_id": node,
                    "node_name": data.tree[node]["name"],
                    "level": int(data.tree[node]["level"]),
                    "all_paper_pmid_count": int(data.counts[node]),
                    "within_community_core_score": float(scores[node]),
                    "core_rule": "按社群内已纳入弦的证据强度排序；取前 max(1, ceil(节点数×15%)) 个，最多15个。",
                }
            )
    return (pd.DataFrame(rows), scores, cores)


def _community_classification_paths(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    membership: pd.DataFrame,
    membership_column: str,
    node_scores: dict[str, float],
    structure: str,
    node_pmids: dict[str, set[int]],
) -> pd.DataFrame:
    membership_map = {
        str(row.node_id): _clean_community_id(getattr(row, membership_column))
        for row in membership.itertuples()
    }
    index_of_id = {node: position for (position, node) in enumerate(graph_data.ids)}
    rows: list[dict[str, Any]] = []
    for group in sorted({value for value in membership_map.values() if value}):
        positions = [
            position
            for (position, node) in enumerate(graph_data.ids)
            if membership_map.get(node, "") == group
        ]
        position_set = set(positions)
        children: dict[int, list[int]] = {position: [] for position in positions}
        for position in positions:
            parent = index_of_id.get(data.parent.get(graph_data.ids[position]))
            if parent in position_set:
                children[parent].append(position)
        leaves = [position for position in positions if not children[position]]
        leaves.sort(
            key=lambda position: (
                -int(data.tree[graph_data.ids[position]]["level"]),
                graph_data.ids[position],
            )
        )
        for ordinal, leaf in enumerate(leaves, start=1):
            path: list[int] = []
            current = leaf
            while True:
                path.append(current)
                parent = index_of_id.get(data.parent.get(graph_data.ids[current]))
                if parent not in position_set:
                    break
                current = int(parent)
            representative = max(
                path,
                key=lambda position: (
                    node_scores[graph_data.ids[position]],
                    int(data.tree[graph_data.ids[position]]["level"]),
                    graph_data.ids[position],
                ),
            )
            pmids = (
                set().union(
                    *(node_pmids[graph_data.ids[position]] for position in path)
                )
                if path
                else set()
            )
            path_nodes = [graph_data.ids[position] for position in path]
            rows.append(
                {
                    "structure": structure,
                    "community_id": group,
                    "classification_path_id": f"{group}_Path{ordinal}",
                    "path_node_count": int(len(path_nodes)),
                    "path_levels": " → ".join(
                        (str(data.tree[node]["level"]) for node in reversed(path_nodes))
                    ),
                    "path_node_ids": " → ".join(reversed(path_nodes)),
                    "path_node_names": " → ".join(
                        (data.tree[node]["name"] for node in reversed(path_nodes))
                    ),
                    "path_unique_all_paper_pmid_count": int(len(pmids)),
                    "representative_node_id": graph_data.ids[representative],
                    "representative_node_name": data.tree[
                        graph_data.ids[representative]
                    ]["name"],
                    "representative_node_level": int(
                        data.tree[graph_data.ids[representative]]["level"]
                    ),
                    "representative_node_core_score": float(
                        node_scores[graph_data.ids[representative]]
                    ),
                    "path_definition": "社群内仅沿主题树真实直接父子边由叶至根追溯；中间节点不在社群时停止，不跨越连接祖父与孙节点。",
                }
            )
    return pd.DataFrame(rows)


def build_community_catalog(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    membership: pd.DataFrame,
    membership_column: str,
    chords: pd.DataFrame,
    active_sources: list[set[int]] | None,
    *,
    structure: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[tuple[str, int, int]]]:
    node_pmids = _node_pmid_sets(data, graph_data)
    (core_table, node_scores, cores) = _select_community_cores(
        data, graph_data, membership, membership_column, chords
    )
    paths = _community_classification_paths(
        data,
        graph_data,
        membership,
        membership_column,
        node_scores,
        structure,
        node_pmids,
    )
    membership_map = {
        str(row.node_id): _clean_community_id(getattr(row, membership_column))
        for row in membership.itertuples()
    }
    admitted = chords[
        chords.admitted_to_structure.astype(bool)
        & chords.community_chord_role.eq("core_chord")
    ].copy()
    core_edge_keys = {_edge_key(row) for row in admitted.itertuples()}
    rows: list[dict[str, Any]] = []
    for group in sorted({value for value in membership_map.values() if value}):
        nodes = [
            node for node in graph_data.ids if membership_map.get(node, "") == group
        ]
        all_pmids = (
            set().union(*(node_pmids[node] for node in nodes)) if nodes else set()
        )
        core_nodes = cores.get(group, [])
        core_pmids = (
            set().union(*(node_pmids[node] for node in core_nodes))
            if core_nodes
            else set()
        )
        internal = admitted[admitted.community_context_id.eq(group)]
        horizontal_count = int(internal.edge_orientation.eq("horizontal").sum())
        vertical_count = int(internal.edge_orientation.eq("vertical").sum())
        edge_pmids: set[int] = set()
        for edge in internal.itertuples():
            edge_pmids |= _edge_pmid_indices(
                edge, data, graph_data, node_pmids, active_sources
            )
        group_paths = paths[paths.community_id.eq(group)] if not paths.empty else paths
        path_sizes = " | ".join(
            (
                f"{row.classification_path_id}:{row.path_node_count}"
                for row in group_paths.itertuples()
            )
        )
        path_pmid_counts = " | ".join(
            (
                f"{row.classification_path_id}:{row.path_unique_all_paper_pmid_count}"
                for row in group_paths.itertuples()
            )
        )
        path_representatives = " | ".join(
            (
                f"{row.classification_path_id}:{row.representative_node_name}"
                for row in group_paths.itertuples()
            )
        )
        touching = chords[
            chords.admitted_to_structure.astype(bool)
            & (chords.start_community_id.eq(group) | chords.end_community_id.eq(group))
            & ~chords.community_chord_role.eq("core_chord")
        ]
        row = {
            "structure": structure,
            "community_id": group,
            "node_count": int(len(nodes)),
            "level_distribution": " | ".join(
                (
                    f"L{level}:{count}"
                    for (level, count) in pd.Series(
                        [data.tree[node]["level"] for node in nodes]
                    )
                    .value_counts()
                    .sort_index()
                    .items()
                )
            ),
            "core_node_count": int(len(core_nodes)),
            "core_node_names": " | ".join(
                (data.tree[node]["name"] for node in core_nodes)
            ),
            "core_node_ids": " | ".join(core_nodes),
            "core_node_unique_all_paper_pmid_count": int(len(core_pmids)),
            "all_node_unique_all_paper_pmid_count": int(len(all_pmids)),
            "internal_chord_count": int(len(internal)),
            "internal_horizontal_chord_count": horizontal_count,
            "internal_vertical_chord_count": vertical_count,
            "internal_chord_unique_evidence_pmid_count": int(len(edge_pmids)),
            "intercommunity_or_bridge_chord_count_touching": int(len(touching)),
            "classification_path_count": int(len(group_paths)),
            "classification_path_node_counts": path_sizes,
            "classification_path_unique_all_paper_pmid_counts": path_pmid_counts,
            "classification_path_representative_nodes": path_representatives,
        }
        rows.append(row)
    return (pd.DataFrame(rows), paths, core_table, core_edge_keys)


def apply_posthoc_community_screen(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    audited = summary.copy()
    conditions = pd.DataFrame(
        {
            "criterion_unique_evidence_pmids_ge_3": audited.internal_chord_unique_evidence_pmid_count.ge(
                3
            ),
            "criterion_internal_chords_ge_3": audited.internal_chord_count.ge(3),
            "criterion_internal_horizontal_chords_ge_2": audited.internal_horizontal_chord_count.ge(
                2
            ),
            "criterion_classification_paths_ge_3": audited.classification_path_count.ge(
                3
            ),
        }
    )
    for column in conditions:
        audited[column] = conditions[column]
    audited["posthoc_confirmation_rule"] = (
        "去重弦贡献 PMID≥3；内部弦≥3；内部横向弦≥2；连续分类路径≥3。发现后筛选，不重新运行社群发现，不合并小结构。"
    )
    audited["formal_community_confirmed"] = conditions.all(axis=1)
    reasons: list[str] = []
    for row in conditions.itertuples(index=False):
        failed = []
        if not row.criterion_unique_evidence_pmids_ge_3:
            failed.append("弦贡献 PMID<3")
        if not row.criterion_internal_chords_ge_3:
            failed.append("内部弦<3")
        if not row.criterion_internal_horizontal_chords_ge_2:
            failed.append("内部横向弦<2")
        if not row.criterion_classification_paths_ge_3:
            failed.append("连续分类路径<3")
        reasons.append("通过全部确认条件" if not failed else "；".join(failed))
    audited["posthoc_screening_result"] = reasons
    audited["posthoc_structure_class"] = np.where(
        audited.formal_community_confirmed,
        "confirmed_multibranch_community",
        np.where(
            audited.internal_chord_count.eq(0),
            "singleton_or_partition_fragment",
            "micro_evidence_structure",
        ),
    )
    confirmed = set(
        audited.loc[audited.formal_community_confirmed, "community_id"].astype(str)
    )
    return (audited, confirmed)


def restrict_membership_to_confirmed(
    membership: pd.DataFrame,
    membership_column: str,
    status_column: str,
    confirmed: set[str],
) -> pd.DataFrame:
    result = membership.copy()
    result["discovered_community_id"] = result[membership_column].map(
        _clean_community_id
    )
    result["formal_community_confirmed"] = result.discovered_community_id.isin(
        confirmed
    )
    result[membership_column] = np.where(
        result.formal_community_confirmed, result.discovered_community_id, ""
    )
    result["posthoc_structure_class"] = np.where(
        result.formal_community_confirmed,
        "confirmed_multibranch_community",
        np.where(
            result.discovered_community_id.ne(""),
            "excluded_discovered_structure",
            "not_assigned_at_discovery",
        ),
    )
    result[status_column] = np.where(
        result.formal_community_confirmed,
        result[status_column],
        np.where(
            result.discovered_community_id.ne(""),
            "发现后未通过正式社群筛选",
            result[status_column],
        ),
    )
    return result


def export_reader_focused_pmid_catalog(
    data: TopicTreeDataset,
    out: Output,
    graph_data: HypergraphData,
    horizontal: pd.DataFrame,
    vertical: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    structure: str,
    membership_column: str,
    status_column: str,
    active_sources: list[set[int]] | None,
) -> dict[str, Any]:
    chords = build_readable_pmid_chord_list(
        data,
        graph_data,
        horizontal,
        vertical,
        membership,
        structure=structure,
        membership_column=membership_column,
        status_column=status_column,
    )
    (summary, paths, cores, core_edge_keys) = build_community_catalog(
        data,
        graph_data,
        membership,
        membership_column,
        chords,
        active_sources,
        structure=structure,
    )
    (audited_summary, confirmed) = apply_posthoc_community_screen(summary)
    formal_members = restrict_membership_to_confirmed(
        membership, membership_column, status_column, confirmed
    )
    status_by_community = audited_summary.set_index(
        "community_id"
    ).posthoc_structure_class.to_dict()
    chords["start_community_posthoc_class"] = chords.start_community_id.map(
        status_by_community
    ).fillna("")
    chords["end_community_posthoc_class"] = chords.end_community_id.map(
        status_by_community
    ).fillna("")
    chords["formal_community_context_id"] = np.where(
        chords.community_chord_role.eq("core_chord")
        & chords.community_context_id.isin(confirmed),
        chords.community_context_id,
        "",
    )
    chords["formal_community_chord_role"] = np.where(
        chords.formal_community_context_id.ne(""),
        "core_chord",
        np.where(
            chords.community_chord_role.eq("core_chord"),
            "excluded_structure_chord",
            chords.community_chord_role,
        ),
    )
    formal_summary = audited_summary[audited_summary.formal_community_confirmed].copy()
    formal_paths = paths[paths.community_id.isin(confirmed)].copy()
    formal_cores = cores[cores.community_id.isin(confirmed)].copy()
    formal_core_edge_keys = {
        _edge_key(row)
        for row in chords[
            chords.admitted_to_structure.astype(bool)
            & chords.formal_community_chord_role.eq("core_chord")
        ].itertuples()
    }
    prefix = "16_S社群"
    out.table(audited_summary, f"{prefix}_发现后筛选审计")
    out.table(formal_summary, f"{prefix}_完整清单")
    out.table(formal_cores, f"{prefix}_核心节点清单")
    out.table(formal_paths, f"{prefix}_连续分类路径清单")
    out.table(
        chords[chords.statistically_significant.astype(bool)].copy(),
        f"{prefix}_显著性筛选弦列表",
    )
    out.table(chords, f"{prefix}_全部弦列表")
    return {
        "summary": {
            row.community_id: row._asdict()
            for row in formal_summary.itertuples(index=False)
        },
        "path_counts": formal_paths.groupby("community_id").size().to_dict()
        if not formal_paths.empty
        else {},
        "core_edge_keys": formal_core_edge_keys,
        "membership": {
            str(row.node_id): _clean_community_id(getattr(row, membership_column))
            for row in formal_members.itertuples()
        },
        "confirmed_ids": confirmed,
        "formal_members": formal_members,
        "formal_summary": formal_summary,
        "formal_paths": formal_paths,
        "screening_audit": audited_summary,
    }


def run_significance_calibrated_pmid_layer(
    data: TopicTreeDataset,
    out: Output,
    pmid: CommunityResult | HypergraphData,
    seed: int,
) -> dict[str, Any]:
    graph_data = pmid.data if isinstance(pmid, CommunityResult) else pmid
    calibration = levelwise_pmid_significance(data, graph_data)
    horizontal = calibration["horizontal"]
    vertical = significant_vertical_evidence(
        data, graph_data, calibration["active_sources"]
    )
    (members, blocks, _, scale_calibration) = significant_pmid_blocks(
        data, graph_data, horizontal, vertical, seed
    )
    index_to_id = dict(enumerate(graph_data.ids))
    index_to_name = {
        index: data.tree[node]["name"] for (index, node) in index_to_id.items()
    }
    for frame, left, right in [
        (horizontal, "node_1_index", "node_2_index"),
        (vertical, "parent_index", "child_index"),
    ]:
        frame.insert(0, "node_1_id", frame[left].map(index_to_id))
        frame.insert(1, "node_1_name", frame[left].map(index_to_name))
        frame.insert(2, "node_2_id", frame[right].map(index_to_id))
        frame.insert(3, "node_2_name", frame[right].map(index_to_name))
    out.table(calibration["level_summary"], "12_PMID逐层显著性_层级汇总")
    out.table(horizontal, "12_PMID逐层显著性_全部已观测节点对")
    out.table(
        horizontal[horizontal.significant_horizontal_relation],
        "12_PMID逐层显著性_高置信横向关系",
    )
    out.table(vertical, "12_PMID逐层显著性_全部可检验直接父子关系")
    out.table(
        vertical[vertical.significant_vertical_relation],
        "12_PMID逐层显著性_高置信直接父子关系",
    )
    out.table(scale_calibration, "12_PMID逐层显著性_结构块尺度校准")
    reader_catalog = export_reader_focused_pmid_catalog(
        data,
        out,
        graph_data,
        horizontal,
        vertical,
        members,
        structure="S",
        membership_column="significance_block_id",
        status_column="node_status",
        active_sources=calibration["active_sources"],
    )
    formal_members = reader_catalog["formal_members"]
    formal_blocks = blocks[
        blocks.significance_block_id.astype(str).isin(reader_catalog["confirmed_ids"])
    ].copy()
    screening_fields = reader_catalog["screening_audit"][
        [
            "community_id",
            "formal_community_confirmed",
            "posthoc_structure_class",
            "posthoc_screening_result",
        ]
    ].rename(columns={"community_id": "significance_block_id"})
    formal_blocks = formal_blocks.merge(
        screening_fields, on="significance_block_id", how="left"
    )
    out.table(reader_catalog["screening_audit"], "12_PMID逐层显著性_发现后筛选审计")
    out.table(formal_members, "12_PMID逐层显著性_全部节点状态")
    out.table(formal_blocks, "12_PMID逐层显著性_连续跨层结构块")
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "members": formal_members,
        "blocks": formal_blocks,
        "active_sources": calibration["active_sources"],
        "reader_catalog": reader_catalog,
    }


def factor_pair_layer(
    graph_data: HypergraphData, factor_mask: np.ndarray
) -> sparse.csr_matrix:
    n_tree = len(graph_data.ids)
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for factor_id, nodes in enumerate(graph_data.members):
        if not factor_mask[factor_id] or len(nodes) < 2:
            continue
        total = float(graph_data.factor_weight[factor_id])
        value = total * 2.0 / (len(nodes) * (len(nodes) - 1))
        for left, right in combinations(nodes.tolist(), 2):
            rows.append(left)
            columns.append(right)
            weights.append(value)
    pair = sparse.coo_matrix((weights, (rows, columns)), shape=(n_tree, n_tree)).tocsr()
    return (pair + pair.T).tocsr()


QUALITY_MODEL_SPECS: dict[str, str] = {
    "JCR_Q": "JCR 分区权重：Q1=4，Q2=3，Q3=2，Q4=1，缺失=1。",
    "IF_year": "发表年份内 IF 百分位权重：最低 IF=1、最高 IF=4，中间值按 1+3×年内百分位线性映射；缺失=1。",
}


def build_quality_weight_audit(data: TopicTreeDataset) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quartile_weight = {"Q1": 4.0, "Q2": 3.0, "Q3": 2.0, "Q4": 1.0, "未分区或缺失": 1.0}
    for source, pmid in enumerate(data.pmids):
        record = data.papers[str(pmid)]
        quartile = normalise_jcr_quartile(record)
        impact = parse_impact_factor(record)
        rows.append(
            {
                "pmid_index": int(source),
                "pmid": str(pmid),
                "publication_year": int(data.years[source]),
                "journal": norm_name(
                    record.get("Journal") or record.get("journal") or ""
                ),
                "JCR_quartile": quartile,
                "impact_factor_raw": impact,
                "weight_JCR_Q": float(quartile_weight.get(quartile, 1.0)),
            }
        )
    audit = pd.DataFrame(rows)
    audit["IF_year_reference_n"] = 0
    audit["IF_year_percentile"] = np.nan
    for year, positions in audit.groupby("publication_year").groups.items():
        index = list(positions)
        values = audit.loc[index, "impact_factor_raw"].to_numpy(float)
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        audit.loc[index, "IF_year_reference_n"] = int(len(finite))
        if len(finite) < 2 or float(np.max(finite)) <= float(np.min(finite)):
            continue
        ranks = stats.rankdata(finite, method="average")
        percentile = (ranks - ranks.min()) / max(float(ranks.max() - ranks.min()), 1.0)
        year_percentile = np.full(len(index), np.nan, dtype=float)
        year_percentile[finite_mask] = percentile
        audit.loc[index, "IF_year_percentile"] = year_percentile
    audit["IF_weight_reference"] = np.where(
        audit.IF_year_percentile.notna(),
        "publication_year_percentile",
        "missing_or_nonvarying_IF",
    )
    audit["weight_IF_year"] = np.where(
        np.isfinite(audit.impact_factor_raw.to_numpy(float)),
        1.0 + 3.0 * audit.IF_year_percentile.fillna(0.0).to_numpy(float),
        1.0,
    )
    audit["quality_metadata_status"] = np.select(
        [
            audit.JCR_quartile.ne("未分区或缺失") & audit.impact_factor_raw.notna(),
            audit.JCR_quartile.ne("未分区或缺失"),
            audit.impact_factor_raw.notna(),
        ],
        ["JCR_and_IF_available", "JCR_only", "IF_only"],
        default="JCR_and_IF_missing",
    )
    return audit


def quality_weighted_levelwise_significance(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    pmid_weight: np.ndarray,
    model: str,
    alpha: float = 0.05,
    min_independent_pmids: int = 2,
) -> dict[str, Any]:
    n_nodes = len(graph_data.ids)
    levels = graph_data.hyperedges["level"].to_numpy(dtype=int)
    active_sources: list[set[int]] = [set() for _ in range(n_nodes)]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for level in sorted(np.unique(levels)):
        factor_ids = np.flatnonzero(levels == level)
        factor_nodes = [graph_data.members[int(factor_id)] for factor_id in factor_ids]
        sizes = np.asarray([len(nodes) for nodes in factor_nodes], dtype=np.int32)
        sources = graph_data.factor_source[factor_ids].astype(int)
        factor_quality = pmid_weight[sources]
        stubs = np.concatenate(factor_nodes)
        degree = np.bincount(stubs, minlength=n_nodes).astype(float)
        total_stubs = float(len(stubs))
        ordered_slot_pairs = float(np.sum(sizes * (sizes - 1)))
        pair_values: dict[tuple[int, int], list[float]] = {}
        for factor_id, nodes, source_weight in zip(
            factor_ids, factor_nodes, factor_quality
        ):
            size = len(nodes)
            pair_mass = float(
                graph_data.factor_weight[int(factor_id)]
                * source_weight
                * 2.0
                / (size * (size - 1))
            )
            for left, right in combinations(nodes.tolist(), 2):
                if left > right:
                    (left, right) = (right, left)
                item = pair_values.setdefault((left, right), [0.0, 0.0, 0])
                item[0] += float(source_weight)
                item[1] += pair_mass
                item[2] += 1
        keys = list(pair_values)
        if not keys or total_stubs < 2:
            summaries.append(
                {
                    "quality_model": model,
                    "level": int(level),
                    "tested_pairs": 0,
                    "significant_pairs": 0,
                    "status": "没有可检验的同层 PMID 节点对",
                }
            )
            continue
        left = np.asarray([key[0] for key in keys], dtype=np.int32)
        right = np.asarray([key[1] for key in keys], dtype=np.int32)
        observed_weighted = np.asarray(
            [pair_values[key][0] for key in keys], dtype=float
        )
        observed_mass = np.asarray([pair_values[key][1] for key in keys], dtype=float)
        expected_raw = (
            degree[left]
            * degree[right]
            * ordered_slot_pairs
            / (total_stubs * (total_stubs - 1.0))
        )
        (mean_weight, second_moment) = (
            float(np.mean(factor_quality)),
            float(np.mean(factor_quality**2)),
        )
        expected_weighted = expected_raw * mean_weight
        variance_weighted = np.maximum(expected_raw * second_moment, 1e-12)
        z_values = (observed_weighted - expected_weighted) / np.sqrt(variance_weighted)
        p_values = _normal_sf(z_values, alpha)
        q_values = adjust_pvalues(p_values, "BY")
        raw_support = np.asarray([pair_values[key][2] for key in keys], dtype=int)
        expected_mass = (
            2.0
            * float(np.sum(graph_data.factor_weight[factor_ids] * factor_quality))
            * degree[left]
            * degree[right]
            / (total_stubs * (total_stubs - 1.0))
        )
        supported = (
            (q_values < alpha) & (z_values > 0) & (raw_support >= min_independent_pmids)
        )
        significant_keys = {keys[position] for position in np.flatnonzero(supported)}
        for factor_id, nodes in zip(factor_ids, factor_nodes):
            source = int(graph_data.factor_source[int(factor_id)])
            for left_node, right_node in combinations(nodes.tolist(), 2):
                key = (
                    (left_node, right_node)
                    if left_node < right_node
                    else (right_node, left_node)
                )
                if key in significant_keys:
                    active_sources[left_node].add(source)
                    active_sources[right_node].add(source)
        for position, (node_left, node_right) in enumerate(keys):
            rows.append(
                {
                    "quality_model": model,
                    "level": int(level),
                    "node_1_index": int(node_left),
                    "node_2_index": int(node_right),
                    "independent_pmid_support": int(raw_support[position]),
                    "weighted_observed_pmid_support": float(
                        observed_weighted[position]
                    ),
                    "observed_fractional_pair_mass": float(observed_mass[position]),
                    "null_expected_support": float(expected_weighted[position]),
                    "weighted_null_variance": float(variance_weighted[position]),
                    "null_expected_fractional_pair_mass": float(
                        expected_mass[position]
                    ),
                    "excess_support": float(
                        observed_weighted[position] - expected_weighted[position]
                    ),
                    "fold_over_null": float(
                        observed_weighted[position]
                        / max(expected_weighted[position], 1e-12)
                    ),
                    "evidence_z": float(z_values[position]),
                    "p_configuration_approx": float(p_values[position]),
                    "q_BY_within_level": float(q_values[position]),
                    "significant_horizontal_relation": bool(supported[position]),
                }
            )
        summaries.append(
            {
                "quality_model": model,
                "level": int(level),
                "factors": int(len(factor_ids)),
                "tested_pairs": int(len(keys)),
                "significant_pairs": int(supported.sum()),
                "mean_pmid_quality_weight": mean_weight,
                "status": "加权复合 Poisson 配置零模型；PMID 质量权重随 PMID 保持不变",
            }
        )
    backend = _gpu_state()
    return {
        "horizontal": pd.DataFrame(rows),
        "level_summary": pd.DataFrame(summaries),
        "active_sources": active_sources,
        "compute_backend": {
            "p_value_kernel": "scipy_cpu_reference",
            "cuda_numeric_validation_enabled": bool(backend["validation_enabled"]),
            "cuda_numeric_divergence": bool(backend["diverged"]),
        },
    }


def quality_weighted_vertical_evidence(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    active_sources: list[set[int]],
    pmid_weight: np.ndarray,
    model: str,
    alpha: float = 0.05,
    min_independent_pmids: int = 2,
) -> pd.DataFrame:
    index = {node: position for (position, node) in enumerate(graph_data.ids)}
    rows: list[dict[str, Any]] = []
    for child, parent in data.parent.items():
        if child not in index or parent not in index:
            continue
        (child_index, parent_index) = (index[child], index[parent])
        pool = np.asarray(data.node_papers[parent], dtype=int)
        parent_active = active_sources[parent_index] & set(pool.tolist())
        child_active = active_sources[child_index] & set(pool.tolist())
        if not parent_active or not child_active:
            continue
        (population, draws) = (len(pool), len(child_active))
        if population <= 1 or draws == 0:
            continue
        values = np.asarray(
            [
                pmid_weight[source] if int(source) in parent_active else 0.0
                for source in pool
            ],
            dtype=float,
        )
        observed_sources = parent_active & child_active
        observed = (
            float(np.sum(pmid_weight[list(observed_sources)]))
            if observed_sources
            else 0.0
        )
        expected = float(draws * values.mean())
        population_variance = float(values.var(ddof=1)) if population > 1 else 0.0
        variance = float(draws * (1.0 - draws / population) * population_variance)
        z_value = (observed - expected) / math.sqrt(max(variance, 1e-12))
        p_value = float(stats.norm.sf(z_value))
        rows.append(
            {
                "quality_model": model,
                "parent_index": int(parent_index),
                "child_index": int(child_index),
                "parent_active_pmids": int(len(parent_active)),
                "child_active_pmids": int(draws),
                "effective_pmid_support": int(len(observed_sources)),
                "weighted_effective_pmid_support": observed,
                "null_expected_support": expected,
                "null_variance": variance,
                "evidence_z": float(z_value),
                "p_tree_constrained_weighted_normal": p_value,
            }
        )
    vertical = pd.DataFrame(
        rows,
        columns=[
            "quality_model",
            "parent_index",
            "child_index",
            "parent_active_pmids",
            "child_active_pmids",
            "effective_pmid_support",
            "weighted_effective_pmid_support",
            "null_expected_support",
            "null_variance",
            "evidence_z",
            "p_tree_constrained_weighted_normal",
        ],
    )
    if len(vertical):
        vertical["q_BY_all_direct_parent_child_tests"] = adjust_pvalues(
            vertical.p_tree_constrained_weighted_normal, "BY"
        )
        vertical["significant_vertical_relation"] = (
            (vertical.q_BY_all_direct_parent_child_tests < alpha)
            & (vertical.evidence_z > 0)
            & (vertical.effective_pmid_support >= min_independent_pmids)
        )
    else:
        vertical["q_BY_all_direct_parent_child_tests"] = pd.Series(dtype=float)
        vertical["significant_vertical_relation"] = pd.Series(dtype=bool)
    return vertical


def quality_block_summary(
    data: TopicTreeDataset,
    graph_data: HypergraphData,
    members: pd.DataFrame,
    blocks: pd.DataFrame,
    horizontal: pd.DataFrame,
    vertical: pd.DataFrame,
    pmid_weight: np.ndarray,
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    library_weight_by_year = (
        pd.Series(pmid_weight, index=data.years).groupby(level=0).sum()
    )
    members.set_index("node_id").significance_block_id.fillna("").astype(str).to_dict()
    for block in blocks.significance_block_id.astype(str):
        node_ids = (
            members.loc[members.significance_block_id.astype(str).eq(block), "node_id"]
            .astype(str)
            .tolist()
        )
        positions = np.asarray(
            [graph_data.ids.index(node) for node in node_ids], dtype=int
        )
        h = horizontal[
            horizontal.significant_horizontal_relation
            & horizontal.node_1_index.isin(positions)
            & horizontal.node_2_index.isin(positions)
        ]
        v = vertical[
            vertical.significant_vertical_relation
            & vertical.parent_index.isin(positions)
            & vertical.child_index.isin(positions)
        ]
        pmids = (
            set().union(*(set(data.node_papers[node].tolist()) for node in node_ids))
            if node_ids
            else set()
        )
        pmid_index = np.asarray(sorted(pmids), dtype=int)
        years = data.years[pmid_index] if len(pmid_index) else np.asarray([], dtype=int)
        weighted_mass = float(pmid_weight[pmid_index].sum()) if len(pmid_index) else 0.0
        degree = collections.Counter(
            h.node_1_index.tolist()
            + h.node_2_index.tolist()
            + v.parent_index.tolist()
            + v.child_index.tolist()
        )
        core = sorted(
            node_ids,
            key=lambda node: (
                -degree.get(graph_data.ids.index(node), 0),
                int(data.tree[node]["level"]),
                node,
            ),
        )[:15]
        summaries.append(
            {
                "quality_model": model,
                "quality_block_id": f"{model}_{block}",
                "source_block_id": block,
                "node_count": int(len(node_ids)),
                "lowest_level": int(
                    min((data.tree[node]["level"] for node in node_ids), default=0)
                ),
                "highest_level": int(
                    max((data.tree[node]["level"] for node in node_ids), default=0)
                ),
                "unique_all_paper_pmid_count": int(len(pmid_index)),
                "quality_weighted_unique_pmid_mass": weighted_mass,
                "internal_significant_horizontal_chord_count": int(len(h)),
                "internal_significant_vertical_chord_count": int(len(v)),
                "internal_quality_evidence_z": float(
                    h.evidence_z.sum() + v.evidence_z.sum()
                ),
                "core_node_names": " | ".join(
                    (data.tree[node]["name"] for node in core)
                ),
                "core_node_ids": " | ".join(core),
                "definition": "探索性质量加权显著弦构成的候选结构块；不替代正式 S 社群。",
            }
        )
        for year in data.year_values:
            mask = years == year
            annual_rows.append(
                {
                    "quality_model": model,
                    "quality_block_id": f"{model}_{block}",
                    "year": int(year),
                    "unique_pmids": int(mask.sum()),
                    "quality_weighted_pmid_mass": float(
                        pmid_weight[pmid_index[mask]].sum()
                    )
                    if mask.any()
                    else 0.0,
                    "library_quality_weighted_pmid_mass": float(
                        library_weight_by_year.get(year, np.nan)
                    ),
                }
            )
    annual = pd.DataFrame(annual_rows)
    if len(annual):
        annual["quality_weighted_share"] = (
            annual.quality_weighted_pmid_mass
            / annual.library_quality_weighted_pmid_mass.clip(lower=1e-12)
        )
    return (pd.DataFrame(summaries), annual)


def quality_robustness_comparison(
    out: Output,
    model: str,
    horizontal: pd.DataFrame,
    blocks: pd.DataFrame,
    members: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_horizontal = pd.read_csv(
        out.tables / "12_PMID逐层显著性_高置信横向关系.csv"
    )
    original_keys = set(
        zip(
            original_horizontal.node_1_index.astype(int),
            original_horizontal.node_2_index.astype(int),
        )
    )
    weighted_sig = horizontal[horizontal.significant_horizontal_relation]
    weighted_keys = set(
        zip(
            weighted_sig.node_1_index.astype(int), weighted_sig.node_2_index.astype(int)
        )
    )
    intersection = len(original_keys & weighted_keys)
    union = len(original_keys | weighted_keys)
    edge = pd.DataFrame(
        [
            {
                "quality_model": model,
                "formal_significant_horizontal_edges": len(original_keys),
                "quality_weighted_significant_horizontal_edges": len(weighted_keys),
                "edge_intersection": intersection,
                "edge_union": union,
                "edge_jaccard": intersection / union if union else np.nan,
                "formal_edge_retention": intersection / max(len(original_keys), 1),
                "quality_edge_overlap_with_formal": intersection
                / max(len(weighted_keys), 1),
            }
        ]
    )
    official = pd.read_csv(out.tables / "12_PMID逐层显著性_全部节点状态.csv")
    official = official[official.formal_community_confirmed.astype(bool)].copy()
    official_sets = {
        key: set(part.node_id.astype(str))
        for (key, part) in official.groupby("significance_block_id")
    }
    candidate_sets = {
        key: set(part.node_id.astype(str))
        for (key, part) in members[
            members.significance_block_id.astype(str).ne("")
        ].groupby("significance_block_id")
    }
    rows: list[dict[str, Any]] = []
    for block, nodes in candidate_sets.items():
        (match, overlap) = ("", 0)
        for formal, formal_nodes in official_sets.items():
            current = len(nodes & formal_nodes)
            if current > overlap:
                (match, overlap) = (formal, current)
        denominator = len(nodes | official_sets[match]) if match else 0
        rows.append(
            {
                "quality_model": model,
                "quality_block_id": f"{model}_{block}",
                "source_block_id": block,
                "quality_block_node_count": len(nodes),
                "best_formal_S_community_id": match,
                "intersection_node_count": overlap,
                "best_match_jaccard": overlap / denominator if denominator else 0.0,
                "quality_block_coverage_by_best_formal": overlap / max(len(nodes), 1),
                "formal_block_coverage_by_quality": overlap
                / max(len(official_sets.get(match, set())), 1)
                if match
                else 0.0,
            }
        )
    return (edge, pd.DataFrame(rows))


def run_quality_weighted_exploration(
    data: TopicTreeDataset, out: Output, seed: int
) -> dict[str, Any]:
    _require_table_files(
        out.tables,
        [
            "12_PMID逐层显著性_结构块尺度校准",
            "12_PMID逐层显著性_高置信横向关系",
            "12_PMID逐层显著性_全部节点状态",
        ],
        "探索性质量加权分析",
    )
    graph_data = build_pmid_hypergraph(data)
    audit = build_quality_weight_audit(data)
    out.table(audit, "22a_探索性质量加权_PMID权重审计")
    results: dict[str, Any] = {}
    (all_edges, all_matches, all_blocks) = ([], [], [])
    for model, description in QUALITY_MODEL_SPECS.items():
        weight_column = {"JCR_Q": "weight_JCR_Q", "IF_year": "weight_IF_year"}[model]
        weight = audit.sort_values("pmid_index")[weight_column].to_numpy(float)
        calibration = quality_weighted_levelwise_significance(
            data, graph_data, weight, model
        )
        vertical = quality_weighted_vertical_evidence(
            data, graph_data, calibration["active_sources"], weight, model
        )
        formal_resolution = pd.read_csv(
            out.tables / "12_PMID逐层显著性_结构块尺度校准.csv"
        )
        selected = formal_resolution.loc[
            formal_resolution.selected.astype(bool), "resolution_gamma"
        ]
        frozen_gamma = (
            float(selected.iloc[0])
            if len(selected)
            else float(formal_resolution.resolution_gamma.iloc[0])
        )
        (members, blocks, _, resolution) = significant_pmid_blocks(
            data,
            graph_data,
            calibration["horizontal"],
            vertical,
            seed,
            resolution_grid=(frozen_gamma,),
            trial_count=3,
        )
        resolution["resolution_policy"] = (
            "冻结正式 S 流程的已选 CPM gamma；每个质量模型仅进行3次种子稳定性复核"
        )
        quality_chords = build_readable_pmid_chord_list(
            data,
            graph_data,
            calibration["horizontal"],
            vertical,
            members,
            structure=f"quality_weighted_{model}",
            membership_column="significance_block_id",
            status_column="node_status",
        )
        (quality_catalog, quality_paths, quality_cores, _) = build_community_catalog(
            data,
            graph_data,
            members,
            "significance_block_id",
            quality_chords,
            calibration["active_sources"],
            structure=f"quality_weighted_{model}",
        )
        (quality_screen, quality_confirmed) = apply_posthoc_community_screen(
            quality_catalog
        )
        quality_formal_members = restrict_membership_to_confirmed(
            members, "significance_block_id", "node_status", quality_confirmed
        )
        quality_formal = quality_screen[
            quality_screen.formal_community_confirmed
        ].copy()
        (summary, annual) = quality_block_summary(
            data,
            graph_data,
            members,
            blocks,
            calibration["horizontal"],
            vertical,
            weight,
            model,
        )
        summary = summary.merge(
            quality_screen[
                [
                    "community_id",
                    "formal_community_confirmed",
                    "posthoc_structure_class",
                    "posthoc_screening_result",
                    "classification_path_count",
                    "internal_chord_unique_evidence_pmid_count",
                ]
            ],
            left_on="source_block_id",
            right_on="community_id",
            how="left",
        ).drop(columns="community_id")
        (edge_compare, block_match) = quality_robustness_comparison(
            out, model, calibration["horizontal"], blocks, quality_formal_members
        )
        prefix = f"22b_探索性质量加权_{model}"
        out.table(calibration["level_summary"], f"{prefix}_逐层显著性汇总")
        out.table(calibration["horizontal"], f"{prefix}_全部横向弦")
        out.table(
            calibration["horizontal"][
                calibration["horizontal"].significant_horizontal_relation
            ],
            f"{prefix}_显著横向弦",
        )
        out.table(vertical, f"{prefix}_直接父子弦")
        out.table(resolution, f"{prefix}_结构块尺度校准")
        out.table(members, f"{prefix}_候选结构块节点")
        out.table(quality_screen, f"{prefix}_发现后筛选审计")
        out.table(quality_formal_members, f"{prefix}_正式候选社群节点")
        out.table(quality_formal, f"{prefix}_正式候选社群清单")
        out.table(
            quality_paths[quality_paths.community_id.isin(quality_confirmed)],
            f"{prefix}_正式候选连续分类路径",
        )
        out.table(
            quality_chords[quality_chords.statistically_significant],
            f"{prefix}_显著弦清单",
        )
        out.table(summary, f"{prefix}_候选结构块清单")
        out.table(annual, f"{prefix}_候选结构块年度质量加权论文量")
        all_edges.append(edge_compare)
        all_matches.append(block_match)
        all_blocks.append(summary)
        results[model] = {
            "description": description,
            "horizontal": calibration["horizontal"],
            "vertical": vertical,
            "blocks": summary,
            "edge_comparison": edge_compare,
            "block_match": block_match,
        }
    edge_table = pd.concat(all_edges, ignore_index=True)
    match_table = pd.concat(all_matches, ignore_index=True)
    block_table = pd.concat(all_blocks, ignore_index=True)
    model_overview_rows: list[dict[str, Any]] = []
    for model in QUALITY_MODEL_SPECS:
        model_blocks = block_table[block_table.quality_model.eq(model)].copy()
        confirmed = model_blocks[
            model_blocks.formal_community_confirmed.fillna(False).astype(bool)
        ]
        edge_row = edge_table[edge_table.quality_model.eq(model)].iloc[0]
        match_rows = match_table[match_table.quality_model.eq(model)]
        model_overview_rows.append(
            {
                "quality_model": model,
                "pmid_count": int(len(audit)),
                "JCR_and_IF_available_pmid_count": int(
                    audit.quality_metadata_status.eq("JCR_and_IF_available").sum()
                ),
                "frozen_formal_gamma": frozen_gamma,
                "formal_significant_horizontal_edge_count": int(
                    edge_row.formal_significant_horizontal_edges
                ),
                "quality_weighted_significant_horizontal_edge_count": int(
                    edge_row.quality_weighted_significant_horizontal_edges
                ),
                "formal_edge_retention": float(edge_row.formal_edge_retention),
                "edge_jaccard_with_formal": float(edge_row.edge_jaccard),
                "discovery_block_count": int(len(model_blocks)),
                "posthoc_confirmed_candidate_community_count": int(len(confirmed)),
                "confirmed_median_node_count": float(confirmed.node_count.median())
                if len(confirmed)
                else np.nan,
                "confirmed_median_unique_pmid_count": float(
                    confirmed.unique_all_paper_pmid_count.median()
                )
                if len(confirmed)
                else np.nan,
                "confirmed_exact_match_to_formal_S_count": int(
                    (match_rows.best_match_jaccard == 1.0).sum()
                ),
                "confirmed_median_best_formal_S_jaccard": float(
                    match_rows.best_match_jaccard.median()
                )
                if len(match_rows)
                else np.nan,
            }
        )
    out.table(edge_table, "22c_探索性质量加权_正式S显著弦稳健性对照")
    out.table(match_table, "22c_探索性质量加权_正式S社群最佳匹配")
    out.table(block_table, "22c_探索性质量加权_候选结构块汇总")
    out.table(pd.DataFrame(model_overview_rows), "22c_探索性质量加权_模型总览")
    return results


def burst_path(
    counts: np.ndarray,
    totals: np.ndarray,
    s: float = 2.0,
    gamma: float = 1.0,
    max_state: int = 2,
    direction: str = "increase",
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(np.rint(counts), dtype=int)
    totals = np.asarray(totals, dtype=int)
    baseline = float(np.clip(counts.sum() / max(1, totals.sum()), 1e-06, 0.995))
    state_factors = s ** np.arange(max_state + 1)
    if direction == "increase":
        rates = np.minimum(0.995, baseline * state_factors)
    elif direction == "decrease":
        rates = np.maximum(1e-06, baseline / state_factors)
    else:
        raise ValueError(f"未知的Kleinberg方向：{direction}")
    emission = -stats.binom.logpmf(counts[:, None], totals[:, None], rates[None, :])
    cost = np.full((len(counts), max_state + 1), np.inf)
    previous = np.zeros((len(counts), max_state + 1), dtype=np.int16)
    cost[0] = emission[0]
    penalty = gamma * math.log(len(counts) + 1)
    for time in range(1, len(counts)):
        for state in range(max_state + 1):
            candidates = cost[time - 1] + penalty * np.maximum(
                0, state - np.arange(max_state + 1)
            )
            previous[time, state] = int(np.argmin(candidates))
            cost[time, state] = (
                candidates[previous[time, state]] + emission[time, state]
            )
    states = np.zeros(len(counts), dtype=np.int16)
    states[-1] = int(np.argmin(cost[-1]))
    for time in range(len(states) - 1, 0, -1):
        states[time - 1] = previous[time, states[time]]
    gains = stats.binom.logpmf(counts, totals, rates[states]) - stats.binom.logpmf(
        counts, totals, rates[0]
    )
    return (states, gains)


def parse_impact_factor(record: dict[str, Any]) -> float:
    raw = str(
        record.get("影响因子") or record.get("impact_factor") or record.get("IF") or ""
    ).strip()
    match = re.search("\\d+(?:\\.\\d+)?", raw)
    return float(match.group()) if match else np.nan


def normalise_jcr_quartile(record: dict[str, Any]) -> str:
    raw = (
        norm_name(record.get("JCR分区") or record.get("JCR") or "")
        .upper()
        .replace(" ", "")
    )
    match = re.search("Q([1-4])", raw)
    if match:
        return f"Q{match.group(1)}"
    match = re.search("([1-4])区", raw)
    if match:
        return f"Q{match.group(1)}"
    return "未分区或缺失"


def fit_annual_binomial_trend(
    years: np.ndarray, counts: np.ndarray, totals: np.ndarray
) -> dict[str, float]:
    year = np.asarray(years, dtype=float)
    count = np.asarray(counts, dtype=float)
    total = np.asarray(totals, dtype=float)
    valid = (total > 0) & np.isfinite(count) & np.isfinite(year)
    (year, count, total) = (year[valid], count[valid], total[valid])
    if (
        len(year) < 3
        or count.sum() <= 0
        or np.all(count <= 0)
        or np.all(count >= total)
    ):
        return {
            "trend_log_odds_per_year": np.nan,
            "trend_se_log_odds_per_year": np.nan,
            "trend_odds_ratio_per_year": np.nan,
            "trend_or_ci95_low": np.nan,
            "trend_or_ci95_high": np.nan,
            "trend_z": np.nan,
            "trend_p_value": np.nan,
        }
    x = year - float(year.mean())
    design = np.column_stack((np.ones(len(x)), x))
    pooled = float(np.clip(count.sum() / total.sum(), 1e-06, 1 - 1e-06))
    beta = np.asarray([math.log(pooled / (1.0 - pooled)), 0.0], dtype=float)
    for _ in range(100):
        eta = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weights = np.maximum(total * probability * (1.0 - probability), 1e-09)
        hessian = design.T @ (design * weights[:, None])
        score = design.T @ (count - total * probability)
        try:
            update = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            return {
                "trend_log_odds_per_year": np.nan,
                "trend_se_log_odds_per_year": np.nan,
                "trend_odds_ratio_per_year": np.nan,
                "trend_or_ci95_low": np.nan,
                "trend_or_ci95_high": np.nan,
                "trend_z": np.nan,
                "trend_p_value": np.nan,
            }
        beta += update
        if float(np.max(np.abs(update))) < 1e-09:
            break
    eta = np.clip(design @ beta, -30.0, 30.0)
    probability = 1.0 / (1.0 + np.exp(-eta))
    hessian = design.T @ (
        design * np.maximum(total * probability * (1.0 - probability), 1e-09)[:, None]
    )
    try:
        covariance = np.linalg.inv(hessian)
        standard_error = float(math.sqrt(max(float(covariance[1, 1]), 0.0)))
    except np.linalg.LinAlgError:
        standard_error = np.nan
    slope = float(beta[1])
    z_value = (
        slope / standard_error
        if np.isfinite(standard_error) and standard_error > 0
        else np.nan
    )
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        "trend_log_odds_per_year": slope,
        "trend_se_log_odds_per_year": standard_error,
        "trend_odds_ratio_per_year": float(math.exp(np.clip(slope, -30, 30))),
        "trend_or_ci95_low": float(
            math.exp(np.clip(slope - 1.96 * standard_error, -30, 30))
        )
        if np.isfinite(standard_error)
        else np.nan,
        "trend_or_ci95_high": float(
            math.exp(np.clip(slope + 1.96 * standard_error, -30, 30))
        )
        if np.isfinite(standard_error)
        else np.nan,
        "trend_z": z_value,
        "trend_p_value": p_value,
    }


def odds_ratio_and_fisher(
    exposed_event: int,
    exposed_total: int,
    reference_event: int,
    reference_total: int,
    *,
    alternative: str = "two-sided",
) -> dict[str, float]:
    a = max(int(exposed_event), 0)
    b = max(int(exposed_total) - a, 0)
    c = max(int(reference_event), 0)
    d = max(int(reference_total) - c, 0)
    if a + b == 0 or c + d == 0:
        return {"odds_ratio": np.nan, "p_value": np.nan}
    (_, p_value) = fisher_exact([[a, b], [c, d]], alternative=alternative)
    corrected_or = (a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))
    return {"odds_ratio": float(corrected_or), "p_value": float(p_value)}


LAST_COMPLETE_TREND_YEAR = TIME.observation_end
BURST_COMPARISON_WINDOWS = (3, 4, 5)
MIN_DECLINE_BURST_STATE = 2
BURST_WINDOW_DESCRIPTION = "、".join(
    (str(window) for window in BURST_COMPARISON_WINDOWS)
)
BURST_ACTIVITY_ANY_COLUMN = (
    "Kleinberg_burst_active_in_recent_"
    + "_or_".join((str(window) for window in BURST_COMPARISON_WINDOWS))
    + "_years"
)
DECLINE_BURST_ACTIVITY_ANY_COLUMN = (
    "Kleinberg_decline_burst_active_in_recent_"
    + "_or_".join((str(window) for window in BURST_COMPARISON_WINDOWS))
    + "_years"
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def classify_annual_share_time_trends(
    annual: pd.DataFrame,
    *,
    analysis_end_year: int = LAST_COMPLETE_TREND_YEAR,
    community_column: str = "community_id",
) -> pd.DataFrame:
    required = {community_column, "year", "unique_community_pmids", "library_pmids"}
    missing = required - set(annual.columns)
    if missing:
        raise ValueError(f"年度趋势重算缺少字段：{sorted(missing)}")
    working = annual[
        [community_column, "year", "unique_community_pmids", "library_pmids"]
    ].copy()
    working[community_column] = working[community_column].astype(str).str.strip()
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    working["unique_community_pmids"] = pd.to_numeric(
        working["unique_community_pmids"], errors="coerce"
    ).fillna(0.0)
    working["library_pmids"] = pd.to_numeric(working["library_pmids"], errors="coerce")
    working = working.loc[working.year.notna() & working.library_pmids.notna()].copy()
    working["year"] = working.year.astype(int)
    analysis = working.loc[
        working.year.between(TIME.history_start, int(analysis_end_year))
    ].copy()
    if analysis.empty:
        raise ValueError(
            f"年度趋势重算在截止年 {analysis_end_year} 前没有可用年度记录。"
        )
    years = np.sort(analysis.year.unique()).astype(int)
    required_years = 2 * max(BURST_COMPARISON_WINDOWS)
    if len(years) < required_years:
        raise ValueError(
            f"最近{BURST_WINDOW_DESCRIPTION}年窗口比较至少需要{required_years}个完整统计年。"
        )
    library_counts = analysis.groupby("year", sort=True).library_pmids.agg(
        ["first", "nunique"]
    )
    if (library_counts["nunique"] != 1).any():
        inconsistent = library_counts.index[library_counts["nunique"] != 1].tolist()
        raise ValueError(
            f"同一年 TNBC 文献总量不一致，无法计算年度份额：{inconsistent}"
        )
    totals = library_counts["first"].reindex(years).to_numpy(dtype=int)
    window_years = {
        window: (years[-window:], years[-2 * window : -window])
        for window in BURST_COMPARISON_WINDOWS
    }
    rows: list[dict[str, Any]] = []
    for community_id, part in analysis.groupby(community_column, sort=False):
        counts = (
            part.groupby("year", sort=True)
            .unique_community_pmids.sum()
            .reindex(years, fill_value=0.0)
            .to_numpy(dtype=int)
        )
        trend = fit_annual_binomial_trend(years, counts, totals)

        def to_episodes(
            state_path: np.ndarray, state_gains: np.ndarray
        ) -> list[tuple[int, int, int, float]]:
            output: list[tuple[int, int, int, float]] = []
            start: int | None = None
            for position, state in enumerate(np.r_[state_path, 0]):
                if state > 0 and start is None:
                    start = position
                elif start is not None and (state == 0 or position == len(state_path)):
                    end = position - 1
                    output.append(
                        (
                            start,
                            end,
                            int(state_path[start : end + 1].max()),
                            float(state_gains[start : end + 1].sum()),
                        )
                    )
                    start = None
            return output

        (states, gains) = burst_path(counts, totals, direction="increase")
        (decline_states, decline_gains) = burst_path(
            counts, totals, direction="decrease"
        )
        episodes = to_episodes(states, gains)
        decline_episodes = to_episodes(decline_states, decline_gains)
        best_episode = max(episodes, key=lambda item: item[3], default=None)
        recent_episodes = [
            episode
            for episode in episodes
            if episode[1] >= len(years) - max(BURST_COMPARISON_WINDOWS)
        ]
        best_recent_episode = max(
            recent_episodes, key=lambda item: item[3], default=None
        )
        best_decline_episode = max(
            decline_episodes, key=lambda item: item[3], default=None
        )
        recent_decline_episodes = [
            episode
            for episode in decline_episodes
            if episode[1] >= len(years) - max(BURST_COMPARISON_WINDOWS)
        ]
        best_recent_decline_episode = max(
            recent_decline_episodes, key=lambda item: item[3], default=None
        )
        row: dict[str, Any] = {
            community_column: str(community_id),
            "time_trend_analysis_start_year": int(years.min()),
            "time_trend_analysis_end_year": int(years.max()),
            **trend,
            "Kleinberg_burst_detected": bool(best_episode is not None),
            "Kleinberg_burst_start_year": int(years[best_episode[0]])
            if best_episode
            else np.nan,
            "Kleinberg_burst_end_year": int(years[best_episode[1]])
            if best_episode
            else np.nan,
            "Kleinberg_max_burst_state": int(best_episode[2]) if best_episode else 0,
            "Kleinberg_burst_strength": float(best_episode[3]) if best_episode else 0.0,
            "Kleinberg_recent_burst_start_year": int(years[best_recent_episode[0]])
            if best_recent_episode
            else np.nan,
            "Kleinberg_recent_burst_end_year": int(years[best_recent_episode[1]])
            if best_recent_episode
            else np.nan,
            "Kleinberg_recent_burst_strength": float(best_recent_episode[3])
            if best_recent_episode
            else 0.0,
            "Kleinberg_decline_burst_detected": bool(best_decline_episode is not None),
            "Kleinberg_decline_burst_start_year": int(years[best_decline_episode[0]])
            if best_decline_episode
            else np.nan,
            "Kleinberg_decline_burst_end_year": int(years[best_decline_episode[1]])
            if best_decline_episode
            else np.nan,
            "Kleinberg_decline_max_burst_state": int(best_decline_episode[2])
            if best_decline_episode
            else 0,
            "Kleinberg_decline_burst_strength": float(best_decline_episode[3])
            if best_decline_episode
            else 0.0,
            "Kleinberg_recent_decline_burst_start_year": int(
                years[best_recent_decline_episode[0]]
            )
            if best_recent_decline_episode
            else np.nan,
            "Kleinberg_recent_decline_burst_end_year": int(
                years[best_recent_decline_episode[1]]
            )
            if best_recent_decline_episode
            else np.nan,
            "Kleinberg_recent_decline_max_burst_state": int(
                best_recent_decline_episode[2]
            )
            if best_recent_decline_episode
            else 0,
            "Kleinberg_recent_decline_burst_strength": float(
                best_recent_decline_episode[3]
            )
            if best_recent_decline_episode
            else 0.0,
        }
        for window, (recent_years, prior_years) in window_years.items():
            recent_mask = np.isin(years, recent_years)
            prior_mask = np.isin(years, prior_years)
            recent_test = odds_ratio_and_fisher(
                int(counts[recent_mask].sum()),
                int(totals[recent_mask].sum()),
                int(counts[prior_mask].sum()),
                int(totals[prior_mask].sum()),
                alternative="greater",
            )
            recent_decline_test = odds_ratio_and_fisher(
                int(counts[recent_mask].sum()),
                int(totals[recent_mask].sum()),
                int(counts[prior_mask].sum()),
                int(totals[prior_mask].sum()),
                alternative="less",
            )
            active = any((episode[1] >= len(years) - window for episode in episodes))
            decline_window_episodes = [
                episode
                for episode in decline_episodes
                if episode[1] >= len(years) - window
            ]
            decline_active = bool(decline_window_episodes)
            decline_peak_state = max(
                (episode[2] for episode in decline_window_episodes), default=0
            )
            row.update(
                {
                    f"recent_{window}yr_window_years": " | ".join(
                        map(str, recent_years)
                    ),
                    f"prior_{window}yr_window_years": " | ".join(map(str, prior_years)),
                    f"recent_{window}yr_odds_ratio": recent_test["odds_ratio"],
                    f"recent_{window}yr_p_value": recent_test["p_value"],
                    f"Kleinberg_burst_active_in_last_{window}_years": active,
                    f"recent_{window}yr_decline_odds_ratio": recent_decline_test[
                        "odds_ratio"
                    ],
                    f"recent_{window}yr_decline_p_value": recent_decline_test[
                        "p_value"
                    ],
                    f"Kleinberg_decline_burst_active_in_last_{window}_years": decline_active,
                    f"Kleinberg_decline_burst_peak_state_in_last_{window}_years": int(
                        decline_peak_state
                    ),
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["q_BY_publication_trend"] = adjust_pvalues(summary.trend_p_value, "BY")
    for window in BURST_COMPARISON_WINDOWS:
        summary[f"q_BY_recent_{window}yr"] = adjust_pvalues(
            summary[f"recent_{window}yr_p_value"], "BY"
        )
        summary[f"q_BY_recent_{window}yr_decline"] = adjust_pvalues(
            summary[f"recent_{window}yr_decline_p_value"], "BY"
        )
        summary[f"burst_condition_pass_last_{window}_years"] = (
            summary[f"Kleinberg_burst_active_in_last_{window}_years"].astype(bool)
            & summary[f"recent_{window}yr_odds_ratio"].gt(1)
            & summary[f"q_BY_recent_{window}yr"].lt(0.05)
        )
        summary[f"decline_burst_condition_pass_last_{window}_years"] = (
            summary[f"Kleinberg_decline_burst_active_in_last_{window}_years"].astype(
                bool
            )
            & summary[f"Kleinberg_decline_burst_peak_state_in_last_{window}_years"].ge(
                MIN_DECLINE_BURST_STATE
            )
            & summary[f"recent_{window}yr_decline_odds_ratio"].lt(1)
            & summary[f"q_BY_recent_{window}yr_decline"].lt(0.05)
        )
    summary[BURST_ACTIVITY_ANY_COLUMN] = summary[
        [
            f"Kleinberg_burst_active_in_last_{window}_years"
            for window in BURST_COMPARISON_WINDOWS
        ]
    ].any(axis=1)
    summary[DECLINE_BURST_ACTIVITY_ANY_COLUMN] = summary[
        [
            f"Kleinberg_decline_burst_active_in_last_{window}_years"
            for window in BURST_COMPARISON_WINDOWS
        ]
    ].any(axis=1)
    summary["burst_condition_pass_any_recent_window"] = summary[
        [
            f"burst_condition_pass_last_{window}_years"
            for window in BURST_COMPARISON_WINDOWS
        ]
    ].any(axis=1)
    summary["decline_burst_condition_pass_any_recent_window"] = summary[
        [
            f"decline_burst_condition_pass_last_{window}_years"
            for window in BURST_COMPARISON_WINDOWS
        ]
    ].any(axis=1)
    summary["recent_burst_window_years"] = summary["recent_3yr_window_years"]
    summary["prior_burst_window_years"] = summary["prior_3yr_window_years"]
    summary["recent_burst_odds_ratio"] = summary["recent_3yr_odds_ratio"]
    summary["recent_burst_p_value"] = summary["recent_3yr_p_value"]
    summary["q_BY_recent_burst"] = summary["q_BY_recent_3yr"]
    is_burst = summary["burst_condition_pass_any_recent_window"].astype(bool)
    is_decline_burst = summary["decline_burst_condition_pass_any_recent_window"].astype(
        bool
    )
    summary["dual_direction_burst_condition"] = is_burst & is_decline_burst
    display_decline_burst = is_decline_burst & ~is_burst
    is_growth = (summary.q_BY_publication_trend < 0.05) & (
        summary.trend_log_odds_per_year > 0
    )
    is_decline = (summary.q_BY_publication_trend < 0.05) & (
        summary.trend_log_odds_per_year < 0
    )
    summary["time_trend_classification"] = np.select(
        [is_burst, display_decline_burst, is_growth, is_decline],
        ["爆发", "爆发性消退", "增长", "消退"],
        default="稳定",
    )
    summary["time_trend_rule"] = (
        f"爆发：在最近{BURST_WINDOW_DESCRIPTION}年窗口中，任一窗口同时满足高发生率Kleinberg突发片段活跃、该窗口相对紧邻前一等长窗口Fisher检验OR>1且BY-q<0.05；爆发性消退：在最近{BURST_WINDOW_DESCRIPTION}年窗口中，任一窗口同时满足低发生率Kleinberg突发片段活跃、近期Kleinberg状态≥{MIN_DECLINE_BURST_STATE}、该窗口相对紧邻前一等长窗口Fisher检验OR<1且BY-q<0.05；否则增长/消退：完整年度份额二项Logistic回归BY-q<0.05且年OR>1/<1；其余稳定。"
    )
    return summary


def _formal_community_catalog(
    out: Output,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, str]]:
    summary_path = out.tables / "16_S社群_完整清单.csv"
    member_path = out.tables / "12_PMID逐层显著性_全部节点状态.csv"
    if not summary_path.is_file() or not member_path.is_file():
        raise FileNotFoundError(
            f"正式 S 社群后续分析需要：{summary_path}；{member_path}"
        )
    summary = pd.read_csv(summary_path)
    members = pd.read_csv(member_path)
    ids = set(summary.community_id.astype(str))
    members = members[
        members.significance_block_id.fillna("").astype(str).isin(ids)
    ].copy()
    members["community_id"] = members.significance_block_id.astype(str)
    return [("S", summary, members, "significance_block_id")]


def formal_community_bibliometrics(
    data: TopicTreeDataset, out: Output
) -> dict[str, pd.DataFrame]:
    catalog = _formal_community_catalog(out)
    library_by_year = (
        pd.Series(data.years)
        .value_counts()
        .reindex(data.year_values, fill_value=0)
        .astype(int)
    )
    detail_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    year_window = min(5, max(3, len(data.year_values) // 4))
    early_years = data.year_values[-2 * year_window : -year_window]
    late_years = data.year_values[-year_window:]
    for structure, catalog_summary, members, _ in catalog:
        summary_lookup = catalog_summary.set_index("community_id", drop=False)
        nodes_by_community = (
            members.groupby("community_id", sort=False).node_id.agg(list).to_dict()
        )
        for community_id, overview in summary_lookup.iterrows():
            community_id = str(community_id)
            nodes = [
                node
                for node in nodes_by_community.get(community_id, [])
                if node in data.node_papers
            ]
            indices = (
                np.unique(np.concatenate([data.node_papers[node] for node in nodes]))
                if nodes and any((len(data.node_papers[node]) for node in nodes))
                else np.empty(0, dtype=int)
            )
            pmids = data.pmids[indices]
            years = data.years[indices]
            records = [data.papers[str(pmid)] for pmid in pmids]
            jcr = np.asarray(
                [normalise_jcr_quartile(record) for record in records], dtype=object
            )
            raw_jcr = np.asarray(
                [norm_name(record.get("JCR分区") or "未查到") for record in records],
                dtype=object,
            )
            impact = np.asarray(
                [parse_impact_factor(record) for record in records], dtype=float
            )
            label = str(overview.get("core_node_names", "")).split(" | ")[0]
            for pmid, year, record, raw_quartile, quartile, factor in zip(
                pmids, years, records, raw_jcr, jcr, impact
            ):
                detail_rows.append(
                    {
                        "structure": structure,
                        "community_id": community_id,
                        "community_label": label,
                        "pmid": str(pmid),
                        "publication_year": int(year),
                        "journal": norm_name(
                            record.get("Journal") or record.get("journal") or ""
                        ),
                        "title": norm_name(
                            record.get("Title") or record.get("title") or ""
                        ),
                        "JCR_partition_raw": raw_quartile,
                        "JCR_quartile": quartile,
                        "impact_factor": factor,
                    }
                )
            counts = np.asarray(
                [(years == year).sum() for year in data.year_values], dtype=int
            )
            q1_counts = np.asarray(
                [((years == year) & (jcr == "Q1")).sum() for year in data.year_values],
                dtype=int,
            )
            classified_counts = np.asarray(
                [
                    ((years == year) & np.isin(jcr, ["Q1", "Q2", "Q3", "Q4"])).sum()
                    for year in data.year_values
                ],
                dtype=int,
            )
            for position, year in enumerate(data.year_values):
                values = impact[years == year]
                valid_if = values[np.isfinite(values)]
                annual_rows.append(
                    {
                        "structure": structure,
                        "community_id": community_id,
                        "community_label": label,
                        "year": int(year),
                        "unique_community_pmids": int(counts[position]),
                        "library_pmids": int(library_by_year.loc[year]),
                        "community_share": float(
                            counts[position] / max(int(library_by_year.loc[year]), 1)
                        ),
                        "JCR_Q1_pmids": int(q1_counts[position]),
                        "JCR_classified_pmids": int(classified_counts[position]),
                        "JCR_Q1_share_among_classified": float(
                            q1_counts[position] / classified_counts[position]
                        )
                        if classified_counts[position]
                        else np.nan,
                        "impact_factor_valid_pmid_count": int(len(valid_if)),
                        "impact_factor_mean": float(valid_if.mean())
                        if len(valid_if)
                        else np.nan,
                        "impact_factor_median": float(np.median(valid_if))
                        if len(valid_if)
                        else np.nan,
                    }
                )
            (early_mask, late_mask) = (
                np.isin(years, early_years),
                np.isin(years, late_years),
            )
            (early_count, late_count) = (int(early_mask.sum()), int(late_mask.sum()))
            (early_total, late_total) = (
                int(library_by_year.loc[early_years].sum()),
                int(library_by_year.loc[late_years].sum()),
            )
            share_test = odds_ratio_and_fisher(
                late_count, late_total, early_count, early_total
            )
            (early_if, late_if) = (
                impact[early_mask & np.isfinite(impact)],
                impact[late_mask & np.isfinite(impact)],
            )
            if len(early_if) >= 3 and len(late_if) >= 3:
                if_test = stats.mannwhitneyu(late_if, early_if, alternative="two-sided")
                if_delta = (
                    2 * float(if_test.statistic) / (len(late_if) * len(early_if)) - 1
                )
                if_p = float(if_test.pvalue)
            else:
                (if_delta, if_p) = (np.nan, np.nan)
            (early_q1, late_q1) = (
                int((jcr[early_mask] == "Q1").sum()),
                int((jcr[late_mask] == "Q1").sum()),
            )
            (early_classified, late_classified) = (
                int(np.isin(jcr[early_mask], ["Q1", "Q2", "Q3", "Q4"]).sum()),
                int(np.isin(jcr[late_mask], ["Q1", "Q2", "Q3", "Q4"]).sum()),
            )
            q1_test = (
                odds_ratio_and_fisher(
                    late_q1, late_classified, early_q1, early_classified
                )
                if early_classified and late_classified
                else {"odds_ratio": np.nan, "p_value": np.nan}
            )
            summary_rows.append(
                {
                    "structure": structure,
                    "community_id": community_id,
                    "community_label": label,
                    "node_count": int(overview.get("node_count", len(nodes))),
                    "all_node_unique_all_paper_pmid_count_reported": int(
                        overview.get("all_node_unique_all_paper_pmid_count", len(pmids))
                    ),
                    "all_node_unique_all_paper_pmid_count_recomputed": int(len(pmids)),
                    "pmid_count_matches_community_list": bool(
                        int(
                            overview.get(
                                "all_node_unique_all_paper_pmid_count", len(pmids)
                            )
                        )
                        == len(pmids)
                    ),
                    "first_publication_year": int(years.min())
                    if len(years)
                    else np.nan,
                    "last_publication_year": int(years.max()) if len(years) else np.nan,
                    "JCR_Q1_pmid_count": int((jcr == "Q1").sum()),
                    "JCR_Q2_pmid_count": int((jcr == "Q2").sum()),
                    "JCR_Q3_pmid_count": int((jcr == "Q3").sum()),
                    "JCR_Q4_pmid_count": int((jcr == "Q4").sum()),
                    "JCR_unclassified_or_missing_pmid_count": int(
                        (jcr == "未分区或缺失").sum()
                    ),
                    "JCR_classified_pmid_count": int(
                        np.isin(jcr, ["Q1", "Q2", "Q3", "Q4"]).sum()
                    ),
                    "JCR_Q1_share_among_classified": float(
                        (jcr == "Q1").sum()
                        / max(np.isin(jcr, ["Q1", "Q2", "Q3", "Q4"]).sum(), 1)
                    ),
                    "impact_factor_valid_pmid_count": int(np.isfinite(impact).sum()),
                    "impact_factor_mean": float(np.nanmean(impact))
                    if np.isfinite(impact).any()
                    else np.nan,
                    "impact_factor_median": float(np.nanmedian(impact))
                    if np.isfinite(impact).any()
                    else np.nan,
                    "impact_factor_early_window_mean": float(early_if.mean())
                    if len(early_if)
                    else np.nan,
                    "impact_factor_late_window_mean": float(late_if.mean())
                    if len(late_if)
                    else np.nan,
                    "impact_factor_late_minus_early": float(
                        late_if.mean() - early_if.mean()
                    )
                    if len(early_if) and len(late_if)
                    else np.nan,
                    "impact_factor_cliffs_delta": if_delta,
                    "impact_factor_p_value": if_p,
                    "JCR_Q1_late_vs_early_odds_ratio": q1_test["odds_ratio"],
                    "JCR_Q1_late_vs_early_p_value": q1_test["p_value"],
                    "early_window_years": " | ".join(map(str, early_years)),
                    "late_window_years": " | ".join(map(str, late_years)),
                    "early_window_unique_pmids": early_count,
                    "late_window_unique_pmids": late_count,
                    "early_window_share": early_count / max(early_total, 1),
                    "late_window_share": late_count / max(late_total, 1),
                    "late_minus_early_share": late_count / max(late_total, 1)
                    - early_count / max(early_total, 1),
                    "late_vs_early_odds_ratio": share_test["odds_ratio"],
                    "late_vs_early_p_value": share_test["p_value"],
                }
            )
    detail = pd.DataFrame(detail_rows)
    annual = pd.DataFrame(annual_rows)
    summary = pd.DataFrame(summary_rows)
    complete_year_trends = classify_annual_share_time_trends(annual)
    summary = summary.merge(
        complete_year_trends, on="community_id", how="left", validate="one_to_one"
    )
    for structure, positions in summary.groupby("structure", sort=False).groups.items():
        index = list(positions)
        summary.loc[index, "q_BY_late_vs_early_share"] = adjust_pvalues(
            summary.loc[index, "late_vs_early_p_value"], "BY"
        )
        summary.loc[index, "q_BY_impact_factor_change"] = adjust_pvalues(
            summary.loc[index, "impact_factor_p_value"], "BY"
        )
        summary.loc[index, "q_BY_JCR_Q1_change"] = adjust_pvalues(
            summary.loc[index, "JCR_Q1_late_vs_early_p_value"], "BY"
        )
    out.table(detail, "19_S正式社群_全部去重PMID及文献计量")
    out.table(annual, "19_S正式社群_年度论文与文献计量")
    out.table(
        summary.sort_values(
            ["structure", "time_trend_classification", "trend_odds_ratio_per_year"],
            ascending=[True, True, False],
        ),
        "19_S正式社群_时间趋势与文献计量汇总",
    )
    return {"papers": detail, "annual": annual, "summary": summary}


NESTED_COVERAGE_MIN = 0.5
NESTED_STRONG_COVERAGE_MIN = 0.8


def _tree_euler_intervals(
    data: TopicTreeDataset,
) -> tuple[dict[str, int], dict[str, int]]:
    entry: dict[str, int] = {}
    exit_: dict[str, int] = {}
    counter = 0

    def visit(node: str) -> None:
        nonlocal counter
        entry[node] = counter
        counter += 1
        for child in data.tree[node]["children_ids"]:
            visit(child)
        exit_[node] = counter

    for root in data.roots:
        visit(root)
    return (entry, exit_)


def _compact_subtree_intervals(
    nodes: Iterable[str], entry: dict[str, int], exit_: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    intervals: list[tuple[int, int]] = []
    last_exit = -1
    for node in sorted(
        set((str(node) for node in nodes if str(node) in entry)), key=entry.__getitem__
    ):
        (start, end) = (entry[node], exit_[node])
        if start >= last_exit:
            intervals.append((start, end))
            last_exit = end
    if not intervals:
        return (np.empty(0, dtype=int), np.empty(0, dtype=int))
    return (
        np.asarray([pair[0] for pair in intervals]),
        np.asarray([pair[1] for pair in intervals]),
    )


def _covered_by_subtree_intervals(
    node: str, starts: np.ndarray, ends: np.ndarray, entry: dict[str, int]
) -> bool:
    if node not in entry or not len(starts):
        return False
    position = int(np.searchsorted(starts, entry[node], side="right") - 1)
    return position >= 0 and entry[node] < ends[position]


def formal_community_nesting(
    data: TopicTreeDataset, out: Output
) -> dict[str, dict[str, pd.DataFrame]]:
    (entry, exit_) = _tree_euler_intervals(data)
    catalog = _formal_community_catalog(out)
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for structure, summary, members, _ in catalog:
        overview = summary.set_index("community_id", drop=False)
        groups = {
            str(key): part.node_id.astype(str).tolist()
            for (key, part) in members.groupby("community_id", sort=False)
        }
        intervals = {
            key: _compact_subtree_intervals(nodes, entry, exit_)
            for (key, nodes) in groups.items()
        }
        interval_extent = {
            key: int((ends - starts).sum())
            for (key, (starts, ends)) in intervals.items()
        }
        pmid_sets = {
            key: set(
                data.pmids[
                    np.unique(
                        np.concatenate([data.node_papers[node] for node in nodes])
                    )
                ].astype(str)
            )
            if nodes and any((len(data.node_papers[node]) for node in nodes))
            else set()
            for (key, nodes) in groups.items()
        }
        candidate_rows: list[dict[str, Any]] = []
        keys = sorted(groups)
        for parent in keys:
            parent_nodes = groups[parent]
            parent_node_count = len(parent_nodes)
            (starts, ends) = intervals[parent]
            for child in keys:
                if parent == child or parent_node_count <= len(groups[child]):
                    continue
                covered = sum(
                    (
                        _covered_by_subtree_intervals(node, starts, ends, entry)
                        for node in groups[child]
                    )
                )
                coverage = covered / max(len(groups[child]), 1)
                if coverage + 1e-12 < NESTED_COVERAGE_MIN:
                    continue
                candidate_rows.append(
                    {
                        "structure": structure,
                        "parent_community_id": parent,
                        "child_community_id": child,
                        "parent_node_count": parent_node_count,
                        "child_node_count": len(groups[child]),
                        "child_nodes_covered_by_parent_subtrees": int(covered),
                        "child_node_coverage_fraction": float(coverage),
                        "nesting_relation": "完全覆盖"
                        if coverage >= 1 - 1e-12
                        else "强部分覆盖"
                        if coverage >= NESTED_STRONG_COVERAGE_MIN
                        else "部分覆盖",
                        "parent_subtree_closure_topic_node_count": interval_extent[
                            parent
                        ],
                        "parent_all_node_unique_all_paper_pmid_count": int(
                            overview.loc[parent, "all_node_unique_all_paper_pmid_count"]
                        ),
                        "child_all_node_unique_all_paper_pmid_count": int(
                            overview.loc[child, "all_node_unique_all_paper_pmid_count"]
                        ),
                        "parent_child_shared_unique_pmid_count": int(
                            len(pmid_sets[parent] & pmid_sets[child])
                        ),
                    }
                )
        candidates = pd.DataFrame(
            candidate_rows,
            columns=[
                "structure",
                "parent_community_id",
                "child_community_id",
                "parent_node_count",
                "child_node_count",
                "child_nodes_covered_by_parent_subtrees",
                "child_node_coverage_fraction",
                "nesting_relation",
                "parent_subtree_closure_topic_node_count",
                "parent_all_node_unique_all_paper_pmid_count",
                "child_all_node_unique_all_paper_pmid_count",
                "parent_child_shared_unique_pmid_count",
            ],
        )
        direct_rows: list[dict[str, Any]] = []
        if len(candidates):
            for child, part in candidates.groupby("child_community_id", sort=False):
                chosen = (
                    part.sort_values(
                        [
                            "child_node_coverage_fraction",
                            "parent_subtree_closure_topic_node_count",
                            "parent_node_count",
                            "parent_community_id",
                        ],
                        ascending=[False, True, True, True],
                        kind="stable",
                    )
                    .iloc[0]
                    .to_dict()
                )
                chosen["is_direct_nesting_edge"] = True
                chosen["direct_parent_selection_rule"] = (
                    "最高覆盖率；并列时选主题树子树闭包最小的容器，再选节点数较少者"
                )
                direct_rows.append(chosen)
        direct = pd.DataFrame(
            direct_rows,
            columns=list(candidates.columns)
            + [
                "is_direct_nesting_edge",
                "direct_parent_selection_rule",
                "parent_nesting_depth",
                "child_nesting_depth",
                "flow_pmid_count",
            ],
        )
        direct_parent = (
            dict(
                zip(
                    direct.child_community_id.astype(str),
                    direct.parent_community_id.astype(str),
                )
            )
            if len(direct)
            else {}
        )
        depth: dict[str, int] = {}

        def calculate_depth(community: str) -> int:
            if community in depth:
                return depth[community]
            depth[community] = (
                0
                if community not in direct_parent
                else calculate_depth(direct_parent[community]) + 1
            )
            return depth[community]

        for key in keys:
            calculate_depth(key)
        node_rows: list[dict[str, Any]] = []
        for key in keys:
            selected_nodes = groups[key]
            levels = [int(data.tree[node]["level"]) for node in selected_nodes]
            anchor = min(
                (entry[node] for node in selected_nodes if node in entry),
                default=np.inf,
            )
            node_rows.append(
                {
                    "structure": structure,
                    "community_id": key,
                    "nesting_depth": int(depth[key]),
                    "direct_parent_community_id": direct_parent.get(key, ""),
                    "node_count": int(overview.loc[key, "node_count"]),
                    "all_node_unique_all_paper_pmid_count": int(
                        overview.loc[key, "all_node_unique_all_paper_pmid_count"]
                    ),
                    "level_range": f"{min(levels)}-{max(levels)}" if levels else "",
                    "core_node_names": str(overview.loc[key, "core_node_names"]),
                    "tree_anchor": float(anchor),
                    "parent_subtree_closure_topic_node_count": interval_extent[key],
                }
            )
        nodes = pd.DataFrame(node_rows)
        if len(direct):
            direct["parent_nesting_depth"] = direct.parent_community_id.map(depth)
            direct["child_nesting_depth"] = direct.child_community_id.map(depth)
            direct["flow_pmid_count"] = (
                direct.child_all_node_unique_all_paper_pmid_count
            )
            direct = direct.sort_values(
                [
                    "parent_nesting_depth",
                    "parent_community_id",
                    "child_nesting_depth",
                    "child_community_id",
                ]
            )
        candidates = (
            candidates.sort_values(
                [
                    "child_node_coverage_fraction",
                    "parent_subtree_closure_topic_node_count",
                ],
                ascending=[False, True],
            )
            if len(candidates)
            else candidates
        )
        out.table(candidates, f"20_{structure}正式社群_子树覆盖嵌套候选关系")
        out.table(direct, f"20_{structure}正式社群_直接多层嵌套关系")
        out.table(
            nodes.sort_values(["nesting_depth", "tree_anchor", "community_id"]),
            f"20_{structure}正式社群_嵌套层级与桑基节点",
        )
        outputs[structure] = {
            "candidates": candidates,
            "direct": direct,
            "nodes": nodes,
        }
    return outputs


def _fit_grouped_binomial_glm(
    design: np.ndarray, counts: np.ndarray, totals: np.ndarray
) -> dict[str, Any] | None:
    X = np.asarray(design, dtype=float)
    y = np.asarray(counts, dtype=float)
    n = np.asarray(totals, dtype=float)
    valid = (n > 0) & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    (X, y, n) = (X[valid], y[valid], n[valid])
    if len(y) <= X.shape[1] or y.sum() <= 0 or np.all(y >= n):
        return None
    pooled = float(np.clip(y.sum() / n.sum(), 1e-06, 1 - 1e-06))
    beta = np.zeros(X.shape[1], dtype=float)
    beta[0] = math.log(pooled / (1 - pooled))
    for _ in range(150):
        eta = np.clip(X @ beta, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(n * probability * (1.0 - probability), 1e-09)
        hessian = X.T @ (X * weight[:, None])
        score = X.T @ (y - n * probability)
        try:
            update = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            return None
        beta += update
        if np.max(np.abs(update)) < 1e-09:
            break
    eta = np.clip(X @ beta, -30, 30)
    probability = 1.0 / (1.0 + np.exp(-eta))
    weight = np.maximum(n * probability * (1.0 - probability), 1e-09)
    hessian = X.T @ (X * weight[:, None])
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.full((X.shape[1], X.shape[1]), np.nan)
    log_likelihood = float(
        np.sum(stats.binom.logpmf(y.astype(int), n.astype(int), probability))
    )
    return {"beta": beta, "covariance": covariance, "log_likelihood": log_likelihood}


def annual_share_changepoint(
    years: np.ndarray, counts: np.ndarray, totals: np.ndarray
) -> dict[str, Any]:
    year = np.asarray(years, dtype=float)
    count = np.asarray(counts, dtype=int)
    total = np.asarray(totals, dtype=int)
    valid = total > 0
    (year, count, total) = (year[valid], count[valid], total[valid])
    if len(year) < 8 or count.sum() <= 0:
        return {
            "changepoint_year": np.nan,
            "pre_changepoint_log_odds_per_year": np.nan,
            "post_changepoint_log_odds_per_year": np.nan,
            "slope_change_log_odds_per_year": np.nan,
            "slope_change_odds_ratio": np.nan,
            "changepoint_LR": np.nan,
            "changepoint_raw_p_value": np.nan,
            "changepoint_within_community_p_value": np.nan,
            "changepoint_candidates_tested": 0,
        }
    centered = year - year.mean()
    base = _fit_grouped_binomial_glm(
        np.column_stack((np.ones(len(year)), centered)), count, total
    )
    if base is None:
        return {
            "changepoint_year": np.nan,
            "pre_changepoint_log_odds_per_year": np.nan,
            "post_changepoint_log_odds_per_year": np.nan,
            "slope_change_log_odds_per_year": np.nan,
            "slope_change_odds_ratio": np.nan,
            "changepoint_LR": np.nan,
            "changepoint_raw_p_value": np.nan,
            "changepoint_within_community_p_value": np.nan,
            "changepoint_candidates_tested": 0,
        }
    fits: list[tuple[float, float, dict[str, Any]]] = []
    for position in range(3, len(year) - 3):
        breakpoint = year[position]
        hinge = np.maximum(0.0, year - breakpoint)
        fit = _fit_grouped_binomial_glm(
            np.column_stack((np.ones(len(year)), centered, hinge)), count, total
        )
        if fit is not None:
            lr = max(0.0, 2.0 * (fit["log_likelihood"] - base["log_likelihood"]))
            fits.append((lr, breakpoint, fit))
    if not fits:
        return {
            "changepoint_year": np.nan,
            "pre_changepoint_log_odds_per_year": np.nan,
            "post_changepoint_log_odds_per_year": np.nan,
            "slope_change_log_odds_per_year": np.nan,
            "slope_change_odds_ratio": np.nan,
            "changepoint_LR": np.nan,
            "changepoint_raw_p_value": np.nan,
            "changepoint_within_community_p_value": np.nan,
            "changepoint_candidates_tested": 0,
        }
    (lr, breakpoint, fit) = max(fits, key=lambda item: item[0])
    raw_p = float(stats.chi2.sf(lr, 1))
    adjusted_p = min(1.0, raw_p * len(fits))
    (pre_slope, slope_change) = (float(fit["beta"][1]), float(fit["beta"][2]))
    return {
        "changepoint_year": int(breakpoint),
        "pre_changepoint_log_odds_per_year": pre_slope,
        "post_changepoint_log_odds_per_year": pre_slope + slope_change,
        "slope_change_log_odds_per_year": slope_change,
        "slope_change_odds_ratio": float(math.exp(np.clip(slope_change, -30, 30))),
        "changepoint_LR": float(lr),
        "changepoint_raw_p_value": raw_p,
        "changepoint_within_community_p_value": adjusted_p,
        "changepoint_candidates_tested": len(fits),
    }


def _percentile_score(values: pd.Series) -> pd.Series:
    result = values.rank(pct=True, method="average")
    return result.fillna(0.0).astype(float)


def formal_community_discussion_screen(
    data: TopicTreeDataset, out: Output
) -> dict[str, pd.DataFrame]:
    annual_path = out.tables / "19_S正式社群_年度论文与文献计量.csv"
    trend_path = out.tables / "19_S正式社群_时间趋势与文献计量汇总.csv"
    block_path = out.tables / "12_PMID逐层显著性_连续跨层结构块.csv"
    required = [annual_path, trend_path, block_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("讨论社群筛选需要：" + "；".join(missing))
    annual = pd.read_csv(annual_path)
    trends = pd.read_csv(trend_path)
    blocks = pd.read_csv(block_path).rename(
        columns={"significance_block_id": "community_id"}
    )
    direct_evidence = blocks[
        [
            "community_id",
            "significant_horizontal_relations",
            "significant_direct_parent_child_relations",
            "total_evidence_z",
        ]
    ].copy()
    changepoint_rows: list[dict[str, Any]] = []
    for community, part in annual.groupby("community_id", sort=False):
        part = part.sort_values("year")
        estimate = annual_share_changepoint(
            part.year.to_numpy(),
            part.unique_community_pmids.to_numpy(),
            part.library_pmids.to_numpy(),
        )
        changepoint_rows.append(
            {"structure": "S", "community_id": str(community), **estimate}
        )
    changepoints = pd.DataFrame(changepoint_rows)
    changepoints["q_BY_changepoint"] = adjust_pvalues(
        changepoints["changepoint_within_community_p_value"], "BY"
    )
    nodes = pd.read_csv(out.tables / "20_S正式社群_嵌套层级与桑基节点.csv")
    direct = pd.read_csv(out.tables / "20_S正式社群_直接多层嵌套关系.csv")
    child_counts = (
        direct.groupby("parent_community_id")
        .child_community_id.nunique()
        .rename("direct_child_community_count")
        .reset_index()
    )
    child_papers = (
        direct.groupby("parent_community_id")
        .flow_pmid_count.sum()
        .rename("direct_child_pmid_sum")
        .reset_index()
    )
    screen = trends.merge(direct_evidence, on="community_id", how="left")
    screen = screen.merge(changepoints, on=["structure", "community_id"], how="left")
    screen = screen.merge(
        nodes[
            [
                "community_id",
                "nesting_depth",
                "direct_parent_community_id",
                "parent_subtree_closure_topic_node_count",
                "core_node_names",
                "level_range",
            ]
        ],
        on="community_id",
        how="left",
    )
    screen = screen.merge(
        child_counts, left_on="community_id", right_on="parent_community_id", how="left"
    ).drop(columns=["parent_community_id"])
    screen = screen.merge(
        child_papers, left_on="community_id", right_on="parent_community_id", how="left"
    ).drop(columns=["parent_community_id"])
    screen["direct_child_community_count"] = screen.direct_child_community_count.fillna(
        0
    ).astype(int)
    screen["direct_child_pmid_sum"] = screen.direct_child_pmid_sum.fillna(0).astype(int)
    screen[
        [
            "significant_horizontal_relations",
            "significant_direct_parent_child_relations",
            "total_evidence_z",
        ]
    ] = screen[
        [
            "significant_horizontal_relations",
            "significant_direct_parent_child_relations",
            "total_evidence_z",
        ]
    ].fillna(0)
    screen["burst_condition_pass_any_recent_window"] = screen[
        "burst_condition_pass_any_recent_window"
    ].map(_as_bool)
    screen["decline_burst_condition_pass_any_recent_window"] = screen[
        "decline_burst_condition_pass_any_recent_window"
    ].map(_as_bool)
    significant_time = (
        (screen.q_BY_publication_trend < 0.05)
        | screen.burst_condition_pass_any_recent_window
        | screen.decline_burst_condition_pass_any_recent_window
        | (screen.q_BY_changepoint < 0.05)
    )
    stable_structural_core = (
        (screen.time_trend_classification == "稳定")
        & (screen.all_node_unique_all_paper_pmid_count_recomputed >= 1000)
        & (screen.direct_child_community_count >= 2)
    )
    standard_coverage = screen.all_node_unique_all_paper_pmid_count_recomputed >= 100
    screen["basic_discussion_eligible"] = standard_coverage & (
        significant_time | stable_structural_core
    )
    screen["basic_eligibility_reason"] = np.select(
        [stable_structural_core, standard_coverage & significant_time],
        [
            "大规模稳定结构核心：≥1000 PMID且至少2个直接子社群",
            "≥100 PMID且具显著趋势、爆发或拐点证据",
        ],
        default="未达到讨论门槛",
    )
    structural_base = 0.65 * _percentile_score(
        np.log1p(screen.direct_child_community_count)
    ) + 0.35 * _percentile_score(np.log1p(screen.direct_child_pmid_sum))
    root_penalty = np.where(
        (screen.nesting_depth == 0) & (screen.direct_child_community_count >= 10),
        0.6,
        1.0,
    )
    screen["structure_component_0_to_25"] = 25 * structural_base * root_penalty
    time_type_weight = np.select(
        [
            screen.time_trend_classification.isin(["爆发", "爆发性消退"]),
            screen.q_BY_changepoint.lt(0.05),
            screen.time_trend_classification.isin(["增长", "消退"]),
        ],
        [1.0, 0.85, 0.7],
        default=0.35,
    )
    time_effect = np.maximum(
        np.abs(screen.trend_log_odds_per_year.fillna(0)),
        np.abs(screen.slope_change_log_odds_per_year.fillna(0)),
    )
    screen["time_component_0_to_25"] = 25 * (
        0.7 * time_type_weight + 0.3 * _percentile_score(time_effect)
    )
    discovery_evidence = (
        0.55 * _percentile_score(screen.total_evidence_z)
        + 0.3 * _percentile_score(np.log1p(screen.significant_horizontal_relations))
        + 0.15
        * _percentile_score(np.log1p(screen.significant_direct_parent_child_relations))
    )
    screen["discovery_evidence_component_0_to_20"] = 20 * discovery_evidence
    screen["coverage_component_0_to_15"] = 15 * _percentile_score(
        np.log1p(screen.all_node_unique_all_paper_pmid_count_recomputed)
    )
    screen["quantitative_priority_0_to_85"] = screen[
        [
            "structure_component_0_to_25",
            "time_component_0_to_25",
            "discovery_evidence_component_0_to_20",
            "coverage_component_0_to_15",
        ]
    ].sum(axis=1)
    screen["clinical_semantic_score_manual_0_to_3"] = np.nan
    screen["clinical_semantic_component_manual_0_to_15"] = np.nan
    screen["final_priority_after_clinical_review_0_to_100"] = np.nan
    screen["suggested_discussion_role"] = np.select(
        [
            screen.time_trend_classification.eq("爆发"),
            screen.time_trend_classification.eq("爆发性消退"),
            screen.q_BY_changepoint.lt(0.05),
            screen.time_trend_classification.eq("增长"),
            screen.time_trend_classification.eq("消退"),
            screen.direct_child_community_count.ge(2),
        ],
        [
            "近期正向爆发前沿",
            "近期爆发性消退",
            "研究范式拐点",
            "持续增长方向",
            "消退/转折方向",
            "结构牵头社群",
        ],
        default="稳定背景主题",
    )
    parent_map = {
        str(row.community_id): str(row.direct_parent_community_id)
        for row in screen.itertuples()
        if str(row.direct_parent_community_id) not in ("", "nan")
    }

    def lineage_root(community: str) -> str:
        seen: set[str] = set()
        current = community
        while current in parent_map and current not in seen:
            seen.add(current)
            current = parent_map[current]
        return current

    screen["lineage_root_community_id"] = [
        lineage_root(row.community_id) for row in screen.itertuples()
    ]
    screen["quantitative_rank"] = screen.quantitative_priority_0_to_85.rank(
        method="first", ascending=False
    ).astype(int)
    eligible = (
        screen[screen.basic_discussion_eligible]
        .copy()
        .sort_values("quantitative_priority_0_to_85", ascending=False)
    )
    eligible["quantitative_rank_among_eligible"] = np.arange(1, len(eligible) + 1)
    eligible["rank_within_lineage"] = (
        eligible.groupby("lineage_root_community_id")
        .quantitative_priority_0_to_85.rank(method="first", ascending=False)
        .astype(int)
    )
    screen = screen.merge(
        eligible[
            ["community_id", "quantitative_rank_among_eligible", "rank_within_lineage"]
        ],
        on="community_id",
        how="left",
    )
    slope = screen.set_index("community_id")[
        [
            "trend_log_odds_per_year",
            "trend_odds_ratio_per_year",
            "time_trend_classification",
            "changepoint_year",
            "q_BY_changepoint",
        ]
    ]
    divergence_rows: list[dict[str, Any]] = []
    for row in direct.itertuples():
        if (
            row.parent_community_id not in slope.index
            or row.child_community_id not in slope.index
        ):
            continue
        (parent, child) = (
            slope.loc[row.parent_community_id],
            slope.loc[row.child_community_id],
        )
        delta = float(child.trend_log_odds_per_year - parent.trend_log_odds_per_year)
        divergence_rows.append(
            {
                "structure": "S",
                "parent_community_id": row.parent_community_id,
                "child_community_id": row.child_community_id,
                "parent_trend_log_odds_per_year": parent.trend_log_odds_per_year,
                "child_trend_log_odds_per_year": child.trend_log_odds_per_year,
                "child_minus_parent_log_odds_per_year": delta,
                "child_vs_parent_growth_OR_ratio": float(
                    math.exp(np.clip(delta, -30, 30))
                ),
                "parent_time_classification": parent.time_trend_classification,
                "child_time_classification": child.time_trend_classification,
                "parent_changepoint_year": parent.changepoint_year,
                "child_changepoint_year": child.changepoint_year,
                "trend_divergence_interpretation": "子社群增长更快"
                if delta > 0.03
                else "子社群消退更快"
                if delta < -0.03
                else "父子年度趋势近似",
            }
        )
    divergence = pd.DataFrame(divergence_rows)
    review_columns = [
        "structure",
        "community_id",
        "community_label",
        "suggested_discussion_role",
        "quantitative_rank_among_eligible",
        "quantitative_priority_0_to_85",
        "lineage_root_community_id",
        "rank_within_lineage",
        "direct_parent_community_id",
        "all_node_unique_all_paper_pmid_count_recomputed",
        "core_node_names",
        "time_trend_classification",
        "trend_odds_ratio_per_year",
        "trend_or_ci95_low",
        "trend_or_ci95_high",
        "q_BY_publication_trend",
        "Kleinberg_burst_strength",
        "q_BY_recent_burst",
        "changepoint_year",
        "slope_change_odds_ratio",
        "q_BY_changepoint",
        "total_evidence_z",
        "significant_horizontal_relations",
        "significant_direct_parent_child_relations",
        "direct_child_community_count",
        "direct_child_pmid_sum",
        "clinical_semantic_score_manual_0_to_3",
        "clinical_semantic_component_manual_0_to_15",
        "manual_clinical_rationale",
        "discussion_decision",
    ]
    for column in ["manual_clinical_rationale", "discussion_decision"]:
        screen[column] = ""
        eligible[column] = ""
    review = eligible.head(30).reindex(columns=review_columns)
    out.table(changepoints, "21_S正式社群_年度份额拐点检验")
    out.table(
        screen.sort_values("quantitative_priority_0_to_85", ascending=False),
        "21_S正式社群_讨论资格与量化优先级",
    )
    out.table(review, "21_S社群_讨论候选人工语义复核模板")
    out.table(divergence, "21_S直接父子社群_年度趋势差异")
    return {
        "changepoints": changepoints,
        "screen": screen,
        "eligible": eligible,
        "divergence": divergence,
        "review": review,
    }


def _profile_weighted_structure_time_trends(
    data: TopicTreeDataset, catalogue: pd.DataFrame, members: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_catalogue = {
        "community_id",
        "node_count",
        "all_node_unique_all_paper_pmid_count",
    }
    required_members = {
        "node_id",
        "significance_block_id",
        "formal_community_confirmed",
    }
    missing = required_catalogue - set(catalogue.columns) | required_members - set(
        members.columns
    )
    if missing:
        raise ValueError(f"图26的JCR结构时间趋势输入缺少字段：{sorted(missing)}")
    catalogue = catalogue.copy()
    catalogue["community_id"] = catalogue.community_id.astype(str).str.strip()
    if catalogue.community_id.duplicated().any():
        raise ValueError("图26要求JCR正式结构清单以 community_id 一对一记录。")
    members = members.copy()
    members["node_id"] = members.node_id.astype(str)
    members["community_id"] = (
        members.significance_block_id.fillna("").astype(str).str.strip()
    )
    confirmed = (
        members.formal_community_confirmed.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    valid_ids = set(catalogue.community_id)
    members = members.loc[confirmed & members.community_id.isin(valid_ids)].copy()
    nodes_by_community = (
        members.groupby("community_id", sort=False).node_id.agg(list).to_dict()
    )
    library_by_year = (
        pd.Series(data.years)
        .value_counts()
        .reindex(data.year_values, fill_value=0)
        .astype(int)
    )
    annual_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for overview in catalogue.itertuples(index=False):
        community_id = str(overview.community_id)
        nodes = [
            node
            for node in nodes_by_community.get(community_id, [])
            if node in data.node_papers
        ]
        indices = (
            np.unique(np.concatenate([data.node_papers[node] for node in nodes]))
            if nodes and any((len(data.node_papers[node]) for node in nodes))
            else np.empty(0, dtype=int)
        )
        years = data.years[indices]
        counts = np.asarray(
            [(years == year).sum() for year in data.year_values], dtype=int
        )
        for position, year in enumerate(data.year_values):
            annual_rows.append(
                {
                    "community_id": community_id,
                    "year": int(year),
                    "unique_community_pmids": int(counts[position]),
                    "library_pmids": int(library_by_year.loc[year]),
                    "community_share": float(
                        counts[position] / max(int(library_by_year.loc[year]), 1)
                    ),
                }
            )
        summary_rows.append(
            {
                "community_id": community_id,
                "node_count": int(getattr(overview, "node_count", len(nodes))),
                "all_node_unique_all_paper_pmid_count_reported": int(
                    getattr(
                        overview, "all_node_unique_all_paper_pmid_count", len(indices)
                    )
                ),
                "all_node_unique_all_paper_pmid_count_recomputed": int(len(indices)),
            }
        )
    annual = pd.DataFrame(annual_rows)
    summary = pd.DataFrame(summary_rows)
    complete_year_trends = classify_annual_share_time_trends(annual)
    recalculated_columns = [
        column for column in complete_year_trends.columns if column != "community_id"
    ]
    summary = summary.drop(
        columns=[
            column for column in recalculated_columns if column in summary.columns
        ],
        errors="ignore",
    )
    summary = summary.merge(
        complete_year_trends, on="community_id", how="left", validate="one_to_one"
    )
    return (annual, summary)
