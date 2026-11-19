from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Hashable, Mapping

(MIN_NODES, MAX_NODES) = (3, 15)
(ALL_PAPER_TARGET, EVIDENCE_TARGET) = (0.75, 0.9)


@dataclass
class Selection:
    nodes: list[str]
    owners: dict[str, Hashable]
    reasons: dict[str, str]
    coverage: dict[Hashable, dict[str, Any]]
    stop_reason: str
    available_nodes: int


def select_coverage_nodes(
    orders: Mapping[Hashable, list[str]],
    papers: Mapping[Hashable, Mapping[str, set]],
    evidence: Mapping[Hashable, set],
    *,
    minimum: int = MIN_NODES,
    maximum: int = MAX_NODES,
    all_target: float = ALL_PAPER_TARGET,
    evidence_target: float = EVIDENCE_TARGET,
) -> Selection:
    if not 1 <= minimum <= maximum:
        raise ValueError("Invalid core-network node bounds.")
    sources = list(orders)
    rank = {
        source: {node: i for (i, node) in enumerate(orders[source])}
        for source in sources
    }
    available = (
        set().union(*(set(values) for values in orders.values())) if sources else set()
    )
    effective_minimum = min(minimum, len(available))
    denominators = {
        source: set().union(
            *(set(papers[source].get(node, set())) for node in orders[source])
        )
        if orders[source]
        else set()
        for source in sources
    }
    covered = {source: set() for source in sources}
    chosen: list[str] = []
    owners: dict[str, Hashable] = {}
    reasons: dict[str, str] = {}

    def fractions(source):
        return (
            len(covered[source]) / max(len(denominators[source]), 1),
            len(covered[source] & evidence[source]) / max(len(evidence[source]), 1),
        )

    def complete():
        return all(
            (
                fractions(source)[0] >= all_target
                and fractions(source)[1] >= evidence_target
                for source in sources
            )
        )

    def add(node, owner, reason):
        chosen.append(node)
        (owners[node], reasons[node]) = (owner, reason)
        for source in sources:
            if node in rank[source]:
                covered[source].update(papers[source].get(node, set()))

    def rebuild():
        for source in sources:
            covered[source] = (
                set().union(
                    *(
                        set(papers[source].get(node, set()))
                        for node in chosen
                        if node in rank[source]
                    )
                )
                if chosen
                else set()
            )

    for source in sources:
        if len(chosen) >= maximum:
            break
        node = next((node for node in orders[source] if node not in owners), None)
        if node is not None:
            add(node, source, "minimum_source_representative")
    while len(chosen) < maximum and (not complete() or len(chosen) < effective_minimum):
        candidates = []
        for node in sorted(available - set(chosen)):
            gains = []
            for index, source in enumerate(sources):
                if node not in rank[source]:
                    continue
                (all_cover, evidence_cover) = fractions(source)
                values = papers[source].get(node, set())
                gain_all = len(values - covered[source]) / max(
                    len(denominators[source]), 1
                )
                gain_evidence = len(
                    (values & evidence[source]) - covered[source]
                ) / max(len(evidence[source]), 1)
                gain = (
                    min(gain_all, max(all_target - all_cover, 0.0)) / all_target
                    + min(gain_evidence, max(evidence_target - evidence_cover, 0.0))
                    / evidence_target
                )
                gains.append((source, gain, rank[source][node], index))
            positive = [value for value in gains if value[1] > 0]
            owner = (
                min(positive, key=lambda value: (-value[1], value[2], value[3]))[0]
                if positive
                else min(gains, key=lambda value: (value[2], value[3]))[0]
            )
            candidates.append(
                (
                    sum((value[1] for value in gains)),
                    len(positive),
                    min((value[2] for value in gains)),
                    node,
                    owner,
                )
            )
        if not candidates:
            break
        (gain, _, _, node, owner) = min(
            candidates, key=lambda value: (-value[0], -value[1], value[2], value[3])
        )
        if gain <= 0 and len(chosen) >= effective_minimum:
            break
        add(
            node,
            owner,
            "adaptive_joint_coverage" if gain > 0 else "minimum_three_nodes",
        )
    for node in sorted(
        list(chosen),
        key=lambda node: (
            -min((rank[source][node] for source in sources if node in rank[source])),
            node,
        ),
    ):
        if len(chosen) <= effective_minimum:
            break
        if sum((owners[value] == owners[node] for value in chosen)) <= 1:
            continue
        position = chosen.index(node)
        chosen.remove(node)
        rebuild()
        if complete():
            del owners[node], reasons[node]
        else:
            chosen.insert(position, node)
            rebuild()
    stop_reason = (
        "coverage_targets_reached"
        if complete()
        else "maximum_nodes_reached"
        if len(chosen) >= maximum
        else "available_nodes_exhausted"
        if len(chosen) == len(available)
        else "no_positive_coverage_gain"
    )
    audit = {}
    for source in sources:
        visible = set(chosen) & set(orders[source])
        (all_cover, evidence_cover) = fractions(source)
        audit[source] = {
            "source_node_count": len(orders[source]),
            "selected_node_count": len(visible),
            "assigned_node_count": sum((owners[node] == source for node in chosen)),
            "all_paper_denominator": len(denominators[source]),
            "selected_all_papers": len(covered[source]),
            "evidence_paper_denominator": len(evidence[source]),
            "selected_evidence_papers": len(covered[source] & evidence[source]),
            "all_paper_coverage": all_cover,
            "evidence_paper_coverage": evidence_cover,
            "all_paper_target_reached": all_cover >= all_target,
            "evidence_target_reached": evidence_cover >= evidence_target,
            "minimum_display_nodes": minimum,
            "maximum_display_nodes": maximum,
            "displayed_node_count": len(chosen),
            "minimum_node_target_reached": len(chosen) >= minimum,
        }
    assert effective_minimum <= len(chosen) <= maximum
    assert len(chosen) == len(set(chosen))
    return Selection(chosen, owners, reasons, audit, stop_reason, len(available))
