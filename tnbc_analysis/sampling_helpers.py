from __future__ import annotations
import re

RUBRIC = [
    [
        "1",
        "主要内容与分类明显不符",
        "主要研究内容几乎未被覆盖",
        "无合理细化关系，或方向明显相反",
    ],
    [
        "2",
        "部分相关，但有实质性错配或无依据推断",
        "遗漏重要研究主题或关键角度",
        "相关但属于并列、泛化或关系不清",
    ],
    [
        "3",
        "主要正确，仅有次要不精确",
        "覆盖核心内容，存在少量次要遗漏",
        "基本构成合理细分，边界或表述略含糊",
    ],
    [
        "4",
        "全部路径有内容依据，无实质错误",
        "充分覆盖核心对象、问题和关键角度，层级适当",
        "给定语境下明确、方向正确的细分",
    ],
]


def extract_text(paper):
    tandem = str(paper.get("TandA") or "")
    match = re.search(
        "\\bTitle\\s*:\\s*(.*?)\\s*;?\\s*Abstract\\s*:\\s*(.*)", tandem, re.I | re.S
    )
    title = str(paper.get("Title") or (match.group(1) if match else "")).strip()
    abstract = str(paper.get("Abstract") or (match.group(2) if match else "")).strip()
    if abstract.lower() in ("none", "nan", "no abstract available", ""):
        abstract = ""
    return (title, abstract)


def allocate(total, sizes, block_size=2):
    keys = sorted(sizes)
    ideal = {k: total * sizes[k] / sum(sizes.values()) for k in keys}
    result = {k: min(int(ideal[k]), sizes[k] // block_size) for k in keys}
    while sum(result.values()) < total:
        choices = [k for k in keys if result[k] < sizes[k] // block_size]
        if not choices:
            raise ValueError("不足1000个合格不同论文。")
        k = max(choices, key=lambda k: (ideal[k] - result[k], k))
        result[k] += 1
    return result


def edge_group_counts(population, cap=50):
    return (
        {"original": cap // 2, "random_mismatch": cap // 2}
        if population >= cap
        else {"original": population}
    )
