from __future__ import annotations
from pathlib import Path
import pandas as pd


def u_main_names(tables, language="zh"):
    path = (
        Path(tables).parent.parent
        / "06_predict/tables/21_所有年度与全量S_JCR_IF_未来结构块统一对齐.csv"
    )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {v: v for v in frame["整合结构块编号"]}


def formal_s_u_ids(tables):
    path = (
        Path(tables).parent.parent
        / "06_predict/tables/21_所有年度与全量S_JCR_IF_未来结构块统一对齐.csv"
    )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(frame["正式s结构块编号"], frame["整合结构块编号"]))


def english_node_names(nodes, tree=None, tree_path=None):
    if tree is None:
        import json
        from .project_config import PROFILE

        tree = json.loads(
            Path(tree_path or PROFILE["tree"]).read_text(encoding="utf-8-sig")
        )
    return [
        str(tree[str(k)].get("english_name") or tree[str(k)].get("name") or k)
        for k in nodes
    ]
