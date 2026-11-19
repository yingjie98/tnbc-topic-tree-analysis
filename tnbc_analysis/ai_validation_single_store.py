from __future__ import annotations
import math
import re

HEADERS = [
    "所属大题编号",
    "小题序号",
    "题干文本",
    "错配标记",
    "AI第一次打分",
    "AI第二次打分",
    "AI第三次打分",
    "专家一打分",
    "专家二打分",
    "AI统一评分",
]
METRICS = {1: "accuracy_score", 2: "coverage_score", 3: "refinement_score"}
EDGE_ID = re.compile("^L(\\d+)-P(\\d+)-(T\\d+)$")
NA = "n/a"
MATERIAL = "\n【题目材料】\n"


def integrated(values):
    if any((v in (None, "") for v in values)):
        return None
    if NA in values:
        return NA
    return sum(values) / 3


def validate_rows(rows):
    (seen, papers) = (set(), {})
    for row in rows:
        if len(row) != 10:
            raise ValueError("Scoring rows must contain exactly ten columns")
        (major, qid, question, mismatch) = row[:4]
        if type(major) is not int or major not in METRICS:
            raise ValueError("Invalid major question number")
        if not isinstance(qid, str) or (major, qid) in seen:
            raise ValueError("Duplicate/invalid question number")
        seen.add((major, qid))
        if (
            not isinstance(question, str)
            or not question
            or len(question) > 32767
            or (MATERIAL not in question)
        ):
            raise ValueError(
                "Question text missing, malformed or longer than an Excel cell"
            )
        if type(mismatch) is not int or mismatch not in (0, 1):
            raise ValueError("Invalid mismatch flag")
        if major == 3 and (not EDGE_ID.fullmatch(qid)):
            raise ValueError("Edge ID must encode Level/source-parent group")
        for value in row[4:9]:
            if value in (None, "", NA):
                continue
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value not in (1, 2, 3, 4)
            ):
                raise ValueError("A round score must be integer 1..4, n/a, or blank")
        expected = integrated(row[4:7])
        if expected is None:
            if row[9] not in (None, ""):
                raise ValueError("Incomplete first three rounds have a nonblank mean")
        elif expected == NA:
            if row[9] != NA:
                raise ValueError("An unable first-three result must be n/a")
        elif not isinstance(row[9], (int, float)) or not math.isclose(
            expected, row[9], abs_tol=1e-12
        ):
            raise ValueError("AI mean must use exactly rounds 1, 2 and 3")
        if major in (1, 2):
            key = (question.split(MATERIAL, 1)[1], mismatch)
            if qid in papers and papers[qid] != key:
                raise ValueError(
                    "The two paper metrics have different material/condition"
                )
            papers[qid] = key
    first = {r[1] for r in rows if r[0] == 1}
    second = {r[1] for r in rows if r[0] == 2}
    if first != second:
        raise ValueError("Paper accuracy and coverage questions do not pair by ID")
    return rows


def question_text(record, major):
    from .sampling_helpers import RUBRIC

    names = {1: "分类准确性", 2: "内容覆盖完整性", 3: "上下级概念细化关系正确性"}
    rubric = "\n".join((f"{r[0]}分：{r[major]}" for r in RUBRIC))
    p = record["payload"]
    if major in (1, 2):
        paths = "\n".join(
            ("[" + q["path_id"] + "] " + q["full_path"] for q in p["paths"])
        )
        material = f"题名：{p['title']}\n摘要：{p['abstract']}\n全部分类路径：\n{paths}"
    else:
        material = f"Level {record['level'] - 1} 至 Level {record['level']}\n完整语境：{p['context']}\n上层概念：{p['parent_concept']}\n待评价下层概念：{p['child_concept']}"
    return (
        f"请评价{names[major]}，仅给1—4整数分；材料不足时给n/a。\n评分标准：\n{rubric}"
        + MATERIAL
        + material
    )


def rows_from_records(records):
    parents = {
        p: i + 1
        for (i, p) in enumerate(
            sorted({r["parent_id"] for r in records if r["task_type"] == "edge"})
        )
    }
    rows = []
    for r in records:
        qid = (
            r["item_id"]
            if r["task_type"] == "paper"
            else f"L{int(r['level']):02}-P{parents[r['parent_id']]:04}-{r['item_id']}"
        )
        for major in (1, 2) if r["task_type"] == "paper" else (3,):
            rows.append(
                [
                    major,
                    qid,
                    question_text(r, major),
                    int(r["condition"] == "random_mismatch"),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
            )
    return validate_rows(sorted(rows, key=lambda r: (r[0], r[1])))
