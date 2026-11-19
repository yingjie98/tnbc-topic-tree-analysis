from __future__ import annotations
import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(
        description="Table-only topic-tree analysis. Inputs: tree JSON keyed by node ID (name, level, children_ids, all_paper_pmid), papers JSON keyed by PMID (year, Title, Abstract; optional JCR/IF). Scoring XLSX columns H/I are manual expert scores. AI scoring requires --stages score and OPENAI_API_KEY; no API call is made by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tree", type=Path)
    p.add_argument("--papers", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "all",
            "descriptive",
            "reliability",
            "observed",
            "weighted",
            "temporal",
            "history",
            "comparison",
            "cores",
            "prepare-score",
            "score",
            "analyze-score",
        ],
        default=["all"],
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        help="Existing local PubMedBERT directory; never copied into this package",
    )
    p.add_argument(
        "--model-id",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    p.add_argument("--skip-reliability", action="store_true")
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--observation-start", type=int, default=2001)
    p.add_argument("--observation-end", type=int, default=2025)
    p.add_argument("--cohort-start", type=int, default=2007)
    p.add_argument("--history-start", type=int, default=2007)
    p.add_argument(
        "--snapshot-ends", type=int, nargs="+", default=[2010, 2015, 2020, 2025]
    )
    p.add_argument(
        "--training-cutoffs", type=int, nargs="+", default=[2010, 2015, 2020]
    )
    p.add_argument("--forecast-start", type=int, default=2026)
    p.add_argument("--forecast-end", type=int, default=2030)
    p.add_argument("--emerging-years", type=int, default=3)
    p.add_argument("--recent-years", type=int, default=5)
    p.add_argument("--structure-start", type=int)
    p.add_argument("--structure-end", type=int)
    p.add_argument(
        "--reliability-scope",
        choices=["complete-years", "all-papers"],
        default="complete-years",
    )
    p.add_argument("--bootstrap-replicates", type=int, default=49999)
    p.add_argument("--mean-replicates", type=int, default=1999)
    p.add_argument("--permutations", type=int, default=49999)
    p.add_argument("--simulations", type=int, default=24)
    p.add_argument("--open-candidates", type=int, default=60)
    p.add_argument(
        "--workbook",
        type=Path,
        help="Single ten-column scoring workbook; editable expert columns are never filled by API",
    )
    p.add_argument("--paper-count", type=int, default=1000)
    p.add_argument("--edge-cap", type=int, default=50)
    p.add_argument("--openai-model", default="gpt-4")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    requested = args.stages
    needs_data = any((s not in {"score", "analyze-score"} for s in requested))
    if needs_data and any((v is None for v in (args.tree, args.papers, args.output))):
        p.error("--tree, --papers and --output are required for analysis")
    if not needs_data:
        if args.workbook is None:
            p.error("--workbook is required for scoring or score analysis")
        args.tree = args.tree or Path("tree.json")
        args.papers = args.papers or Path("papers.json")
        args.output = args.output or args.workbook.parent
    args.snapshot_ends = tuple(args.snapshot_ends)
    args.training_cutoffs = tuple(args.training_cutoffs)
    for name in (
        "bootstrap_replicates",
        "mean_replicates",
        "permutations",
        "simulations",
        "open_candidates",
    ):
        if getattr(args, name) < 1:
            p.error(name + " must be positive")
    from .project_config import configure, TIME

    configure(args)
    order = [
        "descriptive",
        "reliability",
        "observed",
        "weighted",
        "temporal",
        "history",
        "comparison",
        "cores",
        "prepare-score",
        "score",
        "analyze-score",
    ]
    stages = order[:-2] if "all" in requested else [s for s in order if s in requested]
    if "all" in requested and len(requested) > 1:
        p.error("Use all alone, or explicit stages")
    if args.workbook is None and any(
        (s in stages for s in ("prepare-score", "score", "analyze-score"))
    ):
        args.workbook = (
            args.output / "02_reliability/ai_validation/AI内容验证_评分与统计结果.xlsx"
        )
    if needs_data:
        for path in (args.tree, args.papers):
            if not path.is_file():
                p.error("Input file does not exist: " + str(path))
    package = Path(__file__).resolve().parent.parent
    for path in (args.output, args.workbook):
        if path is not None and (
            path.resolve() == package or package in path.resolve().parents
        ):
            p.error("Choose output and scoring paths outside the release package")
    from . import workflow as w
    from .topic_tree_analysis import TopicTreeDataset

    full = None
    complete = None
    for stage in stages:
        print("Running " + stage, flush=True)
        if stage in {"descriptive", "observed", "weighted"}:
            if full is None:
                full = TopicTreeDataset(
                    args.tree, args.papers, TIME.structure_start, TIME.structure_end
                )
        if stage in {"reliability", "prepare-score"}:
            if complete is None:
                complete = TopicTreeDataset(
                    args.tree,
                    args.papers,
                    None
                    if args.reliability_scope == "all-papers"
                    else TIME.cohort_start,
                    None
                    if args.reliability_scope == "all-papers"
                    else TIME.observation_end,
                )
        if stage == "descriptive":
            w.descriptive(full, args.output)
        elif stage == "reliability" and (not args.skip_reliability):
            from .reliability import run

            run(
                complete,
                args.output / "02_reliability/tables",
                args.model_dir,
                args.model_id,
                args.bootstrap_replicates,
                args.mean_replicates,
                args.permutations,
            )
        elif stage == "observed":
            w.observed(full, args.output, args.seed)
        elif stage == "weighted":
            w.weighted(full, args.output, args.seed)
        elif stage == "temporal":
            w.temporal(
                args.tree,
                args.papers,
                args.output,
                args.seed,
                args.simulations,
                args.open_candidates,
            )
        elif stage == "history":
            w.history(args.tree, args.papers, args.output)
        elif stage == "comparison":
            w.comparison(args.tree, args.papers, args.output)
        elif stage == "cores":
            w.core_networks(args.tree, args.papers, args.output)
        elif stage == "prepare-score":
            from .validation import sample_questions, save_book, read_book

            path = (
                args.workbook
                or args.output
                / "02_reliability/ai_validation/AI内容验证_评分与统计结果.xlsx"
            )
            if path.is_file():
                read_book(path)
            else:
                save_book(
                    path,
                    sample_questions(
                        complete, paper_count=args.paper_count, edge_cap=args.edge_cap
                    ),
                )
            print("Manual scores: columns H and I in " + str(path), flush=True)
        elif stage == "score":
            from .validation import score

            score(args.workbook, model=args.openai_model)
        elif stage == "analyze-score":
            from .validation import analyze

            analyze(args.workbook, args.output / "02_reliability/tables")
    return 0
