#!/usr/bin/env python3
"""
Recompute LLM metrics in existing RAGChecker JSON reports.

Applies the fixed filename parser (strips LightRAG knowledge-graph annotations
appended after filenames, e.g. "file.pdf   (Entity ↔ Relation)") and
recalculates retrievedDocuments, recall, precision, F1, hit, MRR for every
LLM run, then re-aggregates per-result and summary.llm.

The matching logic mirrors LlmRunResult.java:
  ret.equalsIgnoreCase(exp) OR ret.lower().contains(exp.lower())

StdDev uses population variance (÷ N), matching AggregatedLlmMetrics.java.

Usage:
    python recompute_llm_metrics.py --run-group 20260227_164245_run
    python recompute_llm_metrics.py --run-group 20260227_164245_run --dry-run
    python recompute_llm_metrics.py --reports /path/to/reports --run-group myrun
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

# Mirrors REFERENCE_PATTERN in LightRagClient.java
REFERENCE_PATTERN = re.compile(r'^[-*]?\s*\[(\d+)]\s*(.+?)\s*$', re.MULTILINE)


def parse_referenced_files(text: str) -> list[str]:
    """Mirrors LightRagClient.parseReferencedFiles() with the annotation fix."""
    if not text:
        return []
    files = []
    for m in REFERENCE_PATTERN.finditer(text):
        ref = m.group(2).strip()
        # Strip LightRAG annotations separated by 2+ spaces
        # e.g. "file.pdf   (Entity ↔ Relation)   (Firma ↔ § 5 EFZG)"
        ann = ref.find("  ")
        if ann > 0:
            ref = ref[:ann].strip()
        if ref:
            files.append(ref)
    # distinct, preserve order
    seen = set()
    result = []
    for f in files:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def matches(retrieved: str, expected: str) -> bool:
    """Mirrors: ret.equalsIgnoreCase(exp) || ret.toLowerCase().contains(exp.toLowerCase())"""
    r, e = retrieved.lower(), expected.lower()
    return r == e or e in r


def compute_run_metrics(retrieved: list[str], expected: list[str]) -> dict:
    """Mirrors LlmRunResult.of() metric calculation."""
    if not expected:
        return dict(recall=0.0, precision=0.0, f1=0.0, hit=False, mrr=0.0)

    hits = sum(1 for exp in expected if any(matches(ret, exp) for ret in retrieved))

    recall    = hits / len(expected)
    precision = hits / len(retrieved) if retrieved else 0.0
    f1        = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    hit       = hits > 0

    # MRR: reciprocal rank of first retrieved doc that matches any expected doc
    mrr = 0.0
    for i, ret in enumerate(retrieved):
        if any(matches(ret, exp) for exp in expected):
            mrr = 1.0 / (i + 1)
            break

    return dict(recall=recall, precision=precision, f1=f1, hit=hit, mrr=mrr)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std_dev(values: list[float]) -> float:
    """Population std dev (÷ N), mirrors AggregatedLlmMetrics.java."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def recompute_result_llm(result: dict) -> dict:
    """Re-derives retrievedDocuments and all LLM metrics for one result entry."""
    expected = result.get("expectedDocuments", [])
    llm = result.get("llm", {})

    new_runs = []
    for run in llm.get("runs", []):
        response_text = run.get("responseText", "")
        new_retrieved = parse_referenced_files(response_text)
        m = compute_run_metrics(new_retrieved, expected)
        new_run = {**run,
                   "retrievedDocuments": new_retrieved,
                   "recall":    round(m["recall"],    6),
                   "precision": round(m["precision"], 6),
                   "f1":        round(m["f1"],        6),
                   "hit":       m["hit"],
                   "mrr":       round(m["mrr"],       6)}
        new_runs.append(new_run)

    recalls    = [r["recall"]    for r in new_runs]
    precisions = [r["precision"] for r in new_runs]
    f1s        = [r["f1"]        for r in new_runs]
    hits       = [1.0 if r["hit"] else 0.0 for r in new_runs]
    mrrs       = [r["mrr"]       for r in new_runs]

    return {**llm,
            "runs":         new_runs,
            "avgRecall":    round(_mean(recalls),    6),
            "stdDevRecall": round(_std_dev(recalls), 6),
            "avgPrecision": round(_mean(precisions),    6),
            "stdDevPrecision": round(_std_dev(precisions), 6),
            "avgF1":        round(_mean(f1s),    6),
            "stdDevF1":     round(_std_dev(f1s), 6),
            "hitRate":      round(_mean(hits),   6),
            "avgMrr":       round(_mean(mrrs),    6),
            "stdDevMrr":    round(_std_dev(mrrs), 6)}


def recompute_summary_llm(results: list[dict], old_summary_llm: dict) -> dict:
    """Re-aggregates summary.llm from updated results, preserving avgDurationSec."""
    llm_list = [r["llm"] for r in results if "llm" in r]
    return {**old_summary_llm,
            "avgRecall":    round(_mean([l["avgRecall"]    for l in llm_list]), 6),
            "avgPrecision": round(_mean([l["avgPrecision"] for l in llm_list]), 6),
            "avgF1":        round(_mean([l["avgF1"]        for l in llm_list]), 6),
            "avgHitRate":   round(_mean([l["hitRate"]      for l in llm_list]), 6),
            "avgMrr":       round(_mean([l["avgMrr"]       for l in llm_list]), 6)}


def find_report_files(base: Path) -> list[Path]:
    files = []
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir():
            candidate = subdir / (subdir.name + ".json")
            if candidate.is_file():
                files.append(candidate)
    return files


def process(path: Path, dry_run: bool) -> tuple[float, float, int]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    old_f1 = data.get("summary", {}).get("llm", {}).get("avgF1", 0.0)

    new_results = []
    changed_runs = 0
    for result in data.get("results", []):
        if "llm" not in result:
            new_results.append(result)
            continue
        new_llm = recompute_result_llm(result)
        # Count runs where retrievedDocuments changed
        for old_run, new_run in zip(result["llm"].get("runs", []), new_llm["runs"]):
            if old_run.get("retrievedDocuments") != new_run["retrievedDocuments"]:
                changed_runs += 1
        new_results.append({**result, "llm": new_llm})

    data["results"] = new_results

    old_summary_llm = data.get("summary", {}).get("llm", {})
    data.setdefault("summary", {})["llm"] = recompute_summary_llm(new_results, old_summary_llm)
    new_f1 = data["summary"]["llm"]["avgF1"]

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return old_f1, new_f1, changed_runs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports",   default="reports",
                        help="Base reports directory (default: reports)")
    parser.add_argument("--run-group", default=None,
                        help="Subdirectory within reports (run group folder name)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show what would change without writing files")
    args = parser.parse_args()

    base = Path(args.reports)
    if args.run_group:
        base = base / args.run_group

    if not base.is_dir():
        print(f"ERROR: Directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    files = find_report_files(base)
    if not files:
        print(f"ERROR: No <subdir>/<subdir>.json files found in {base}", file=sys.stderr)
        sys.exit(1)

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Recomputing LLM metrics for {len(files)} report(s) in {base}\n")
    print(f"{'Label':<50} {'Old F1':>8} {'New F1':>8}  {'Δ':>8}  {'Runs':>6}")
    print("-" * 85)

    total_changed = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            label = json.load(f).get("configuration", {}).get("runLabel", path.stem)
        old_f1, new_f1, changed = process(path, args.dry_run)
        total_changed += changed
        delta = new_f1 - old_f1
        marker = " ✓" if delta > 0.0001 else ""
        print(f"{label:<50} {old_f1:>8.4f} {new_f1:>8.4f}  {delta:>+8.4f}  {changed:>5}x{marker}")

    print("-" * 85)
    print(f"\n{total_changed} run(s) with changed retrievedDocuments across all files.")
    if args.dry_run:
        print("Dry run — no files written. Remove --dry-run to apply.")
    else:
        print("Done. JSON files updated in place.")


if __name__ == "__main__":
    main()
