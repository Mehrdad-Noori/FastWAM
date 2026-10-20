#!/usr/bin/env python3
"""Audit a pinned LIBERO-Plus manifest without simulator dependencies."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
BASE = f"https://raw.githubusercontent.com/sylvestf/LIBERO-plus/{REVISION}/"
FILES = {
    "classification": "libero/libero/benchmark/task_classification.json",
    "task_map": "libero/libero/benchmark/libero_suite_task_map.py",
}
PAPER_V2_COUNTS = {
    "libero_spatial": 2293, "libero_object": 2576,
    "libero_goal": 2569, "libero_10": 2592,
}


def audit(classification_bytes, task_map_bytes):
    classification = json.loads(classification_bytes)
    tree = ast.parse(task_map_bytes.decode("utf-8"))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Expected one literal libero_task_map assignment")
    task_map = ast.literal_eval(assignments[0].value)
    if set(classification) != set(PAPER_V2_COUNTS):
        raise ValueError("Unexpected classification suites")
    totals = Counter()
    suites = {}
    for suite, rows in classification.items():
        names = task_map[suite]
        if len(rows) != len(names) or len(set(names)) != len(names):
            raise ValueError(f"{suite}: duplicate names or unequal manifest lengths")
        for index, row in enumerate(rows):
            if row["id"] != index + 1 or row["name"] != names[index]:
                raise ValueError(f"{suite}: name/index mismatch at index {index}")
            if row["difficulty_level"] is not None and row["difficulty_level"] not in range(1, 6):
                raise ValueError(f"{suite}: invalid difficulty at index {index}")
        counts = Counter(row["category"] for row in rows)
        totals.update(counts)
        suites[suite] = {
            "released_count": len(rows),
            "paper_v2_table7_count": PAPER_V2_COUNTS[suite],
            "difference_from_paper": len(rows) - PAPER_V2_COUNTS[suite],
            "classification_id_base": 1,
            "benchmark_task_index_base": 0,
            "all_names_match_task_map": True,
            "category_counts": dict(sorted(counts.items())),
            "difficulty_counts": dict(sorted(Counter("unknown" if r["difficulty_level"] is None else str(r["difficulty_level"]) for r in rows).items())),
        }
    return {
        "repository": "sylvestf/LIBERO-plus", "revision": REVISION,
        "sources": {
            key: {"url": BASE + FILES[key], "sha256": hashlib.sha256(data).hexdigest()}
            for key, data in (("classification", classification_bytes), ("task_map", task_map_bytes))
        },
        "paper_comparison": "https://arxiv.org/html/2510.13626v2",
        "total_variants": sum(s["released_count"] for s in suites.values()),
        "category_counts": dict(sorted(totals.items())), "suites": suites,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Read a local checkout instead.")
    parser.add_argument("--output", type=Path, help="Write JSON instead of stdout.")
    args = parser.parse_args()
    data = {}
    for key, path in FILES.items():
        if args.source_dir is not None:
            data[key] = (args.source_dir / path).read_bytes()
        else:
            with urllib.request.urlopen(BASE + path, timeout=30) as response:
                data[key] = response.read()
    result = audit(data["classification"], data["task_map"])
    if args.source_dir is not None:
        result["revision"] = None
        result["repository"] = "local source directory; verify revision independently"
        for source in result["sources"].values():
            source.pop("url")
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
