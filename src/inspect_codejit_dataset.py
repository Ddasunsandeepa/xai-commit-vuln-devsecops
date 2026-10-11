#!/usr/bin/env python3
"""CodeJIT metadata and provenance inspection (read-only).

Example:
 python src/inspect_codejit_dataset.py \
   --csv data/raw/jit_candidate/CodeJIT/Data/CodeJIT_Dataset_Commit_Hash.csv \
   --repo-root data/raw/jit_candidate/CodeJIT \
   --out reports/codejit_inspection

Requires: pandas. Does not clone repositories or fetch commit diffs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

REQUIRED = {"commit_id", "commit_date", "repo", "label"}
TERMS = re.compile(r"vulnerab|dangerous|safe|benign|introduc|fixing|fix commit|label|dataset|ground.truth|commit.hash|commit_id|train.test|SZZ", re.I)


def save_csv(df: pd.DataFrame, dest: Path) -> None:
    df.to_csv(dest, index=False)
    print(f"  Saved {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/codejit_inspection"))
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"CSV not found: {args.csv}")
    if not args.repo_root.is_dir():
        parser.error(f"Repository directory not found: {args.repo_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={"commit_id": "string", "repo": "string", "label": "string"}, low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        parser.error(f"Missing required columns: {sorted(missing)}")

    digest = hashlib.sha256()
    with args.csv.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    print("\n========== 1. DATASET AND DUPLICATES ==========")
    print(f"Rows: {len(df):,}")
    print(f"Unique repositories: {df['repo'].nunique():,}")
    print(f"SHA256: {digest.hexdigest()}")
    print(f"Missing values:\n{df[list(REQUIRED)].isna().sum().to_string()}")
    print(f"Label counts:\n{df['label'].value_counts(dropna=False).to_string()}")

    pair_cols = ["repo", "commit_id"]
    duplicates = int(df.duplicated(pair_cols).sum())
    conflict_mask = df.groupby(pair_cols, dropna=False)["label"].nunique(dropna=False).gt(1)
    conflicts = int(conflict_mask.sum())
    print(f"Duplicate (repo, commit_id) rows: {duplicates}")
    print(f"Conflicting labels for same (repo, commit_id): {conflicts}")
    print(f"Unique (repo, commit_id) pairs: {df[pair_cols].drop_duplicates().shape[0]:,}")

    if duplicates:
        dup = df[df.duplicated(pair_cols, keep=False)].sort_values(pair_cols)
        save_csv(dup, args.out / "duplicate_commit_rows.csv")
    if conflicts:
        conflicting_keys = conflict_mask[conflict_mask].index.to_frame(index=False)
        conflict_rows = df.merge(conflicting_keys, on=pair_cols, how="inner")
        save_csv(conflict_rows, args.out / "conflicting_label_rows.csv")

    project = df.groupby("repo", dropna=False).agg(rows=("commit_id", "size"), unique_commits=("commit_id", "nunique"))
    for label in sorted(df["label"].dropna().unique()):
        project[f"label_{label}"] = df.loc[df["label"].eq(label)].groupby("repo").size().reindex(project.index, fill_value=0)
    project = project.sort_values("rows", ascending=False).reset_index()
    print("\nLargest repositories:")
    print(project.head(args.top).to_string(index=False))
    save_csv(project, args.out / "repository_label_distribution.csv")
    print("\nNumber of repositories by size:")
    for cutoff in (1, 5, 10, 20, 50, 100):
        print(f"  >= {cutoff} rows: {int((project['rows'] >= cutoff).sum())}")
    if "label_0" in project and "label_1" in project:
        both = ((project["label_0"] > 0) & (project["label_1"] > 0)).sum()
        print(f"Repositories containing both labels: {int(both)}")

    print("\n========== 2. TIMESTAMP INSPECTION ==========")
    numeric_dates = pd.to_numeric(df["commit_date"], errors="coerce")
    times = pd.to_datetime(numeric_dates, unit="s", utc=True, errors="coerce")
    invalid = int(times.isna().sum())
    print(f"Invalid dates: {invalid}")
    print(f"Earliest commit: {times.min()}")
    print(f"Latest commit:   {times.max()}")
    years = times.dt.year.value_counts(dropna=False).sort_index().rename_axis("year").reset_index(name="rows")
    print("\nRecords by year:")
    print(years.to_string(index=False))
    save_csv(years, args.out / "commits_by_year.csv")
    time_df = df[pair_cols + ["label"]].copy()
    time_df["commit_datetime_utc"] = times.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    save_csv(time_df, args.out / "commit_metadata_with_dates.csv")

    print("\n========== 3. ORIGINAL DOCUMENTATION ==========")
    readmes = [p for p in args.repo_root.iterdir() if p.is_file() and p.name.lower().startswith("readme")]
    excerpts = []
    for readme in sorted(readmes):
        print(f"\n--- {readme.relative_to(args.repo_root)} ---")
        content = readme.read_text(encoding="utf-8", errors="replace")
        print(content[:16000])
        if len(content) > 16000:
            print("[README truncated in console; full file remains in cloned repository]")
        excerpts.append(f"README: {readme.relative_to(args.repo_root)}\n{content}\n")

    print("\nPython files containing potentially relevant terms:")
    matched_files = 0
    for p in sorted(args.repo_root.rglob("*.py")):
        # Skip any vendored environments, if present.
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in p.parts):
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        matches = [(i, line.strip()) for i, line in enumerate(lines, start=1) if TERMS.search(line)]
        if matches:
            matched_files += 1
            print(f"  {p.relative_to(args.repo_root)} ({len(matches)} matching lines)")
            excerpts.append(f"\nFILE: {p.relative_to(args.repo_root)}\n")
            for i, line in matches[:30]:
                excerpts.append(f"  L{i}: {line[:300]}\n")
            if len(matches) > 30:
                excerpts.append(f"  ... {len(matches)-30} further matching lines omitted\n")
    print(f"Matched Python files: {matched_files}")
    (args.out / "provenance_excerpts.txt").write_text("\n".join(excerpts), encoding="utf-8")
    print(f"  Saved {args.out / 'provenance_excerpts.txt'}")

    summary = {
        "source_csv": str(args.csv), "sha256": digest.hexdigest(),
        "rows": int(len(df)), "unique_repositories": int(df["repo"].nunique()),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()},
        "duplicate_repo_commit_rows": duplicates, "conflicting_repo_commit_pairs": conflicts,
        "invalid_timestamps": invalid, "earliest_commit_utc": str(times.min()),
        "latest_commit_utc": str(times.max()), "readme_files": [str(p) for p in readmes],
        "matched_python_files": matched_files,
        "cautions": [
            "Label meaning must be verified from the source paper or dataset construction documentation.",
            "Commit diffs and messages are not present in the metadata CSV.",
            "Repository identifiers are not verified clone URLs.",
            "Do not use future commit information to predict earlier commits.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {args.out / 'summary.json'}")
    print("\nNEXT: Share console output + provenance_excerpts.txt; verify label semantics before extracting Git diffs.")


if __name__ == "__main__":
    main()
