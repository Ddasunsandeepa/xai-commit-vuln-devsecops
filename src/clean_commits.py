"""
STEP 2.3 — DATA CLEANING SCRIPT (v2 — Improved)
=================================================
Save this as: src/clean_commits.py
Run with:    python src/clean_commits.py

Improvements in v2:
  1. label column added (placeholder 0 — ready for real labels later)
  2. files_changed kept as ML feature (churn metric)
  3. author_name + author_date extracted (for activity patterns)
  4. Safe error handling around diff processing
  5. Cleaning report saved to reports/cleaning_report.txt
"""

import pandas as pd
import re
import os
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/commit_dataset.csv"
OUTPUT_FILE  = "data/commit_dataset_cleaned.csv"
REPORT_FILE  = "reports/cleaning_report.txt"

MAX_DIFF_CHARS = 10000   # skip diffs longer than this (too big for CodeBERT)
MIN_DIFF_CHARS = 30      # skip diffs shorter than this (basically empty)

NOISE_KEYWORDS = [
    "merge branch", "merge pull request",
    "whitespace", "formatting", "typo",
    "readme", "changelog", "licence", "license",
    "bump version", "update version",
    ".gitignore", "documentation", "docs only",
]

SKIP_EXTENSIONS = [
    ".md", ".txt", ".rst", ".html", ".css",
    ".png", ".jpg", ".gif", ".svg", ".ico",
    ".pdf", ".lock", ".xml", ".json", ".yaml", ".yml",
    ".min.js",
]

SECURITY_TERMS = [
    "password", "passwd", "secret", "token", "auth",
    "sql", "query", "execute", "cursor",
    "eval(", "exec(", "subprocess", "shell=True",
    "os.system", "pickle", "deserializ",
    "xss", "csrf", "injection", "overflow",
    "strcpy", "gets(", "sprintf", "malloc",
    "chmod", "chown", "setuid",
    "base64", "encrypt", "decrypt", "hash(",
]


# ── HELPERS ──────────────────────────────────────────────────────────────────

def is_noise_message(msg: str) -> bool:
    msg_lower = str(msg).lower()
    return any(kw in msg_lower for kw in NOISE_KEYWORDS)


def count_lines_added(diff: str) -> int:
    try:
        return sum(1 for line in diff.splitlines()
                   if line.startswith("+") and not line.startswith("+++"))
    except Exception as e:
        print(f"    [WARN] count_lines_added failed: {e}")
        return 0


def count_lines_removed(diff: str) -> int:
    try:
        return sum(1 for line in diff.splitlines()
                   if line.startswith("-") and not line.startswith("---"))
    except Exception as e:
        print(f"    [WARN] count_lines_removed failed: {e}")
        return 0


def has_security_keywords(diff: str) -> int:
    try:
        diff_lower = diff.lower()
        return int(any(term in diff_lower for term in SECURITY_TERMS))
    except Exception as e:
        print(f"    [WARN] has_security_keywords failed: {e}")
        return 0


def clean_diff(diff: str) -> str:
    try:
        lines = []
        for line in diff.splitlines():
            if line.startswith("@@") and "@@" in line[2:]:
                continue
            lines.append(line.rstrip())
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    except Exception as e:
        print(f"    [WARN] clean_diff failed: {e}")
        return diff


def save_report(report_lines: list, report_path: str):
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n[5] Cleaning report saved → '{report_path}'")
    except Exception as e:
        print(f"    [WARN] Could not save report: {e}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("  COMMIT DATASET CLEANER  v2")
    print("=" * 60)

    # 1. Load raw data
    df = pd.read_csv(INPUT_FILE)
    print(f"\n[1] Loaded {len(df)} commits from '{INPUT_FILE}'")

    df["diff"]           = df["diff"].fillna("")
    df["commit_message"] = df["commit_message"].fillna("")

    # ── IMPROVEMENT 3: Author columns ────────────────────────────────────────
    # These come from PyDriller when you add them to the extractor.
    # If not present yet, create placeholder columns.
    if "author_name" not in df.columns:
        df["author_name"] = "unknown"
        print("    [INFO] 'author_name' not found — added as placeholder.")
    if "author_date" not in df.columns:
        df["author_date"] = ""
        print("    [INFO] 'author_date' not found — added as placeholder.")

    original_count = len(df)
    removed_reasons = {}

    # 2. Drop empty / very short diffs
    mask = df["diff"].str.strip().str.len() < MIN_DIFF_CHARS
    removed_reasons["empty/too-short diff"] = int(mask.sum())
    df = df[~mask].copy()

    # 3. Drop huge diffs (bad for CodeBERT token limits)
    mask = df["diff"].str.len() > MAX_DIFF_CHARS
    removed_reasons["diff too large (>10k chars)"] = int(mask.sum())
    df = df[~mask].copy()

    # 4. Drop noise commits
    mask = df["commit_message"].apply(is_noise_message)
    removed_reasons["noise commit message"] = int(mask.sum())
    df = df[~mask].copy()

    # 5. Compute feature columns
    print("\n[2] Computing features...")
    df["lines_added"]        = df["diff"].apply(count_lines_added)
    df["lines_removed"]      = df["diff"].apply(count_lines_removed)
    df["diff_length"]        = df["diff"].str.len()
    df["has_security_terms"] = df["diff"].apply(has_security_keywords)

    # ── IMPROVEMENT 2: files_changed ─────────────────────────────────────────
    if "files_changed" not in df.columns:
        df["files_changed"] = 0
        print("    [INFO] 'files_changed' not found — added as 0.")

    # ── IMPROVEMENT 1: Add label placeholder ─────────────────────────────────
    df["label"] = 0
    print("    [INFO] 'label' column added (all 0 — placeholder for JIT-Vul labels).")

    # 6. Clean diff text
    print("    [INFO] Cleaning diff text...")
    df["diff"] = df["diff"].apply(clean_diff)

    # 7. Reset index and reorder columns
    df = df.reset_index(drop=True)

    column_order = [
        "commit_hash", "commit_message",
        "author_name", "author_date",
        "files_changed", "lines_added", "lines_removed",
        "diff_length", "has_security_terms",
        "label", "diff",
    ]
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]

    # ── RESULTS ──────────────────────────────────────────────────────────────
    kept    = len(df)
    dropped = original_count - kept

    print(f"\n[3] Cleaning Results:")
    for reason, count in removed_reasons.items():
        print(f"    - Removed {count:>3} commit(s): {reason}")
    print(f"\n    Original : {original_count}")
    print(f"    Dropped  : {dropped}")
    print(f"    Kept     : {kept}")

    print(f"\n[4] Sample of cleaned dataset:")
    print(df[["commit_hash", "commit_message", "lines_added",
              "lines_removed", "has_security_terms", "label"]].to_string(index=False))

    # 8. Save CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n    Saved cleaned dataset → '{OUTPUT_FILE}'")

    # ── IMPROVEMENT 5: Save cleaning report ──────────────────────────────────
    report_lines = [
        "COMMIT DATASET CLEANING REPORT",
        "=" * 50,
        f"Run time      : {run_time}",
        f"Input file    : {INPUT_FILE}",
        f"Output file   : {OUTPUT_FILE}",
        "",
        "CLEANING SUMMARY",
        "-" * 50,
        f"Original commits  : {original_count}",
        f"Commits dropped   : {dropped}",
        f"Commits kept      : {kept}",
        "",
        "REMOVAL REASONS",
        "-" * 50,
    ]
    for reason, count in removed_reasons.items():
        report_lines.append(f"  {count:>3} removed — {reason}")

    report_lines += [
        "",
        "FEATURE STATS (cleaned dataset)",
        "-" * 50,
        f"  avg lines added   : {df['lines_added'].mean():.1f}",
        f"  avg lines removed : {df['lines_removed'].mean():.1f}",
        f"  avg diff length   : {df['diff_length'].mean():.0f} chars",
        f"  security term hits: {df['has_security_terms'].sum()} / {kept}",
        "",
        "LABEL DISTRIBUTION",
        "-" * 50,
        f"  label=0 (placeholder): {kept}",
        f"  label=1 (vulnerable) : 0  <- will be filled from JIT-Vul",
        "",
        "NEXT STEP",
        "-" * 50,
        "  Move to STEP 2.4 — Load JIT-Vul dataset and map real labels.",
    ]

    save_report(report_lines, REPORT_FILE)

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("  Next step -> STEP 2.4: Load JIT-Vul and add real labels")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()