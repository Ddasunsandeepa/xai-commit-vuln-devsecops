"""
STEP 2.4 — BIG-VUL DATASET EXPLORER (Memory-Safe Version)
===========================================================
Save as:  src/explore_big_vul.py
Run with: python src/explore_big_vul.py

Fix: Loads only a SAMPLE of rows first to avoid crashing
     the container on large datasets.
"""

import pandas as pd
import os
import sys

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH = "data/big_vul/MSR_data_cleaned.csv"

# Load only this many rows first — safe for any container
# Change to None to load ALL (warning: may crash on small containers)
SAMPLE_ROWS = 5000


def check_file_exists(path):
    if not os.path.exists(path):
        print(f"\n  ERROR: File not found: {path}")
        print("  Place MSR_data_cleaned.csv in data/big_vul/\n")
        sys.exit(1)


def explore(df, is_sample=True):

    sample_note = f" (SAMPLE: first {len(df):,} rows)" if is_sample else ""

    print(f"\n[2] DATASET SHAPE{sample_note}")
    print(f"    Rows    : {len(df):,}")
    print(f"    Columns : {df.shape[1]}")

    print("\n[3] ALL COLUMNS")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        print(f"    {col:<30} dtype={dtype:<10} nulls={nulls:,}")

    print(f"\n[4] LABEL DISTRIBUTION{sample_note}")
    if "vul" in df.columns:
        counts = df["vul"].value_counts().sort_index()
        total  = len(df)
        for val, cnt in counts.items():
            label = "VULNERABLE    " if val == 1 else "not vulnerable"
            pct   = cnt / total * 100
            bar   = "█" * int(pct / 3)
            print(f"    vul={val}  {label}  {cnt:>6,}  ({pct:.1f}%)  {bar}")
        vuln  = counts.get(1, 0)
        safe  = counts.get(0, 0)
        ratio = safe // max(vuln, 1)
        print(f"\n    Class imbalance: 1 vulnerable for every {ratio} safe rows")
    else:
        print("    ERROR: 'vul' column not found!")

    print("\n[5] TOP CWE TYPES (vulnerable only)")
    if "cwe_id" in df.columns and "vul" in df.columns:
        top = df[df["vul"] == 1]["cwe_id"].value_counts().head(8)
        if len(top) == 0:
            print("    No vulnerable rows in this sample")
        for cwe, cnt in top.items():
            print(f"    {str(cwe):<25} {cnt:>4,} functions")

    print("\n[6] TOP PROJECTS")
    if "project" in df.columns:
        top = df["project"].value_counts().head(8)
        for proj, cnt in top.items():
            vuln = len(df[(df["project"] == proj) & (df.get("vul", pd.Series()) == 1)])
            print(f"    {str(proj):<25} total={cnt:>5,}  vuln={vuln:>4,}")

    print("\n[7] FUNCTION TEXT COLUMNS")
    for col in ["func_before", "func_after", "patch"]:
        if col in df.columns:
            non_null = df[col].notna().sum()
            avg_len  = int(df[col].dropna().astype(str).str.len().mean())
            print(f"    '{col}': {non_null:,} non-null, avg {avg_len:,} chars")

    print("\n[8] SCHEMA MAPPING — Your Pipeline vs Big-Vul")
    print("    " + "-" * 52)
    print(f"    {'YOUR COLUMN':<22} {'BIG-VUL COLUMN':<22} STATUS")
    print("    " + "-" * 52)
    mapping = [
        ("commit_hash",    "commit_id",      "commit_id"      in df.columns),
        ("commit_message", "commit_message", "commit_message" in df.columns),
        ("diff",           "patch",          "patch"          in df.columns),
        ("label",          "vul",            "vul"            in df.columns),
        ("files_changed",  "files_changed",  "files_changed"  in df.columns),
        ("lines_added",    "add_lines",      "add_lines"      in df.columns),
        ("lines_removed",  "del_lines",      "del_lines"      in df.columns),
        ("(bonus)",        "func_before",    "func_before"    in df.columns),
        ("(bonus)",        "CVE ID",         "CVE ID"         in df.columns),
    ]
    for your_col, bv_col, found in mapping:
        status = "✅ FOUND" if found else "❌ missing"
        print(f"    {your_col:<22} {bv_col:<22} {status}")

    print("\n[9] SAMPLE VULNERABLE FUNCTION")
    if "vul" in df.columns and "func_before" in df.columns:
        vuln_rows = df[df["vul"] == 1]
        if len(vuln_rows) == 0:
            print("    No vulnerable rows in this sample — try increasing SAMPLE_ROWS")
        else:
            sample = vuln_rows.iloc[0]
            print(f"    commit_id : {sample.get('commit_id', 'N/A')}")
            print(f"    project   : {sample.get('project',   'N/A')}")
            print(f"    cwe_id    : {sample.get('cwe_id',    'N/A')}")
            code = str(sample.get("func_before", ""))[:400]
            print(f"    func_before preview:")
            for line in code.splitlines()[:12]:
                print(f"      {line}")


def main():
    print("=" * 60)
    print("   BIG-VUL DATASET EXPLORER  (memory-safe)")
    print("=" * 60)

    check_file_exists(DATASET_PATH)

    # ── Step 1: peek at full row count without loading everything ─────────────
    print(f"\n[1] Counting total rows (without loading full dataset)...")
    try:
        # Count lines in file = fast, no memory issue
        with open(DATASET_PATH, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = sum(1 for _ in f)
        total_rows = total_lines - 1  # subtract header
        print(f"    Total rows in file: {total_rows:,}")
        print(f"    Loading sample of: {SAMPLE_ROWS:,} rows")
    except Exception as e:
        print(f"    Could not count lines: {e}")
        total_rows = "unknown"

    # ── Step 2: load only the sample ─────────────────────────────────────────
    print(f"\n    Loading {SAMPLE_ROWS:,} rows from CSV...")
    try:
        df = pd.read_csv(
            DATASET_PATH,
            nrows=SAMPLE_ROWS,
            low_memory=False,
        )
        print(f"    Loaded OK — shape: {df.shape}")
    except Exception as e:
        print(f"    ERROR loading CSV: {e}")
        sys.exit(1)

    # ── Step 3: explore the sample ────────────────────────────────────────────
    explore(df, is_sample=True)

    print("\n" + "=" * 60)
    print("  EXPLORATION COMPLETE")
    print(f"  Full dataset has ~{total_rows:,} rows total")
    print("  Next: Run  python src/preprocess_big_vul.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()