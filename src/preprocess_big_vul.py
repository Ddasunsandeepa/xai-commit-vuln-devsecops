"""
STEP 2.5 — BIG-VUL PREPROCESSING (Chunk-Based, Memory-Safe) v2
================================================================
Save as:  src/preprocess_big_vul.py
Run with: python src/preprocess_big_vul.py

Improvements in v2:
  1. files_changed null handling  → fillna(1) instead of keeping NaN
  2. Duplicate commit removal     → drop_duplicates on commit_hash
  3. Patch cleaning               → remove +++/---/@@ git diff headers
  4. Security term leak warning   → clearly flags this as a research point
  5. Keyword feature toggle       → USE_SECURITY_TERMS = True/False
"""

import pandas as pd
import re
import os
import sys
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/big_vul/MSR_data_cleaned.csv"
OUTPUT_FILE  = "data/processed/big_vul_ml_ready.csv"
REPORT_FILE  = "reports/big_vul_preprocess_report.txt"

CHUNK_SIZE   = 2000
MAX_VULN     = 3000    # vulnerable rows to collect  (vul=1)
MAX_SAFE     = 6000    # safe rows to collect         (vul=0)

MAX_DIFF_CHARS = 5000
MIN_DIFF_CHARS = 30

KEEP_LANGS   = ["C", "C++"]

# ── IMPROVEMENT 4: Toggle security terms feature ──────────────────────────────
# Set False to train WITHOUT keyword hint — compare results in thesis
USE_SECURITY_TERMS = True

SECURITY_TERMS = [
    "password", "sql", "injection", "overflow", "exec(", "eval(",
    "strcpy", "gets(", "malloc", "free(", "memcpy", "sprintf",
    "auth", "token", "secret", "xss", "csrf", "buffer",
]

COLUMN_MAP = {
    "commit_id"     : "commit_hash",
    "project"       : "project",
    "lang"          : "lang",
    "add_lines"     : "lines_added",
    "del_lines"     : "lines_removed",
    "files_changed" : "files_changed",
    "patch"         : "diff",
    "func_before"   : "func_before",
    "func_after"    : "func_after",
    "CWE ID"        : "cwe_id",
    "CVE ID"        : "cve_id",
    "Score"         : "cvss_score",
    "vul"           : "label",
}


# ── HELPERS ──────────────────────────────────────────────────────────────────

def clean_chunk(chunk):
    """Filter rows: language, patch length, missing code."""
    if "lang" in chunk.columns and KEEP_LANGS:
        chunk = chunk[chunk["lang"].isin(KEEP_LANGS)]
    if "patch" in chunk.columns:
        chunk = chunk[chunk["patch"].notna()]
        chunk = chunk[chunk["patch"].astype(str).str.len() >= MIN_DIFF_CHARS]
        chunk = chunk[chunk["patch"].astype(str).str.len() <= MAX_DIFF_CHARS]
    if "func_before" in chunk.columns:
        chunk = chunk[chunk["func_before"].notna()]
    return chunk


def clean_patch(patch: str) -> str:
    """
    IMPROVEMENT 3 — Remove git diff noise lines:
      - Lines starting with +++  (new file header)
      - Lines starting with ---  (old file header)
      - Lines starting with @@   (hunk position header)
    Keeps:
      - Lines starting with +    (added code)
      - Lines starting with -    (removed code)
      - Context lines            (unchanged code)
    """
    try:
        lines = []
        for line in patch.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("@@") and "@@" in line[2:]:
                continue
            lines.append(line.rstrip())
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    except Exception:
        return patch   # return original if cleaning fails


def add_features(df):
    """
    IMPROVEMENT 4 — Security term feature is toggleable.
    This matters for thesis: compare model WITH vs WITHOUT keyword features.
    If USE_SECURITY_TERMS=False, model must learn purely from code structure.
    """
    df["diff_length"]        = df["diff"].astype(str).str.len()
    df["func_before_length"] = df["func_before"].astype(str).str.len()

    if USE_SECURITY_TERMS:
        df["has_security_terms"] = df["func_before"].astype(str).str.lower().apply(
            lambda x: int(any(t in x for t in SECURITY_TERMS))
        )
    else:
        # Feature set without keyword hint — useful for ablation study
        df["has_security_terms"] = 0

    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 62)
    print("  BIG-VUL PREPROCESSOR  v2  (chunk-based, memory-safe)")
    print("=" * 62)

    if not os.path.exists(INPUT_FILE):
        print(f"\n  ERROR: {INPUT_FILE} not found")
        print("  Place MSR_data_cleaned.csv in data/big_vul/\n")
        sys.exit(1)

    print(f"\n  Target    : {MAX_VULN:,} vulnerable + {MAX_SAFE:,} safe rows")
    print(f"  Chunks    : {CHUNK_SIZE:,} rows at a time")
    print(f"  Languages : {KEEP_LANGS}")
    print(f"  Diff size : {MIN_DIFF_CHARS}–{MAX_DIFF_CHARS} chars")
    print(f"  Sec terms : {'ON  ← keyword feature included' if USE_SECURITY_TERMS else 'OFF ← ablation mode (no keyword hint)'}\n")

    # ── IMPROVEMENT 4 warning ─────────────────────────────────────────────────
    if USE_SECURITY_TERMS:
        print("  ⚠️  RESEARCH NOTE: has_security_terms uses domain keywords.")
        print("      Run again with USE_SECURITY_TERMS=False to compare.")
        print("      Both results belong in your thesis ablation section.\n")

    vuln_chunks, safe_chunks = [], []
    total_read = chunk_num = vuln_collected = safe_collected = 0

    print("[1] Reading dataset in chunks...")
    print("    Progress: ", end="", flush=True)

    try:
        reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)

        for chunk in reader:
            chunk_num  += 1
            total_read += len(chunk)

            if chunk_num % 10 == 0:
                print(".", end="", flush=True)

            if vuln_collected >= MAX_VULN and safe_collected >= MAX_SAFE:
                break

            chunk = clean_chunk(chunk)
            if len(chunk) == 0:
                continue

            vuln_part = chunk[chunk["vul"] == 1]
            safe_part  = chunk[chunk["vul"] == 0]

            if vuln_collected < MAX_VULN and len(vuln_part) > 0:
                need = MAX_VULN - vuln_collected
                vuln_chunks.append(vuln_part.head(need))
                vuln_collected += len(vuln_part.head(need))

            if safe_collected < MAX_SAFE and len(safe_part) > 0:
                need = MAX_SAFE - safe_collected
                safe_chunks.append(safe_part.head(need))
                safe_collected += len(safe_part.head(need))

    except Exception as e:
        print(f"\n  ERROR reading CSV: {e}")
        sys.exit(1)

    print(f"\n\n    Chunks read         : {chunk_num:,}")
    print(f"    Total rows seen     : {total_read:,}")
    print(f"    Vulnerable collected: {vuln_collected:,}")
    print(f"    Safe collected      : {safe_collected:,}")

    if vuln_collected == 0:
        print("\n  ⚠️  0 vulnerable rows found — dataset may be fully sorted.")
        print("  Increase MAX_SAFE or read the full file.\n")

    # ── Combine ───────────────────────────────────────────────────────────────
    print("\n[2] Combining, cleaning, deduplicating...")

    df = pd.concat(vuln_chunks + safe_chunks, ignore_index=True)

    # ── Select and rename ─────────────────────────────────────────────────────
    existing_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(existing_map.keys())].rename(columns=existing_map)

    # ── IMPROVEMENT 1: files_changed null fix ─────────────────────────────────
    # NOTE: In Big-Vul, files_changed is a JSON blob like:
    #   {"sha": "...", "filename": "...", "additions": 20, ...}
    # It is NOT a simple integer. We drop it and derive a proxy instead.
    if "files_changed" in df.columns:
        df = df.drop(columns=["files_changed"])
        print("    [files_changed] column dropped — contains JSON, not usable as-is")
    # Proxy: estimate files changed from add_lines + del_lines being nonzero
    # (1 = at least one file changed, which is always true here)
    df["files_changed"] = 1
    print("    [files_changed] set to 1 (proxy — real value was JSON blob)")

    # ── IMPROVEMENT 2: Remove duplicate FUNCTIONS (not commits) ─────────────
    # WHY NOT commit_hash:
    #   Big-Vul is FUNCTION-LEVEL, not commit-level.
    #   Many functions share the same commit_id — that is CORRECT in this dataset.
    #   Deduping by commit_hash collapses almost all safe rows away (as we saw).
    #
    # CORRECT approach: deduplicate on actual function content + label.
    #   This removes truly identical function rows while keeping multiple
    #   different functions from the same commit.
    before_dedup = len(df)
    if "func_before" in df.columns:
        df = df.drop_duplicates(subset=["func_before", "label"], keep="first")
        removed_dupes = before_dedup - len(df)
        print(f"    [dedup] removed {removed_dupes:,} exact duplicate function rows")
        print(f"            (deduped on func_before+label, NOT commit_hash)")
        print(f"            Multiple functions per commit is correct — Big-Vul is function-level")

    # ── Fill other nulls ──────────────────────────────────────────────────────
    df["diff"]       = df["diff"].fillna("")
    df["cwe_id"]     = df["cwe_id"].fillna("Unknown")     if "cwe_id"     in df.columns else "Unknown"
    df["cve_id"]     = df["cve_id"].fillna("N/A")         if "cve_id"     in df.columns else "N/A"
    df["cvss_score"] = df["cvss_score"].fillna(0.0)       if "cvss_score" in df.columns else 0.0

    # ── IMPROVEMENT 3: Clean patch text ──────────────────────────────────────
    print("    [patch] Cleaning git diff headers from patch column...")
    df["diff"] = df["diff"].astype(str).apply(clean_patch)

    # ── Add feature columns ───────────────────────────────────────────────────
    df = add_features(df)

    # ── Shuffle ───────────────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n[3] Final Dataset:")
    print(f"    Rows    : {len(df):,}")
    print(f"    Columns : {df.columns.tolist()}")

    counts = df["label"].value_counts().sort_index()
    vuln_n = counts.get(1, 0)
    safe_n = counts.get(0, 0)
    ratio  = safe_n // max(vuln_n, 1)

    print(f"\n    Label Distribution:")
    for val, cnt in counts.items():
        tag = "VULNERABLE    " if val == 1 else "not vulnerable"
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 3)
        print(f"    label={val}  {tag}  {cnt:>6,}  ({pct:.1f}%)  {bar}")
    print(f"\n    Imbalance ratio : 1:{ratio}")
    print(f"    ⚠️  Use class_weight='balanced' or SMOTE in ML step")

    if "cwe_id" in df.columns and vuln_n > 0:
        print(f"\n    Top CWE types (vulnerable):")
        for cwe, cnt in df[df["label"]==1]["cwe_id"].value_counts().head(6).items():
            print(f"      {str(cwe):<25} {cnt:>4,}")

    if "project" in df.columns:
        print(f"\n    Top projects:")
        for proj, cnt in df["project"].value_counts().head(6).items():
            v = len(df[(df["project"]==proj) & (df["label"]==1)])
            print(f"      {str(proj):<22} total={cnt:>5,}  vuln={v:>4,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n[4] Saved → '{OUTPUT_FILE}'  ({size_mb:.1f} MB)")

    # ── Report ────────────────────────────────────────────────────────────────
    report = [
        "BIG-VUL PREPROCESSING REPORT  v2",
        "=" * 50,
        f"Run time              : {run_time}",
        f"Input                 : {INPUT_FILE}",
        f"Output                : {OUTPUT_FILE}",
        f"Chunk size            : {CHUNK_SIZE}",
        f"Languages             : {KEEP_LANGS}",
        f"Diff limits           : {MIN_DIFF_CHARS}–{MAX_DIFF_CHARS} chars",
        f"Security terms feature: {'ON' if USE_SECURITY_TERMS else 'OFF (ablation)'}",
        "",
        "COLLECTION RESULTS",
        "-" * 50,
        f"Chunks read        : {chunk_num:,}",
        f"Total rows seen    : {total_read:,}",
        f"Vulnerable (vul=1) : {vuln_collected:,}",
        f"Safe (vul=0)       : {safe_collected:,}",
        f"After dedup        : {len(df):,} rows",
        "",
        "LABEL DISTRIBUTION",
        "-" * 50,
        f"  label=1 (vulnerable)    : {vuln_n:,}",
        f"  label=0 (not vulnerable): {safe_n:,}",
        f"  imbalance ratio         : 1:{ratio}",
        "",
        "IMPROVEMENTS APPLIED",
        "-" * 50,
        "  1. files_changed nulls   → filled with 1",
        "  2. Duplicate functions   → removed on func_before+label (NOT commit_hash)",
        "  3. Patch cleaning        → removed +++/---/@@ headers",
        f" 4. Security terms        → {'included (USE_SECURITY_TERMS=True)' if USE_SECURITY_TERMS else 'excluded (ablation mode)'}",
        "",
        "COLUMN MAPPING",
        "-" * 50,
    ]
    for orig, renamed in existing_map.items():
        report.append(f"  {orig:<20} → {renamed}")
    report += [
        "",
        "FEATURE COLUMNS ADDED",
        "-" * 50,
        "  diff_length          (char count of cleaned patch)",
        "  func_before_length   (char count of vulnerable function)",
        f"  has_security_terms   ({'active' if USE_SECURITY_TERMS else 'set to 0 — ablation'})",
        "",
        "RESEARCH NOTE",
        "-" * 50,
        "  has_security_terms uses domain keywords (sql, overflow, etc.)",
        "  This may cause information leakage in the model.",
        "  Recommended: run two experiments:",
        "    Exp A: USE_SECURITY_TERMS = True",
        "    Exp B: USE_SECURITY_TERMS = False",
        "  Compare F1/AUC-ROC. Discuss difference in thesis.",
        "",
        "NEXT STEP",
        "-" * 50,
        "  STEP 3: Baseline ML Model",
        "  Features: lines_added, lines_removed, diff_length,",
        "            func_before_length, has_security_terms",
        "  Target  : label (0/1)",
        "  Models  : Logistic Regression → Random Forest",
        "  Metrics : F1, Precision, Recall, AUC-ROC  (NOT accuracy)",
    ]

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report))
    print(f"    Report → '{REPORT_FILE}'")

    print("\n" + "=" * 62)
    print("  DONE ✅  Next → STEP 3: Baseline ML Model")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()