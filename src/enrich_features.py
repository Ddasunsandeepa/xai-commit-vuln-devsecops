"""
STEP 3b — FEATURE ENRICHMENT
==============================
Save as:  src/enrich_features.py
Run with: python src/enrich_features.py

Problem:
  Exp-C only used 2 features: diff_length, func_before_length
  SHAP showed the model mostly learned "function size"
  A supervisor will ask: "is that all?"

This script adds richer structural features:
  1. cyclomatic_complexity  — number of branches/decisions in function
  2. num_parameters         — how many arguments the function takes
  3. num_function_calls     — how many other functions it calls
  4. nesting_depth          — max depth of nested blocks
  5. lines_added            — from the diff
  6. lines_removed          — from the diff
  7. token_diversity        — unique tokens / total tokens (code variety)
  8. security_keyword_count — count of security-relevant terms (not just 0/1)
  9. comment_ratio          — comments as fraction of total lines

Then reruns the baseline model + SHAP on the enriched feature set.
If func_before_length still dominates → research finding.
If other features emerge → stronger model.
"""

import pandas as pd
import numpy as np
import os
import re
import joblib
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/processed/big_vul_ml_ready.csv"
OUTPUT_FILE  = "data/processed/big_vul_enriched.csv"
REPORT_FILE  = "reports/feature_enrichment_report.txt"
RANDOM_STATE = 42

SECURITY_TERMS = [
    "strcpy", "strcat", "sprintf", "gets", "memcpy", "memset",
    "malloc", "free", "realloc", "alloca",
    "password", "passwd", "secret", "token", "auth", "key",
    "sql", "query", "exec", "eval", "system", "popen",
    "overflow", "injection", "xss", "csrf",
    "setuid", "chmod", "chown",
]


# ── FEATURE EXTRACTORS ───────────────────────────────────────────────────────

def cyclomatic_complexity(code: str) -> int:
    """
    Approximate cyclomatic complexity.
    Counts decision points: if, else if, for, while, case, &&, ||
    CC = decision_points + 1
    """
    try:
        code_lower = code.lower()
        decisions = len(re.findall(
            r'\b(if|else\s+if|for|while|case|catch)\b|\&\&|\|\||\?',
            code_lower
        ))
        return decisions + 1
    except Exception:
        return 1


def count_parameters(code: str) -> int:
    """
    Estimate number of function parameters from the first function signature.
    Looks for pattern: funcname(param1, param2, ...) {
    """
    try:
        # Find first opening paren before a brace
        match = re.search(r'\(([^)]*)\)\s*\{', code[:500])
        if match:
            params = match.group(1).strip()
            if not params or params == "void":
                return 0
            # Count commas + 1
            return len(params.split(","))
        return 0
    except Exception:
        return 0


def count_function_calls(code: str) -> int:
    """
    Count number of function calls: word followed by '('
    Excludes keywords: if, for, while, switch
    """
    try:
        keywords = {"if", "for", "while", "switch", "return", "sizeof", "typeof"}
        calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        return sum(1 for c in calls if c.lower() not in keywords)
    except Exception:
        return 0


def nesting_depth(code: str) -> int:
    """Compute maximum brace nesting depth."""
    try:
        max_depth = 0
        depth = 0
        for ch in code:
            if ch == '{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == '}':
                depth = max(0, depth - 1)
        return max_depth
    except Exception:
        return 0


def token_diversity(code: str) -> float:
    """
    Unique tokens / total tokens.
    High diversity = varied code (complex logic).
    Low diversity = repetitive patterns.
    """
    try:
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)
    except Exception:
        return 0.0


def security_keyword_count(code: str) -> int:
    """Count total occurrences of security-relevant terms (not just 0/1)."""
    try:
        code_lower = code.lower()
        return sum(code_lower.count(term) for term in SECURITY_TERMS)
    except Exception:
        return 0


def comment_ratio(code: str) -> float:
    """
    Fraction of lines that are comments (// or /* style).
    High comment ratio might indicate well-documented safe code.
    """
    try:
        lines = code.splitlines()
        if not lines:
            return 0.0
        comment_lines = sum(
            1 for line in lines
            if line.strip().startswith("//") or line.strip().startswith("*")
               or line.strip().startswith("/*")
        )
        return comment_lines / len(lines)
    except Exception:
        return 0.0


def count_diff_lines(diff: str):
    """Count added and removed lines from the diff."""
    try:
        added   = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
        return added, removed
    except Exception:
        return 0, 0


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 62)
    print("  FEATURE ENRICHMENT")
    print("=" * 62)

    if not os.path.exists(INPUT_FILE):
        print(f"\n  ERROR: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"\n  Loaded {len(df):,} rows")

    func_col = "func_before"
    diff_col = "diff"

    if func_col not in df.columns:
        print(f"  ERROR: '{func_col}' column not found.")
        return

    df[func_col] = df[func_col].fillna("").astype(str)
    df[diff_col] = df[diff_col].fillna("").astype(str) if diff_col in df.columns else ""

    print("\n  Computing enriched features...")
    print("  (this may take 1–2 minutes for 9k rows)")

    # Existing features kept
    df["diff_length"]        = df[diff_col].str.len()
    df["func_before_length"] = df[func_col].str.len()

    # New features
    print("  → cyclomatic_complexity...")
    df["cyclomatic_complexity"]  = df[func_col].apply(cyclomatic_complexity)

    print("  → num_parameters...")
    df["num_parameters"]         = df[func_col].apply(count_parameters)

    print("  → num_function_calls...")
    df["num_function_calls"]     = df[func_col].apply(count_function_calls)

    print("  → nesting_depth...")
    df["nesting_depth"]          = df[func_col].apply(nesting_depth)

    print("  → token_diversity...")
    df["token_diversity"]        = df[func_col].apply(token_diversity)

    print("  → security_keyword_count...")
    df["security_keyword_count"] = df[func_col].apply(security_keyword_count)

    print("  → comment_ratio...")
    df["comment_ratio"]          = df[func_col].apply(comment_ratio)

    print("  → diff lines added/removed...")
    diff_lines = df[diff_col].apply(count_diff_lines)
    df["lines_added"]   = diff_lines.apply(lambda x: x[0])
    df["lines_removed"] = diff_lines.apply(lambda x: x[1])

    # ── Show stats ────────────────────────────────────────────────────────────
    NEW_FEATURES = [
        "diff_length", "func_before_length",
        "cyclomatic_complexity", "num_parameters",
        "num_function_calls", "nesting_depth",
        "token_diversity", "security_keyword_count",
        "comment_ratio", "lines_added", "lines_removed",
    ]

    print(f"\n  Feature statistics by label:")
    print(f"\n  {'Feature':<25} {'Vuln mean':>12} {'Safe mean':>12} {'Difference':>12}")
    print(f"  {'-' * 64}")

    report_lines = [
        "FEATURE ENRICHMENT REPORT",
        "=" * 55,
        f"Run time : {run_time}",
        "",
        "NEW FEATURES ADDED",
        "-" * 55,
    ]

    vuln_df = df[df["label"] == 1]
    safe_df = df[df["label"] == 0]

    for feat in NEW_FEATURES:
        if feat not in df.columns:
            continue
        vuln_mean = vuln_df[feat].mean()
        safe_mean = safe_df[feat].mean()
        diff      = vuln_mean - safe_mean
        flag      = "↑" if diff > 0 else "↓"
        print(f"  {feat:<25} {vuln_mean:>12.2f} {safe_mean:>12.2f} {diff:>+12.2f}  {flag}")
        report_lines.append(f"  {feat:<25}  vuln={vuln_mean:.2f}  safe={safe_mean:.2f}  diff={diff:+.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024*1024)
    print(f"\n  Saved → '{OUTPUT_FILE}'  ({size_mb:.1f} MB)")

    report_lines += [
        "",
        "NEXT STEP",
        "  Re-run train_baseline_model.py with INPUT_FILE = big_vul_enriched.csv",
        "  Update FEATURE_COLS to include all new features",
        "  Re-run explain_shap.py to see if func_before_length still dominates",
        "  If it does → research finding (structure = key signal)",
        "  If other features emerge → stronger model",
    ]

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report → '{REPORT_FILE}'")
    print("\n  DONE ✅")
    print("\n  Next: update FEATURE_COLS in train_baseline_model.py to:")
    print(f"  {NEW_FEATURES}\n")


if __name__ == "__main__":
    main()