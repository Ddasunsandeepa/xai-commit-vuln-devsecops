"""
STEP 4 — SHAP EXPLAINABILITY  (Enriched 11-Feature Version)
=============================================================
Save as:  src/explain_shap.py
Run with: python src/explain_shap.py

Changes from previous version:
  - Uses big_vul_enriched.csv (11 features)
  - Loads Exp-C enriched model (structural + complexity)
  - SHAP now explains ALL 11 features
  - Shows which software engineering metrics matter most
  - More meaningful thesis finding than 2-feature version
"""

import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/processed/big_vul_enriched.csv"
MODEL_FILE  = "models/ExpC_best_model.pkl"
SCALER_FILE = "models/ExpC_scaler.pkl"
REPORT_FILE = "reports/shap_explanation_report_enriched.txt"
PLOTS_DIR   = "reports/shap_plots_enriched"

# All 11 features — matches Exp-C enriched in train_baseline_model.py
FEATURE_COLS = [
    "func_before_length", "cyclomatic_complexity",
    "num_parameters", "num_function_calls",
    "nesting_depth", "token_diversity", "comment_ratio",
]

TARGET_COL   = "label"
RANDOM_STATE = 42
MAX_EXPLAIN  = 500

FEATURE_DESC = {
    "diff_length":            "Length of code patch (chars)",
    "func_before_length":     "Length of vulnerable function (chars)",
    "cyclomatic_complexity":  "Number of branches/decision points",
    "num_parameters":         "Number of function arguments",
    "num_function_calls":     "Number of calls to other functions",
    "nesting_depth":          "Maximum brace nesting depth",
    "token_diversity":        "Unique tokens / total tokens (code variety)",
    "security_keyword_count": "Count of security-sensitive API calls",
    "comment_ratio":          "Fraction of lines that are comments",
    "lines_added":            "Lines added in the patch",
    "lines_removed":          "Lines removed in the patch",
}


# ── HELPERS ──────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print('=' * 62)


def project_wise_split(df, test_frac=0.25):
    projects = df["project"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(projects)
    n_test  = max(1, int(len(projects) * test_frac))
    test_p  = set(projects[:n_test])
    train_p = set(projects[n_test:])
    return df[df["project"].isin(train_p)].copy(), df[df["project"].isin(test_p)].copy()


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    report_lines = [
        "SHAP EXPLAINABILITY REPORT — ENRICHED FEATURES",
        "=" * 55,
        f"Run time : {run_time}",
        f"Model    : {MODEL_FILE}",
        f"Features : {FEATURE_COLS}",
        "",
        "SHAP values: positive = toward VULNERABLE, negative = toward SAFE",
        "",
    ]

    print_section("STEP 4 — SHAP EXPLAINABILITY  (11 Features)")

    if not os.path.exists(INPUT_FILE):
        print(f"  ERROR: {INPUT_FILE} not found. Run enrich_features.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"\n  Loaded {len(df):,} rows")

    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        return

    train_df, test_df = project_wise_split(df)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train = train_df[FEATURE_COLS].fillna(0).values
    X_test  = test_df[FEATURE_COLS].fillna(0).values
    y_test  = test_df[TARGET_COL].values

    if not os.path.exists(MODEL_FILE):
        print(f"\n  ERROR: {MODEL_FILE} not found.")
        print("  Run train_baseline_model.py first — it saves the Exp-C model.")
        return

    model = joblib.load(MODEL_FILE)
    print(f"  Model: {type(model).__name__}")

    if os.path.exists(SCALER_FILE):
        scaler  = joblib.load(SCALER_FILE)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)
        print("  Scaler applied")
    else:
        print("  No scaler — using raw features")

    # Sample test set
    np.random.seed(RANDOM_STATE)
    if len(X_test) > MAX_EXPLAIN:
        idx       = np.random.choice(len(X_test), MAX_EXPLAIN, replace=False)
        X_explain = X_test[idx]
        y_explain = y_test[idx]
    else:
        X_explain = X_test
        y_explain = y_test

    print(f"  Explaining {len(X_explain)} samples")

    # ── Compute SHAP values ───────────────────────────────────────────────────
    print("\n  Computing SHAP values...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    if isinstance(model, LogisticRegression):
        explainer   = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_explain)
        print("  LinearExplainer (exact)")
    elif isinstance(model, RandomForestClassifier):
        explainer     = shap.TreeExplainer(model)
        shap_vals_all = explainer.shap_values(X_explain)
        shap_values   = shap_vals_all[1] if isinstance(shap_vals_all, list) else shap_vals_all
        print("  TreeExplainer (exact)")
    else:
        background  = shap.sample(X_train, 100, random_state=RANDOM_STATE)
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_explain[:50])[:, :, 1]
        X_explain   = X_explain[:50]
        y_explain   = y_explain[:50]
        print("  KernelExplainer (fallback)")

    # ── Global importance ─────────────────────────────────────────────────────
    print_section("GLOBAL FEATURE IMPORTANCE")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1]

    print(f"\n  {'Rank':<6} {'Feature':<30} {'Mean |SHAP|':>12}  Description")
    print(f"  {'-' * 80}")
    report_lines.append("\nGLOBAL FEATURE IMPORTANCE")
    report_lines.append("-" * 55)

    for rank, i in enumerate(sorted_idx):
        feat = FEATURE_COLS[i]
        imp  = mean_abs_shap[i]
        bar  = "█" * max(1, int(imp * 30))
        desc = FEATURE_DESC.get(feat, feat)
        print(f"  #{rank+1:<5} {feat:<30} {imp:>12.4f}  {desc}")
        report_lines.append(f"  #{rank+1}  {feat:<30}  {imp:.4f}  — {desc}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")

    # Beeswarm
    plt.figure(figsize=(10, max(4, len(FEATURE_COLS) * 0.6)))
    shap.summary_plot(shap_values, X_explain, feature_names=FEATURE_COLS,
                      show=False, plot_type="dot", max_display=len(FEATURE_COLS))
    plt.title("SHAP Feature Impact — Enriched Structural Model", fontsize=12, pad=10)
    plt.tight_layout()
    p1 = f"{PLOTS_DIR}/shap_beeswarm_enriched.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm → {p1}")

    # Bar
    plt.figure(figsize=(9, max(3, len(FEATURE_COLS) * 0.5)))
    shap.summary_plot(shap_values, X_explain, feature_names=FEATURE_COLS,
                      show=False, plot_type="bar", max_display=len(FEATURE_COLS))
    plt.title("Mean SHAP Feature Importance — Enriched Model", fontsize=12, pad=10)
    plt.tight_layout()
    p2 = f"{PLOTS_DIR}/shap_bar_enriched.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar     → {p2}")

    # ── Local explanations ────────────────────────────────────────────────────
    print_section("LOCAL EXPLANATIONS  (TP / FP / TN / FN)")

    y_pred  = (model.predict_proba(X_explain)[:, 1] >= 0.5).astype(int)
    tp_idx  = np.where((y_pred == 1) & (y_explain == 1))[0]
    fp_idx  = np.where((y_pred == 1) & (y_explain == 0))[0]
    tn_idx  = np.where((y_pred == 0) & (y_explain == 0))[0]
    fn_idx  = np.where((y_pred == 0) & (y_explain == 1))[0]

    cases = [
        ("TRUE POSITIVE  — correctly flagged vulnerable", tp_idx, "TP"),
        ("FALSE POSITIVE — wrongly flagged vulnerable",   fp_idx, "FP"),
        ("TRUE NEGATIVE  — correctly flagged safe",       tn_idx, "TN"),
        ("FALSE NEGATIVE — missed vulnerability",         fn_idx, "FN"),
    ]

    report_lines.append("\n\nLOCAL EXPLANATIONS")
    report_lines.append("-" * 55)

    for case_name, indices, label in cases:
        if len(indices) == 0:
            print(f"  [{label}] No examples found")
            continue
        i    = indices[0]
        prob = model.predict_proba(X_explain[i:i+1])[0, 1]
        sv   = shap_values[i]

        print(f"\n  [{label}] {case_name}")
        print(f"  Vuln probability: {prob:.4f}")
        # Show top 5 contributors
        top_contrib = sorted(zip(FEATURE_COLS, X_explain[i], sv),
                             key=lambda x: abs(x[2]), reverse=True)[:5]
        for feat, val, shap_val in top_contrib:
            direction = "→ VULN" if shap_val > 0 else "→ SAFE"
            bar = ("▲" if shap_val > 0 else "▼") * min(15, max(1, int(abs(shap_val) * 10)))
            print(f"    {feat:<30} scaled={val:>6.2f}  SHAP={shap_val:>+.4f}  {direction}  {bar}")

        report_lines += [f"\n  [{label}] {case_name}", f"  Vuln prob: {prob:.4f}"]
        for feat, val, shap_val in top_contrib:
            report_lines.append(f"    {feat}: SHAP={shap_val:+.4f}")

    # Waterfall for TP
    if len(tp_idx) > 0:
        try:
            i = tp_idx[0]
            exp_obj = shap.Explanation(
                values        = shap_values[i],
                base_values   = explainer.expected_value if hasattr(explainer, "expected_value") else 0,
                data          = X_explain[i],
                feature_names = FEATURE_COLS,
            )
            plt.figure(figsize=(8, max(4, len(FEATURE_COLS) * 0.5)))
            shap.waterfall_plot(exp_obj, show=False)
            plt.title("SHAP Waterfall — True Positive (Enriched Model)", fontsize=11)
            plt.tight_layout()
            p3 = f"{PLOTS_DIR}/shap_waterfall_enriched.png"
            plt.savefig(p3, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  Waterfall → {p3}")
        except Exception as e:
            print(f"  Waterfall skipped: {e}")

    # ── SHAP by class ─────────────────────────────────────────────────────────
    print_section("SHAP BY CLASS  (vuln vs safe)")

    vuln_mask = y_explain == 1
    safe_mask = y_explain == 0

    if vuln_mask.sum() > 0 and safe_mask.sum() > 0:
        print(f"\n  {'Feature':<30} {'Vuln SHAP':>12} {'Safe SHAP':>12} {'Gap':>10}")
        print(f"  {'-' * 68}")
        report_lines.append("\n\nSHAP BY CLASS")
        report_lines.append("-" * 55)
        for i, feat in enumerate(FEATURE_COLS):
            v = shap_values[vuln_mask, i].mean()
            s = shap_values[safe_mask, i].mean()
            d = v - s
            flag = "↑ vuln" if d > 0 else "↓ safe"
            print(f"  {feat:<30} {v:>+12.4f} {s:>+12.4f} {d:>+10.4f}  {flag}")
            report_lines.append(f"  {feat}: vuln={v:+.4f}  safe={s:+.4f}  diff={d:+.4f}")

    # ── Research insights ─────────────────────────────────────────────────────
    print_section("KEY THESIS INSIGHTS")
    top_feat = FEATURE_COLS[sorted_idx[0]]
    print(f"""
  Most important feature: {top_feat}
  Mean |SHAP| = {mean_abs_shap[sorted_idx[0]]:.4f}

  COMPARE WITH 2-FEATURE VERSION:
  - Old: only func_before_length and diff_length
  - New: {len(FEATURE_COLS)} software engineering metrics
  - If {top_feat} still dominates → "function complexity
    is the primary structural vulnerability indicator"
  - If complexity features emerge → richer model, stronger thesis

  USE IN THESIS:
  - Beeswarm plot: shows all features by SHAP impact
  - Bar plot: clean summary for methodology section
  - Waterfall: one specific prediction explained
  - SHAP by class table: systematic vulnerability patterns
    """)

    report_lines += [
        "\n\nPLOTS SAVED",
        f"  {p1}",
        f"  {p2}",
        f"  {PLOTS_DIR}/shap_waterfall_enriched.png",
        "\nNEXT STEP",
        "  → CodeBERT fine-tuning on func_before text",
    ]

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report → '{REPORT_FILE}'")
    print(f"  Plots  → '{PLOTS_DIR}/'")
    print("\n  DONE ✅  Enriched SHAP complete.\n")


if __name__ == "__main__":
    main()