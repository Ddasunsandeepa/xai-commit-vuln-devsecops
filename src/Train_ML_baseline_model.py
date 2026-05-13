"""
STEP 3 — BASELINE ML MODEL  (Enriched Features Version)
=========================================================
Save as:  src/train_baseline_model.py
Run with: python src/train_baseline_model.py

Changes from previous version:
  - INPUT_FILE now points to big_vul_enriched.csv (11 features)
  - EXPERIMENTS updated to use enriched feature set
  - Exp-A: all 11 features (check for leakage with richer set)
  - Exp-B: no patch-size features (lines_added/removed removed)
  - Exp-C: structural + complexity (honest enriched baseline)
  - Exp-D: complexity only (pure code metrics, no size at all)
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/processed/big_vul_enriched.csv"   # ← enriched dataset
REPORT_FILE = "reports/baseline_results_enriched.txt"
MODEL_DIR   = "models"
TARGET_COL  = "label"
RANDOM_STATE = 42

# ── ENRICHED EXPERIMENT SETS ─────────────────────────────────────────────────
EXPERIMENTS = {

    "Exp-A: All 11 features": [
        # Shows what leakage looks like with enriched feature set
        "diff_length", "func_before_length",
        "cyclomatic_complexity", "num_parameters", "num_function_calls",
        "nesting_depth", "token_diversity", "security_keyword_count",
        "comment_ratio", "lines_added", "lines_removed",
    ],

    "Exp-B: No patch-size (lines_added/removed)": [
        # Removes the known size-bias shortcuts
        "diff_length", "func_before_length",
        "cyclomatic_complexity", "num_parameters", "num_function_calls",
        "nesting_depth", "token_diversity", "security_keyword_count",
        "comment_ratio",
    ],

    "Exp-C: Structural + complexity (honest enriched)": [
        # No size shortcuts, no security keywords
        # Pure software engineering metrics
        "func_before_length", "cyclomatic_complexity",
        "num_parameters", "num_function_calls",
        "nesting_depth", "token_diversity", "comment_ratio",
    ],

    "Exp-D: Complexity only (no size at all)": [
        # Completely removes function length
        # Tests: does complexity alone predict vulnerability?
        "cyclomatic_complexity", "num_parameters", "num_function_calls",
        "nesting_depth", "token_diversity", "comment_ratio",
    ],
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
    train_df = df[df["project"].isin(train_p)].copy()
    test_df  = df[df["project"].isin(test_p)].copy()
    print(f"\n  Project-wise split:")
    print(f"    Train: {len(train_p)} projects → {len(train_df):,} rows")
    print(f"    Test : {len(test_p)} projects → {len(test_df):,} rows")
    return train_df, test_df


def evaluate(name, model, X_test, y_test, report_lines):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    p   = precision_score(y_test, y_pred, zero_division=0)
    r   = recall_score(y_test, y_pred, zero_division=0)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  {name}")
    print(f"  Precision : {p:.4f}  Recall : {r:.4f}  F1 : {f1:.4f}  AUC : {auc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    report_lines += [
        f"\n  {name}",
        f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  AUC={auc:.4f}",
        f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}",
    ]
    return {"name": name, "p": p, "r": r, "f1": f1, "auc": auc}


def show_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(model.coef_[0])
    else:
        return
    paired = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature Importances (top 6):")
    for feat, imp in paired[:6]:
        bar = "█" * max(1, int(imp * 40))
        print(f"    {feat:<30} {imp:.4f}  {bar}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "BASELINE ML RESULTS — ENRICHED FEATURES",
        "=" * 55,
        f"Run time : {run_time}",
        f"Input    : {INPUT_FILE}",
        "",
        "ENRICHED FEATURES USED",
        "-" * 55,
        "  diff_length, func_before_length, cyclomatic_complexity,",
        "  num_parameters, num_function_calls, nesting_depth,",
        "  token_diversity, security_keyword_count, comment_ratio,",
        "  lines_added, lines_removed",
        "",
    ]

    print_section("STEP 3 — BASELINE ML  (Enriched 11 Features)")

    if not os.path.exists(INPUT_FILE):
        print(f"\n  ERROR: {INPUT_FILE} not found.")
        print("  Run src/enrich_features.py first.\n")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"\n  Loaded {len(df):,} rows  |  Columns: {len(df.columns)}")

    train_df, test_df = project_wise_split(df)

    for split_name, split_df in [("Train", train_df), ("Test", test_df)]:
        counts = split_df[TARGET_COL].value_counts().sort_index()
        print(f"  {split_name}: safe={counts.get(0,0):,}  vuln={counts.get(1,0):,}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    all_results = []

    for exp_name, features in EXPERIMENTS.items():
        print_section(exp_name)
        print(f"  Features ({len(features)}): {features}")

        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"  SKIP — missing: {missing}")
            continue

        X_train = train_df[features].fillna(0).values
        y_train = train_df[TARGET_COL].values
        X_test  = test_df[features].fillna(0).values
        y_test  = test_df[TARGET_COL].values

        if len(np.unique(y_test)) < 2:
            print("  SKIP — test set has only one class")
            continue

        scaler  = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        report_lines += [f"\n{'='*55}", f"EXPERIMENT: {exp_name}", f"Features: {features}"]
        exp_results = []

        # Logistic Regression
        print("\n  --- Logistic Regression ---")
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_train_s, y_train)
        exp_results.append(evaluate("Logistic Regression", lr, X_test_s, y_test, report_lines))
        show_importances(lr, features)
        cv = cross_val_score(
            LogisticRegression(class_weight="balanced", max_iter=1000),
            X_train_s, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="f1",
        )
        print(f"\n  Train CV F1: {cv.mean():.4f} ± {cv.std():.4f}")
        report_lines.append(f"    LR Train CV F1: {cv.mean():.4f} ± {cv.std():.4f}")

        # Random Forest
        print("\n  --- Random Forest ---")
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            max_depth=10, random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_train_s, y_train)
        exp_results.append(evaluate("Random Forest", rf, X_test_s, y_test, report_lines))
        show_importances(rf, features)
        cv_rf = cross_val_score(
            RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=10),
            X_train_s, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="f1",
        )
        print(f"\n  Train CV F1: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
        report_lines.append(f"    RF Train CV F1: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

        all_results.append((exp_name, exp_results))

        # Save best model
        best_model = rf if exp_results[1]["f1"] >= exp_results[0]["f1"] else lr
        best_scaler = scaler
        tag = exp_name.split(":")[0].strip().replace(" ", "_").replace("-", "")
        joblib.dump(best_model,  f"{MODEL_DIR}/{tag}_best_model.pkl")
        joblib.dump(best_scaler, f"{MODEL_DIR}/{tag}_scaler.pkl")
        print(f"\n  Saved → models/{tag}_best_model.pkl")

    # ── Final comparison ──────────────────────────────────────────────────────
    print_section("CROSS-EXPERIMENT COMPARISON")
    print(f"\n  {'Experiment':<45} {'Model':<22} {'F1':>6}  {'AUC':>6}")
    print(f"  {'-'*82}")

    report_lines += ["\n\nFINAL COMPARISON", "=" * 55]
    for exp_name, exp_res in all_results:
        for r in exp_res:
            flag = "  ⚠️ check leakage" if r["f1"] > 0.95 else ("  ← realistic" if r["f1"] < 0.85 else "")
            print(f"  {exp_name:<45} {r['name']:<22} {r['f1']:>6.4f}  {r['auc']:>6.4f}{flag}")
            report_lines.append(f"  {exp_name} | {r['name']} | F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    print(f"""
  KEY QUESTION: Does enriched Exp-C beat old Exp-C (F1=0.494)?
    If yes → richer features genuinely help
    If same → function size still dominates
    Either way → valid thesis finding
    """)

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report → '{REPORT_FILE}'\n  DONE ✅\n")


if __name__ == "__main__":
    main()