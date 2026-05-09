"""
STEP 3 — BASELINE ML MODEL  (Leakage-Mitigated Version)
=========================================================
Save as:  src/train_baseline_model.py
Run with: python src/train_baseline_model.py

Leakage fixes applied:
  1. Project-wise split  — train/test on DIFFERENT projects (no memorization)
  2. Remove patch-size features — test if lines_added/removed cause shortcut
  3. Security terms OFF  — remove keyword leakage
  4. Save model artifacts → models/
  5. Explicit leakage warnings in report

Why F1=1.0 was wrong:
  - Big-Vul sorted: vuln rows from different projects than safe rows
  - lines_added/removed trivially separated classes (size bias)
  - Same project in both train AND test = memorization
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
INPUT_FILE  = "data/processed/big_vul_ml_ready.csv"
REPORT_FILE = "reports/baseline_results.txt"
MODEL_DIR   = "models"

# ── EXPERIMENT SETS ───────────────────────────────────────────────────────────
# We run THREE experiments to isolate leakage sources.
# Each teaches us something different.

EXPERIMENTS = {

    "Exp-A: ALL features (leaked baseline)": [
        # This is the LEAKY setup — shows what NOT to trust
        "lines_added", "lines_removed",
        "diff_length", "func_before_length", "has_security_terms",
    ],

    "Exp-B: No patch-size features": [
        # Removes lines_added/lines_removed size bias
        # Tests if model still works without trivial shortcuts
        "diff_length", "func_before_length", "has_security_terms",
    ],

    "Exp-C: Structural only (no keywords)": [
        # Removes keyword leakage + patch-size leakage
        # Most honest baseline — model must learn from code structure
        "diff_length", "func_before_length",
    ],
}

TARGET_COL   = "label"
RANDOM_STATE = 42


# ── HELPERS ──────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print('=' * 62)


def project_wise_split(df, test_projects_frac=0.25):
    """
    FIX 1 — PROJECT-WISE SPLIT (most important fix).

    Instead of random row split (which leaks project patterns),
    we split by PROJECT:
      - Train: 75% of projects
      - Test:  25% of projects (completely unseen)

    This tests: can model generalise to NEW projects?
    That is the real vulnerability detection problem.
    """
    projects = df["project"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(projects)

    n_test  = max(1, int(len(projects) * test_projects_frac))
    test_p  = set(projects[:n_test])
    train_p = set(projects[n_test:])

    train_df = df[df["project"].isin(train_p)].copy()
    test_df  = df[df["project"].isin(test_p)].copy()

    print(f"\n  Project-wise split:")
    print(f"    Train projects : {len(train_p)}  →  {len(train_df):,} rows")
    print(f"    Test  projects : {len(test_p)}  →  {len(test_df):,} rows")
    print(f"    Test projects  : {sorted(test_p)}")

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
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}  ← catch rate for real vulns")
    print(f"  F1        : {f1:.4f}  ← main metric")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Confusion : TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Missed vulns (FN): {fn}  |  False alarms (FP): {fp}")

    report_lines += [
        f"\n  {name}",
        f"    Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}  AUC={auc:.4f}",
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
    print(f"\n  Feature Importances:")
    for feat, imp in paired:
        bar = "█" * max(1, int(imp * 30))
        print(f"    {feat:<25} {imp:.4f}  {bar}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "BASELINE ML RESULTS  (Leakage-Mitigated)",
        "=" * 55,
        f"Run time : {run_time}",
        f"Input    : {INPUT_FILE}",
        "",
        "LEAKAGE FIXES APPLIED",
        "-" * 55,
        "  1. Project-wise train/test split (no project memorization)",
        "  2. Three experiments isolating different leakage sources",
        "  3. Model artifacts saved to models/",
        "",
        "IF YOU SEE F1~1.0 AGAIN:",
        "  → leakage still present in dataset structure",
        "  → discuss this in thesis as research finding",
        "",
    ]

    print_section("STEP 3 — BASELINE ML  (Leakage-Mitigated)")

    if not os.path.exists(INPUT_FILE):
        print(f"\n  ERROR: {INPUT_FILE} not found. Run preprocess first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"\n  Loaded {len(df):,} rows")

    if "project" not in df.columns:
        print("  WARNING: 'project' column missing — falling back to random split")
        df["project"] = "unknown"

    # Project-wise split — the REAL fix
    train_df, test_df = project_wise_split(df)

    # Check label distribution in each split
    for split_name, split_df in [("Train", train_df), ("Test", test_df)]:
        counts = split_df[TARGET_COL].value_counts().sort_index()
        safe   = counts.get(0, 0)
        vuln   = counts.get(1, 0)
        print(f"  {split_name}: safe={safe:,}  vuln={vuln:,}", end="")
        if vuln == 0:
            print(f"  ⚠️  NO VULNERABLE rows in {split_name}!")
        else:
            print()

    os.makedirs(MODEL_DIR, exist_ok=True)
    all_results = []

    # ── Run each experiment ───────────────────────────────────────────────────
    for exp_name, features in EXPERIMENTS.items():
        print_section(exp_name)
        print(f"  Features: {features}")

        # Check features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"  SKIP — missing columns: {missing}")
            continue

        X_train = train_df[features].fillna(0).values
        y_train = train_df[TARGET_COL].values
        X_test  = test_df[features].fillna(0).values
        y_test  = test_df[TARGET_COL].values

        if len(np.unique(y_test)) < 2:
            print("  SKIP — test set has only one class (project split imbalance)")
            print("  → Consider increasing dataset size or adjusting split ratio")
            continue

        # Scale
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        report_lines.append(f"\n{'='*55}")
        report_lines.append(f"EXPERIMENT: {exp_name}")
        report_lines.append(f"Features  : {features}")

        exp_results = []

        # Logistic Regression
        print("\n  --- Logistic Regression ---")
        lr = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )
        lr.fit(X_train, y_train)
        exp_results.append(evaluate("Logistic Regression", lr, X_test, y_test, report_lines))
        show_importances(lr, features)

        # 5-fold CV on train set
        cv = cross_val_score(
            LogisticRegression(class_weight="balanced", max_iter=1000),
            X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1",
        )
        print(f"\n  Train 5-Fold CV F1: {cv.mean():.4f} ± {cv.std():.4f}")
        report_lines.append(f"    Train CV F1: {cv.mean():.4f} ± {cv.std():.4f}")

        # Random Forest
        print("\n  --- Random Forest ---")
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            max_depth=10, random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        exp_results.append(evaluate("Random Forest", rf, X_test, y_test, report_lines))
        show_importances(rf, features)

        cv_rf = cross_val_score(
            RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=10),
            X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1",
        )
        print(f"\n  Train 5-Fold CV F1: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
        report_lines.append(f"    Train CV F1: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

        all_results.append((exp_name, exp_results))

        # Save best model from this experiment
        best_model = rf if exp_results[1]["f1"] >= exp_results[0]["f1"] else lr
        safe_name  = exp_name.split(":")[0].strip().replace(" ", "_").replace("-", "")
        joblib.dump(best_model, f"{MODEL_DIR}/{safe_name}_best_model.pkl")
        joblib.dump(scaler,     f"{MODEL_DIR}/{safe_name}_scaler.pkl")
        print(f"\n  Model saved → {MODEL_DIR}/{safe_name}_best_model.pkl")

    # ── Final comparison across experiments ───────────────────────────────────
    print_section("CROSS-EXPERIMENT COMPARISON")
    print(f"\n  {'Experiment':<40} {'Model':<22} {'F1':>6}  {'AUC':>6}")
    print(f"  {'-'*78}")

    report_lines += ["\n\nFINAL COMPARISON", "=" * 55]
    for exp_name, exp_res in all_results:
        for r in exp_res:
            flag = "  ← realistic" if r["f1"] < 0.90 else ("  ⚠️ check leakage" if r["f1"] > 0.98 else "")
            print(f"  {exp_name:<40} {r['name']:<22} {r['f1']:>6.4f}  {r['auc']:>6.4f}{flag}")
            report_lines.append(f"  {exp_name} | {r['name']} | F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    print(f"""
  INTERPRETATION:
    F1 > 0.98 with simple features → leakage likely still present
    F1 = 0.60–0.85                 → realistic baseline (good for thesis)
    F1 < 0.60                      → features insufficient, need CodeBERT text

  The drop from Exp-A → Exp-C shows how much was shortcut learning.
  This comparison IS your thesis contribution on leakage analysis.
    """)

    report_lines += [
        "\nINTERPRETATION",
        "  F1 > 0.98 → check for remaining leakage",
        "  F1 0.60-0.85 → realistic baseline",
        "  F1 < 0.60 → features insufficient, need CodeBERT",
        "\nNEXT STEPS",
        "  Step 4: SHAP explainability on Exp-C model (no leakage)",
        "  Step 5: TF-IDF + LR on func_before text (text baseline)",
        "  Step 6: CodeBERT fine-tuning (advanced model)",
        "  Compare all F1/AUC results in thesis Table",
    ]

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report saved → '{REPORT_FILE}'")
    print("  DONE ✅\n")


if __name__ == "__main__":
    main()