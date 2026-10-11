#!/usr/bin/env python3
"""Reproducible CodeJIT pilot input-ablation baseline.

Usage:
 python src/run_codejit_input_ablation.py \
   --input reports/codejit_pilot_200/pilot_commits.csv \
   --out reports/codejit_input_ablation

Dependencies: pandas, numpy, scikit-learn.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def extract_changed_code(patch: str) -> str:
    """Keep +/- changed lines; exclude +++ and --- patch file headers."""
    lines = []
    for line in str(patch).splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            lines.append(line)
    return "\n".join(lines)


def run(input_path: Path, output_dir: Path, seed: int, folds_count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    required = {"repo", "commit_id", "label", "status", "diff_truncated", "diff", "commit_message"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    initial_rows = len(df)
    # Parse boolean reliably even if the CSV column was loaded as text.
    truncated = df["diff_truncated"].astype(str).str.strip().str.lower().eq("true")
    df = df.loc[df["status"].eq("ok") & ~truncated].copy().reset_index(drop=True)
    after_status_filter = len(df)
    df["commit_message"] = df["commit_message"].fillna("").astype(str)
    df["diff"] = df["diff"].fillna("").astype(str)
    df["code_input"] = df["diff"].map(extract_changed_code)
    df["message_input"] = df["commit_message"]
    df["combined_input"] = (
        "COMMIT_MESSAGE\n" + df["message_input"] + "\nCODE_CHANGES\n" + df["code_input"]
    )
    df = df.loc[df["code_input"].str.strip().ne("")].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable records after filtering.")
    if df.duplicated(["repo", "commit_id"]).any():
        raise ValueError("Duplicate repo/commit keys detected; stop and investigate.")
    y = df["label"].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1.")
    if min(np.bincount(y)) < folds_count:
        raise ValueError("Each label must have at least n_splits examples.")

    cv = StratifiedKFold(n_splits=folds_count, shuffle=True, random_state=seed)
    folds = list(cv.split(df, y))
    manifest = df[["repo", "commit_id", "label"]].copy()
    manifest["fold"] = -1
    for fold_id, (_, test_idx) in enumerate(folds):
        manifest.loc[test_idx, "fold"] = fold_id
    manifest.to_csv(output_dir / "fold_manifest.csv", index=False)
    manifest.drop(columns=["fold"]).to_csv(output_dir / "matched_commits.csv", index=False)

    experiments = {
        "message_only": "message_input",
        "code_only": "code_input",
        "message_plus_code": "combined_input",
    }
    summary = []
    for name, column in experiments.items():
        X = df[column].to_numpy(dtype=str)
        probabilities = np.zeros(len(df), dtype=float)
        for train_idx, test_idx in folds:
            model = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ])
            model.fit(X[train_idx], y[train_idx])
            probabilities[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        cm = confusion_matrix(y, predictions, labels=[0, 1])
        metrics = {
            "experiment": name,
            "rows": int(len(df)),
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "macro_f1": float(f1_score(y, predictions, average="macro")),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "pr_auc": float(average_precision_score(y, probabilities)),
        }
        summary.append(metrics)
        predictions_df = manifest[["repo", "commit_id", "label", "fold"]].copy()
        predictions_df["probability"] = probabilities
        predictions_df["prediction"] = predictions
        predictions_df.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
            output_dir / f"{name}_confusion_matrix.csv"
        )
        print(f"\n=== {name} ===\n{json.dumps(metrics, indent=2)}\nConfusion matrix:\n{cm}")

    pd.DataFrame(summary).to_csv(output_dir / "comparison.csv", index=False)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path),
        "original_rows": initial_rows,
        "rows_after_success_and_truncation_filter": after_status_filter,
        "matched_rows": len(df),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "seed": seed,
        "folds": folds_count,
        "cv": "StratifiedKFold; identical folds for all three inputs; out-of-fold predictions",
        "model": "TF-IDF 1-2 grams, min_df=2, max_features=5000; balanced LogisticRegression max_iter=1000",
        "input_representation": "changed +/- lines only, excluding +++/--- file headers; commit message text; concatenation",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "limitations": [
            "Pilot is nonrandom and not representative of full CodeJIT dataset.",
            "Dataset label provenance must be verified before interpreting vulnerability detection.",
            "Truncated patches excluded; exclusion is label-imbalanced.",
            "Only changed lines are modeled, not full patch context or graph structure.",
            "No separate held-out external/project-scale deployment evaluation.",
            "CV is stratified by label; not chronological or representative of project distributions.",
        ],
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\n=== FAIR INPUT COMPARISON ===")
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(f"\nSaved reports to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/codejit_pilot_200/pilot_commits.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/codejit_input_ablation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    run(args.input, args.out, args.seed, args.folds)


if __name__ == "__main__":
    main()
