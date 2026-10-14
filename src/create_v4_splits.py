#!/usr/bin/env python3
"""Create immutable, row-aware, project-disjoint Big-Vul evaluation splits.

Run from repository root:
  python experiments/create_v4_splits.py
  python experiments/create_v4_splits.py --seeds 42 7 21 84 123

Outputs are independent of downstream model training. Existing manifests are
verified and reused, never silently overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_KEYS = ["project", "label", "func_before"]


def digest(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def project_table(df):
    t = df.groupby("project", sort=True).agg(rows=("label", "size"), positives=("label", "sum"))
    t["negatives"] = t.rows - t.positives
    return t


def allocate(df, seed, attempts, fractions, min_rows, max_share, prevalence_tolerance):
    stats = project_table(df)
    projects = stats.index.to_numpy()
    n = len(df)
    global_prev = df.label.mean()
    target = np.asarray(fractions, dtype=float)
    target_rows = target * n
    rng = np.random.default_rng(seed)
    best = None
    best_score = np.inf
    # Stochastic greedy bin packing: largest projects placed first, with
    # randomized tie/order perturbations. A candidate is selected using only
    # project sizes and class counts, never model predictions.
    for attempt in range(attempts):
        jitter = rng.uniform(0.55, 1.45, size=len(projects))
        order = np.argsort(-(stats.rows.to_numpy() * jitter))
        groups = [[], [], []]
        sizes = np.zeros(3, dtype=int)
        positives = np.zeros(3, dtype=int)
        for ix in order:
            p = projects[ix]
            count = int(stats.loc[p, "rows"])
            pos = int(stats.loc[p, "positives"])
            options = []
            for k in range(3):
                new_sizes = sizes.copy(); new_sizes[k] += count
                new_pos = positives.copy(); new_pos[k] += pos
                row_loss = np.sum(((new_sizes - target_rows) / n) ** 2)
                # Small prevalence weight keeps prevalence reasonably similar,
                # without treating the target prevalence as an exact constraint.
                prevalence_loss = sum(
                    ((new_pos[j] / new_sizes[j] - global_prev) ** 2) * 0.04
                    for j in range(3) if new_sizes[j]
                )
                options.append(row_loss + prevalence_loss + rng.uniform(0, 0.00015))
            k = int(np.argmin(options))
            groups[k].append(str(p)); sizes[k] += count; positives[k] += pos
        if np.any(sizes == 0):
            continue
        rates = positives / sizes
        if np.any(positives == 0) or np.any(positives == sizes):
            continue
        if sizes[1] < min_rows[0] or sizes[2] < min_rows[1]:
            continue
        if abs(rates[1] - global_prev) > prevalence_tolerance or abs(rates[2] - global_prev) > prevalence_tolerance:
            continue
        shares = [max(stats.loc[groups[k], "rows"]) / sizes[k] for k in (1, 2)]
        if shares[0] > max_share[0] or shares[1] > max_share[1]:
            continue
        score = np.sum(((sizes / n - target) ** 2)) + 0.2 * np.sum((rates - global_prev) ** 2)
        if score < best_score:
            best_score = score
            best = groups
    if best is None:
        raise RuntimeError(
            f"Seed {seed}: no valid split in {attempts} attempts. "
            "Inspect project sizes; increase attempts or explicitly revise constraints."
        )
    return best, best_score


def audit_duplicates(df, assignments):
    report = {}
    for col in ("commit_hash", "func_before"):
        if col not in df.columns:
            report[col] = {"status": "column_missing"}
            continue
        series = df[col].fillna("").astype(str).str.strip()
        if col == "func_before":
            # Exact normalized code, not fuzzy near-duplicates.
            series = series.map(lambda s: hashlib.sha256(s.encode()).hexdigest() if s else "")
        tmp = pd.DataFrame({"key": series, "split": assignments})
        tmp = tmp[tmp.key != ""]
        shared = tmp.groupby("key").split.nunique()
        shared_keys = set(shared[shared > 1].index)
        report[col] = {
            "cross_split_unique_keys": len(shared_keys),
            "cross_split_rows": int(tmp.key.isin(shared_keys).sum()),
            "examples": list(sorted(shared_keys))[:5] if col == "commit_hash" else [],
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/big_vul_enriched.csv")
    ap.add_argument("--out", default="reports/v4_splits")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 21, 84, 123])
    ap.add_argument("--attempts", type=int, default=3000)
    ap.add_argument("--min-val", type=int, default=650)
    ap.add_argument("--min-test", type=int, default=1200)
    ap.add_argument("--max-val-share", type=float, default=0.40)
    ap.add_argument("--max-test-share", type=float, default=0.35)
    ap.add_argument("--prevalence-tolerance", type=float, default=0.12)
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    missing = set(FEATURE_KEYS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[FEATURE_KEYS].isna().any().any():
        raise ValueError("Missing project/label/func_before; fix preprocessing before splitting")
    if not set(df.label.unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0/1")
    if df.project.astype(str).str.strip().eq("").any():
        raise ValueError("Empty project IDs")
    df = df.reset_index(drop=True)
    source_sha = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {len(df):,} rows, {df.project.nunique()} projects, positive={df.label.mean():.3f}")
    print(f"SHA256: {source_sha}")
    for seed in args.seeds:
        groups, score = allocate(df, seed, args.attempts, [0.70, 0.10, 0.20],
                                 [args.min_val, args.min_test],
                                 [args.max_val_share, args.max_test_share],
                                 args.prevalence_tolerance)
        names = ("train", "validation", "test")
        mapping = {project: name for name, group in zip(names, groups) for project in group}
        assignments = df.project.map(mapping)
        assert assignments.notna().all()
        assert len(mapping) == df.project.nunique()
        report = {"seed": seed, "source_sha256": source_sha,
                  "split_algorithm": "v4_stochastic_greedy_v1",
                  "score": score, "splits": {}, "duplicate_audit": audit_duplicates(df, assignments)}
        for name, group in zip(names, groups):
            sub = df[assignments == name]
            counts = sub.project.value_counts()
            report["splits"][name] = {
                "rows": len(sub), "projects_count": len(group),
                "positive_rate": float(sub.label.mean()),
                "positive_rows": int(sub.label.sum()),
                "negative_rows": int(len(sub) - sub.label.sum()),
                "largest_project": str(counts.index[0]),
                "largest_project_share": float(counts.iloc[0] / len(sub)),
                "projects": sorted(group),
                "row_indices_sha256": digest(sub.index.tolist()),
            }
        path = out / f"seed_{seed}.json"
        content = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"{path} exists with different contents. Refusing overwrite.")
            print(f"Seed {seed}: existing manifest verified")
        else:
            path.write_text(content, encoding="utf-8")
        print(f"\nSEED {seed} | split score={score:.6f}")
        for name in names:
            s = report["splits"][name]
            print(f"  {name:10} rows={s['rows']:5} projects={s['projects_count']:3} "
                  f"positive={s['positive_rate']:.3f} largest={s['largest_project_share']:.1%}")
        for key, value in report["duplicate_audit"].items():
            print(f"  duplicate audit {key}: {value}")
        print(f"  Manifest: {path}")
    print("\nDone. Audit any cross-split duplicates before model training.")


if __name__ == "__main__":
    main()
