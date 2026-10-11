#!/usr/bin/env python3
"""Analyze paired out-of-fold CodeJIT input-ablation predictions.

Run from repository root:
  python src/analyze_codejit_ablation.py \
    --predictions-dir reports/codejit_input_ablation \
    --pilot-csv reports/codejit_pilot_200/pilot_commits.csv \
    --out reports/codejit_ablation_analysis --bootstrap 3000 --seed 42

Dependencies: numpy, pandas, scikit-learn, scipy (optional; McNemar exact
p-value computed using Python standard library if scipy is absent).
"""
import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)

MODELS = ('message_only', 'code_only', 'message_plus_code')
KEYS = ['repo', 'commit_id']
METRICS = ('accuracy', 'f1', 'macro_f1', 'roc_auc', 'pr_auc')


def exact_mcnemar_p(b, c):
    """Two-sided exact McNemar p-value for discordant counts b,c."""
    n = int(b + c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def scores(y, pred, prob):
    result = {
        'accuracy': float(accuracy_score(y, pred)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'roc_auc': float('nan'),
        'pr_auc': float('nan'),
    }
    if len(np.unique(y)) == 2:
        result['roc_auc'] = float(roc_auc_score(y, prob))
        result['pr_auc'] = float(average_precision_score(y, prob))
    return result


def read_predictions(directory):
    tables = {}
    reference = None
    for model in MODELS:
        path = directory / f'{model}_predictions.csv'
        table = pd.read_csv(path, dtype={'repo': str, 'commit_id': str})
        required = set(KEYS + ['label', 'prediction', 'probability'])
        if not required.issubset(table.columns):
            raise ValueError(f'{path}: missing {sorted(required - set(table.columns))}')
        if table.duplicated(KEYS).any():
            raise ValueError(f'{path}: duplicate repository-commit identifiers')
        if not table['label'].isin([0, 1]).all() or not table['prediction'].isin([0, 1]).all():
            raise ValueError(f'{path}: labels and predictions must be binary')
        if table['probability'].isna().any() or not table['probability'].between(0, 1).all():
            raise ValueError(f'{path}: probabilities must be in [0,1]')
        table = table.set_index(KEYS).sort_index()
        if reference is None:
            reference = table
        else:
            if not reference.index.equals(table.index):
                raise ValueError(f'{path}: unmatched commit identifiers across models')
            if not np.array_equal(reference['label'].to_numpy(), table['label'].to_numpy()):
                raise ValueError(f'{path}: inconsistent labels across models')
            if 'fold' in reference.columns and 'fold' in table.columns:
                if not np.array_equal(reference['fold'].to_numpy(), table['fold'].to_numpy()):
                    raise ValueError(f'{path}: inconsistent fold assignments')
        tables[model] = table
    return tables


def add_message_flags(comparison, pilot_csv):
    if pilot_csv is None:
        return comparison
    pilot = pd.read_csv(pilot_csv, dtype={'repo': str, 'commit_id': str})
    if pilot.duplicated(KEYS).any():
        raise ValueError('Pilot CSV has duplicate repository-commit keys')
    keep = KEYS + ['commit_message', 'diff']
    available = [c for c in keep if c in pilot.columns]
    pilot = pilot[available].copy()
    if 'commit_message' in pilot:
        messages = pilot['commit_message'].fillna('').astype(str)
        pilot['mentions_fix'] = messages.str.contains(
            r'\b(?:fix(?:es|ed|ing)?|patch(?:ed)?|repair(?:ed)?)\b', case=False, regex=True)
        pilot['mentions_security'] = messages.str.contains(
            r'\b(?:cve-\d{4}-\d+|vulnerab\w*|security|overflow|use.after.free|out.of.bounds)\b',
            case=False, regex=True)
        pilot['message_length'] = messages.str.len()
    if 'diff' in pilot:
        pilot['diff_length'] = pilot['diff'].fillna('').astype(str).str.len()
    pilot = pilot.drop(columns=[c for c in ('commit_message', 'diff') if c in pilot])
    return comparison.merge(pilot, on=KEYS, how='left', validate='one_to_one')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, default=Path('reports/codejit_input_ablation'))
    parser.add_argument('--pilot-csv', type=Path, default=Path('reports/codejit_pilot_200/pilot_commits.csv'))
    parser.add_argument('--out', type=Path, default=Path('reports/codejit_ablation_analysis'))
    parser.add_argument('--bootstrap', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('--bootstrap must be >= 100')
    args.out.mkdir(parents=True, exist_ok=True)

    tables = read_predictions(args.predictions_dir)
    base = tables[MODELS[0]].reset_index()
    comparison = base[KEYS + ['label'] + (['fold'] if 'fold' in base else [])].copy()
    for model in MODELS:
        t = tables[model].reset_index()
        comparison = comparison.merge(
            t[KEYS + ['prediction', 'probability']].rename(columns={
                'prediction': f'{model}_prediction', 'probability': f'{model}_probability'}),
            on=KEYS, validate='one_to_one')
        comparison[f'{model}_correct'] = comparison[f'{model}_prediction'].eq(comparison['label'])

    comparison['combined_only_correct'] = (
        comparison['message_plus_code_correct']
        & ~comparison['message_only_correct'] & ~comparison['code_only_correct'])
    comparison['combined_only_wrong'] = (
        ~comparison['message_plus_code_correct']
        & comparison['message_only_correct'] & comparison['code_only_correct'])
    comparison = add_message_flags(comparison, args.pilot_csv)
    comparison.to_csv(args.out / 'paired_predictions.csv', index=False)
    comparison[comparison['combined_only_correct']].to_csv(args.out / 'combined_only_correct.csv', index=False)
    comparison[comparison['combined_only_wrong']].to_csv(args.out / 'combined_only_wrong.csv', index=False)
    comparison[(comparison[[f'{m}_correct' for m in MODELS]].nunique(axis=1) > 1)].to_csv(
        args.out / 'disagreement_review.csv', index=False)

    y = comparison['label'].to_numpy(dtype=int)
    arrays = {m: (comparison[f'{m}_prediction'].to_numpy(dtype=int),
                  comparison[f'{m}_probability'].to_numpy(dtype=float)) for m in MODELS}
    baseline = {m: scores(y, *arrays[m]) for m in MODELS}
    rows = []
    rng = np.random.default_rng(args.seed)
    n = len(y)
    # Stratified paired bootstrap: identical sampled indices across all models.
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(idx0) or not len(idx1):
        raise ValueError('Both classes required for paired bootstrap')
    comparisons = [('message_plus_code', 'message_only'),
                   ('message_plus_code', 'code_only'), ('code_only', 'message_only')]
    draws = {(a, b, metric): [] for a, b in comparisons for metric in METRICS}
    for _ in range(args.bootstrap):
        sample = np.concatenate((rng.choice(idx0, size=len(idx0), replace=True),
                                 rng.choice(idx1, size=len(idx1), replace=True)))
        ys = y[sample]
        boot = {m: scores(ys, arrays[m][0][sample], arrays[m][1][sample]) for m in MODELS}
        for a, b in comparisons:
            for metric in METRICS:
                draws[(a, b, metric)].append(boot[a][metric] - boot[b][metric])

    for a, b in comparisons:
        correct_a = comparison[f'{a}_correct'].to_numpy(dtype=bool)
        correct_b = comparison[f'{b}_correct'].to_numpy(dtype=bool)
        a_only = int(np.sum(correct_a & ~correct_b))
        b_only = int(np.sum(~correct_a & correct_b))
        p = exact_mcnemar_p(a_only, b_only)
        for metric in METRICS:
            distribution = np.asarray(draws[(a, b, metric)], dtype=float)
            lo, hi = np.percentile(distribution, [2.5, 97.5])
            rows.append({
                'model_a': a, 'model_b': b, 'metric': metric,
                'score_a': baseline[a][metric], 'score_b': baseline[b][metric],
                'difference_a_minus_b': baseline[a][metric] - baseline[b][metric],
                'bootstrap_ci_95_low': float(lo), 'bootstrap_ci_95_high': float(hi),
                'a_correct_b_wrong': a_only, 'a_wrong_b_correct': b_only,
                'mcnemar_exact_p_for_accuracy': p,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / 'paired_metric_comparisons.csv', index=False)
    pd.DataFrame([{'model': m, **baseline[m]} for m in MODELS]).to_csv(
        args.out / 'model_metrics.csv', index=False)

    by_label = []
    for label, subset in comparison.groupby('label'):
        for m in MODELS:
            by_label.append({'label': int(label), 'model': m, 'n': len(subset),
                             'error_rate': float((~subset[f'{m}_correct']).mean())})
    pd.DataFrame(by_label).to_csv(args.out / 'errors_by_label.csv', index=False)
    by_keyword = []
    for flag in ('mentions_fix', 'mentions_security'):
        if flag not in comparison.columns:
            continue
        for value, subset in comparison.groupby(flag):
            for m in MODELS:
                by_keyword.append({'keyword': flag, 'present': bool(value),
                                   'model': m, 'n': len(subset),
                                   'error_rate': float((~subset[f'{m}_correct']).mean())})
    pd.DataFrame(by_keyword, columns=['keyword', 'present', 'model', 'n', 'error_rate']).to_csv(
        args.out / 'errors_by_keywords.csv', index=False)

    input_files = [args.predictions_dir / f'{m}_predictions.csv' for m in MODELS]
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'prediction_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        'pilot_csv': str(args.pilot_csv), 'n_commits': n, 'label_counts':
        {str(k): int(v) for k, v in comparison['label'].value_counts().sort_index().items()},
        'bootstrap': {'method': 'paired stratified row bootstrap of OOF predictions',
                      'resamples': args.bootstrap, 'seed': args.seed, 'ci': 'percentile 95%'},
        'mcnemar': 'exact two-sided binomial test of discordant paired correctness',
        'limitations': [
            'Nonrepresentative pilot; results are exploratory.',
            'Bootstrap resamples fixed out-of-fold predictions; it does not refit models or capture training variability.',
            'P-values and confidence intervals are conditional on the selected sample; not proof of cross-project generalization.',
            'Multiple metrics/comparisons are exploratory; no multiple-testing correction.',
            'Truncated patches excluded, disproportionately removing positive-label commits.',
            'CodeJIT label provenance and operational validity require independent verification.',
        ],
    }
    (args.out / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('\n=== MODEL METRICS ===')
    print(pd.DataFrame([{'model': m, **baseline[m]} for m in MODELS]).round(4).to_string(index=False))
    print('\n=== PAIRED DIFFERENCES (95% bootstrap CI) ===')
    print(summary[['model_a','model_b','metric','difference_a_minus_b',
                   'bootstrap_ci_95_low','bootstrap_ci_95_high']].round(4).to_string(index=False))
    print('\n=== PAIRED ERROR ANALYSIS ===')
    print('Combined correct, both individual models wrong:', int(comparison['combined_only_correct'].sum()))
    print('Combined wrong, both individual models correct:', int(comparison['combined_only_wrong'].sum()))
    for a, b in comparisons:
        row = summary[(summary.model_a == a) & (summary.model_b == b)].iloc[0]
        print(f'{a} vs {b}: discordant correctness {row.a_correct_b_wrong}/{row.a_wrong_b_correct}, '
              f'exact McNemar p={row.mcnemar_exact_p_for_accuracy:.4f}')
    print(f'\nSaved analysis to {args.out}')


if __name__ == '__main__':
    main()
