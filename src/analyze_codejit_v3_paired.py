#!/usr/bin/env python3
"""E07: paired, repository-clustered analysis of six CodeJIT v3 validation runs.

Only reads development validation predictions; never opens held-out test data.
Usage: python src/analyze_codejit_v3_paired.py
"""
import argparse
import hashlib
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

FAMILIES = ('codebert', 'graphcodebert')
MODES = ('message', 'code', 'combined')
KEY = ['repo', 'commit_id']
REQUIRED = KEY + ['label', 'predicted', 'probability_positive']
EXPECTED_SHA = {
    'train': '22fe1a4a193865c47ab1276752d82f4c4b7ca65c9128882a363df938be124981',
    'val': '1e284c9c6647b615fce51a9bb33a9da7112edf0fca33bdbb72d866e818ab15c1',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, pred, prob):
    y, pred, prob = np.asarray(y), np.asarray(pred), np.asarray(prob)
    out = {'f1': float(f1_score(y, pred, zero_division=0))}
    out['roc_auc'] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float('nan')
    out['pr_auc'] = float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else float('nan')
    return out


def load_predictions(root, family, mode):
    path = root / f'codejit_{family}_v3' / mode / mode / 'validation_predictions.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing predictions: {path}')
    df = pd.read_csv(path, dtype={'repo': str, 'commit_id': str})
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if df[KEY].isna().any().any() or df.duplicated(KEY).any():
        raise ValueError(f'{path}: missing or duplicate repo/commit_id keys')
    if df[REQUIRED].isna().any().any():
        raise ValueError(f'{path}: missing required values')
    for col in ('label', 'predicted'):
        df[col] = pd.to_numeric(df[col], errors='raise').astype(int)
        if not df[col].isin([0, 1]).all():
            raise ValueError(f'{path}: invalid binary {col}')
    df['probability_positive'] = pd.to_numeric(df['probability_positive'], errors='raise')
    if not df['probability_positive'].between(0, 1).all():
        raise ValueError(f'{path}: probabilities outside [0,1]')
    expected_pred = (df['probability_positive'].to_numpy() >= 0.5).astype(int)
    if not np.array_equal(expected_pred, df['predicted'].to_numpy()):
        raise ValueError(f'{path}: predicted differs from probability >= 0.5')
    return df[REQUIRED].copy(), path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reports-root', default='reports')
    ap.add_argument('--val-manifest', default='reports/codejit_train_val_v3/val_manifest.csv')
    ap.add_argument('--train-manifest', default='reports/codejit_train_val_v3/train_manifest.csv')
    ap.add_argument('--out', default='reports/codejit_v3_paired_analysis')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error('--bootstrap must be at least 100')

    root = Path(args.reports_root)
    out = Path(args.out)
    for split, path in [('train', Path(args.train_manifest)), ('val', Path(args.val_manifest))]:
        actual = digest(path)
        if actual != EXPECTED_SHA[split]:
            raise ValueError(f'{split} manifest SHA mismatch: {actual}')
    manifest = pd.read_csv(args.val_manifest, dtype={'repo': str, 'commit_id': str})
    if not set(KEY + ['label']).issubset(manifest.columns):
        raise ValueError('Validation manifest missing repo, commit_id, or label')
    if len(manifest) != 323 or manifest.duplicated(KEY).any():
        raise ValueError('Expected 323 unique validation commit keys')
    manifest['label'] = pd.to_numeric(manifest['label']).astype(int)
    if manifest['repo'].nunique() != 63:
        raise ValueError('Expected 63 validation repositories')
    manifest = manifest[KEY + ['label']].sort_values(KEY).reset_index(drop=True)
    manifest_keys = set(zip(manifest.repo, manifest.commit_id))
    models, paths = {}, {}
    for family in FAMILIES:
        for mode in MODES:
            name = f'{family}_{mode}'
            df, path = load_predictions(root, family, mode)
            keys = set(zip(df.repo, df.commit_id))
            if keys != manifest_keys:
                raise ValueError(f'{name}: key mismatch (missing={len(manifest_keys - keys)}, extra={len(keys - manifest_keys)})')
            aligned = manifest.merge(df, on=KEY, how='left', validate='one_to_one', suffixes=('_manifest', '_prediction'))
            if not np.array_equal(aligned.label_manifest.to_numpy(), aligned.label_prediction.to_numpy()):
                raise ValueError(f'{name}: labels disagree with frozen v3 manifest')
            models[name] = {
                'pred': aligned.predicted.to_numpy(dtype=int),
                'prob': aligned.probability_positive.to_numpy(dtype=float),
            }
            paths[name] = {'path': str(path), 'sha256': digest(path)}
    y = manifest.label.to_numpy(dtype=int)
    repos = manifest.repo.to_numpy()
    unique_repos = np.unique(repos)
    repo_indices = [np.flatnonzero(repos == repo) for repo in unique_repos]
    summary = []
    for name, values in models.items():
        pred = values['pred']
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        m = metrics(y, pred, values['prob'])
        summary.append({'model': name, 'n': len(y), 'repositories': len(unique_repos), **m,
                        'tn': cm[0][0], 'fp': cm[0][1], 'fn': cm[1][0], 'tp': cm[1][1]})
    summary_df = pd.DataFrame(summary)

    # Explicit research contrasts, defined before inspecting inferential output.
    contrasts = [
        ('graphcodebert_message', 'codebert_message'),
        ('graphcodebert_code', 'codebert_code'),
        ('graphcodebert_combined', 'codebert_combined'),
        ('codebert_combined', 'codebert_message'),
        ('graphcodebert_combined', 'graphcodebert_message'),
        ('codebert_code', 'codebert_message'),
        ('graphcodebert_code', 'graphcodebert_message'),
        ('graphcodebert_combined', 'codebert_combined'),
    ]
    rng = np.random.default_rng(args.seed)
    # Common bootstrap samples for all contrasts, resampling entire repositories.
    boot_indices = []
    for _ in range(args.bootstrap):
        chosen = rng.integers(0, len(unique_repos), size=len(unique_repos))
        boot_indices.append(np.concatenate([repo_indices[j] for j in chosen]))
    comparisons = []
    for a, b in contrasts:
        va, vb = models[a], models[b]
        ma = metrics(y, va['pred'], va['prob'])
        mb = metrics(y, vb['pred'], vb['prob'])
        # McNemar: A correct / B wrong, and vice versa.
        a_correct = va['pred'] == y
        b_correct = vb['pred'] == y
        a_only = int(np.sum(a_correct & ~b_correct))
        b_only = int(np.sum(~a_correct & b_correct))
        discordant = a_only + b_only
        mcnemar_p = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
        diffs = {k: [] for k in ('f1', 'roc_auc', 'pr_auc')}
        for idx in boot_indices:
            ya = y[idx]
            if len(np.unique(ya)) < 2:
                continue
            mba = metrics(ya, va['pred'][idx], va['prob'][idx])
            mbb = metrics(ya, vb['pred'][idx], vb['prob'][idx])
            for metric in diffs:
                diffs[metric].append(mba[metric] - mbb[metric])
        result = {'model_a': a, 'model_b': b, 'a_correct_b_wrong': a_only,
                  'a_wrong_b_correct': b_only, 'mcnemar_exact_p': mcnemar_p,
                  'valid_bootstrap_replicates': len(diffs['f1'])}
        for metric, values in diffs.items():
            arr = np.asarray(values)
            if len(arr) < 100:
                raise ValueError('Too few valid bootstrap resamples')
            result[f'{metric}_delta_a_minus_b'] = ma[metric] - mb[metric]
            result[f'{metric}_ci95_low'] = float(np.quantile(arr, 0.025))
            result[f'{metric}_ci95_high'] = float(np.quantile(arr, 0.975))
        comparisons.append(result)

    # Positive-label recovery: message succeeds while code fails, by model family.
    recovery = []
    for family in FAMILIES:
        msg = models[f'{family}_message']['pred']
        code = models[f'{family}_code']['pred']
        combined = models[f'{family}_combined']['pred']
        pos = y == 1
        recovery.append({
            'family': family, 'positive_total': int(pos.sum()),
            'message_correct_code_wrong': int(np.sum(pos & (msg == 1) & (code == 0))),
            'code_correct_message_wrong': int(np.sum(pos & (code == 1) & (msg == 0))),
            'combined_correct_code_wrong': int(np.sum(pos & (combined == 1) & (code == 0))),
            'code_correct_combined_wrong': int(np.sum(pos & (code == 1) & (combined == 0))),
            'missed_by_all_three': int(np.sum(pos & (msg == 0) & (code == 0) & (combined == 0))),
        })

    out.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out / 'model_summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out / 'paired_comparisons.csv', index=False)
    pd.DataFrame(recovery).to_csv(out / 'positive_recovery.csv', index=False)
    aligned = manifest.copy()
    for name, values in models.items():
        aligned[f'{name}_predicted'] = values['pred']
        aligned[f'{name}_probability_positive'] = values['prob']
    aligned.to_csv(out / 'aligned_validation_predictions.csv', index=False)
    audit = {
        'purpose': 'E07 exploratory CodeJIT v3 development-validation paired analysis',
        'test_cohort_accessed': False,
        'n_validation': len(y), 'n_repositories': len(unique_repos),
        'label_counts': {str(k): int(v) for k, v in manifest.label.value_counts().items()},
        'train_manifest_sha256': EXPECTED_SHA['train'],
        'val_manifest_sha256': EXPECTED_SHA['val'],
        'prediction_sources': paths,
        'bootstrap_method': 'repository cluster bootstrap; percentile 95% confidence interval',
        'bootstrap_replicates': args.bootstrap, 'seed': args.seed,
        'threshold': 0.5, 'mcnemar_method': 'exact two-sided binomial on discordant correctness',
        'notes': [
            'Validation was used for training checkpoint selection; intervals are exploratory.',
            'Multiple comparisons are unadjusted; do not claim confirmatory significance.',
            'CodeJIT label zero is not representative of arbitrary benign pull requests.',
            'GraphCodeBERT uses a sequence classifier, not graph/dataflow architecture.',
        ],
    }
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('\n=== E07 MODEL SUMMARY (fixed threshold 0.5) ===')
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print('\n=== E07 PAIRED COMPARISONS (A minus B) ===')
    print(pd.DataFrame(comparisons)[['model_a','model_b','f1_delta_a_minus_b','f1_ci95_low','f1_ci95_high','mcnemar_exact_p']].to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print('\n=== POSITIVE-LABEL RECOVERY ===')
    print(pd.DataFrame(recovery).to_string(index=False))
    print(f'\nSaved to: {out}')

if __name__ == '__main__':
    main()
