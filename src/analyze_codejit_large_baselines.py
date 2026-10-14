#!/usr/bin/env python3
"""Diagnostic analysis of frozen CodeJIT large-baseline validation predictions.

No model retraining, no test data access, and no modifications to input artifacts.
Requires pandas, numpy, scikit-learn, scipy.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

MODELS = ('message', 'code', 'combined')
PAIRS = (('message', 'combined'), ('message', 'code'), ('combined', 'code'))
PATTERN = re.compile(r'\b(?:fix|fixed|fixes|fixing|bugfix|patch|patched|security|vulnerab\w*|cve|exploit\w*|overflow|crash|sanitize\w*)\b', re.I)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, prob):
    pred = (prob >= .5).astype(int)
    return {'f1': float(f1_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, prob)),
            'pr_auc': float(average_precision_score(y, prob)),
            'accuracy': float((pred == y).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='reports/codejit_large_baselines_v2')
    p.add_argument('--out', default='reports/codejit_large_diagnostics_v1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error('--bootstrap must be at least 100')
    root, out = Path(args.baseline), Path(args.out)
    paths = {name: root / name for name in ('validation_predictions.csv', 'input_coverage.csv', 'top_tfidf_features.csv', 'baseline_report.json')}
    for path in paths.values():
        if not path.is_file():
            p.error(f'Missing input: {path}')
    predictions = pd.read_csv(paths['validation_predictions.csv'], dtype={'repo': str, 'commit_id': str})
    coverage = pd.read_csv(paths['input_coverage.csv'], dtype={'repo': str, 'commit_id': str})
    features = pd.read_csv(paths['top_tfidf_features.csv'])
    report = json.loads(paths['baseline_report.json'].read_text(encoding='utf-8'))
    if set(predictions.model) != set(MODELS):
        raise ValueError('Expected exactly message, code, combined prediction sets')
    wide = None
    for model in MODELS:
        d = predictions[predictions.model == model].copy()
        if d.duplicated(['repo', 'commit_id']).any():
            raise ValueError(f'Duplicate prediction keys: {model}')
        d = d[['repo', 'commit_id', 'label', 'predicted', 'probability_positive']].rename(
            columns={'label': f'label_{model}', 'predicted': f'pred_{model}',
                     'probability_positive': f'prob_{model}'})
        wide = d if wide is None else wide.merge(d, on=['repo', 'commit_id'], how='inner', validate='one_to_one')
    if len(wide) != len(predictions) // 3:
        raise ValueError('Prediction models do not have identical commit keys')
    if not ((wide.label_message == wide.label_code) & (wide.label_code == wide.label_combined)).all():
        raise ValueError('Label mismatch between models')
    if coverage.duplicated(['repo', 'commit_id']).any():
        raise ValueError('Duplicate coverage keys')
    val_coverage = coverage[coverage.subset == 'val'].copy()
    wide = wide.merge(val_coverage[['repo', 'commit_id', 'patch_bytes', 'patch_truncated', 'message_chars']],
                      on=['repo', 'commit_id'], how='left', validate='one_to_one')
    if wide.patch_bytes.isna().any():
        raise ValueError('Validation input coverage incomplete')
    wide['label'] = wide.label_message.astype(int)
    for model in MODELS:
        wide[f'correct_{model}'] = (wide[f'pred_{model}'] == wide.label)
    wide['patch_truncated'] = wide.patch_truncated.astype(str).str.lower().eq('true')
    wide['patch_size_group'] = np.where(wide.patch_truncated, 'truncated', 'not_truncated')
    y = wide.label.to_numpy()
    baseline = {m: metrics(y, wide[f'prob_{m}'].to_numpy()) for m in MODELS}
    pair_rows = []
    for a, b in PAIRS:
        ac = wide[f'correct_{a}'].to_numpy()
        bc = wide[f'correct_{b}'].to_numpy()
        a_only = int((ac & ~bc).sum())
        b_only = int((~ac & bc).sum())
        n = a_only + b_only
        pval = float(binomtest(min(a_only, b_only), n=n, p=.5).pvalue) if n else 1.0
        pair_rows.append({'a': a, 'b': b, 'a_only_correct': a_only, 'b_only_correct': b_only,
                          'both_correct': int((ac & bc).sum()), 'both_wrong': int((~ac & ~bc).sum()),
                          'mcnemar_exact_p': pval})
    rng = np.random.default_rng(args.seed)
    # Cluster bootstrap: resample entire repositories with replacement.
    # This is a conditional CI for this validation cohort, not unseen-project generalization.
    groups = [grp.index.to_numpy() for _, grp in wide.groupby('repo', sort=True)]
    draws = {pair: {metric: [] for metric in ('f1', 'roc_auc', 'pr_auc')} for pair in PAIRS}
    skipped = 0
    for _ in range(args.bootstrap):
        chosen = rng.integers(0, len(groups), len(groups))
        ix = np.concatenate([groups[j] for j in chosen])
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            skipped += 1
            continue
        sample_metrics = {m: metrics(yy, wide[f'prob_{m}'].to_numpy()[ix]) for m in MODELS}
        for pair in PAIRS:
            a, b = pair
            for metric in draws[pair]:
                draws[pair][metric].append(sample_metrics[a][metric] - sample_metrics[b][metric])
    ci_rows = []
    for pair in PAIRS:
        a, b = pair
        for metric, vals in draws[pair].items():
            if not vals:
                raise ValueError('No valid bootstrap replicates')
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ci_rows.append({'model_a': a, 'model_b': b, 'metric': metric,
                            'observed_delta_a_minus_b': baseline[a][metric] - baseline[b][metric],
                            'cluster_bootstrap_ci_low': float(lo), 'cluster_bootstrap_ci_high': float(hi),
                            'valid_replicates': len(vals)})
    subgroup_rows = []
    for group_name, d in wide.groupby('patch_size_group'):
        yy = d.label.to_numpy()
        for model in MODELS:
            pred = d[f'pred_{model}'].to_numpy()
            subgroup_rows.append({'patch_group': group_name, 'model': model, 'n': len(d),
                                  'label0': int((yy == 0).sum()), 'label1': int((yy == 1).sum()),
                                  'accuracy': float((pred == yy).mean()),
                                  'f1': float(f1_score(yy, pred, zero_division=0))})
    terms = features.copy()
    terms['shortcut_keyword'] = terms.term.astype(str).apply(lambda s: bool(PATTERN.search(s)))
    shortcut_summary = terms.groupby(['model', 'direction'], as_index=False).agg(
        top_terms=('term', 'size'), shortcut_terms=('shortcut_keyword', 'sum'))
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / 'paired_validation_errors.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(out / 'mcnemar_pairs.csv', index=False)
    pd.DataFrame(ci_rows).to_csv(out / 'cluster_bootstrap_intervals.csv', index=False)
    pd.DataFrame(subgroup_rows).to_csv(out / 'truncation_subgroups.csv', index=False)
    terms.to_csv(out / 'top_features_shortcut_flags.csv', index=False)
    shortcut_summary.to_csv(out / 'shortcut_feature_summary.csv', index=False)
    result = {'status': 'completed', 'validation_commits': len(wide), 'repositories': len(groups),
              'validation_label_counts': {str(k): int(v) for k, v in wide.label.value_counts().items()},
              'baseline_metrics_recomputed': baseline,
              'cluster_bootstrap': {'replicates_requested': args.bootstrap, 'replicates_skipped_single_class': skipped,
                                    'resampling_unit': 'repository', 'seed': args.seed},
              'input_sha256': {name: digest(path) for name, path in paths.items()},
              'limitations': ['Exploratory validation-only analysis; no test access.',
                              'Keyword flags describe top TF-IDF coefficients, not causal explanations.',
                              'Subgroup comparisons are descriptive, not randomized tests of truncation effects.',
                              'Cluster bootstrap is conditional on observed validation repositories.',
                              'No message masking or retraining is performed in this diagnostic.',
                              'Curated CodeJIT negative selection can create message shortcuts.']}
    (out / 'diagnostic_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'validation_commits': len(wide), 'repositories': len(groups),
                      'paired_tests': pair_rows, 'bootstrap_intervals': ci_rows,
                      'truncation_subgroups': subgroup_rows}, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    main()
