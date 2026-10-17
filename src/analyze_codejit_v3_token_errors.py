#!/usr/bin/env python3
"""E08: Read-only CodeJIT v3 validation token truncation and error audit.

Run from repository root. Does not load held-out test data or train models.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

FAMILIES = ('codebert', 'graphcodebert')
MODES = ('message', 'code', 'combined')
KEY = ['repo', 'commit_id']

def require(cond, message):
    if not cond:
        raise ValueError(message)

def load_csv(path):
    require(path.is_file(), f'Missing file: {path}')
    return pd.read_csv(path, dtype={'repo': str, 'commit_id': str})

def bool_series(series, label):
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    require(converted.notna().all(), f'Invalid boolean in {label}')
    return converted.astype(bool)

def unique(df, name):
    require(not df.duplicated(KEY).any(), f'Duplicate repo/commit_id in {name}')

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports-root', default='reports')
    p.add_argument('--val-manifest', default='reports/codejit_train_val_v3/val_manifest.csv')
    p.add_argument('--out', default='reports/codejit_v3_token_error_analysis')
    args = p.parse_args()
    root = Path(args.reports_root)
    out = Path(args.out)
    val = load_csv(Path(args.val_manifest))
    unique(val, 'validation manifest')
    require({'repo', 'commit_id', 'label'}.issubset(val.columns), 'Manifest missing key/label columns')
    val = val[KEY + ['label']].copy()
    val['label'] = pd.to_numeric(val['label'], errors='raise').astype(int)
    require(val.label.isin([0, 1]).all(), 'Manifest contains nonbinary labels')
    require(len(val) == 323, f'Expected 323 validation rows; got {len(val)}')
    expected = set(map(tuple, val[KEY].itertuples(index=False, name=None)))
    require(val.repo.nunique() == 63, f'Expected 63 validation repositories; got {val.repo.nunique()}')

    all_models = {}
    audit = {'validation_rows': len(val), 'validation_repositories': val.repo.nunique(), 'model_files': {}}
    for family in FAMILIES:
        for mode in MODES:
            name = f'{family}_{mode}'
            folder = root / f'codejit_{family}_v3' / mode / mode
            pp = folder / 'validation_predictions.csv'
            cp = folder / 'token_coverage.csv'
            pred = load_csv(pp)
            cov = load_csv(cp)
            unique(pred, str(pp))
            require({'label', 'predicted', 'probability_positive'}.issubset(pred.columns), f'Missing prediction fields: {pp}')
            require({'subset', 'message_tokens_truncated', 'code_tokens_truncated', 'model_tokens'}.issubset(cov.columns), f'Missing coverage fields: {cp}')
            # The coverage table contains BOTH train and validation; only val may be used for error analysis.
            subsets = cov.subset.astype(str).str.lower().str.strip()
            require(set(subsets.unique()).issubset({'train', 'val', 'validation'}), f'Unexpected subset values in {cp}: {set(subsets.unique())}')
            coverage = cov.loc[subsets.isin(['val', 'validation'])].copy()
            unique(coverage, str(cp) + ' validation subset')
            require(len(coverage) == len(val), f'{name}: expected {len(val)} coverage validation rows, got {len(coverage)}')
            for frame, description in [(pred, 'predictions'), (coverage, 'coverage')]:
                require(set(map(tuple, frame[KEY].itertuples(index=False, name=None))) == expected, f'{name}: {description} keys do not match frozen validation manifest')
            pred['label'] = pd.to_numeric(pred['label'], errors='raise').astype(int)
            pred['predicted'] = pd.to_numeric(pred['predicted'], errors='raise').astype(int)
            pred['probability_positive'] = pd.to_numeric(pred['probability_positive'], errors='raise')
            require(pred.label.isin([0,1]).all() and pred.predicted.isin([0,1]).all(), f'{name}: invalid label/prediction')
            require(pred.probability_positive.between(0,1).all(), f'{name}: invalid probabilities')
            require((pred.predicted.to_numpy() == (pred.probability_positive.to_numpy() >= .5).astype(int)).all(), f'{name}: predicted values disagree with fixed threshold 0.5')
            joined = val.merge(pred[KEY + ['label', 'predicted', 'probability_positive']], on=KEY, how='left', validate='one_to_one', suffixes=('_manifest', '_pred'))
            require(joined.label_pred.notna().all(), f'{name}: missing predictions')
            require((joined.label_manifest == joined.label_pred).all(), f'{name}: validation labels disagree')
            joined = joined.drop(columns=['label_pred']).rename(columns={'label_manifest': 'label'})
            coverage['message_tokens_truncated'] = bool_series(coverage['message_tokens_truncated'], f'{name} message truncation')
            coverage['code_tokens_truncated'] = bool_series(coverage['code_tokens_truncated'], f'{name} code truncation')
            coverage['model_tokens'] = pd.to_numeric(coverage['model_tokens'], errors='raise').astype(int)
            joined = joined.merge(coverage[KEY + ['message_tokens_truncated', 'code_tokens_truncated', 'model_tokens']], on=KEY, how='left', validate='one_to_one')
            require(joined.model_tokens.notna().all(), f'{name}: missing coverage')
            joined['correct'] = joined.label == joined.predicted
            joined['error_type'] = np.select(
                [(joined.label == 1) & (joined.predicted == 0), (joined.label == 0) & (joined.predicted == 1)],
                ['FN', 'FP'], default='correct')
            joined['relevant_truncation'] = (joined.message_tokens_truncated if mode == 'message'
                                             else joined.code_tokens_truncated if mode == 'code'
                                             else (joined.message_tokens_truncated | joined.code_tokens_truncated))
            joined['family'] = family
            joined['mode'] = mode
            all_models[name] = joined
            audit['model_files'][name] = {'predictions': str(pp), 'coverage': str(cp), 'coverage_total_rows': len(cov), 'validation_rows': len(joined), 'model_token_min': int(joined.model_tokens.min()), 'model_token_max': int(joined.model_tokens.max())}

    out.mkdir(parents=True, exist_ok=True)
    detail = pd.concat(all_models.values(), ignore_index=True)
    detail.to_csv(out / 'validation_token_error_details.csv', index=False)

    summary = []
    error_by_trunc = []
    for name, df in all_models.items():
        family, mode = df.family.iloc[0], df['mode'].iloc[0]
        tn, fp, fn, tp = confusion_matrix(df.label, df.predicted, labels=[0,1]).ravel()
        summary.append({
            'model': name, 'n': len(df), 'repositories': df.repo.nunique(),
            'message_truncated_n': int(df.message_tokens_truncated.sum()) if mode in ('message','combined') else None,
            'message_truncated_rate': float(df.message_tokens_truncated.mean()) if mode in ('message','combined') else None,
            'code_truncated_n': int(df.code_tokens_truncated.sum()) if mode in ('code','combined') else None,
            'code_truncated_rate': float(df.code_tokens_truncated.mean()) if mode in ('code','combined') else None,
            'relevant_truncated_n': int(df.relevant_truncation.sum()),
            'relevant_truncated_rate': float(df.relevant_truncation.mean()),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        })
        for label_value, label_name in [(0, 'negative'), (1, 'positive')]:
            for truncated in [False, True]:
                group = df[(df.label == label_value) & (df.relevant_truncation == truncated)]
                error_by_trunc.append({
                    'model': name, 'true_label': label_value, 'label_group': label_name,
                    'relevant_truncated': truncated, 'n': len(group),
                    'errors': int((~group.correct).sum()),
                    'error_rate': float((~group.correct).mean()) if len(group) else None,
                    'error_kind': 'FP' if label_value == 0 else 'FN',
                })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / 'truncation_summary.csv', index=False)
    pd.DataFrame(error_by_trunc).to_csv(out / 'errors_by_truncation.csv', index=False)

    wide = val.copy()
    for name, df in all_models.items():
        renamed = df[KEY + ['predicted', 'probability_positive', 'correct', 'relevant_truncation']].rename(columns={
            'predicted': f'{name}__predicted',
            'probability_positive': f'{name}__probability',
            'correct': f'{name}__correct',
            'relevant_truncation': f'{name}__truncated',
        })
        wide = wide.merge(renamed, on=KEY, how='left', validate='one_to_one')
    wide.to_csv(out / 'aligned_error_matrix.csv', index=False)

    recoveries = []
    for family in FAMILIES:
        positives = wide[wide.label == 1]
        m = positives[f'{family}_message__correct'].astype(bool)
        c = positives[f'{family}_code__correct'].astype(bool)
        b = positives[f'{family}_combined__correct'].astype(bool)
        recoveries.append({
            'family': family, 'positive_total': len(positives),
            'message_correct_code_wrong': int((m & ~c).sum()),
            'code_correct_message_wrong': int((c & ~m).sum()),
            'combined_correct_code_wrong': int((b & ~c).sum()),
            'code_correct_combined_wrong': int((c & ~b).sum()),
            'missed_by_all_three': int((~m & ~c & ~b).sum()),
        })
    pd.DataFrame(recoveries).to_csv(out / 'positive_recovery.csv', index=False)

    repo_rows = []
    for name, df in all_models.items():
        for repo, g in df.groupby('repo'):
            repo_rows.append({'model': name, 'repo': repo, 'n': len(g),
                              'errors': int((~g.correct).sum()), 'fp': int((g.error_type == 'FP').sum()),
                              'fn': int((g.error_type == 'FN').sum()),
                              'relevant_truncated_n': int(g.relevant_truncation.sum())})
    pd.DataFrame(repo_rows).to_csv(out / 'repository_error_summary.csv', index=False)
    write_json(out / 'audit.json', audit)
    print('\n=== E08 VALIDATION TOKEN TRUNCATION ===')
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n=== E08 ERROR RATES BY RELEVANT TRUNCATION ===')
    print(pd.DataFrame(error_by_trunc).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n=== POSITIVE LABEL RECOVERY (cross-check E07) ===')
    print(pd.DataFrame(recoveries).to_string(index=False))
    print('\nSaved to:', out)
    print('Descriptive associations only; truncation does not establish causation.')

if __name__ == '__main__':
    main()
