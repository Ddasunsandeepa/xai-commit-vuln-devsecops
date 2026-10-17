#!/usr/bin/env python3
"""Paired, validation-only comparison of prefix and hunk-aware CodeJIT transformers.

Requires pandas, numpy, scipy, scikit-learn. Never reads held-out test.
Run from repository root:
 python src/analyze_codejit_hunk_aware_ablation.py --bootstrap 2000 --seed 42
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score

KEY = ['repo', 'commit_id']
FAMILIES = ('codebert', 'graphcodebert')
MODES = ('code', 'combined')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load(folder, expected_train, expected_val):
    pred_path = folder / 'validation_predictions.csv'
    audit_path = folder / 'data_audit.json'
    results_path = folder / 'results.json'
    for path in (pred_path, audit_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text())
    for field, expected in [('train_manifest_sha256', expected_train), ('val_manifest_sha256', expected_val)]:
        if audit.get(field) != expected:
            raise ValueError(f'{folder}: {field} mismatch: {audit.get(field)} != {expected}')
    pred = pd.read_csv(pred_path, dtype={'repo': str, 'commit_id': str}, keep_default_na=False)
    required = KEY + ['label', 'predicted', 'probability_positive']
    if not set(required).issubset(pred.columns):
        raise ValueError(f'{pred_path} missing {set(required) - set(pred.columns)}')
    if pred.duplicated(KEY).any():
        raise ValueError(f'Duplicate commit keys in {pred_path}')
    for col in ('label', 'predicted'):
        pred[col] = pd.to_numeric(pred[col], errors='raise').astype(int)
        if not pred[col].isin([0, 1]).all():
            raise ValueError(f'Invalid {col}: {pred_path}')
    pred['probability_positive'] = pd.to_numeric(pred['probability_positive'], errors='raise')
    if pred['probability_positive'].isna().any() or not pred['probability_positive'].between(0, 1).all():
        raise ValueError(f'Invalid probabilities: {pred_path}')
    return pred[required], audit, {'predictions_sha256': sha(pred_path), 'results_sha256': sha(results_path)}

def metric(y, p, score):
    return {'n': int(len(y)), 'accuracy': float(accuracy_score(y, p)),
            'f1': float(f1_score(y, p, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            'pr_auc': float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None,
            'confusion_matrix': confusion_matrix(y, p, labels=[0, 1]).tolist()}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--codebert', default='reports/codejit_codebert_v1')
    ap.add_argument('--graphcodebert', default='reports/codejit_graphcodebert_v1')
    ap.add_argument('--hunk-aware', default='reports/codejit_hunk_aware_v1')
    ap.add_argument('--train', default='reports/codejit_train_val_v2/train_manifest.csv')
    ap.add_argument('--val', default='reports/codejit_train_val_v2/val_manifest.csv')
    ap.add_argument('--out', default='reports/codejit_hunk_aware_comparison_v1')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error('--bootstrap must be >=100')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_hash, val_hash = sha(args.train), sha(args.val)
    train = pd.read_csv(args.train, dtype={'repo': str, 'commit_id': str}, keep_default_na=False)
    val = pd.read_csv(args.val, dtype={'repo': str, 'commit_id': str}, keep_default_na=False)
    for name, frame in [('train', train), ('val', val)]:
        if frame.duplicated(KEY).any():
            raise ValueError(f'Duplicate {name} manifest keys')
        if not frame.label.astype(int).isin([0, 1]).all():
            raise ValueError(f'Invalid {name} labels')
    if set(zip(train.repo, train.commit_id)) & set(zip(val.repo, val.commit_id)):
        raise ValueError('Train and validation commits overlap')
    if set(train.repo) & set(val.repo):
        raise ValueError('Train and validation repositories overlap')
    expected = val[KEY + ['label']].copy()
    expected['label'] = expected.label.astype(int)
    expected = expected.sort_values(KEY).reset_index(drop=True)
    wide = expected.copy()
    hashes, scores, comparisons, recovered_rows, coverage_rows = {}, {}, [], [], []
    rng = np.random.default_rng(args.seed)
    for family in FAMILIES:
        for mode in MODES:
            prefix_folder = Path(args.codebert if family == 'codebert' else args.graphcodebert) / mode
            hunk_folder = Path(args.hunk_aware) / family / mode
            pair = {}
            for rep, folder in [('prefix', prefix_folder), ('hunk', hunk_folder)]:
                df, audit, file_hashes = load(folder, train_hash, val_hash)
                aligned = df.sort_values(KEY).reset_index(drop=True)
                if not aligned[KEY + ['label']].equals(expected):
                    raise ValueError(f'Prediction keys or labels differ from current validation manifest: {folder}')
                pair[rep] = aligned
                hashes[f'{family}_{mode}_{rep}'] = file_hashes
                y = aligned.label.to_numpy(int)
                p = aligned.predicted.to_numpy(int)
                s = aligned.probability_positive.to_numpy(float)
                scores[f'{family}_{mode}_{rep}'] = metric(y, p, s)
                for col in ('predicted', 'probability_positive'):
                    wide[f'{family}_{mode}_{rep}_{col}'] = aligned[col]
                covpath = folder / 'token_coverage.csv'
                if covpath.exists():
                    cov = pd.read_csv(covpath, dtype={'repo': str, 'commit_id': str}, keep_default_na=False)
                    c = cov[cov.subset.eq('val')].copy()
                    if c.duplicated(KEY).any() or set(zip(c.repo, c.commit_id)) != set(zip(expected.repo, expected.commit_id)):
                        raise ValueError(f'Token coverage validation keys mismatch: {covpath}')
                    for column in ('code_tokens_truncated', 'message_tokens_truncated'):
                        if column in c:
                            flags = c[column].astype(str).str.lower()
                            if not flags.isin(['true', 'false', '0', '1']).all():
                                raise ValueError(f'Invalid {column} in {covpath}')
                            coverage_rows.append({'family': family, 'mode': mode, 'representation': rep,
                                                  'field': column, 'n': len(c),
                                                  'truncated_n': int(flags.isin(['true', '1']).sum())})
            y = expected.label.to_numpy(int)
            old = pair['prefix']; new = pair['hunk']
            po, pn = old.predicted.to_numpy(int), new.predicted.to_numpy(int)
            so, sn = old.probability_positive.to_numpy(float), new.probability_positive.to_numpy(float)
            old_correct, new_correct = po == y, pn == y
            new_only = int((new_correct & ~old_correct).sum())
            old_only = int((old_correct & ~new_correct).sum())
            discord = new_only + old_only
            mcnemar_p = float(binomtest(min(new_only, old_only), discord, 0.5).pvalue) if discord else 1.0
            repos = expected.repo.to_numpy()
            groups = {repo: np.flatnonzero(repos == repo) for repo in np.unique(repos)}
            repo_ids = np.array(list(groups))
            draws = {'f1': [], 'roc_auc': [], 'pr_auc': []}
            for _ in range(args.bootstrap):
                chosen = rng.choice(repo_ids, size=len(repo_ids), replace=True)
                ix = np.concatenate([groups[r] for r in chosen])
                yy = y[ix]
                draws['f1'].append(float(f1_score(yy, pn[ix], zero_division=0) - f1_score(yy, po[ix], zero_division=0)))
                if len(np.unique(yy)) == 2:
                    draws['roc_auc'].append(float(roc_auc_score(yy, sn[ix]) - roc_auc_score(yy, so[ix])))
                    draws['pr_auc'].append(float(average_precision_score(yy, sn[ix]) - average_precision_score(yy, so[ix])))
            before = scores[f'{family}_{mode}_prefix']; after = scores[f'{family}_{mode}_hunk']
            row = {'family': family, 'mode': mode, 'n': len(y), 'repos': len(repo_ids),
                   'hunk_only_correct': new_only, 'prefix_only_correct': old_only,
                   'both_correct': int((new_correct & old_correct).sum()),
                   'both_wrong': int((~new_correct & ~old_correct).sum()),
                   'mcnemar_exact_p': mcnemar_p,
                   'positive_recovered': int(((y == 1) & (po == 0) & (pn == 1)).sum()),
                   'positive_newly_missed': int(((y == 1) & (po == 1) & (pn == 0)).sum()),
                   'negative_recovered': int(((y == 0) & (po == 1) & (pn == 0)).sum()),
                   'negative_newly_wrong': int(((y == 0) & (po == 0) & (pn == 1)).sum())}
            for m in ('f1', 'roc_auc', 'pr_auc'):
                if not draws[m]:
                    raise ValueError('Bootstrap never sampled both labels')
                lo, hi = np.quantile(draws[m], [.025, .975])
                row.update({f'prefix_{m}': before[m], f'hunk_{m}': after[m],
                            f'delta_{m}': after[m] - before[m],
                            f'delta_{m}_ci_low': float(lo), f'delta_{m}_ci_high': float(hi)})
            comparisons.append(row)
            for i in range(len(y)):
                if y[i] == 1 and po[i] != pn[i]:
                    recovered_rows.append({'family': family, 'mode': mode,
                                           'repo': expected.repo.iloc[i], 'commit_id': expected.commit_id.iloc[i],
                                           'status': 'recovered_by_hunk' if pn[i] == 1 else 'newly_missed_by_hunk',
                                           'prefix_probability': float(so[i]), 'hunk_probability': float(sn[i])})
    wide.to_csv(out / 'aligned_validation_predictions.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out / 'paired_representation_comparisons.csv', index=False)
    pd.DataFrame(recovered_rows, columns=['family', 'mode', 'repo', 'commit_id', 'status', 'prefix_probability', 'hunk_probability']).to_csv(out / 'positive_prediction_changes.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(out / 'token_coverage_comparison.csv', index=False)
    report = {'scope': 'Validation only; frozen held-out test not accessed',
              'train_commits': len(train), 'train_repositories': int(train.repo.nunique()),
              'validation_commits': len(val), 'validation_repositories': int(val.repo.nunique()),
              'manifest_sha256': {'train': train_hash, 'val': val_hash},
              'bootstrap': args.bootstrap, 'seed': args.seed,
              'metrics': scores, 'comparisons': comparisons, 'input_file_sha256': hashes,
              'limitations': [
                  'Hunk-aware selection uses full verified patches before applying the byte cap; prefix uses first bytes only. This is a representation and source-coverage intervention, not a same-prefix filtering control.',
                  'Single-seed checkpoints and thresholds selected using validation; confidence intervals conditional on observed validation repositories.',
                  'Four exploratory paired comparisons; no multiple-comparison adjustment.',
                  'McNemar tests paired correctness at fixed 0.5; they do not test ROC-AUC or F1 directly.',
                  'GraphCodeBERT uses a sequence classifier without explicit dataflow graphs.',
                  'CodeJIT curated negative commits do not represent ordinary benign CI/CD commits.',
                  'No held-out test accessed.'
              ]}
    (out / 'comparison_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'train_commits': len(train), 'validation_commits': len(val),
                      'validation_repositories': int(val.repo.nunique()),
                      'comparisons': comparisons, 'output': str(out)}, indent=2))

if __name__ == '__main__':
    main()
