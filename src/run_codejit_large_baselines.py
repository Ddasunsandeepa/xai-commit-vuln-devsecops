#!/usr/bin/env python3
"""Matched-input CodeJIT message/code/combined TF-IDF logistic baselines.

Uses ONLY frozen verified train/val manifests and compressed patch + metadata
files. Never accesses test data. Requires pandas, numpy, scikit-learn.

Example:
 python src/run_codejit_large_baselines.py \
   --train reports/codejit_train_val_v1/train_manifest.csv \
   --val reports/codejit_train_val_v1/val_manifest.csv \
   --out reports/codejit_large_baselines_v1
"""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def key_for(repo, commit):
    return hashlib.sha256(f'{repo}\0{commit}'.encode()).hexdigest()[:24]


def load_manifest(path):
    df = pd.read_csv(path, dtype={'repo': str, 'commit_id': str}, keep_default_na=False)
    required = {'repo', 'commit_id', 'label', 'patch_path', 'patch_valid'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path} missing {required - set(df.columns)}')
    if df.duplicated(['repo', 'commit_id']).any():
        raise ValueError(f'Duplicate commit keys in {path}')
    if not df.label.astype(str).isin(['0', '1']).all():
        raise ValueError(f'Invalid labels in {path}')
    if not df.patch_valid.astype(str).str.lower().eq('true').all():
        raise ValueError(f'Unverified patch in {path}')
    return df.sort_values(['repo', 'commit_id']).reset_index(drop=True)


def metadata_message(metadata):
    # Fail closed rather than silently train on blank messages due to schema mismatch.
    for field in ('message', 'commit_message', 'full_message'):
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f'Commit message missing in metadata (keys={list(metadata)})')


def load_records(df, extraction, max_patch_bytes):
    messages, patches, audit = [], [], []
    records_dir = extraction / 'records'
    for r in df.itertuples(index=False):
        key = key_for(str(r.repo), str(r.commit_id))
        expected = records_dir / f'{key}.diff.gz'
        declared = Path(str(r.patch_path))
        if not expected.is_file():
            raise FileNotFoundError(f'Missing {expected}')
        # Audit manifest path may be relative to Kaggle project root; the key-derived
        # path is canonical. Verify filename equality without trusting arbitrary paths.
        if declared.name != expected.name:
            raise ValueError(f'Unexpected patch filename for {r.repo}/{r.commit_id}')
        metadata = json.loads((records_dir / f'{key}.json').read_text(encoding='utf-8'))
        if (str(metadata.get('repo')) != str(r.repo) or
                str(metadata.get('commit_id')) != str(r.commit_id) or
                int(metadata.get('label', -1)) != int(r.label) or
                metadata.get('status') != 'ok'):
            raise ValueError(f'Metadata mismatch for {r.repo}/{r.commit_id}')
        msg = metadata_message(metadata)
        # Read complete gzip patch and verify it matches audit size; selection for
        # the model is separately capped to avoid huge TF-IDF memory usage.
        with gzip.open(expected, 'rb') as f:
            raw = f.read()
        if len(raw) != int(r.patch_bytes):
            raise ValueError(f'Patch length mismatch for {r.repo}/{r.commit_id}')
        if b'diff --git ' not in raw:
            raise ValueError(f'No git diff header for {r.repo}/{r.commit_id}')
        # Deterministic prefix window. This is a *baseline*, not a claim that
        # prefix truncation is optimal or unbiased.
        selected = raw[:max_patch_bytes].decode('utf-8', errors='replace')
        messages.append(msg)
        patches.append(selected)
        audit.append({'repo': r.repo, 'commit_id': r.commit_id, 'label': int(r.label),
                      'message_chars': len(msg), 'patch_bytes': len(raw),
                      'model_patch_bytes': min(len(raw), max_patch_bytes),
                      'patch_truncated': len(raw) > max_patch_bytes})
    return messages, patches, pd.DataFrame(audit)


def scores(y, pred, prob):
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'roc_auc': float(roc_auc_score(y, prob)),
            'pr_auc': float(average_precision_score(y, prob)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--extraction', default='reports/codejit_extraction_v1')
    p.add_argument('--out', default='reports/codejit_large_baselines_v1')
    p.add_argument('--max-patch-bytes', type=int, default=32768,
                   help='Deterministic patch prefix cap per commit; raw patch is preserved (default 32768)')
    p.add_argument('--max-features', type=int, default=15000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.max_patch_bytes < 1 or args.max_features < 10:
        p.error('max-patch-bytes must be >=1 and max-features >=10')
    train, val = load_manifest(args.train), load_manifest(args.val)
    if set(zip(train.repo, train.commit_id)) & set(zip(val.repo, val.commit_id)):
        raise ValueError('Train/val commit overlap')
    if set(train.repo) & set(val.repo):
        raise ValueError('Train/val repository overlap')
    if set(train.label.astype(int)) != {0, 1} or set(val.label.astype(int)) != {0, 1}:
        raise ValueError('Both classes required in train and val')
    extraction = Path(args.extraction)
    train_msg, train_patch, train_audit = load_records(train, extraction, args.max_patch_bytes)
    val_msg, val_patch, val_audit = load_records(val, extraction, args.max_patch_bytes)
    train_audit['subset'] = 'train'
    val_audit['subset'] = 'val'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([train_audit, val_audit], ignore_index=True).to_csv(out / 'input_coverage.csv', index=False)
    y_train, y_val = train.label.astype(int).to_numpy(), val.label.astype(int).to_numpy()
    input_sets = {
        'message': (train_msg, val_msg),
        'code': (train_patch, val_patch),
        'combined': ([f'MESSAGE {m}\nCODE {c}' for m, c in zip(train_msg, train_patch)],
                     [f'MESSAGE {m}\nCODE {c}' for m, c in zip(val_msg, val_patch)])
    }
    all_metrics = {}
    prediction_rows = []
    features_rows = []
    for name, (x_train, x_val) in input_sets.items():
        model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=args.max_features,
                                      lowercase=True, token_pattern=r'(?u)\b\w\w+\b',
                                      sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                       solver='liblinear', random_state=args.seed))
        ])
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_val)[:, list(model.named_steps['clf'].classes_).index(1)]
        preds = (probs >= 0.5).astype(int)
        all_metrics[name] = scores(y_val, preds, probs)
        terms = model.named_steps['tfidf'].get_feature_names_out()
        weights = model.named_steps['clf'].coef_[0]
        for direction, indices in [('positive', np.argsort(weights)[-30:][::-1]),
                                   ('negative', np.argsort(weights)[:30])]:
            for rank, idx in enumerate(indices, 1):
                features_rows.append({'model': name, 'direction': direction, 'rank': rank,
                                      'term': str(terms[idx]), 'coefficient': float(weights[idx])})
        for i, r in enumerate(val.itertuples(index=False)):
            prediction_rows.append({'model': name, 'repo': r.repo, 'commit_id': r.commit_id,
                                    'label': int(y_val[i]), 'predicted': int(preds[i]),
                                    'probability_positive': float(probs[i])})
        print(f'{name}: F1={all_metrics[name]["f1"]:.4f} ROC-AUC={all_metrics[name]["roc_auc"]:.4f} '
              f'PR-AUC={all_metrics[name]["pr_auc"]:.4f}')
    pd.DataFrame(prediction_rows).to_csv(out / 'validation_predictions.csv', index=False)
    pd.DataFrame(features_rows).to_csv(out / 'top_tfidf_features.csv', index=False)
    coverage = pd.concat([train_audit, val_audit], ignore_index=True)
    report = {
        'experiment': 'CodeJIT repository-disjoint matched-input TF-IDF logistic baseline',
        'evaluation': 'frozen validation only; held-out test not accessed',
        'source_sha256': {'train_manifest': sha_file(args.train), 'val_manifest': sha_file(args.val)},
        'configuration': {'seed': args.seed, 'max_patch_bytes': args.max_patch_bytes,
                          'max_features': args.max_features, 'vectorizer': 'word TF-IDF 1-2grams, sublinear_tf',
                          'classifier': 'class-weighted logistic regression, liblinear, max_iter=2000',
                          'threshold': 0.5, 'patch_selection': 'first N raw bytes decoded with replacement'},
        'sample_counts': {'train': len(train), 'val': len(val),
                          'train_repositories': int(train.repo.nunique()), 'val_repositories': int(val.repo.nunique()),
                          'train_label0': int((y_train == 0).sum()), 'train_label1': int((y_train == 1).sum()),
                          'val_label0': int((y_val == 0).sum()), 'val_label1': int((y_val == 1).sum())},
        'input_coverage': {'truncated_train': int(train_audit.patch_truncated.sum()),
                           'truncated_val': int(val_audit.patch_truncated.sum()),
                           'truncated_total': int(coverage.patch_truncated.sum()),
                           'max_full_patch_bytes': int(coverage.patch_bytes.max())},
        'validation_metrics': all_metrics,
        'limitations': ['CodeJIT negative class is curated, not representative of normal CI traffic.',
                        'Group-disjoint is not chronological and may share code across forks.',
                        'Raw patch prefix truncation may omit security-relevant hunks.',
                        'TF-IDF coefficients are correlational and not causal explanations.',
                        'Validation comparison is exploratory; no statistical significance asserted.',
                        'Original 18 unrecovered commits excluded; selection bias possible.']
    }
    (out / 'baseline_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'sample_counts': report['sample_counts'],
                      'input_coverage': report['input_coverage'],
                      'validation_metrics': all_metrics}, indent=2))
    print(f'Outputs: {out}')


if __name__ == '__main__':
    main()
