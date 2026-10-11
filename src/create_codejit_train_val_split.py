#!/usr/bin/env python3
"""Create a reproducible, repository-disjoint train/validation split from audited CodeJIT patches.

Requires pandas and scikit-learn. Uses ONLY the verified training manifest; never reads test patches.

Example:
 python src/create_codejit_train_val_split.py \
   --manifest reports/codejit_cohort_v1/cohort_manifest.csv \
   --verified reports/codejit_training_audit_v3/verified_training_manifest.csv \
   --audit-summary reports/codejit_training_audit_v3/audit_summary.json \
   --out reports/codejit_train_val_v1 --seed 42 --val-fraction 0.20
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

KEYS = ['repo', 'commit_id']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_table(path):
    return pd.read_csv(path, dtype={'repo': 'string', 'commit_id': 'string', 'split': 'string'}, keep_default_na=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--verified', required=True)
    ap.add_argument('--audit-summary', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-fraction', type=float, default=0.20)
    ap.add_argument('--candidates', type=int, default=3000)
    ap.add_argument('--max-size-deviation', type=float, default=0.05,
                    help='Max absolute deviation from requested validation fraction; default 0.05')
    args = ap.parse_args()
    require(0 < args.val_fraction < 0.5, 'val-fraction must be between 0 and 0.5')
    require(args.candidates >= 1, 'candidates must be positive')
    require(0 <= args.max_size_deviation < 1, 'max-size-deviation must be between 0 and 1')
    manifest_path = Path(args.manifest)
    verified_path = Path(args.verified)
    summary_path = Path(args.audit_summary)
    manifest = read_table(manifest_path)
    verified = read_table(verified_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    require(summary.get('manifest_sha256') == digest(manifest_path), 'Audit manifest hash does not match input manifest')
    for name, frame in [('cohort manifest', manifest), ('verified manifest', verified)]:
        require(set(KEYS + ['label']).issubset(frame.columns), f'{name} missing required columns')
        require(not frame.duplicated(KEYS).any(), f'{name} contains duplicate repo+commit keys')
        require(frame['repo'].astype(str).str.len().gt(0).all() and frame['commit_id'].astype(str).str.len().gt(0).all(), f'{name} contains blank keys')
        require(frame['label'].astype(str).isin(['0', '1']).all(), f'{name} contains invalid labels')
    require('split' in manifest.columns, 'cohort manifest missing split')
    require(set(manifest['split'].astype(str)) == {'train', 'test'}, 'Expected exactly train and test in frozen cohort manifest')
    original_train = manifest[manifest['split'].eq('train')].copy()
    original_test = manifest[manifest['split'].eq('test')].copy()
    require(not (set(original_train['repo']) & set(original_test['repo'])), 'Frozen train/test repo overlap')
    require(len(verified) == int(summary['verified_patches']), 'Verified row count differs from audit summary')
    require('patch_valid' in verified.columns, 'Verified manifest missing patch_valid flag')
    require(verified['patch_valid'].astype(str).str.lower().eq('true').all(), 'Verified manifest contains invalid patches')
    require('integrity_issue' not in verified.columns or verified['integrity_issue'].astype(str).eq('').all(), 'Verified manifest contains integrity issues')
    joined = verified.merge(original_train[KEYS + ['label']], on=KEYS, how='left', validate='one_to_one', indicator=True, suffixes=('_verified', '_original'))
    require(joined['_merge'].eq('both').all(), 'Verified manifest contains commits outside frozen training cohort')
    require(joined['label_verified'].astype(int).eq(joined['label_original'].astype(int)).all(), 'Verified label mismatch')
    require(not (set(verified['repo']) & set(original_test['repo'])), 'Verified training overlaps frozen test repos')
    require(verified['repo'].nunique() >= 3, 'Not enough repositories for a disjoint split')
    # Use only audited training records. No patch content or held-out test features are read.
    work = verified.copy().sort_values(KEYS, kind='mergesort').reset_index(drop=True)
    labels = work['label'].astype(int).to_numpy()
    groups = work['repo'].astype(str).to_numpy()
    n = len(work)
    prevalence = float(labels.mean())
    splitter = GroupShuffleSplit(n_splits=args.candidates, test_size=args.val_fraction, random_state=args.seed)
    best = None
    evaluated = 0
    for train_idx, val_idx in splitter.split(np.zeros(n), labels, groups):
        evaluated += 1
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        if len(set(train_labels)) != 2 or len(set(val_labels)) != 2:
            continue
        fraction = len(val_idx) / n
        if abs(fraction - args.val_fraction) > args.max_size_deviation:
            continue
        # Balance the sample fraction and class prevalence without using model performance.
        score = abs(fraction - args.val_fraction) / args.val_fraction + 2 * abs(float(val_labels.mean()) - prevalence)
        if best is None or score < best[0] - 1e-12:
            best = (score, train_idx, val_idx, fraction)
    require(best is not None, 'No valid split found; increase --candidates or --max-size-deviation')
    _, train_idx, val_idx, actual_fraction = best
    work['subset'] = 'train'
    work.loc[val_idx, 'subset'] = 'val'
    train = work[work['subset'].eq('train')].copy()
    val = work[work['subset'].eq('val')].copy()
    require(not (set(train.repo) & set(val.repo)), 'Train/val repository overlap')
    require(not ((set(train.repo) | set(val.repo)) & set(original_test.repo)), 'Train/val/test repository overlap')
    require(len(train) + len(val) == len(verified), 'Missing assignments')
    require(set(train.label.astype(int)) == {0, 1} and set(val.label.astype(int)) == {0, 1}, 'Missing class in split')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Keep patch paths and original audit metadata, plus explicit subset assignment.
    work.to_csv(out / 'train_val_assignments.csv', index=False)
    train.to_csv(out / 'train_manifest.csv', index=False)
    val.to_csv(out / 'val_manifest.csv', index=False)
    counts = pd.crosstab(work['subset'], work['label'].astype(int)).reindex(index=['train', 'val'], columns=[0, 1], fill_value=0)
    counts.to_csv(out / 'split_label_counts.csv')
    repo_summary = work.groupby(['repo', 'subset'], as_index=False).agg(commits=('commit_id', 'size'), label0=('label', lambda s: int((s.astype(int) == 0).sum())), label1=('label', lambda s: int((s.astype(int) == 1).sum())))
    repo_summary.to_csv(out / 'repository_assignments.csv', index=False)
    report = {
        'status': 'validated', 'seed': args.seed, 'algorithm': 'GroupShuffleSplit, best of candidate splits by size and label prevalence',
        'candidates_evaluated': evaluated, 'requested_val_fraction': args.val_fraction,
        'actual_val_fraction': actual_fraction, 'max_size_deviation': args.max_size_deviation,
        'source': {'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path),
                   'verified_manifest': str(verified_path), 'verified_manifest_sha256': digest(verified_path),
                   'audit_summary': str(summary_path), 'audit_summary_sha256': digest(summary_path)},
        'verified_records': n, 'verified_repositories': int(work.repo.nunique()),
        'train': {'commits': len(train), 'repositories': int(train.repo.nunique()), 'label0': int((train.label.astype(int) == 0).sum()), 'label1': int((train.label.astype(int) == 1).sum()), 'positive_fraction': float(train.label.astype(int).mean())},
        'val': {'commits': len(val), 'repositories': int(val.repo.nunique()), 'label0': int((val.label.astype(int) == 0).sum()), 'label1': int((val.label.astype(int) == 1).sum()), 'positive_fraction': float(val.label.astype(int).mean())},
        'frozen_test': {'commits_in_cohort_manifest': len(original_test), 'repositories': int(original_test.repo.nunique()), 'patches_not_read': True},
        'checks': {'train_val_repo_overlap': 0, 'train_test_repo_overlap': 0, 'val_test_repo_overlap': 0, 'integrity_issues_in_verified_input': 0},
        'outputs': {'assignments_sha256': digest(out / 'train_val_assignments.csv'), 'train_sha256': digest(out / 'train_manifest.csv'), 'val_sha256': digest(out / 'val_manifest.csv')},
        'limitations': ['Group-disjoint split is not chronological and does not guarantee independence across forks or shared code.',
                        'CodeJIT negative examples are curated from vulnerability-fixing commits; do not interpret validation scores as deployment performance.',
                        'The 18 missing records are excluded; extraction selection bias remains possible.',
                        'Do not tune on the frozen test cohort.']
    }
    (out / 'split_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('\nOutputs saved in', out)


if __name__ == '__main__':
    main()
