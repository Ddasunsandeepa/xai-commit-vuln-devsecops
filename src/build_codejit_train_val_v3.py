#!/usr/bin/env python3
"""Build a repository-disjoint CodeJIT development v3 from verified extraction records.

Preserves all existing v2 assignments, including repository membership. Never opens
held-out test extraction records or test patches. Writes only to a new v3 directory.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter

import pandas as pd

KEY = ['repo', 'commit_id']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', default='reports/codejit_cohort_v1/cohort_manifest.csv')
    ap.add_argument('--train-v2', default='reports/codejit_train_val_v2/train_manifest.csv')
    ap.add_argument('--val-v2', default='reports/codejit_train_val_v2/val_manifest.csv')
    ap.add_argument('--records', default='reports/codejit_extraction_v1/records')
    ap.add_argument('--out', default='reports/codejit_train_val_v3')
    ap.add_argument('--validation-fraction', type=float, default=0.20)
    args = ap.parse_args()
    if not (0 < args.validation_fraction < 1):
        ap.error('validation-fraction must be between zero and one')
    out = Path(args.out)
    if out.exists():
        ap.error(f'Output already exists: {out}. Refusing to overwrite any existing version.')
    cohort = pd.read_csv(args.cohort, dtype={'repo': str, 'commit_id': str})
    old_train = pd.read_csv(args.train_v2, dtype={'repo': str, 'commit_id': str})
    old_val = pd.read_csv(args.val_v2, dtype={'repo': str, 'commit_id': str})
    assert set(['repo','commit_id','label','split']).issubset(cohort.columns)
    assert not cohort.duplicated(KEY).any(), 'Duplicate frozen cohort keys'
    dev = cohort.loc[cohort['split'].eq('train'), KEY + ['label']].copy()
    test_repos = set(cohort.loc[cohort['split'].eq('test'), 'repo'])
    assert not (set(dev.repo) & test_repos), 'Frozen development/test repository overlap'
    assert not old_train.duplicated(KEY).any() and not old_val.duplicated(KEY).any()
    train_repos, val_repos = set(old_train.repo), set(old_val.repo)
    assert not train_repos & val_repos, 'v2 train/validation repository overlap'
    dev_index = dev.set_index(KEY)
    for name, old in [('train', old_train), ('validation', old_val)]:
        old_keys = pd.MultiIndex.from_frame(old[KEY])
        assert old_keys.isin(dev_index.index).all(), f'{name} contains keys outside frozen development'
        expected_labels = dev_index.loc[old_keys, 'label'].to_numpy()
        assert (expected_labels == old['label'].to_numpy()).all(), f'{name} labels disagree with frozen cohort'
    assert not (set(old_train.repo) | set(old_val.repo)) & test_repos
    # Read only records with keys in the frozen DEVELOPMENT pool. No test patch access.
    dev_keys = set(map(tuple, dev[KEY].itertuples(index=False, name=None)))
    metadata = {}
    for fp in sorted(Path(args.records).glob('*.json')):
        r = json.loads(fp.read_text(encoding='utf-8'))
        k = (str(r.get('repo')), str(r.get('commit_id')))
        if k not in dev_keys:
            continue
        if k in metadata:
            raise RuntimeError(f'Duplicate metadata key: {k}')
        metadata[k] = r
    valid, rejected = [], Counter()
    for r in dev.itertuples(index=False):
        key = (r.repo, r.commit_id)
        m = metadata.get(key)
        if m is None:
            rejected['no_metadata'] += 1
            continue
        if m.get('split') != 'train' or int(m.get('label', -1)) != int(r.label):
            rejected['metadata_mismatch'] += 1
            continue
        if m.get('status') != 'ok':
            rejected['extraction_failed'] += 1
            continue
        if not str(m.get('message') or '').strip():
            rejected['empty_message'] += 1
            continue
        patch_str = str(m.get('patch_path') or '')
        if not patch_str:
            rejected['missing_patch_path'] += 1
            continue
        patch_path = Path(patch_str)
        if not patch_path.is_absolute():
            patch_path = Path.cwd() / patch_path
        if not patch_path.is_file():
            rejected['missing_patch_file'] += 1
            continue
        try:
            with gzip.open(patch_path, 'rb') as f:
                raw = f.read()
            if not raw:
                rejected['empty_patch'] += 1
                continue
            if hashlib.sha256(raw).hexdigest() != str(m.get('diff_sha256')):
                rejected['patch_sha_mismatch'] += 1
                continue
            if len(raw) != int(m.get('diff_bytes')):
                rejected['patch_byte_count_mismatch'] += 1
                continue
        except Exception:
            rejected['invalid_patch'] += 1
            continue
        valid.append({'repo': r.repo, 'commit_id': r.commit_id,
                      'label': int(r.label), 'status': 'ok', 'error': '',
                      'patch_bytes': len(raw), 'patch_valid': True,
                      'integrity_issue': '', 'patch_path': patch_str})
    data = pd.DataFrame(valid)
    if data.empty:
        raise RuntimeError('No valid records found')
    assert not data.duplicated(KEY).any()
    old_train_keys = set(map(tuple, old_train[KEY].itertuples(index=False, name=None)))
    old_val_keys = set(map(tuple, old_val[KEY].itertuples(index=False, name=None)))
    new_keys = set(map(tuple, data[KEY].itertuples(index=False, name=None)))
    if not (old_train_keys | old_val_keys).issubset(new_keys):
        missing = (old_train_keys | old_val_keys) - new_keys
        raise RuntimeError(f'{len(missing)} v2 rows failed integrity checks; refusing to silently drop them. Example: {next(iter(missing))}')
    repo_stats = data.groupby('repo').agg(n=('label','size'), pos=('label','sum')).reset_index()
    assignment = {r: 'train' for r in train_repos}
    assignment.update({r: 'validation' for r in val_repos})
    # Greedy deterministic repository-level allocation, balancing size and labels.
    # This is exploratory development validation, NOT an untouched test set.
    counts = {'train': [0,0], 'validation': [0,0]}
    for row in repo_stats.itertuples(index=False):
        if row.repo in assignment:
            s = assignment[row.repo]
            counts[s][0] += int(row.n)
            counts[s][1] += int(row.pos)
    unseen = repo_stats.loc[~repo_stats.repo.isin(assignment)].copy()
    unseen['stable_key'] = unseen.repo.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    unseen = unseen.sort_values(['n','stable_key'], ascending=[False, True])
    total_n, total_pos = len(data), int(data.label.sum())
    goal_n = args.validation_fraction * total_n
    goal_pos = args.validation_fraction * total_pos
    for row in unseen.itertuples(index=False):
        def cost(s):
            v_n = counts['validation'][0] + (row.n if s == 'validation' else 0)
            v_p = counts['validation'][1] + (row.pos if s == 'validation' else 0)
            return abs(v_n - goal_n) / total_n + abs(v_p - goal_pos) / max(total_pos, 1)
        # Don't force an arbitrary 20% target by moving existing validation repositories.
        s = min(['train','validation'], key=lambda name: (cost(name), name != 'train'))
        assignment[row.repo] = s
        counts[s][0] += int(row.n)
        counts[s][1] += int(row.pos)
    data['subset'] = data.repo.map(assignment)
    assert data.subset.notna().all()
    train = data[data.subset.eq('train')].sort_values(KEY).reset_index(drop=True)
    val = data[data.subset.eq('validation')].sort_values(KEY).reset_index(drop=True)
    assert not set(train.repo) & set(val.repo)
    assert not (set(train.repo) | set(val.repo)) & test_repos
    assert old_train_keys.issubset(set(map(tuple, train[KEY].itertuples(index=False, name=None))))
    assert old_val_keys.issubset(set(map(tuple, val[KEY].itertuples(index=False, name=None))))
    out.mkdir(parents=True, exist_ok=False)
    train.to_csv(out / 'train_manifest.csv', index=False)
    val.to_csv(out / 'val_manifest.csv', index=False)
    pd.DataFrame([{'repo': k, 'subset': v} for k,v in sorted(assignment.items())]).to_csv(out / 'repo_assignments.csv', index=False)
    report = {
        'purpose': 'Exploratory development v3; held-out test untouched',
        'frozen_cohort_sha256': sha(args.cohort),
        'v2_train_manifest_sha256': sha(args.train_v2),
        'v2_val_manifest_sha256': sha(args.val_v2),
        'v3_train_manifest_sha256': sha(out / 'train_manifest.csv'),
        'v3_val_manifest_sha256': sha(out / 'val_manifest.csv'),
        'frozen_development_rows': len(dev),
        'valid_rows': len(data),
        'rejected_counts': dict(rejected),
        'train_rows': len(train), 'val_rows': len(val),
        'train_label_counts': {str(k): int(v) for k,v in train.label.value_counts().items()},
        'val_label_counts': {str(k): int(v) for k,v in val.label.value_counts().items()},
        'train_repositories': int(train.repo.nunique()),
        'val_repositories': int(val.repo.nunique()),
        'new_repositories_assigned': int(len(unseen)),
        'v2_train_rows_preserved': len(old_train),
        'v2_val_rows_preserved': len(old_val),
        'train_val_repository_overlap': 0,
        'development_test_repository_overlap': 0,
    }
    (out / 'build_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
