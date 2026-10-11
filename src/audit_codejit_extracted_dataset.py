#!/usr/bin/env python3
"""Audit frozen CodeJIT training extraction; never reads held-out test patches.

python src/audit_codejit_extracted_dataset.py \
 --manifest reports/codejit_cohort_v1/cohort_manifest.csv \
 --extraction reports/codejit_extraction_v1 \
 --out reports/codejit_training_audit_v1
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--extraction', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extraction = Path(args.extraction)
    manifest = pd.read_csv(args.manifest, dtype={'repo': str, 'commit_id': str, 'split': str})
    expected_cols = {'repo', 'commit_id', 'label', 'split'}
    if not expected_cols.issubset(manifest):
        raise ValueError(f'Manifest missing {expected_cols - set(manifest)}')
    if manifest.duplicated(['repo', 'commit_id']).any():
        raise ValueError('Duplicate manifest keys')
    train = manifest[manifest['split'].eq('train')].copy()
    test = manifest[manifest['split'].eq('test')].copy()
    if set(train.repo) & set(test.repo):
        raise ValueError('Train/test repository overlap')
    index_path = extraction / 'train_extraction_index.csv'
    index = pd.read_csv(index_path, dtype={'repo': str, 'commit_id': str, 'split': str})
    if index.duplicated(['repo', 'commit_id']).any():
        raise ValueError('Duplicate extraction index keys')
    if not set(index['split'].dropna()).issubset({'train'}):
        raise ValueError('Index contains non-training rows')
    merged = train.merge(index, on=['repo', 'commit_id'], how='left', suffixes=('_manifest', '_extracted'), validate='one_to_one', indicator=True)
    if len(merged) != len(train):
        raise ValueError('Manifest join changed row count')
    extra = index.merge(train[['repo','commit_id']], on=['repo','commit_id'], how='left', indicator=True)
    if extra['_merge'].eq('left_only').any():
        raise ValueError('Extraction index has keys absent from training manifest')
    if (merged.loc[merged['_merge'].eq('both'), 'label_manifest'].astype(int) != merged.loc[merged['_merge'].eq('both'), 'label_extracted'].astype(int)).any():
        raise ValueError('Label mismatch between manifest and extraction index')
    records = []
    for r in merged.to_dict('records'):
        status = r.get('status') if r['_merge'] == 'both' else 'not_attempted'
        status = str(status) if pd.notna(status) else 'not_attempted'
        row = {'repo': r['repo'], 'commit_id': r['commit_id'], 'label': int(r['label_manifest']), 'status': status,
               'error': '' if pd.isna(r.get('error')) else str(r.get('error')),
               'patch_bytes': None, 'patch_valid': False, 'integrity_issue': '',
               'patch_path': '' if pd.isna(r.get('patch_path')) else str(r.get('patch_path'))}
        if status == 'ok':
            key = hashlib.sha256(f"{r['repo']}\0{r['commit_id']}".encode()).hexdigest()[:24]
            expected = extraction / 'records' / f'{key}.diff.gz'
            try:
                if not expected.is_file():
                    raise ValueError('patch missing at expected records location')
                with gzip.open(expected, 'rb') as f:
                    raw = f.read()
                if b'diff --git ' not in raw:
                    raise ValueError('missing diff --git header')
                recorded_bytes = int(r['diff_bytes'])
                if len(raw) != recorded_bytes:
                    raise ValueError(f'byte count mismatch {len(raw)} != {recorded_bytes}')
                actual_sha = hashlib.sha256(raw).hexdigest()
                if actual_sha != str(r['diff_sha256']):
                    raise ValueError('SHA256 mismatch')
                metadata_path = extraction / 'records' / f'{key}.json'
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                if any(metadata.get(k) != v for k, v in [('repo',r['repo']),('commit_id',r['commit_id']),('label',int(r['label_manifest'])),('status','ok')]):
                    raise ValueError('metadata identity/status mismatch')
                row['patch_bytes'] = len(raw)
                row['patch_valid'] = True
                row['patch_path'] = str(expected)
            except Exception as e:
                row['integrity_issue'] = f'{type(e).__name__}: {e}'
        records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(out / 'all_training_records_audit.csv', index=False)
    valid = result[result['patch_valid']].copy()
    valid.to_csv(out / 'verified_training_manifest.csv', index=False)
    failures = result[~result['patch_valid']].copy()
    failures.to_csv(out / 'failures_and_integrity_issues.csv', index=False)
    by_label = pd.crosstab(result['label'], result['status']).reindex(index=[0,1], fill_value=0)
    by_label.to_csv(out / 'extraction_by_label.csv')
    by_repo = result.groupby('repo').agg(selected=('commit_id','size'), verified=('patch_valid','sum'), label0=('label',lambda x:int((x==0).sum())),label1=('label',lambda x:int((x==1).sum()))).reset_index()
    by_repo['failed_or_missing'] = by_repo['selected'] - by_repo['verified']
    by_repo.sort_values(['failed_or_missing','selected'],ascending=False).to_csv(out / 'repository_coverage.csv', index=False)
    sizes = valid['patch_bytes'].astype(float)
    summary = {
        'manifest_sha256': sha(Path(args.manifest)), 'index_sha256': sha(index_path),
        'training_selected': int(len(train)), 'test_metadata_rows_not_inspected': int(len(test)),
        'training_attempted': int(result['status'].ne('not_attempted').sum()),
        'status_counts': {str(k):int(v) for k,v in result['status'].value_counts().items()},
        'verified_patches': int(len(valid)), 'failed_or_missing_or_corrupt': int(len(failures)),
        'integrity_issues': int(result['integrity_issue'].ne('').sum()),
        'verified_label_counts': {str(k):int(v) for k,v in valid['label'].value_counts().sort_index().items()},
        'selected_label_counts': {str(k):int(v) for k,v in result['label'].value_counts().sort_index().items()},
        'repositories_selected': int(result['repo'].nunique()), 'repositories_with_verified_patch': int(valid['repo'].nunique()),
        'patch_bytes': {k: (float(sizes.quantile(q)) if len(sizes) else None) for k,q in [('p50',.5),('p90',.9),('p95',.95),('p99',.99)]},
        'patch_bytes_max': int(sizes.max()) if len(sizes) else None,
        'notes': ['This audits patch integrity, not ground-truth vulnerability label correctness.',
                  'The original CodeJIT negative class is curated; results are not representative of everyday CI traffic.',
                  'Do not use the held-out test cohort for feature selection or tuning.',
                  'Do not treat verified patches as an unbiased sample without checking failure patterns.']
    }
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('\nLargest repository failure clusters:')
    print(by_repo.sort_values('failed_or_missing',ascending=False).head(12).to_string(index=False))

if __name__ == '__main__':
    main()
