#!/usr/bin/env python3
"""Audit extracted CodeJIT pilot CSV. Read-only, no network or model training."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='reports/codejit_pilot_200/pilot_commits.csv')
    ap.add_argument('--out', default='reports/codejit_pilot_quality')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, dtype={'repo': str, 'commit_id': str}, keep_default_na=False)
    required = {'repo', 'commit_id', 'label', 'status', 'commit_message', 'diff', 'diff_truncated', 'error'}
    if missing := required - set(df.columns):
        raise SystemExit(f'Missing columns: {sorted(missing)}')
    ok = df[df.status == 'ok'].copy()
    ok['diff_chars'] = ok['diff'].str.len()
    ok['message_chars'] = ok['commit_message'].str.len()
    ok['diff_sha256'] = ok['diff'].map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest() if s else '')
    ok['diff_files'] = ok['diff'].str.count(r'(?m)^diff --git ')
    ok['hunks'] = ok['diff'].str.count(r'(?m)^@@ ')
    ok['added_lines'] = ok['diff'].str.count(r'(?m)^\+(?!\+\+)')
    ok['removed_lines'] = ok['diff'].str.count(r'(?m)^-(?!--)')
    ok['has_patch_header'] = ok['diff'].str.contains(r'(?m)^diff --git ', regex=True)
    ok['truncated'] = ok['diff_truncated'].astype(str).str.lower().isin(['true','1','yes'])
    ok['message_mentions_fix'] = ok['commit_message'].str.contains(r'\b(fix(?:es|ed|ing)?|patch(?:ed)?|repair(?:ed)?)\b', case=False, regex=True)
    ok['message_mentions_security'] = ok['commit_message'].str.contains(r'\b(cve-\d{4}-\d+|vulnerab\w*|security|overflow|use.after.free|out.of.bounds)\b', case=False, regex=True)
    ok['repeated_patch'] = ok.duplicated('diff_sha256', keep=False) & ok['diff_sha256'].ne('')
    ok['risk_flag'] = ok['truncated'] | ~ok['has_patch_header'] | ok['repeated_patch']
    columns = ['repo','commit_id','label','status','diff_chars','diff_files','hunks','added_lines','removed_lines','truncated','has_patch_header','repeated_patch','message_mentions_fix','message_mentions_security','risk_flag','diff_sha256']
    ok[columns].to_csv(out/'commit_quality.csv', index=False)
    df[df.status != 'ok'][['repo','commit_id','label','status','error']].to_csv(out/'failures.csv', index=False)
    ok[ok.risk_flag][columns].to_csv(out/'review_queue.csv', index=False)
    dup = df.duplicated(['repo','commit_id']).sum()
    summary = {
        'source': args.input,
        'rows': int(len(df)),
        'status_counts': {str(k):int(v) for k,v in df.status.value_counts().items()},
        'successful_rows': int(len(ok)),
        'successful_label_counts': {str(k):int(v) for k,v in ok.label.value_counts().items()},
        'unique_successful_repositories': int(ok.repo.nunique()),
        'duplicate_repo_commit_pairs': int(dup),
        'truncated_diffs': int(ok.truncated.sum()),
        'missing_diff_headers': int((~ok.has_patch_header).sum()),
        'repeated_patch_rows': int(ok.repeated_patch.sum()),
        'messages_mention_fix': int(ok.message_mentions_fix.sum()),
        'messages_mention_security': int(ok.message_mentions_security.sum()),
        'diff_length_chars': {k:float(v) for k,v in ok.diff_chars.describe(percentiles=[.5,.9,.95,.99]).to_dict().items()} if len(ok) else {},
        'notes': [
            'Keyword flags are heuristic, not evidence of true vulnerability labels.',
            'Truncated patches must be recovered before final dataset construction.',
            'Git commit parent count and true merge status cannot be inferred reliably from this CSV.',
            'Repository URLs and original label construction remain to be verified.',
            'The pilot selection is not representative of the full CodeJIT population.'
        ]
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'\nFiles saved in {out}')

if __name__ == '__main__':
    main()
