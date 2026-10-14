#!/usr/bin/env python3
"""Small, auditable CodeJIT commit-diff extraction pilot. Never trains a model."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

DEFAULT_CSV = 'data/raw/jit_candidate/CodeJIT/Data/CodeJIT_Dataset_Commit_Hash.csv'
SHA = re.compile(r'^[0-9a-fA-F]{40}$')

def git(args, cwd, timeout=60):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True,
                          errors='replace', timeout=timeout, check=False)

def repo_url(name, overrides):
    if name in overrides:
        return overrides[name]
    parts = name.split('___')
    if len(parts) != 2 or not all(re.fullmatch(r'[A-Za-z0-9_.-]+', p) for p in parts):
        return None
    return f'https://github.com/{parts[0]}/{parts[1]}.git'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=DEFAULT_CSV)
    p.add_argument('--out', default='reports/codejit_pilot')
    p.add_argument('--max-commits', type=int, default=24)
    p.add_argument('--max-per-repo', type=int, default=2)
    p.add_argument('--max-diff-chars', type=int, default=16000)
    p.add_argument('--timeout', type=int, default=90)
    p.add_argument('--repo-map', help='Optional CSV with repo,url columns for ambiguous/non-GitHub repositories')
    a = p.parse_args()
    if a.max_commits < 1 or a.max_per_repo < 1:
        p.error('limits must be positive')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = out / 'git_cache'; cache.mkdir(exist_ok=True)
    overrides = {}
    if a.repo_map:
        mapping = pd.read_csv(a.repo_map)
        if not {'repo', 'url'} <= set(mapping.columns):
            p.error('repo map requires repo,url columns')
        overrides = dict(zip(mapping.repo.astype(str), mapping.url.astype(str)))
    df = pd.read_csv(a.csv, dtype={'commit_id': str, 'repo': str})
    required = {'commit_id', 'repo', 'label', 'commit_date'}
    if not required <= set(df.columns):
        p.error(f'Missing columns: {sorted(required - set(df.columns))}')
    df = df.drop_duplicates(['repo', 'commit_id']).copy()
    df['url'] = df.repo.map(lambda x: repo_url(x, overrides))
    # Favor modest-sized repositories: don't clone FFmpeg, QEMU, Linux, etc. in pilot.
    counts = df.repo.value_counts()
    df['repo_size'] = df.repo.map(counts)
    df = df[(df.repo_size.between(2, 100)) & df.url.notna() & df.commit_id.map(lambda x: bool(SHA.fullmatch(x)))].copy()
    df = df.sort_values(['repo_size', 'repo', 'commit_id'], kind='stable')
    # Spread across repositories rather than consuming the first few only.
    df['rank'] = df.groupby('repo').cumcount()
    df = df[df['rank'] < a.max_per_repo].sort_values(['rank', 'repo_size', 'repo']).head(a.max_commits)
    records = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        name, sha, url = r.repo, r.commit_id, r.url
        target = cache / hashlib.sha256(url.encode()).hexdigest()[:16]
        item = {'repo': name, 'commit_id': sha, 'label': int(r.label),
                'commit_date': r.commit_date, 'url': url, 'status': None,
                'commit_message': '', 'diff': '', 'diff_truncated': False,
                'error': ''}
        print(f'[{i}/{len(df)}] {name} {sha[:10]}', flush=True)
        try:
            if not (target / '.git').exists():
                target.mkdir(parents=True, exist_ok=True)
                init = git(['init', '-q'], target, a.timeout)
                if init.returncode:
                    raise RuntimeError(init.stderr[-350:])
            # Fetch only requested history. Server may reject SHA fetch: record failure.
            fetch = git(['-c', 'protocol.version=2', 'fetch', '--no-tags', '--depth=2', url, sha], target, a.timeout)
            if fetch.returncode:
                raise RuntimeError('fetch: ' + fetch.stderr[-500:].strip())
            show = git(['show', '--format=%B', '--no-patch', sha], target, a.timeout)
            diff = git(['show', '--format=', '--no-ext-diff', '--no-renames', '--find-copies=0',
                        '--unified=3', sha, '--'], target, a.timeout)
            if show.returncode or diff.returncode:
                raise RuntimeError('git show failed: ' + (show.stderr + diff.stderr)[-500:])
            item['commit_message'] = show.stdout[:4000]
            item['diff_truncated'] = len(diff.stdout) > a.max_diff_chars
            item['diff'] = diff.stdout[:a.max_diff_chars]
            item['status'] = 'ok' if diff.stdout.strip() else 'empty_diff'
        except (subprocess.TimeoutExpired, RuntimeError, OSError) as e:
            item['status'] = 'failed'
            item['error'] = str(e)[-600:]
        records.append(item)
    result = pd.DataFrame(records)
    result.to_csv(out / 'pilot_commits.csv', index=False)
    report = {'source_csv': a.csv, 'utc_generated': datetime.now(timezone.utc).isoformat(),
              'selected': len(records), 'status_counts': result.status.value_counts().to_dict(),
              'note': 'GitHub URL derived heuristically from repo name; verify identity and label provenance.'}
    (out / 'pilot_summary.json').write_text(json.dumps(report, indent=2))
    print('\n', json.dumps(report, indent=2))
    print('Saved:', out / 'pilot_commits.csv')

if __name__ == '__main__':
    main()
