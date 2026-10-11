#!/usr/bin/env python3
"""Resume-safe Git extraction for a frozen CodeJIT cohort.

Writes one JSON metadata file and one gzip patch per successful commit, plus
summary CSV/JSON. Does not alter labels, sample composition, or train/test split.
Requires git. Repository URL inference from owner___repo is heuristic; supply
--repo-map CSV (repo,url) to override moved/inaccessible repositories.

Default --split train deliberately leaves the held-out test cohort untouched.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def run_git(args, cwd=None, timeout=90):
    env = os.environ.copy()
    env.update({'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never', 'GIT_LFS_SKIP_SMUDGE': '1'})
    try:
        result = subprocess.run(['git', *args], cwd=cwd, env=env, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=timeout)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[-700:])
        return result.stdout
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'git timed out after {timeout}s') from exc


def safe_repo_url(repo, overrides):
    if repo in overrides:
        url = overrides[repo]
    else:
        parts = repo.split('___')
        if len(parts) != 2 or not all(re.fullmatch(r'[A-Za-z0-9_.-]+', p) for p in parts):
            raise ValueError('cannot safely infer GitHub URL; provide --repo-map')
        url = f'https://github.com/{parts[0]}/{parts[1]}.git'
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('repo URL must be an HTTPS URL without credentials')
    return url


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def load_map(path):
    if not path:
        return {}
    df = pd.read_csv(path).fillna('')
    if not {'repo', 'url'}.issubset(df.columns):
        raise ValueError('--repo-map must have repo,url columns')
    return dict(zip(df['repo'].astype(str), df['url'].astype(str)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', default='reports/codejit_cohort_v1/cohort_manifest.csv')
    ap.add_argument('--out', default='reports/codejit_extraction_v1')
    ap.add_argument('--split', choices=['train', 'test', 'all'], default='train')
    ap.add_argument('--limit', type=int, default=100, help='Max NEW records attempted in this invocation; 0 = all')
    ap.add_argument('--timeout', type=int, default=90)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--retry-failed', action='store_true', help='Retry prior failures')
    ap.add_argument('--repo-map', default=None, help='Optional CSV with repo,url for renamed repositories')
    ap.add_argument('--retry-only-mapped', action='store_true', help='With --retry-failed, retry ONLY failed repositories present in --repo-map')
    ap.add_argument('--max-diff-bytes', type=int, default=0,
                    help='Reject patches exceeding this limit; 0 means no limit (never silently truncate)')
    args = ap.parse_args()
    if args.limit < 0 or args.retries < 0 or args.timeout < 1 or args.max_diff_bytes < 0:
        ap.error('limit/retries/max-diff-bytes must be nonnegative; timeout positive')
    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path, dtype={'repo': str, 'commit_id': str, 'split': str})
    required = {'repo', 'commit_id', 'label', 'split'}
    if not required.issubset(manifest.columns):
        raise ValueError(f'Missing manifest columns: {required - set(manifest.columns)}')
    if manifest.duplicated(['repo', 'commit_id']).any():
        raise ValueError('Manifest contains duplicate repository/commit pairs')
    if not set(manifest['split']).issubset({'train', 'test'}):
        raise ValueError('Unexpected split values')
    if set(manifest.loc[manifest.split.eq('train'), 'repo']) & set(manifest.loc[manifest.split.eq('test'), 'repo']):
        raise ValueError('Train/test repository overlap detected')
    if args.split != 'all':
        manifest = manifest[manifest['split'].eq(args.split)]
    manifest = manifest.sort_values(['repo', 'commit_id']).reset_index(drop=True)
    out = Path(args.out)
    records_dir = out / 'records'
    cache_dir = out / 'git_cache'
    records_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    overrides = load_map(args.repo_map)
    if args.retry_only_mapped and (not args.retry_failed or not overrides):
        ap.error('--retry-only-mapped requires --retry-failed and nonempty --repo-map')
    new_attempts = 0
    processed = []
    for row in manifest.itertuples(index=False):
        repo, commit, label, split = str(row.repo), str(row.commit_id), int(row.label), str(row.split)
        if not re.fullmatch(r'[a-fA-F0-9]{7,64}', commit):
            raise ValueError(f'Invalid SHA in manifest: {commit}')
        key = hashlib.sha256(f'{repo}\0{commit}'.encode()).hexdigest()[:24]
        metadata_path = records_dir / f'{key}.json'
        patch_path = records_dir / f'{key}.diff.gz'
        if metadata_path.exists():
            record = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (record.get('repo'), record.get('commit_id'), record.get('label'), record.get('split')) != (repo, commit, label, split):
                raise ValueError(f'Cached metadata mismatch: {metadata_path}')
            if record.get('status') == 'ok' and not patch_path.exists():
                raise ValueError(f'Missing patch for cached success: {patch_path}')
            if record.get('status') == 'ok' or not args.retry_failed:
                processed.append(record)
                continue
        if args.retry_only_mapped and repo not in overrides:
            continue
        if args.limit and new_attempts >= args.limit:
            continue
        new_attempts += 1
        record = {'repo': repo, 'commit_id': commit, 'label': label, 'split': split,
                  'status': 'failed', 'error': '', 'message': '', 'patch_path': '',
                  'diff_bytes': 0, 'diff_sha256': '', 'parent_count': None,
                  'is_merge': None, 'attempted_at_utc': pd.Timestamp.now(tz='UTC').isoformat()}
        try:
            url = safe_repo_url(repo, overrides)
            record['url'] = url
            git_dir = cache_dir / hashlib.sha256(url.encode()).hexdigest()[:24]
            if not git_dir.exists():
                git_dir.mkdir(parents=True)
                run_git(['init', '--bare', '--quiet', str(git_dir.resolve())], timeout=args.timeout)
            last_err = None
            for attempt in range(args.retries + 1):
                try:
                    run_git(['-C', str(git_dir), 'fetch', '--no-tags', '--depth=2', url, commit], timeout=args.timeout)
                    last_err = None
                    break
                except RuntimeError as exc:
                    last_err = exc
                    if attempt < args.retries:
                        time.sleep(min(2 ** attempt, 4))
            if last_err is not None:
                raise last_err
            parents_line = run_git(['-C', str(git_dir), 'show', '-s', '--format=%P', commit], timeout=args.timeout).strip()
            parents = parents_line.split() if parents_line else []
            record['parent_count'] = len(parents)
            record['is_merge'] = len(parents) > 1
            record['message'] = run_git(['-C', str(git_dir), 'show', '-s', '--format=%B', commit], timeout=args.timeout).strip()
            # No file filters or truncation: retain complete Git patch for provenance.
            patch = run_git(['-C', str(git_dir), 'show', '--format=', '--no-ext-diff', '--binary', commit], timeout=args.timeout)
            raw = patch.encode('utf-8', errors='replace')
            if not raw or b'diff --git ' not in raw:
                raise RuntimeError('empty patch or missing diff --git header')
            if args.max_diff_bytes and len(raw) > args.max_diff_bytes:
                raise RuntimeError(f'patch too large: {len(raw)} bytes; NOT truncated')
            with gzip.open(patch_path, 'wb') as f:
                f.write(raw)
            record.update(status='ok', error='', patch_path=str(patch_path),
                          diff_bytes=len(raw), diff_sha256=sha256_bytes(raw))
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {str(exc)[:700]}'
        write_json_atomic(metadata_path, record)
        processed.append(record)
        print(f'[{new_attempts}] {record["status"]}: {repo} {commit[:10]} '
              f'bytes={record["diff_bytes"]}', flush=True)

    # Rebuild summary from all cached records in the selected manifest, including earlier invocations.
    selected = {(str(r.repo), str(r.commit_id)) for r in manifest.itertuples(index=False)}
    all_records = []
    for path in records_dir.glob('*.json'):
        rec = json.loads(path.read_text(encoding='utf-8'))
        if (rec.get('repo'), rec.get('commit_id')) in selected:
            all_records.append(rec)
    all_records.sort(key=lambda r: (r['repo'], r['commit_id']))
    columns = ['repo', 'commit_id', 'label', 'split', 'status', 'error', 'url',
               'message', 'patch_path', 'diff_bytes', 'diff_sha256', 'parent_count', 'is_merge']
    pd.DataFrame(all_records, columns=columns).to_csv(out / f'{args.split}_extraction_index.csv', index=False)
    counts = Counter(r['status'] for r in all_records)
    summary = {'manifest': str(manifest_path), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'selected_split': args.split, 'selected_manifest_rows': len(manifest),
               'cached_or_attempted_rows': len(all_records), 'new_attempts_this_run': new_attempts,
               'status_counts': dict(counts), 'missing_not_attempted': len(manifest) - len(all_records),
               'merge_commits': sum(r.get('is_merge') is True for r in all_records),
               'complete_patches_saved': counts.get('ok', 0),
               'limitations': ['Repository URL inference may fail; use --repo-map overrides',
                               'Full patches are preserved, but transformer input windows still require controlled truncation',
                               'This extraction does not validate SZZ-derived labels or create realistic deployment negatives',
                               'Never tune models using held-out test split']}
    write_json_atomic(out / f'{args.split}_extraction_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
