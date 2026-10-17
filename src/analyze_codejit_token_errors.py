#!/usr/bin/env python3
"""Validation-only token-budget and paired error analysis for CodeJIT.

Run from repository root. Does not access held-out test or retrain models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

FAMILIES = ('codebert', 'graphcodebert')
MODES = ('code', 'combined')
KEY = ['repo', 'commit_id']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={'repo': str, 'commit_id': str}, keep_default_na=False)

def check_keys(df, expected, path):
    if df.duplicated(KEY).any():
        raise ValueError(f'Duplicate keys: {path}')
    keys = set(map(tuple, df[KEY].to_numpy()))
    if keys != expected:
        raise ValueError(f'Validation keys mismatch: {path} (missing={len(expected-keys)}, extra={len(keys-expected)})')

def bool_series(series, path):
    s = series.astype(str).str.lower()
    if not s.isin(['true', 'false', '1', '0']).all():
        raise ValueError(f'Unexpected boolean values in {path}')
    return s.isin(['true', '1'])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='reports/codejit_train_val_v2/train_manifest.csv')
    p.add_argument('--val', default='reports/codejit_train_val_v2/val_manifest.csv')
    p.add_argument('--codebert', default='reports/codejit_codebert_v1')
    p.add_argument('--graphcodebert', default='reports/codejit_graphcodebert_v1')
    p.add_argument('--hunk-aware', default='reports/codejit_hunk_aware_v1')
    p.add_argument('--out', default='reports/codejit_token_error_analysis_v1')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train = read_csv(a.train); val = read_csv(a.val)
    if train.duplicated(KEY).any() or val.duplicated(KEY).any():
        raise ValueError('Duplicate manifest keys')
    if set(train.repo) & set(val.repo):
        raise ValueError('Train/validation repository overlap')
    expected = set(map(tuple, val[KEY].to_numpy()))
    th, vh = sha(a.train), sha(a.val)
    details, summary, repo_rows, coverage_summary, selection_summary = [], [], [], [], []
    for family in FAMILIES:
        for mode in MODES:
            pair = {}
            for representation in ('prefix', 'hunk'):
                folder = ((Path(a.codebert if family == 'codebert' else a.graphcodebert) / mode)
                          if representation == 'prefix' else (Path(a.hunk_aware) / family / mode))
                audit = json.loads((folder / 'data_audit.json').read_text())
                if audit.get('train_manifest_sha256') != th or audit.get('val_manifest_sha256') != vh:
                    raise ValueError(f'Manifest hash mismatch: {folder}')
                predictions = read_csv(folder / 'validation_predictions.csv')
                check_keys(predictions, expected, folder)
                coverage = read_csv(folder / 'token_coverage.csv')
                if 'subset' not in coverage:
                    raise ValueError(f'Missing subset in {folder}/token_coverage.csv')
                coverage = coverage[coverage.subset == 'val'].copy()
                check_keys(coverage, expected, folder / 'token_coverage.csv')
                for field in ('message_tokens_truncated', 'code_tokens_truncated'):
                    if field not in coverage:
                        raise ValueError(f'Missing {field}: {folder}')
                    coverage[field] = bool_series(coverage[field], folder)
                    coverage_summary.append({'family':family,'mode':mode,'representation':representation,
                        'field':field,'validation_n':len(coverage),'truncated_n':int(coverage[field].sum()),
                        'truncated_pct':round(100*coverage[field].mean(),2)})
                frame = predictions.merge(coverage[KEY+['message_tokens_truncated','code_tokens_truncated']],
                    on=KEY,validate='one_to_one')
                frame['label'] = frame.label.astype(int)
                frame['predicted'] = frame.predicted.astype(int)
                frame['probability_positive'] = pd.to_numeric(frame.probability_positive)
                if not frame.label.isin([0,1]).all() or not frame.predicted.isin([0,1]).all():
                    raise ValueError(f'Invalid label or prediction: {folder}')
                frame = frame.sort_values(KEY).reset_index(drop=True)
                pair[representation] = frame
                if representation == 'hunk':
                    selection = folder / 'hunk_selection_coverage.csv'
                    if selection.is_file():
                        sel = read_csv(selection)
                        sel = sel[sel.subset == 'val'].copy()
                        check_keys(sel, expected, selection)
                        entry = {'family':family,'mode':mode,'validation_n':len(sel)}
                        for column in sel.columns:
                            if column not in KEY + ['subset','label'] and pd.api.types.is_numeric_dtype(sel[column]):
                                entry[f'{column}_mean'] = float(sel[column].mean())
                                entry[f'{column}_median'] = float(sel[column].median())
                        selection_summary.append(entry)
            old, new = pair['prefix'], pair['hunk']
            if not old[KEY+['label']].equals(new[KEY+['label']]):
                raise ValueError(f'Labels differ: {family} {mode}')
            merged = old[KEY+['label','predicted','probability_positive','message_tokens_truncated','code_tokens_truncated']].merge(
                new[KEY+['predicted','probability_positive','message_tokens_truncated','code_tokens_truncated']],
                on=KEY,validate='one_to_one',suffixes=('_prefix','_hunk'))
            merged['family'] = family; merged['mode'] = mode
            merged['prefix_correct'] = merged.predicted_prefix.eq(merged.label)
            merged['hunk_correct'] = merged.predicted_hunk.eq(merged.label)
            merged['status'] = 'unchanged'
            merged.loc[~merged.prefix_correct & merged.hunk_correct,'status'] = 'recovered'
            merged.loc[merged.prefix_correct & ~merged.hunk_correct,'status'] = 'new_error'
            merged['positive_status'] = 'not_applicable'
            pos = merged.label.eq(1)
            merged.loc[pos & merged.status.eq('recovered'),'positive_status'] = 'positive_recovered'
            merged.loc[pos & merged.status.eq('new_error'),'positive_status'] = 'positive_newly_missed'
            merged.loc[pos & merged.status.eq('unchanged') & ~merged.hunk_correct,'positive_status'] = 'positive_missed_by_both'
            merged.loc[pos & merged.status.eq('unchanged') & merged.hunk_correct,'positive_status'] = 'positive_correct_by_both'
            details.append(merged)
            for label in (0,1):
                for trunc in (False,True):
                    part = merged[merged.label.eq(label) & merged.code_tokens_truncated_prefix.eq(trunc)]
                    summary.append({'family':family,'mode':mode,'label':label,
                        'prefix_code_truncated':trunc,'n':len(part),
                        'prefix_correct':int(part.prefix_correct.sum()),
                        'hunk_correct':int(part.hunk_correct.sum()),
                        'recovered':int(part.status.eq('recovered').sum()),
                        'new_error':int(part.status.eq('new_error').sum()),
                        'note':'Descriptive subgroup; not causal or inferential'})
            for repo, group in merged.groupby('repo'):
                repo_rows.append({'family':family,'mode':mode,'repo':repo,'n':len(group),
                    'positives':int(group.label.sum()),
                    'prefix_correct':int(group.prefix_correct.sum()),
                    'hunk_correct':int(group.hunk_correct.sum()),
                    'recovered':int(group.status.eq('recovered').sum()),
                    'new_error':int(group.status.eq('new_error').sum())})
    all_details = pd.concat(details,ignore_index=True)
    all_details.to_csv(out/'paired_commit_error_details.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'errors_by_prefix_truncation.csv',index=False)
    pd.DataFrame(repo_rows).to_csv(out/'repository_error_summary.csv',index=False)
    pd.DataFrame(coverage_summary).to_csv(out/'token_truncation_summary.csv',index=False)
    pd.DataFrame(selection_summary).to_csv(out/'hunk_selection_summary.csv',index=False)
    report={'scope':'Validation-only, no held-out test accessed',
        'train_commits':len(train),'validation_commits':len(val),'validation_repositories':int(val.repo.nunique()),
        'train_manifest_sha256':th,'val_manifest_sha256':vh,
        'token_truncation':coverage_summary,
        'positive_error_changes':[{ 'family':f,'mode':m,
            'recovered':int(len(g[g.positive_status=='positive_recovered'])),
            'newly_missed':int(len(g[g.positive_status=='positive_newly_missed'])),
            'missed_by_both':int(len(g[g.positive_status=='positive_missed_by_both']))}
            for (f,m),g in all_details.groupby(['family','mode'])],
        'limitations':['Truncation comparisons are descriptive, not causal.',
            'Hunk selection can draw from full raw patches beyond the prefix byte window.',
            'Single seed, curated CodeJIT negatives, exploratory validation results.',
            'No held-out test accessed.']}
    (out/'analysis_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print('Output:',out)

if __name__ == '__main__':
    main()
