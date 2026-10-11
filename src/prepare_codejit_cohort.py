#!/usr/bin/env python3
"""Prepare an auditable, repository-disjoint CodeJIT extraction cohort.

This script selects metadata records only. It does not fetch diffs, train models,
validate SZZ labels, or create representative deployment negatives.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ALIASES = {
    'repo': ['repo','repository','project','project_name','repo_name'],
    'commit_id': ['commit_id','commit_hash','hash','commit','sha'],
    'label': ['label','bug','vul','is_vulnerable','target'],
    'commit_date': ['commit_date','date','timestamp','commit_time','author_date'],
}

def get_col(df, key):
    mapping = {c.lower().strip(): c for c in df.columns}
    for alias in ALIASES[key]:
        if alias in mapping:
            return mapping[alias]
    if key == 'commit_date':
        return None
    raise ValueError(f'Missing {key}. Available columns: {list(df.columns)}')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/raw/jit_candidate/CodeJIT/Data/CodeJIT_Dataset_Commit_Hash.csv')
    p.add_argument('--out',default='reports/codejit_cohort_v1')
    p.add_argument('--target',type=int,default=2000)
    p.add_argument('--max-per-repo',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--test-fraction',type=float,default=.2)
    args=p.parse_args()
    if args.target < 10 or args.max_per_repo < 1 or not 0 < args.test_fraction < 1:
        p.error('target >= 10, max-per-repo >= 1, and 0 < test-fraction < 1 required')
    source=Path(args.input); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(source,low_memory=False)
    cols={k:get_col(raw,k) for k in ALIASES}
    data=pd.DataFrame({k:raw[v] for k,v in cols.items() if v is not None})
    data['repo']=data['repo'].astype('string').str.strip()
    data['commit_id']=data['commit_id'].astype('string').str.strip()
    data['label']=pd.to_numeric(data['label'],errors='coerce')
    valid=data['repo'].notna() & data['repo'].ne('') & data['commit_id'].notna() & data['commit_id'].str.fullmatch('[a-fA-F0-9]{7,64}').fillna(False) & data['label'].isin([0,1])
    invalid_count=int((~valid).sum())
    data=data[valid].copy()
    data['label']=data['label'].astype(int)
    duplicates=int(data.duplicated(['repo','commit_id']).sum())
    conflicts=int(data.groupby(['repo','commit_id'])['label'].nunique().gt(1).sum())
    if conflicts: raise ValueError(f'{conflicts} conflicting repo/commit labels; stop and inspect source')
    data=data.drop_duplicates(['repo','commit_id']).reset_index(drop=True)
    rng=np.random.default_rng(args.seed)
    # Randomize within each repo and class; round-robin across repositories and labels.
    pools={}
    for (repo,label), grp in data.groupby(['repo','label'],sort=True):
        ids=grp.index.to_numpy().copy();rng.shuffle(ids);pools[(str(repo),int(label))]=list(ids)
    keys=sorted(pools)
    rng.shuffle(keys)
    selected=[];counts={}; class_counts={0:0,1:0}
    # Keep approximately balanced class counts without assuming the original distribution.
    while len(selected)<args.target:
        progress=False
        for key in keys:
            if len(selected)>=args.target:break
            repo,label=key
            if counts.get(repo,0)>=args.max_per_repo or not pools[key]:continue
            if class_counts[label] > class_counts[1-label]:continue
            idx=pools[key].pop()
            selected.append(idx);counts[repo]=counts.get(repo,0)+1
            class_counts[label]+=1;progress=True
        if not progress:
            # If class balancing blocks further progress, relax it explicitly.
            for key in keys:
                if len(selected)>=args.target:break
                repo,label=key
                if counts.get(repo,0)>=args.max_per_repo or not pools[key]:continue
                idx=pools[key].pop();selected.append(idx)
                counts[repo]=counts.get(repo,0)+1;class_counts[label]+=1;progress=True
        if not progress:break
    cohort=data.loc[selected].copy().reset_index(drop=True)
    if cohort['repo'].nunique()<2:raise ValueError('Need at least two repositories for group-disjoint split')
    splitter=GroupShuffleSplit(n_splits=1,test_size=args.test_fraction,random_state=args.seed)
    train_idx,test_idx=next(splitter.split(cohort,cohort['label'],groups=cohort['repo']))
    cohort['split']='train';cohort.loc[test_idx,'split']='test'
    train_repos=set(cohort.loc[train_idx,'repo']);test_repos=set(cohort.loc[test_idx,'repo'])
    assert not train_repos.intersection(test_repos)
    assert not cohort.duplicated(['repo','commit_id']).any()
    # Warn if project splitting leaves a single class in a partition.
    warnings=[]
    for split in ['train','test']:
        if cohort.loc[cohort['split']==split,'label'].nunique()<2:
            warnings.append(f'{split} split has one class only; choose a different seed before modeling')
    for split in ['train','test']:
        cohort[cohort['split']==split].to_csv(out/f'{split}_metadata.csv',index=False)
    cohort.to_csv(out/'cohort_manifest.csv',index=False)
    audit={
        'source':str(source),'source_sha256':sha256(source),
        'raw_rows':len(raw),'invalid_rows':invalid_count,'duplicate_repo_commit_rows':duplicates,
        'label_conflicts':conflicts,'valid_unique_rows':len(data),
        'seed':args.seed,'requested_target':args.target,'actual_cohort_rows':len(cohort),
        'max_per_repo':args.max_per_repo,'requested_test_project_fraction':args.test_fraction,
        'cohort_labels':{str(k):int(v) for k,v in cohort['label'].value_counts().sort_index().items()},
        'cohort_repositories':int(cohort['repo'].nunique()),
        'split_summary':{s:{'rows':int(len(g)),'repos':int(g['repo'].nunique()),
            'labels':{str(k):int(v) for k,v in g['label'].value_counts().sort_index().items()}}
            for s,g in cohort.groupby('split')},
        'warnings':warnings,
        'limitations':['Metadata selection only; no commit retrieval or diff validation',
          'Original CodeJIT safe class consists of curated fixing commits, not ordinary production negatives',
          'Class-balanced, per-repository capped selection is deliberately not population representative',
          'Project-disjoint split is not a chronological split',
          'Splits must be frozen before model selection; do not tune against held-out test data',
          'SZZ-derived labels may contain noise and require provenance review']}
    (out/'cohort_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    print('Saved cohort manifests to',out)

if __name__=='__main__':main()
