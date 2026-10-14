#!/usr/bin/env python3
"""Generate explicit Android Gitiles URL overrides from a frozen CodeJIT manifest.
Does not alter manifest, labels or split. URLs are hypotheses until git fetch succeeds.
"""
import argparse
import csv
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--manifest',default='reports/codejit_cohort_v1/cohort_manifest.csv')
p.add_argument('--out',default='reports/codejit_android_repo_map.csv')
a=p.parse_args()
df=pd.read_csv(a.manifest,dtype=str)
repos=sorted(set(df['repo']))
rows=[]
for repo in repos:
    if repo.startswith('platform___'):
        parts=repo.split('___')
        # platform___frameworks___av => platform/frameworks/av
        path='/'.join(parts)
        rows.append({'repo':repo,'url':'https://android.googlesource.com/'+path})
path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
with path.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['repo','url']);w.writeheader();w.writerows(rows)
print(f'Wrote {len(rows)} Android repository mappings to {path}')
print('These mappings require validation; some historical commits may be missing.')
