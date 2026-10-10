#!/usr/bin/env python3
"""Read-only schema/provenance audit for a candidate JIT vulnerability dataset.
Usage: python src/audit_jit_dataset.py --input data/raw/jit_candidate/file.csv
Supports CSV, JSON, JSONL, parquet. Never assigns labels or trains a model.
"""
import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd

ALIASES = {
 'commit': ['commit_hash','commit_id','commit','sha','hash','commit_sha','revision'],
 'project': ['project','repo','repository','repo_name','repository_name','project_name','repo_url'],
 'label': ['label','vul','vulnerable','is_vulnerable','dangerous','is_dangerous','buggy','target'],
 'diff': ['diff','patch','code_diff','commit_diff','changes'],
 'message': ['commit_message','message','msg'],
 'time': ['commit_date','committed_date','timestamp','date','commit_time','author_date'],
 'before_code': ['func_before','before_code','code_before','source_before','old_code'],
 'after_code': ['func_after','after_code','code_after','source_after','new_code'],
 'cve': ['cve','cve_id'],
 'cwe': ['cwe','cwe_id'],
}

def load(path):
 s=path.suffix.lower()
 if s=='.csv': return pd.read_csv(path, low_memory=False)
 if s=='.jsonl': return pd.read_json(path, lines=True)
 if s=='.json':
  obj=json.loads(path.read_text(encoding='utf-8'))
  if isinstance(obj,list): return pd.json_normalize(obj, sep='.')
  if isinstance(obj,dict):
   for key in ('data','records','items','samples','commits'):
    if isinstance(obj.get(key),list): return pd.json_normalize(obj[key],sep='.')
  raise ValueError('JSON structure is not a list of records; inspect manually')
 if s=='.parquet': return pd.read_parquet(path)
 raise ValueError('Supported: .csv, .json, .jsonl, .parquet')

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--input',required=True)
 ap.add_argument('--out',default='reports/jit_dataset_audit')
 ap.add_argument('--source-url',default='',help='Provenance URL supplied by researcher')
 args=ap.parse_args()
 path=Path(args.input)
 if not path.is_file(): raise SystemExit(f'File not found: {path}')
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 df=load(path)
 cols=list(df.columns)
 mapped={k:next((c for c in cols if c.lower() in names),None) for k,names in ALIASES.items()}
 report={'input':str(path.resolve()),'source_url':args.source_url or 'NOT PROVIDED',
         'sha256':sha256(path),'rows':int(len(df)),'columns':cols,
         'candidate_columns':mapped,'missing_fraction':{},'unique_counts':{},
         'label_distribution':{},'warnings':[]}
 for c in cols:
  report['missing_fraction'][c]=round(float(df[c].isna().mean()),5)
  try: report['unique_counts'][c]=int(df[c].nunique(dropna=True))
  except TypeError: report['unique_counts'][c]=None
 for key in ('label','project','commit','cve'):
  c=mapped[key]
  if c:
   vc=df[c].astype(str).value_counts(dropna=False).head(20)
   report[f'{key}_top_values']={str(k):int(v) for k,v in vc.items()}
 if mapped['label']:
  c=mapped['label'];report['label_distribution']={str(k):int(v) for k,v in df[c].astype(str).value_counts(dropna=False).items()}
  if c.lower()=='buggy': report['warnings'].append('BUGGY denotes defects in many datasets; do not assume security vulnerability.')
 else: report['warnings'].append('No familiar label column detected. Inspect schema before training.')
 if mapped['commit']:
  c=mapped['commit'];report['duplicate_commit_rows']=int(df[c].duplicated(keep=False).sum())
  report['warnings'].append('Repeated commits may be legitimate file/function records; inspect unit of analysis.')
 if mapped['diff'] is None:report['warnings'].append('No obvious diff/patch column: transformer commit-input construction may require git extraction.')
 if mapped['time'] is None:report['warnings'].append('No obvious timestamp: chronological validation may require repository mining.')
 if mapped['project'] is None:report['warnings'].append('No obvious project ID: project-disjoint evaluation needs verified repository identifiers.')
 report['warnings'].append('Column names cannot establish label semantics. Read the original dataset paper and construction code.')
 report['warnings'].append('Do not train or merge datasets until prediction unit, label meaning, and feature availability are verified.')
 (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
 with (out/'audit.txt').open('w',encoding='utf-8') as f:
  f.write(f'DATASET: {path}\nSHA256: {report["sha256"]}\nROWS: {len(df):,}\nCOLUMNS ({len(cols)}): {cols}\n\n')
  for k,c in mapped.items():f.write(f'{k:15} -> {c or "NOT FOUND"}\n')
  f.write('\nLABEL COUNTS:\n'+json.dumps(report['label_distribution'],indent=2)+'\n')
  f.write('\nWARNINGS:\n'+'\n'.join('- '+w for w in report['warnings'])+'\n')
 print((out/'audit.txt').read_text(encoding='utf-8'))
 print(f'Full JSON report: {out/"audit.json"}')
 print('\nPREVIEW (truncated non-sensitive scalar fields):')
 preview=df.head(3).copy()
 for c in preview.columns:
  preview[c]=preview[c].map(lambda x: str(x).replace('\n',' ')[:100] if not pd.isna(x) else '<NA>')
 print(preview.to_string(index=False,max_cols=12))

if __name__=='__main__':main()
