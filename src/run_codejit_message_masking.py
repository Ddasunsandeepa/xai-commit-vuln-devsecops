#!/usr/bin/env python3
"""Controlled CodeJIT message masking ablation (train/validation only).

Run with the exact manifests used for your baseline. Requires pandas, numpy,
scikit-learn. Raw extracted metadata is read-only. No test data is accessed.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline

FIX = re.compile(r'(?i)\b(?:fix(?:es|ed|ing)?|bug[\s_-]?fix(?:es|ed|ing)?|patch(?:es|ed|ing)?|repair(?:s|ed|ing)?|resolv(?:e|es|ed|ing))\b')
SECURITY = re.compile(r'(?i)\b(?:secur(?:ity|e)|vulnerabilit(?:y|ies)|vuln(?:s)?|exploit(?:s|ed|ation)?|cve[\s_-]?\d{4}[\s_-]?\d{4,}|cwe[\s_-]?\d+|overflow|buffer[\s_-]?overflow)\b')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path, records):
    df = pd.read_csv(path, dtype={'repo':str, 'commit_id':str}, keep_default_na=False)
    if not {'repo','commit_id','label'}.issubset(df):
        raise ValueError('Manifest missing repo, commit_id or label')
    if df.duplicated(['repo','commit_id']).any():
        raise ValueError('Duplicate commit keys')
    if not df.label.astype(str).isin(['0','1']).all():
        raise ValueError('Unexpected labels')
    df = df.sort_values(['repo','commit_id']).reset_index(drop=True)
    messages=[]
    for r in df.itertuples(index=False):
        key=hashlib.sha256(f'{r.repo}\0{r.commit_id}'.encode()).hexdigest()[:24]
        p=records / f'{key}.json'
        meta=json.loads(p.read_text(encoding='utf-8'))
        if (meta.get('repo')!=r.repo or meta.get('commit_id')!=r.commit_id or
            str(meta.get('label'))!=str(r.label) or meta.get('status')!='ok'):
            raise ValueError(f'Metadata mismatch: {r.repo}/{r.commit_id}')
        msg=next((meta[k] for k in ('message','commit_message','full_message')
                  if isinstance(meta.get(k),str) and meta[k].strip()),None)
        if msg is None:
            raise ValueError(f'Missing message: {r.repo}/{r.commit_id}. Use matched-input eligible manifests.')
        messages.append(msg)
    return df, messages

def masked(msg, variant):
    if variant=='original': return msg
    msg=FIX.sub(' MASKFIXTOKEN ', msg)
    if variant=='fix_security_masked':
        msg=SECURITY.sub(' MASKSECURITYTOKEN ', msg)
    return msg

def metrics(y,pred,prob):
    return {'n':int(len(y)), 'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'roc_auc':float(roc_auc_score(y,prob)),
        'pr_auc':float(average_precision_score(y,prob)),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',required=True)
    ap.add_argument('--val',required=True)
    ap.add_argument('--extraction',default='reports/codejit_extraction_v1')
    ap.add_argument('--out',default='reports/codejit_message_masking_v1')
    ap.add_argument('--baseline',default='',help='Optional baseline folder for matching prediction-key checks')
    ap.add_argument('--max-features',type=int,default=15000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.max_features<10: ap.error('--max-features must be >= 10')
    records=Path(args.extraction)/'records'
    train,tm=load(args.train,records)
    val,vm=load(args.val,records)
    if set(train.repo)&set(val.repo): raise ValueError('Train/val repository overlap')
    if set(zip(train.repo,train.commit_id))&set(zip(val.repo,val.commit_id)):
        raise ValueError('Train/val commit overlap')
    if set(train.label.astype(int))!={0,1} or set(val.label.astype(int))!={0,1}:
        raise ValueError('Both labels needed in each partition')
    if args.baseline:
        p=Path(args.baseline)/'validation_predictions.csv'
        prior=pd.read_csv(p,dtype={'repo':str,'commit_id':str})
        keys=set(zip(val.repo,val.commit_id))
        for model, group in prior.groupby('model'):
            other=set(zip(group.repo,group.commit_id))
            if other!=keys or len(group)!=len(keys):
                raise ValueError(f'Baseline {model} validation keys differ; use matching manifests')
    ytr=train.label.astype(int).to_numpy()
    yv=val.label.astype(int).to_numpy()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]; impacts=[]; report={}
    for variant in ('original','fix_masked','fix_security_masked'):
        xtr=[masked(m,variant) for m in tm]
        xv=[masked(m,variant) for m in vm]
        pipe=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=args.max_features,
            lowercase=True,token_pattern=r'(?u)\b\w\w+\b',sublinear_tf=True)),
            ('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=args.seed))])
        pipe.fit(xtr,ytr)
        prob=pipe.predict_proba(xv)[:,1]
        pred=(prob>=0.5).astype(int)
        report[variant]=metrics(yv,pred,prob)
        report[variant]['train_messages_changed']=sum(a!=b for a,b in zip(tm,xtr))
        report[variant]['val_messages_changed']=sum(a!=b for a,b in zip(vm,xv))
        for r,p,pr in zip(val.itertuples(index=False),pred,prob):
            rows.append({'variant':variant,'repo':r.repo,'commit_id':r.commit_id,
                         'label':int(r.label),'predicted':int(p),'probability_positive':float(pr)})
    for split,df,msgs in (('train',train,tm),('val',val,vm)):
        for r,m in zip(df.itertuples(index=False),msgs):
            impacts.append({'subset':split,'repo':r.repo,'commit_id':r.commit_id,'label':int(r.label),
                'fix_term_matches':len(FIX.findall(m)),
                'security_term_matches':len(SECURITY.findall(m)),
                'message_chars':len(m)})
    pd.DataFrame(rows).to_csv(out/'validation_predictions.csv',index=False)
    pd.DataFrame(impacts).to_csv(out/'masking_coverage.csv',index=False)
    result={'status':'completed','train_count':len(train),'val_count':len(val),
        'train_repositories':int(train.repo.nunique()),'val_repositories':int(val.repo.nunique()),
        'train_manifest_sha256':digest(args.train),'val_manifest_sha256':digest(args.val),
        'seed':args.seed,'max_features':args.max_features,
        'masking_rules':{'fix':FIX.pattern,'security':SECURITY.pattern,
            'replacement_tokens':['MASKFIXTOKEN','MASKSECURITYTOKEN']},
        'validation_metrics':report,
        'limitations':['Masking is not a causal proof of shortcut learning.',
            'Shared replacement tokens may themselves retain class-associated information.',
            'Validation has been repeatedly examined; results are exploratory.',
            'CodeJIT curated negative class does not represent routine CI commits.',
            'Held-out test data was not accessed.']}
    (out/'masking_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('Saved:',out)
if __name__=='__main__': main()
