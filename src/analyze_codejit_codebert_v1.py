#!/usr/bin/env python3
"""Reproducible CodeJIT CodeBERT validation diagnostics; NEVER reads held-out test.

Run: python src/analyze_codejit_codebert_v1.py --root reports/codejit_codebert_v1 --out reports/codejit_codebert_diagnostics_v1
Requires: pandas, numpy, scipy, scikit-learn.
"""
import argparse
import hashlib
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

MODES = ('message', 'code', 'combined')
KEY = ['repo', 'commit_id']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def metric(y, p, prob):
    d = {'n': int(len(y)), 'accuracy': float(accuracy_score(y,p)),
         'f1': float(f1_score(y,p,zero_division=0)),
         'confusion_matrix': confusion_matrix(y,p,labels=[0,1]).tolist()}
    if len(set(y)) == 2:
        d['roc_auc'] = float(roc_auc_score(y,prob))
        d['pr_auc'] = float(average_precision_score(y,prob))
    else:
        d['roc_auc'] = None
        d['pr_auc'] = None
    return d

def load(root, mode):
    base = root / mode
    pp, cc = base/'validation_predictions.csv', base/'token_coverage.csv'
    if not pp.exists() or not cc.exists():
        raise FileNotFoundError(f'Missing predictions/coverage for {mode} in {base}')
    pred = pd.read_csv(pp, dtype={'repo':str,'commit_id':str},keep_default_na=False)
    cov = pd.read_csv(cc, dtype={'repo':str,'commit_id':str},keep_default_na=False)
    for col in KEY + ['label','predicted','probability_positive']:
        if col not in pred: raise ValueError(f'{pp} missing {col}')
    for col in KEY + ['subset','message_tokens_truncated','code_tokens_truncated','model_tokens']:
        if col not in cov: raise ValueError(f'{cc} missing {col}')
    if pred.duplicated(KEY).any() or cov.duplicated(['subset']+KEY).any():
        raise ValueError(f'Duplicate keys for {mode}')
    if len(pred)!=314: print(f'WARNING: {mode} has {len(pred)} predictions (expected 314 for current cohort)')
    if not pred['label'].astype(int).isin([0,1]).all(): raise ValueError('Invalid labels')
    if not pred['predicted'].astype(int).isin([0,1]).all(): raise ValueError('Invalid predictions')
    pred['label'] = pred['label'].astype(int)
    pred['predicted'] = pred['predicted'].astype(int)
    pred['probability_positive'] = pd.to_numeric(pred['probability_positive'])
    if not pred['probability_positive'].between(0,1).all(): raise ValueError('Invalid probabilities')
    if 'predicted_tuned' in pred:
        pred['predicted_tuned'] = pred['predicted_tuned'].astype(int)
    for flag in ['message_tokens_truncated','code_tokens_truncated']:
        raw = cov[flag].astype(str).str.strip().str.lower()
        if not raw.isin(['true','false','1','0']).all(): raise ValueError(f'Invalid {flag} in {mode}')
        cov[flag] = raw.isin(['true','1'])
    cov['model_tokens'] = pd.to_numeric(cov['model_tokens'],errors='raise').astype(int)
    if not cov['subset'].isin(['train','val']).all(): raise ValueError(f'Unknown subset in {mode}')
    val = cov[cov['subset']=='val'].copy()
    if len(val)!=len(pred): raise ValueError(f'Coverage count mismatch for {mode}')
    merged = pred.merge(val[KEY+['message_tokens_truncated','code_tokens_truncated','model_tokens']],on=KEY,how='left',validate='one_to_one',indicator=True)
    if not merged['_merge'].eq('both').all(): raise ValueError(f'Coverage key mismatch for {mode}')
    return merged.drop(columns='_merge'), cov, {'predictions':sha256(pp),'coverage':sha256(cc)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='reports/codejit_codebert_v1')
    parser.add_argument('--out',default='reports/codejit_codebert_diagnostics_v1')
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-length',type=int,default=256)
    args=parser.parse_args()
    if args.bootstrap<100: parser.error('--bootstrap must be >= 100')
    root,out=Path(args.root),Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    frames,coverage,hashes={}, {}, {}
    for mode in MODES:
        frames[mode],coverage[mode],hashes[mode]=load(root,mode)
    baseline=frames['message'][KEY+['label']].sort_values(KEY).reset_index(drop=True)
    for mode in MODES[1:]:
        check=frames[mode][KEY+['label']].sort_values(KEY).reset_index(drop=True)
        if not baseline.equals(check): raise ValueError(f'Mismatched validation keys or labels: {mode}')
    wide=baseline.copy()
    for mode in MODES:
        cols=KEY+['predicted','probability_positive','message_tokens_truncated','code_tokens_truncated','model_tokens']
        if 'predicted_tuned' in frames[mode]: cols+=['predicted_tuned']
        x=frames[mode][cols].rename(columns={c:f'{mode}_{c}' for c in cols if c not in KEY})
        wide=wide.merge(x,on=KEY,validate='one_to_one')
        wide[f'{mode}_correct']=(wide[f'{mode}_predicted']==wide.label)
    wide.to_csv(out/'aligned_validation_predictions.csv',index=False)
    report={'scope':'frozen validation only; test never read', 'n_validation':len(wide),
            'n_repositories':int(wide.repo.nunique()),'seed':args.seed,'bootstrap_draws':args.bootstrap,
            'max_length':args.max_length,'source_sha256':hashes,'models':{},'pairs':{},'limitations':[
            'All estimates are exploratory and conditional on the observed validation repositories.',
            'Checkpoint selection and threshold tuning used validation labels; tuned scores are optimistic.',
            'Truncation associations are descriptive and confounded by patch size, labels and repositories.',
            'TF-IDF and CodeBERT have different token/character budgets; architecture-only claims are unsupported.',
            'CodeJIT curated negative class is not representative of ordinary CI traffic.',
            'A 256-token limit does not establish which omitted patch regions contain security-relevant evidence.' ]}
    coverage_rows=[]
    for mode in MODES:
        m=wide
        pred=m[f'{mode}_predicted'].to_numpy(dtype=int)
        prob=m[f'{mode}_probability_positive'].to_numpy(dtype=float)
        entry={'validation_fixed_05':metric(m.label.to_numpy(),pred,prob)}
        if f'{mode}_predicted_tuned' in m:
            entry['validation_tuned_predictions_descriptive']=metric(m.label.to_numpy(),m[f'{mode}_predicted_tuned'].to_numpy(dtype=int),prob)
        for subset in ('train','val'):
            c=coverage[mode].loc[coverage[mode].subset==subset]
            item={'mode':mode,'subset':subset,'n':int(len(c)),
                  'message_truncated_n':int(c.message_tokens_truncated.sum()),
                  'message_truncated_rate':float(c.message_tokens_truncated.mean()),
                  'code_truncated_n':int(c.code_tokens_truncated.sum()),
                  'code_truncated_rate':float(c.code_tokens_truncated.mean()),
                  'at_max_length_n':int((c.model_tokens==args.max_length).sum()),
                  'at_max_length_rate':float((c.model_tokens==args.max_length).mean()),
                  'mean_model_tokens':float(c.model_tokens.mean())}
            coverage_rows.append(item)
        report['models'][mode]=entry
    pd.DataFrame(coverage_rows).to_csv(out/'token_coverage_summary.csv',index=False)
    report['token_coverage_summary']=coverage_rows
    rng=np.random.default_rng(args.seed)
    groups={k:np.flatnonzero(wide.repo.to_numpy()==k) for k in wide.repo.unique()}
    repos=np.array(list(groups))
    pair_rows=[]
    for a,b in combinations(MODES,2):
        ac=wide[f'{a}_correct'].to_numpy(dtype=bool)
        bc=wide[f'{b}_correct'].to_numpy(dtype=bool)
        aonly=int((ac & ~bc).sum()); bonly=int((~ac & bc).sum())
        discord=aonly+bonly
        pvalue=float(binomtest(min(aonly,bonly),discord,0.5).pvalue) if discord else 1.0
        point={}; samples={k:[] for k in ('f1','roc_auc','pr_auc')}
        y=wide.label.to_numpy(dtype=int)
        for mode in (a,b):
            point[mode]=metric(y,wide[f'{mode}_predicted'].to_numpy(dtype=int),wide[f'{mode}_probability_positive'].to_numpy(dtype=float))
        for k in samples: point[k+'_difference']=point[a][k]-point[b][k]
        for _ in range(args.bootstrap):
            picked=rng.choice(repos,size=len(repos),replace=True)
            indices=np.concatenate([groups[r] for r in picked])
            yy=y[indices]
            if len(np.unique(yy))<2: continue
            for mode in (a,b):
                pass
            for k,fun in [('f1',lambda y,p,s:f1_score(y,p,zero_division=0)),
                          ('roc_auc',lambda y,p,s:roc_auc_score(y,s)),
                          ('pr_auc',lambda y,p,s:average_precision_score(y,s))]:
                av=fun(yy,wide[f'{a}_predicted'].to_numpy(dtype=int)[indices],wide[f'{a}_probability_positive'].to_numpy(dtype=float)[indices])
                bv=fun(yy,wide[f'{b}_predicted'].to_numpy(dtype=int)[indices],wide[f'{b}_probability_positive'].to_numpy(dtype=float)[indices])
                samples[k].append(float(av-bv))
        cis={k:[float(v) for v in np.quantile(values,[.025,.975])] if values else None for k,values in samples.items()}
        result={'a_only_correct':aonly,'b_only_correct':bonly,'both_correct':int((ac&bc).sum()),
                'both_wrong':int((~ac&~bc).sum()),'mcnemar_exact_p':pvalue,
                'point_differences_a_minus_b':{k:point[k+'_difference'] for k in samples},
                'repository_cluster_bootstrap_95ci':cis}
        report['pairs'][f'{a}_vs_{b}']=result
        pair_rows.append({'a':a,'b':b,**{k:v for k,v in result.items() if not isinstance(v,dict)},
                          **{f'{k}_difference':point[k+'_difference'] for k in samples},
                          **{f'{k}_ci_low':cis[k][0] if cis[k] else np.nan for k in samples},
                          **{f'{k}_ci_high':cis[k][1] if cis[k] else np.nan for k in samples}})
    pd.DataFrame(pair_rows).to_csv(out/'paired_model_comparisons.csv',index=False)
    subgroup=[]
    for mode in MODES:
        for field in ('code_tokens_truncated','message_tokens_truncated'):
            col=f'{mode}_{field}'
            for flag in (False,True):
                s=wide[wide[col]==flag]
                if s.empty: continue
                met=metric(s.label.to_numpy(),s[f'{mode}_predicted'].to_numpy(dtype=int),s[f'{mode}_probability_positive'].to_numpy(dtype=float))
                subgroup.append({'mode':mode,'group':field,'truncated':flag,'n':len(s),
                                 'label0':int((s.label==0).sum()),'label1':int((s.label==1).sum()),
                                 'accuracy':met['accuracy'],'f1':met['f1'],
                                 'roc_auc':met['roc_auc'],'pr_auc':met['pr_auc']})
    pd.DataFrame(subgroup).to_csv(out/'truncation_subgroup_metrics.csv',index=False)
    false_neg=wide[wide.label==1].copy()
    false_neg['models_missed']=sum((false_neg[f'{m}_predicted']==0).astype(int) for m in MODES)
    false_neg.sort_values(['models_missed','repo','commit_id'],ascending=[False,True,True]).head(100).to_csv(out/'false_negative_cases.csv',index=False)
    report['false_negatives_all_three']=int((false_neg.models_missed==3).sum())
    (out/'diagnostics_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'validation':report['n_validation'],'repositories':report['n_repositories'],
                      'metrics':{m:report['models'][m]['validation_fixed_05'] for m in MODES},
                      'coverage':coverage_rows,'pairs':report['pairs'],
                      'false_negatives_all_three':report['false_negatives_all_three'],
                      'output':str(out)},indent=2))

if __name__=='__main__': main()
