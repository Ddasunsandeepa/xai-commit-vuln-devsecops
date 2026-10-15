#!/usr/bin/env python3
"""Compare CodeBERT and GraphCodeBERT on the same CodeJIT validation commits.

No test data is read. Requires numpy, pandas, scipy, scikit-learn.
Example:
 python src/analyze_codejit_transformer_comparison.py --codebert reports/codejit_codebert_v1 --graphcodebert reports/codejit_graphcodebert_v1 --out reports/codejit_transformer_comparison_v1
"""
import argparse
import hashlib
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

MODES=('message','code','combined')
KEY=['repo','commit_id']

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def boolean(s):
    z=s.astype(str).str.lower().str.strip()
    if not z.isin(['true','false','0','1']).all():raise ValueError('Invalid boolean coverage values')
    return z.isin(['true','1'])

def load(root,family,mode):
    folder=root/mode
    pp=folder/'validation_predictions.csv'; cp=folder/'token_coverage.csv'; rp=folder/'results.json'
    for p in (pp,cp,rp):
        if not p.exists():raise FileNotFoundError(p)
    pred=pd.read_csv(pp,dtype={'repo':str,'commit_id':str},keep_default_na=False)
    cov=pd.read_csv(cp,dtype={'repo':str,'commit_id':str},keep_default_na=False)
    result=json.loads(rp.read_text())
    for c in KEY+['label','predicted','probability_positive']:
        if c not in pred:raise ValueError(f'{pp} missing {c}')
    for c in KEY+['subset','message_tokens_truncated','code_tokens_truncated','model_tokens']:
        if c not in cov:raise ValueError(f'{cp} missing {c}')
    if pred.duplicated(KEY).any() or cov.duplicated(['subset']+KEY).any():raise ValueError(f'Duplicate keys: {family}/{mode}')
    pred['label']=pd.to_numeric(pred.label).astype(int)
    pred['predicted']=pd.to_numeric(pred.predicted).astype(int)
    pred['probability_positive']=pd.to_numeric(pred.probability_positive)
    if not pred.label.isin([0,1]).all() or not pred.predicted.isin([0,1]).all() or not pred.probability_positive.between(0,1).all():raise ValueError('Invalid predictions')
    for col in ('message_tokens_truncated','code_tokens_truncated'):cov[col]=boolean(cov[col])
    cov['model_tokens']=pd.to_numeric(cov.model_tokens).astype(int)
    val=cov.loc[cov.subset=='val',KEY+['message_tokens_truncated','code_tokens_truncated','model_tokens']]
    if len(val)!=len(pred):raise ValueError('Coverage and prediction count mismatch')
    joined=pred.merge(val,on=KEY,how='left',validate='one_to_one',indicator=True)
    if not joined._merge.eq('both').all():raise ValueError('Coverage keys mismatch')
    joined=joined.drop(columns='_merge')
    return joined,cov,result,{'predictions':digest(pp),'coverage':digest(cp),'results':digest(rp)}

def metrics(y,p,s):
    d={'n':len(y),'accuracy':float(accuracy_score(y,p)),'f1':float(f1_score(y,p,zero_division=0)), 'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist()}
    d['roc_auc']=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None
    d['pr_auc']=float(average_precision_score(y,s)) if len(np.unique(y))==2 else None
    return d

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--codebert',default='reports/codejit_codebert_v1')
    ap.add_argument('--graphcodebert',default='reports/codejit_graphcodebert_v1')
    ap.add_argument('--out',default='reports/codejit_transformer_comparison_v1')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--max-length',type=int,default=256)
    args=ap.parse_args()
    if args.bootstrap<100:ap.error('--bootstrap must be at least 100')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    roots={'CodeBERT':Path(args.codebert),'GraphCodeBERT':Path(args.graphcodebert)}
    models={}; hashes={}; summaries={}; coverage_rows=[]; manifest_signatures={}
    for family,root in roots.items():
        for mode in MODES:
            name=f'{family}_{mode}'
            frame,cov,res,sha=load(root,family,mode)
            models[name]=frame;hashes[name]=sha
            if len(frame)!=314:raise ValueError(f'Expected 314 validation commits for {name}, got {len(frame)}')
            audit_path=root/mode/'data_audit.json'
            if audit_path.exists():
                audit=json.loads(audit_path.read_text())
                for key in ('train_manifest_sha256','val_manifest_sha256'):
                    if key in audit:manifest_signatures.setdefault(key,set()).add(audit[key])
            for subset in ('train','val'):
                c=cov.loc[cov.subset==subset]
                coverage_rows.append({'model':family,'mode':mode,'subset':subset,'n':int(len(c)),
                    'message_truncated_n':int(c.message_tokens_truncated.sum()),
                    'message_truncated_rate':float(c.message_tokens_truncated.mean()),
                    'code_truncated_n':int(c.code_tokens_truncated.sum()),
                    'code_truncated_rate':float(c.code_tokens_truncated.mean()),
                    'at_max_length_n':int((c.model_tokens==args.max_length).sum())})
    for key,values in manifest_signatures.items():
        if len(values)!=1:raise ValueError(f'Manifest SHA mismatch: {key}: {values}')
    first=next(iter(models.values()))[KEY+['label']].sort_values(KEY).reset_index(drop=True)
    for name,df in models.items():
        current=df[KEY+['label']].sort_values(KEY).reset_index(drop=True)
        if not first.equals(current):raise ValueError(f'Validation keys/labels differ: {name}')
    wide=first.copy()
    for name,df in models.items():
        keep=KEY+['predicted','probability_positive','message_tokens_truncated','code_tokens_truncated','model_tokens']
        renamed=df[keep].rename(columns={c:f'{name}_{c}' for c in keep if c not in KEY})
        wide=wide.merge(renamed,on=KEY,validate='one_to_one')
        wide[f'{name}_correct']=wide[f'{name}_predicted']==wide.label
    wide.to_csv(out/'aligned_validation_predictions.csv',index=False)
    pd.DataFrame(coverage_rows).to_csv(out/'token_coverage_summary.csv',index=False)
    y=wide.label.to_numpy(dtype=int)
    scores={};metric_rows=[]
    for name in models:
        p=wide[f'{name}_predicted'].to_numpy(dtype=int)
        s=wide[f'{name}_probability_positive'].to_numpy(dtype=float)
        scores[name]=metrics(y,p,s)
        metric_rows.append({'model':name,**{k:v for k,v in scores[name].items() if k!='confusion_matrix'}})
    pd.DataFrame(metric_rows).to_csv(out/'model_metrics.csv',index=False)
    groups={r:np.flatnonzero(wide.repo.to_numpy()==r) for r in wide.repo.unique()}
    repos=np.array(list(groups));rng=np.random.default_rng(args.seed)
    comparisons=[];pair_details={}
    # Primary controlled comparisons: model families within identical input mode.
    # Secondary comparisons: input modes within each model family.
    pairs=[(f'GraphCodeBERT_{m}',f'CodeBERT_{m}') for m in MODES]
    pairs += [(f'{family}_{a}',f'{family}_{b}') for family in roots for a,b in combinations(MODES,2)]
    for a,b in pairs:
        ac=wide[f'{a}_correct'].to_numpy(dtype=bool);bc=wide[f'{b}_correct'].to_numpy(dtype=bool)
        aonly=int((ac&~bc).sum());bonly=int((~ac&bc).sum());discord=aonly+bonly
        pval=float(binomtest(min(aonly,bonly),discord,.5).pvalue) if discord else 1.0
        diffs={k:scores[a][k]-scores[b][k] for k in ('f1','roc_auc','pr_auc')}
        draws={k:[] for k in diffs}
        apred=wide[f'{a}_predicted'].to_numpy(dtype=int);bpred=wide[f'{b}_predicted'].to_numpy(dtype=int)
        ascore=wide[f'{a}_probability_positive'].to_numpy(dtype=float);bscore=wide[f'{b}_probability_positive'].to_numpy(dtype=float)
        for _ in range(args.bootstrap):
            chosen=rng.choice(repos,size=len(repos),replace=True)
            ix=np.concatenate([groups[r] for r in chosen]); yy=y[ix]
            if len(np.unique(yy))<2:continue
            draws['f1'].append(float(f1_score(yy,apred[ix],zero_division=0)-f1_score(yy,bpred[ix],zero_division=0)))
            draws['roc_auc'].append(float(roc_auc_score(yy,ascore[ix])-roc_auc_score(yy,bscore[ix])))
            draws['pr_auc'].append(float(average_precision_score(yy,ascore[ix])-average_precision_score(yy,bscore[ix])))
        ci={k:[float(x) for x in np.quantile(v,[.025,.975])] for k,v in draws.items()}
        row={'a':a,'b':b,'a_only_correct':aonly,'b_only_correct':bonly,'both_correct':int((ac&bc).sum()),'both_wrong':int((~ac&~bc).sum()),'mcnemar_exact_p':pval}
        for k in diffs:row.update({f'{k}_difference':diffs[k],f'{k}_ci_low':ci[k][0],f'{k}_ci_high':ci[k][1]})
        comparisons.append(row);pair_details[f'{a}_vs_{b}']={'mcnemar_exact_p':pval,'differences':diffs,'cluster_bootstrap_95ci':ci,'discordant':[aonly,bonly]}
    pd.DataFrame(comparisons).to_csv(out/'paired_model_comparisons.csv',index=False)
    positive=wide.loc[wide.label==1].copy()
    positive['models_missing']=sum((positive[f'{n}_predicted']==0).astype(int) for n in models)
    positive.sort_values(['models_missing','repo','commit_id'],ascending=[False,True,True]).to_csv(out/'false_negative_cases.csv',index=False)
    report={'scope':'validation only; frozen test never accessed','validation_commits':len(wide),'repositories':len(repos),'bootstrap':args.bootstrap,'seed':args.seed,
        'manifest_sha256':{k:list(v)[0] for k,v in manifest_signatures.items()},'source_file_sha256':hashes,'metrics':scores,'coverage':coverage_rows,'pairs':pair_details,
        'false_negatives_all_six':int((positive.models_missing==6).sum()),'limitations':[
        'Single-seed results; checkpoint selection and tuned thresholds used validation labels.',
        'Paired repository bootstrap intervals are exploratory, conditional on observed repositories and do not account for training variability.',
        'Nine paired comparisons are exploratory; no multiplicity adjustment was performed.',
        'GraphCodeBERT used as a sequence classifier without explicit data-flow graph inputs.',
        'Heavy code truncation is descriptive; it does not establish a causal effect on predictive performance.',
        'Curated CodeJIT negative commits are not representative of real CI/CD benign commits.',
        'No held-out test data accessed.' ]}
    (out/'comparison_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'validation_commits':len(wide),'repositories':len(repos),'metrics':scores,
        'graphcodebert_minus_codebert':{m:pair_details[f'GraphCodeBERT_{m}_vs_CodeBERT_{m}'] for m in MODES},
        'false_negatives_all_six':report['false_negatives_all_six'],'output':str(out)},indent=2))
if __name__=='__main__':main()
