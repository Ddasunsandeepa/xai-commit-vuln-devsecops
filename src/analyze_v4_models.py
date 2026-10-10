#!/usr/bin/env python3
"""Paired CodeBERT vs GraphCodeBERT v4 analysis. CPU only.

Run from repository root:
 python src/analyze_v4_models.py --root reports/v4_models --out reports/v4_analysis

For prediction files lacking a stable sample identifier, inspect their format first.
Only use --allow-positional if BOTH files were generated from the same ordered
held-out dataframe, with no reordering or dropped rows.

Dependencies: pandas, numpy, scikit-learn, matplotlib.
"""
from __future__ import annotations
import argparse
import json
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = [42, 7, 21, 84, 123]
MODELS = ('codebert', 'graphcodebert')
LABELS = ['label','y_true','true_label','target','actual','ground_truth','true','y']
PROBS = ['probability','prob','y_prob','positive_probability','positive_prob',
         'prob_vulnerable','prob_vuln','score','prediction_probability','p1','proba']
PRED = ['y_pred','pred','prediction','predicted_label']
PROJECT = ['project','project_name','repo','repository']
IDS = ['row_id','sample_id','dataset_index','original_index','source_index',
       'commit_hash','commit_id','index','id']


def choose(df, names):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    return None


def normalize_id(s):
    return s.astype('string').str.strip()


def load_predictions(path, seed, model, allow_positional):
    if not path.exists():
        raise FileNotFoundError(f'Missing {model} seed {seed}: {path}')
    df = pd.read_csv(path)
    lc, pc = choose(df, LABELS), choose(df, PROBS)
    if lc is None or pc is None:
        raise ValueError(f'{path}: cannot identify label/probability columns. Found: {list(df.columns)}. '
                         'Update LABELS and PROBS near top of script to match your CSV.')
    ic, gc = choose(df, IDS), choose(df, PROJECT)
    y = pd.to_numeric(df[lc], errors='raise').astype(int)
    p = pd.to_numeric(df[pc], errors='raise').astype(float)
    if not y.isin([0,1]).all() or not np.isfinite(p).all() or not p.between(0,1).all():
        raise ValueError(f'{path}: labels must be 0/1 and probabilities within [0,1].')
    result = pd.DataFrame({'label': y, 'prob': p})
    if ic is not None:
        result['sample_id'] = normalize_id(df[ic])
        if result.sample_id.isna().any() or result.sample_id.duplicated().any():
            raise ValueError(f'{path}: sample identifiers contain nulls or duplicates.')
    elif allow_positional:
        result['sample_id'] = pd.Series(np.arange(len(df)), dtype='string')
    else:
        raise ValueError(f'{path}: no stable sample identifier found. For paired analysis, '
                         'add original dataset row_id/commit_hash to predictions, or use '
                         '--allow-positional ONLY after verifying identical row order.')
    if gc is not None:
        result['project'] = normalize_id(df[gc])
    result['model'] = model
    result['seed'] = seed
    predcol = choose(df, PRED)
    if predcol is not None:
        result['saved_pred'] = pd.to_numeric(df[predcol], errors='coerce')
    return result


def load_threshold(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing result metadata: {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    def search(o):
        if isinstance(o, dict):
            for k,v in o.items():
                if k.lower() in ('threshold','best_threshold','val_threshold','selected_threshold') and isinstance(v,(float,int)):
                    return float(v)
            for v in o.values():
                found = search(v)
                if found is not None: return found
        return None
    t = search(obj)
    if t is None or not 0 <= t <= 1:
        raise ValueError(f'No valid threshold found in {path}. Inspect results.json and update load_threshold().')
    return t


def metric(y, p, threshold):
    y, p = np.asarray(y,dtype=int), np.asarray(p,dtype=float)
    yp = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, yp, average='binary',zero_division=0)
    tn,fp,fn,tp = confusion_matrix(y,yp,labels=[0,1]).ravel()
    return {'n':int(len(y)), 'prevalence':float(y.mean()),
            'threshold':float(threshold), 'precision':float(precision),
            'recall':float(recall),'f1':float(f1),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
            'pr_auc':float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan,
            'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}


def attach_projects(pair, dataset_path):
    # The pair already has project columns when CSVs exported them.
    if 'project' in pair.columns and pair.project.notna().all():
        return pair
    if dataset_path is None:
        return pair
    ds = pd.read_csv(dataset_path, low_memory=False)
    project_col = choose(ds, PROJECT)
    if project_col is None:
        return pair
    for key in ('commit_hash','commit_id','row_id','sample_id','dataset_index','original_index'):
        c = choose(ds,[key])
        if c is None: continue
        lookup = pd.DataFrame({'sample_id':normalize_id(ds[c]),'project_from_data':normalize_id(ds[project_col])})
        lookup = lookup.dropna().drop_duplicates()
        if lookup.sample_id.duplicated().any(): continue
        mapped = pair[['sample_id']].merge(lookup,on='sample_id',how='left',validate='many_to_one')
        if mapped.project_from_data.notna().all():
            pair = pair.copy()
            pair['project'] = mapped.project_from_data.to_numpy()
            return pair
    return pair


def bootstrap(pair, ta, tb, reps, seed):
    if 'project' not in pair or pair.project.isna().any():
        return None
    groups = {k:grp.index.to_numpy() for k,grp in pair.groupby('project',sort=False)}
    keys = list(groups)
    if len(keys) < 2: return None
    rng = np.random.default_rng(seed)
    diffs = {m:[] for m in ('f1','roc_auc','pr_auc')}
    for _ in range(reps):
        sampled = rng.choice(len(keys),size=len(keys),replace=True)
        indices = np.concatenate([groups[keys[i]] for i in sampled])
        sub = pair.loc[indices]
        y = sub.label.to_numpy()
        if len(np.unique(y)) < 2: continue
        ma = metric(y,sub.prob_codebert.to_numpy(),ta)
        mb = metric(y,sub.prob_graphcodebert.to_numpy(),tb)
        for m in diffs: diffs[m].append(mb[m]-ma[m])
    if len(diffs['f1']) < max(30,reps//4):
        warnings.warn('Too few valid bootstrap replicates for seed '+str(seed))
        return None
    out = {'seed':seed,'n_projects':len(keys),'valid_replicates':len(diffs['f1'])}
    for m,vals in diffs.items():
        out[m+'_delta_ci_low'],out[m+'_delta_ci_high'] = map(float,np.quantile(vals,[.025,.975]))
        out[m+'_delta_bootstrap_mean'] = float(np.mean(vals))
    return out


def plot_comparison(metrics_df, out):
    for metric_name in ('f1','roc_auc','pr_auc'):
        pivot=metrics_df.pivot(index='seed',columns='model',values=metric_name).reindex(SEEDS)
        fig,ax=plt.subplots(figsize=(8,4.6))
        x=np.arange(len(pivot)); w=.36
        ax.bar(x-w/2,pivot['codebert'],width=w,label='CodeBERT')
        ax.bar(x+w/2,pivot['graphcodebert'],width=w,label='GraphCodeBERT')
        ax.set_xticks(x,pivot.index.astype(str));ax.set_xlabel('Frozen split seed')
        ax.set_ylabel(metric_name.upper().replace('_','-'));ax.set_ylim(0,1)
        ax.legend();fig.tight_layout()
        fig.savefig(out/f'paired_{metric_name}.png',dpi=160);plt.close(fig)


def plot_confusions(metrics_df,out):
    for model in MODELS:
        subset=metrics_df[metrics_df.model==model].sort_values('seed')
        fig,ax=plt.subplots(figsize=(8,4.5))
        x=np.arange(len(subset));w=.2
        for j,k in enumerate(('tp','fp','fn','tn')):
            ax.bar(x+(j-1.5)*w,subset[k],width=w,label=k.upper())
        ax.set_xticks(x,subset.seed.astype(str));ax.set_xlabel('Split seed')
        ax.set_ylabel('Number of test examples');ax.legend(ncol=4)
        fig.tight_layout();fig.savefig(out/f'confusion_counts_{model}.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reports/v4_models'))
    parser.add_argument('--out',type=Path,default=Path('reports/v4_analysis'))
    parser.add_argument('--dataset',type=Path,default=Path('data/processed/big_vul_enriched.csv'))
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--allow-positional',action='store_true')
    args=parser.parse_args()
    if args.bootstrap<100: parser.error('--bootstrap must be >= 100')
    args.out.mkdir(parents=True,exist_ok=True)
    all_metrics=[]; paired_rows=[]; ci_rows=[]; error_rows=[]; project_rows=[]; paired_frames=[]
    for seed in args.seeds:
        a=load_predictions(args.root/'codebert'/f'seed_{seed}'/'test_predictions.csv',seed,'codebert',args.allow_positional)
        b=load_predictions(args.root/'graphcodebert'/f'seed_{seed}'/'test_predictions.csv',seed,'graphcodebert',args.allow_positional)
        ta=load_threshold(args.root/'codebert'/f'seed_{seed}'/'results.json')
        tb=load_threshold(args.root/'graphcodebert'/f'seed_{seed}'/'results.json')
        if ('project' in a) and ('project' in b):
            aa=a.rename(columns={'project':'project_a'});bb=b.rename(columns={'project':'project_b'})
        else: aa=a;bb=b
        pair=aa.merge(bb,on='sample_id',how='outer',suffixes=('_codebert','_graphcodebert'),validate='one_to_one',indicator=True)
        if not pair._merge.eq('both').all():
            raise ValueError(f'Seed {seed}: model test samples do not match exactly. Aborting paired analysis.')
        pair=pair.drop(columns='_merge').reset_index(drop=True)
        if not (pair.label_codebert == pair.label_graphcodebert).all():
            raise ValueError(f'Seed {seed}: labels disagree for matching sample IDs.')
        pair['label']=pair.label_codebert.astype(int)
        if 'project_a' in pair and 'project_b' in pair:
            if not (pair.project_a.fillna('')==pair.project_b.fillna('')).all():
                raise ValueError(f'Seed {seed}: project names disagree for matching IDs.')
            pair['project']=pair.project_a
        elif 'project_codebert' in pair: pair['project']=pair.project_codebert
        elif 'project_graphcodebert' in pair: pair['project']=pair.project_graphcodebert
        pair=attach_projects(pair,args.dataset if args.dataset.exists() else None)
        pair['pred_codebert']=(pair.prob_codebert>=ta).astype(int)
        pair['pred_graphcodebert']=(pair.prob_graphcodebert>=tb).astype(int)
        pair['seed']=seed
        ma=metric(pair.label,pair.prob_codebert,ta)
        mb=metric(pair.label,pair.prob_graphcodebert,tb)
        all_metrics.extend([{'model':'codebert','seed':seed,**ma}, {'model':'graphcodebert','seed':seed,**mb}])
        paired_rows.append({'seed':seed,'n':len(pair),
                            **{f'delta_{m}':mb[m]-ma[m] for m in ('f1','precision','recall','roc_auc','pr_auc')},
                            'both_correct':int(((pair.pred_codebert==pair.label)&(pair.pred_graphcodebert==pair.label)).sum()),
                            'codebert_only_correct':int(((pair.pred_codebert==pair.label)&(pair.pred_graphcodebert!=pair.label)).sum()),
                            'graphcodebert_only_correct':int(((pair.pred_codebert!=pair.label)&(pair.pred_graphcodebert==pair.label)).sum()),
                            'both_wrong':int(((pair.pred_codebert!=pair.label)&(pair.pred_graphcodebert!=pair.label)).sum())})
        for model in MODELS:
            pred=pair[f'pred_{model}']
            condition=np.select([(pair.label==0)&(pred==1),(pair.label==1)&(pred==0),
                                 (pair.label==1)&(pred==1)],['FP','FN','TP'],default='TN')
            pair[f'outcome_{model}']=condition
            for _,r in pair.loc[condition!='TN'].iterrows():
                error_rows.append({'seed':seed,'model':model,'sample_id':r.sample_id,
                                   'project':r.get('project',None),'label':int(r.label),
                                   'prob':float(r[f'prob_{model}']),'outcome':r[f'outcome_{model}']})
            if 'project' in pair:
                for proj,grp in pair.groupby('project'):
                    pm=metric(grp.label,grp[f'prob_{model}'],ta if model=='codebert' else tb)
                    project_rows.append({'seed':seed,'model':model,'project':proj,**pm})
        ci=bootstrap(pair,ta,tb,args.bootstrap,seed)
        if ci: ci_rows.append(ci)
        else: print(f'WARNING seed {seed}: no project bootstrap; project identifiers missing. '
                    'Add project to prediction exports or use --dataset with matching commit_hash.')
        paired_frames.append(pair)
        print(f'Seed {seed}: CodeBERT F1={ma["f1"]:.4f}, GraphCodeBERT F1={mb["f1"]:.4f}, '
              f'delta={mb["f1"]-ma["f1"]:+.4f}; projects={pair.project.nunique() if "project" in pair else "UNKNOWN"}')
    metrics_df=pd.DataFrame(all_metrics)
    paired_df=pd.DataFrame(paired_rows)
    metrics_df.to_csv(args.out/'per_seed_metrics.csv',index=False)
    paired_df.to_csv(args.out/'paired_differences.csv',index=False)
    pd.DataFrame(error_rows).to_csv(args.out/'errors.csv',index=False)
    if project_rows: pd.DataFrame(project_rows).to_csv(args.out/'per_project_metrics.csv',index=False)
    if ci_rows: pd.DataFrame(ci_rows).to_csv(args.out/'project_cluster_bootstrap.csv',index=False)
    for frame in paired_frames:
        seed=int(frame.seed.iloc[0])
        frame.to_csv(args.out/f'paired_predictions_seed_{seed}.csv',index=False)
    summary=metrics_df.groupby('model')[['f1','precision','recall','roc_auc','pr_auc']].agg(['mean','std'])
    summary.to_csv(args.out/'model_summary.csv')
    plot_comparison(metrics_df,args.out)
    plot_confusions(metrics_df,args.out)
    report=['# Big-Vul v4: Paired CodeBERT / GraphCodeBERT analysis','',
            '## Evaluation design',
            'Five project-disjoint seed partitions (same partition for both models per seed).',
            'Each model uses its own validation-selected threshold. Test thresholds are not optimized.',
            'This is function-level classification on `func_before`, not proof of commit-introducing detection.',
            'Repeated seed test partitions overlap: they are not five independent external datasets.','',
            '## Mean and standard deviation (across seeds)','',summary.to_string(),'','## Paired differences',
            'Positive delta means GraphCodeBERT minus CodeBERT.','',paired_df.to_string(index=False),'']
    if ci_rows:
        report+=['## Within-seed paired project-cluster bootstrap (95% percentile intervals)','',
                 pd.DataFrame(ci_rows).to_string(index=False),'',
                 'These intervals describe project resampling WITHIN each test split. They do not',
                 'account for all uncertainty from model initialization, split construction, or',
                 'repeated overlapping test partitions. Avoid claiming statistical significance',
                 'solely because an interval from one seed excludes zero.','']
    else:
        report+=['## Project-level uncertainty unavailable','',
                 'No trustworthy project IDs were available. Do not interpret sample-level bootstrap',
                 'as a substitute for clustered inference.','']
    report+=['## Limitations','',
             '- GraphCodeBERT here uses standard sequence classification, not explicit graph-guided data-flow extraction.',
             '- Big-Vul function-level labels do not establish JIT vulnerability-introducing commit prediction.',
             '- Label provenance, feature leakage, project domain shift, and deployment prevalence need separate audits.',
             '- Thresholds can vary substantially across seeds; deployment requires calibration and policy validation.','',
             '## Recommended next experiment','',
             'Audit JIT-Vul label semantics, then establish commit-level baselines on leakage-safe splits.']
    (args.out/'research_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n',summary.to_string())
    print('\nSaved:',args.out.resolve())

if __name__=='__main__': main()
