#!/usr/bin/env python3
"""Frozen-manifest Big-Vul v4 CodeBERT / GraphCodeBERT comparison.
Run from repository root. GraphCodeBERT here uses sequence classification,
not explicit graph/data-flow input construction.
"""
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def metrics(y,p,t):
    z=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return dict(threshold=float(t),precision=float(precision_score(y,z,zero_division=0)),recall=float(recall_score(y,z,zero_division=0)),f1=float(f1_score(y,z,zero_division=0)),roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))

def threshold_for(y,p):
    grid=np.linspace(.01,.99,99)
    scores=[f1_score(y,p>=t,zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/processed/big_vul_enriched.csv')
    ap.add_argument('--manifest-dir',default='reports/v4_splits')
    ap.add_argument('--out',default='reports/v4_models')
    ap.add_argument('--models',nargs='+',choices=['codebert','graphcodebert'],default=['codebert'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,7,21,84,123])
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--grad-accum',type=int,default=4)
    ap.add_argument('--max-length',type=int,default=256)
    ap.add_argument('--learning-rate',type=float,default=2e-5)
    ap.add_argument('--patience',type=int,default=2)
    ap.add_argument('--model-seed',type=int,default=42)
    ap.add_argument('--smoke-test',action='store_true',help='One epoch, first seed, first model; results not for publication')
    args=ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for this training runner')
    df=pd.read_csv(args.data).reset_index(drop=True)
    sha=hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    for c in ['project','label','func_before']:
        if c not in df or df[c].isna().any(): raise ValueError(f'Missing/null {c}')
    if set(df.label.unique())!={0,1}: raise ValueError('Expected both binary labels')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    model_names={'codebert':'microsoft/codebert-base','graphcodebert':'microsoft/graphcodebert-base'}
    results=[]
    seeds=args.seeds[:1] if args.smoke_test else args.seeds
    models=args.models[:1] if args.smoke_test else args.models
    for seed in seeds:
        manifest=json.loads((Path(args.manifest_dir)/f'seed_{seed}.json').read_text())
        if manifest['source_sha256']!=sha: raise ValueError(f'Seed {seed}: dataset SHA256 differs from frozen manifest')
        split={}
        names=['train','validation','test']
        project_sets={name:set(manifest['splits'][name]['projects']) for name in names}
        if any(project_sets[a]&project_sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]): raise ValueError('Project overlap')
        for name in names:
            sub=df[df.project.isin(project_sets[name])].copy()
            if len(sub)!=manifest['splits'][name]['rows'] or sub.label.sum()!=manifest['splits'][name]['positive_rows']: raise ValueError(f'{name} manifest count mismatch')
            split[name]=sub
        if sum(map(len,split.values()))!=len(df): raise ValueError('Manifest does not cover all rows')
        train=split['train']; pos=train[train.label==1]; neg=train[train.label==0]; n=min(len(pos),len(neg))
        train=pd.concat([pos.sample(n=n,random_state=args.model_seed),neg.sample(n=n,random_state=args.model_seed)]).sample(frac=1,random_state=args.model_seed).reset_index(drop=True)
        print(f'\nSEED {seed}: train-balanced={len(train)} val={len(split["validation"])} test={len(split["test"])}',flush=True)
        for short in models:
            run_dir=out/short/f'seed_{seed}'
            if (run_dir/'results.json').exists() and not args.smoke_test:
                print(f'Skipping existing completed {run_dir}; delete only if intentional',flush=True)
                results.append(json.loads((run_dir/'results.json').read_text()));continue
            run_dir.mkdir(parents=True,exist_ok=True)
            random.seed(args.model_seed);np.random.seed(args.model_seed);torch.manual_seed(args.model_seed);torch.cuda.manual_seed_all(args.model_seed)
            model_id=model_names[short]; print(f'Loading {model_id}',flush=True)
            tokenizer=AutoTokenizer.from_pretrained(model_id)
            model=AutoModelForSequenceClassification.from_pretrained(model_id,num_labels=2).cuda()
            def collate(items):
                text,labels=zip(*items)
                encoded=tokenizer(list(text),padding=True,truncation=True,max_length=args.max_length,return_tensors='pt')
                encoded['labels']=torch.tensor(labels,dtype=torch.long)
                return encoded
            def loader(frame,shuffle,batch):
                data=list(zip(frame.func_before.astype(str).tolist(),frame.label.astype(int).tolist()))
                gen=torch.Generator().manual_seed(args.model_seed)
                return DataLoader(data,batch_size=batch,shuffle=shuffle,collate_fn=collate,generator=gen)
            tr=loader(train,True,args.batch_size);va=loader(split['validation'],False,args.batch_size);te=loader(split['test'],False,args.batch_size)
            @torch.no_grad()
            def predict(dl):
                model.eval();out_probs=[]
                for batch in dl:
                    batch={k:v.cuda() for k,v in batch.items() if k!='labels'}
                    with torch.autocast('cuda',dtype=torch.float16): logits=model(**batch).logits
                    out_probs.append(torch.softmax(logits.float(),dim=-1)[:,1].cpu().numpy())
                return np.concatenate(out_probs)
            opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01)
            epochs=1 if args.smoke_test else args.epochs
            steps=epochs*math.ceil(len(tr)/args.grad_accum)
            scheduler=get_linear_schedule_with_warmup(opt,num_warmup_steps=int(.1*steps),num_training_steps=steps)
            scaler=torch.amp.GradScaler('cuda')
            best=-1;stale=0;history=[];best_path=run_dir/'best_model'
            for epoch in range(1,epochs+1):
                model.train();opt.zero_grad(set_to_none=True);losses=[]
                for i,batch in enumerate(tr):
                    batch={k:v.cuda() for k,v in batch.items()}
                    group_start=(i//args.grad_accum)*args.grad_accum
                    group_size=min(args.grad_accum,len(tr)-group_start)
                    with torch.autocast('cuda',dtype=torch.float16): raw_loss=model(**batch).loss
                    scaler.scale(raw_loss/group_size).backward();losses.append(float(raw_loss.item()))
                    if (i+1)%args.grad_accum==0 or i+1==len(tr):
                        scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                        before=scaler.get_scale();scaler.step(opt);scaler.update()
                        if scaler.get_scale()>=before: scheduler.step()
                        opt.zero_grad(set_to_none=True)
                vp=predict(va);vt=threshold_for(split['validation'].label.to_numpy(),vp)
                vm=metrics(split['validation'].label.to_numpy(),vp,vt)
                history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':vm})
                print(f'  {short} seed={seed} epoch={epoch} loss={np.mean(losses):.4f} val_F1={vm["f1"]:.4f} val_AUC={vm["roc_auc"]:.4f} thr={vt:.2f}',flush=True)
                if vm['f1']>best:
                    best=vm['f1'];stale=0;model.save_pretrained(best_path);tokenizer.save_pretrained(best_path)
                    selection={'epoch':epoch,'threshold':vt,'validation':vm}
                else:
                    stale+=1
                    if stale>=args.patience: break
            del model;torch.cuda.empty_cache()
            model=AutoModelForSequenceClassification.from_pretrained(best_path).cuda()
            vp=predict(va);tp=predict(te)
            # All decisions, including checkpoint and threshold, came from validation.
            val_y=split['validation'].label.to_numpy();test_y=split['test'].label.to_numpy()
            final_thr=threshold_for(val_y,vp)
            record={'model':short,'model_id':model_id,'split_seed':seed,'model_seed':args.model_seed,'source_sha256':sha,'manifest':str(Path(args.manifest_dir)/f'seed_{seed}.json'),'selection':selection,'validation':metrics(val_y,vp,final_thr),'test_tuned':metrics(test_y,tp,final_thr),'test_default_05':metrics(test_y,tp,.5),'smoke_test':args.smoke_test,'seconds':None}
            pd.DataFrame({'project':split['test'].project.to_numpy(),'label':test_y,'probability':tp}).to_csv(run_dir/'test_predictions.csv',index=False)
            (run_dir/'history.json').write_text(json.dumps(history,indent=2))
            (run_dir/'results.json').write_text(json.dumps(record,indent=2))
            print(f'  TEST {short} seed={seed}: F1={record["test_tuned"]["f1"]:.4f} AUC={record["test_tuned"]["roc_auc"]:.4f} AP={record["test_tuned"]["pr_auc"]:.4f}',flush=True)
            results.append(record);del model,opt,scheduler,scaler;torch.cuda.empty_cache()
    if not args.smoke_test:
        rows=[]
        for r in results:
            row={'model':r['model'],'seed':r['split_seed']}
            for k in ['f1','precision','recall','roc_auc','pr_auc','threshold']:row[k]=r['test_tuned'][k]
            rows.append(row)
        table=pd.DataFrame(rows).sort_values(['model','seed']);table.to_csv(out/'all_results.csv',index=False)
        summary=table.groupby('model')[['f1','roc_auc','pr_auc']].agg(['mean','std']);summary.to_csv(out/'summary.csv')
        print('\nFINAL SUMMARY\n',summary.to_string())
    else: print('SMOKE TEST ONLY: results are not suitable for reporting.')
if __name__=='__main__':main()
