#!/usr/bin/env python3
"""CodeJIT CodeBERT, repository-disjoint TRAIN/VALIDATION ONLY.

Uses exactly the same manifest validation and verified extraction loader as
run_codejit_large_baselines.py. No test cohort is loaded or accessed.
Run one mode at a time: --input-mode message|code|combined.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metric(y, p, threshold):
    pred = (np.asarray(p) >= threshold).astype(int)
    y = np.asarray(y).astype(int)
    return dict(n=int(len(y)), threshold=float(threshold),
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
                roc_auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def tune_threshold(y, p):
    # Tune only on validation, deterministic, report 0.5 alongside tuned.
    thresholds = np.arange(5, 96) / 100.0
    values = [f1_score(y, np.asarray(p) >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(values))])


def build_tokens(tokenizer, message, patch, mode, max_length, message_budget):
    # The baseline uses first max_patch_bytes raw bytes, verified before decoding.
    # Encode with explicit special tokens; combined reserves a message budget so
    # neither modality is accidentally entirely omitted by prefix truncation.
    m = tokenizer.encode(message, add_special_tokens=False)
    c = tokenizer.encode(patch, add_special_tokens=False)
    if mode == 'message':
        capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        ids = tokenizer.build_inputs_with_special_tokens(m[:capacity])
        return ids, len(m) > capacity, False
    if mode == 'code':
        capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        ids = tokenizer.build_inputs_with_special_tokens(c[:capacity])
        return ids, False, len(c) > capacity
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=True)
    mcap = min(len(m), message_budget, capacity - 1)
    ccap = capacity - mcap
    # Redistribute unused code capacity to message, if code is short.
    if len(c) < ccap:
        mcap = min(len(m), capacity - len(c))
        ccap = capacity - mcap
    ids = tokenizer.build_inputs_with_special_tokens(m[:mcap], c[:ccap])
    return ids, len(m) > mcap, len(c) > ccap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--extraction', default='reports/codejit_extraction_v1')
    p.add_argument('--out', default='reports/codejit_codebert_v1')
    p.add_argument('--input-mode', choices=['message', 'code', 'combined'], required=True)
    p.add_argument('--model-name', default='microsoft/codebert-base')
    p.add_argument('--max-patch-bytes', type=int, default=32768)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--message-budget', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-test', action='store_true', help='1 epoch, first 64 train/32 val; NOT research results')
    p.add_argument('--check-data-only', action='store_true', help='Verify inputs and report counts, no model download')
    args = p.parse_args()
    if args.max_length < 16 or args.max_length > 512 or args.message_budget < 1 or args.message_budget >= args.max_length - 4:
        p.error('Require 16<=max-length<=512 and 1<=message-budget<max-length-4')
    if min(args.batch_size, args.grad_accum, args.epochs, args.max_patch_bytes) < 1:
        p.error('batch-size, grad-accum, epochs, max-patch-bytes must be positive')

    # Reuse baseline's exact manifest validation, path checks, message extraction,
    # patch integrity and 32KiB prefix selection. Fail if module unavailable.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_codejit_large_baselines import load_manifest, load_records
    train, val = load_manifest(args.train), load_manifest(args.val)
    train_keys = set(zip(train.repo, train.commit_id))
    val_keys = set(zip(val.repo, val.commit_id))
    if train_keys & val_keys or set(train.repo) & set(val.repo):
        raise ValueError('Train/validation commit or repository overlap')
    if set(train.label.astype(int)) != {0, 1} or set(val.label.astype(int)) != {0, 1}:
        raise ValueError('Both labels required in each split')
    train_msg, train_patch, train_audit = load_records(train, Path(args.extraction), args.max_patch_bytes)
    val_msg, val_patch, val_audit = load_records(val, Path(args.extraction), args.max_patch_bytes)
    ytrain = train.label.astype(int).to_numpy()
    yval = val.label.astype(int).to_numpy()
    out = Path(args.out) / args.input_mode
    if args.smoke_test:
        out = out / 'SMOKE_NOT_FOR_REPORTING'
        train = train.iloc[:64].copy(); val = val.iloc[:32].copy()
        train_msg, train_patch = train_msg[:64], train_patch[:64]
        val_msg, val_patch = val_msg[:32], val_patch[:32]
        ytrain, yval = ytrain[:64], yval[:32]
    out.mkdir(parents=True, exist_ok=True)
    data_report = dict(train=int(len(train)), val=int(len(val)),
                       train_repositories=int(train.repo.nunique()),
                       val_repositories=int(val.repo.nunique()),
                       train_label_counts={str(k): int(v) for k, v in zip(*np.unique(ytrain, return_counts=True))},
                       val_label_counts={str(k): int(v) for k, v in zip(*np.unique(yval, return_counts=True))},
                       train_manifest_sha256=sha256(args.train), val_manifest_sha256=sha256(args.val),
                       max_patch_bytes=args.max_patch_bytes,
                       smoke_test=bool(args.smoke_test), heldout_test_accessed=False)
    (out / 'data_audit.json').write_text(json.dumps(data_report, indent=2) + '\n')
    print('DATA AUDIT:', json.dumps(data_report, indent=2), flush=True)
    if args.check_data_only:
        return

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; enable Kaggle GPU before training')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def encode_all(messages, patches):
        ids, flags = [], []
        for m, c in zip(messages, patches):
            t, mt, ct = build_tokens(tokenizer, m, c, args.input_mode, args.max_length, args.message_budget)
            if len(t) > args.max_length:
                raise AssertionError('Token limit exceeded')
            ids.append(t)
            flags.append((mt, ct, len(t)))
        return ids, flags
    tr_ids, tr_flags = encode_all(train_msg, train_patch)
    va_ids, va_flags = encode_all(val_msg, val_patch)
    class Rows(Dataset):
        def __init__(self, ids, labels):
            self.ids = ids; self.labels = labels
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.ids[i], int(self.labels[i])
    def collate(rows):
        sequences, labels = zip(*rows)
        enc = tokenizer.pad({'input_ids': list(sequences)}, padding=True, return_tensors='pt')
        enc['labels'] = torch.tensor(labels, dtype=torch.long)
        return enc
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Rows(tr_ids, ytrain), batch_size=args.batch_size, shuffle=True,
                              generator=gen, collate_fn=collate)
    val_loader = DataLoader(Rows(va_ids, yval), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).cuda()
    class_counts = np.bincount(ytrain, minlength=2)
    weights = torch.tensor(len(ytrain) / (2 * np.maximum(class_counts, 1)), dtype=torch.float32, device='cuda')
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    epochs = 1 if args.smoke_test else args.epochs
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(.1 * steps_per_epoch * epochs),
                                                 num_training_steps=steps_per_epoch * epochs)
    scaler = torch.amp.GradScaler('cuda')
    @torch.no_grad()
    def predict():
        model.eval(); probs = []
        for batch in val_loader:
            inp = {k: v.cuda() for k, v in batch.items() if k != 'labels'}
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(**inp).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        return np.asarray(probs)
    history = []; best = -1.; stale = 0; best_threshold = .5
    best_path = out / 'best_model'
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True); losses = []
        for idx, batch in enumerate(train_loader):
            group_start = (idx // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                raw_loss = loss_fn(logits.float(), batch['labels'])
            scaler.scale(raw_loss / group_size).backward()
            losses.append(float(raw_loss.item()))
            if (idx + 1) % args.grad_accum == 0 or idx + 1 == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                before = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() >= before:
                    scheduler.step()
                opt.zero_grad(set_to_none=True)
        probs = predict()
        threshold = tune_threshold(yval, probs)
        scores = metric(yval, probs, threshold)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), tuned_threshold=threshold,
                            validation_tuned=scores, validation_default_05=metric(yval, probs, .5)))
        print(f'{args.input_mode} epoch {epoch}: loss={np.mean(losses):.4f} '
              f'val F1@0.5={history[-1]["validation_default_05"]["f1"]:.4f} '
              f'val tuned F1={scores["f1"]:.4f} val AP={scores["pr_auc"]:.4f}', flush=True)
        # Selection criterion: tuned validation F1, exploratory, no test involved.
        if scores['f1'] > best:
            best = scores['f1']; stale = 0; best_threshold = threshold
            model.save_pretrained(best_path); tokenizer.save_pretrained(best_path)
        else:
            stale += 1
            if stale >= args.patience: break
    del model
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(best_path).cuda()
    probs = predict()
    pred = (probs >= .5).astype(int)
    pd.DataFrame(dict(model=args.input_mode, repo=val.repo.to_numpy(), commit_id=val.commit_id.to_numpy(),
                      label=yval, predicted=pred, probability_positive=probs,
                      predicted_tuned=(probs >= best_threshold).astype(int))).to_csv(
                          out / 'validation_predictions.csv', index=False)
    coverage_rows = []
    for subset, frame, flags in [('train', train, tr_flags), ('val', val, va_flags)]:
        for r, (mt, ct, length) in zip(frame.itertuples(index=False), flags):
            coverage_rows.append(dict(subset=subset, repo=r.repo, commit_id=r.commit_id,
                                      message_tokens_truncated=mt, code_tokens_truncated=ct,
                                      model_tokens=length))
    pd.DataFrame(coverage_rows).to_csv(out / 'token_coverage.csv', index=False)
    result = dict(experiment='CodeJIT CodeBERT repository-disjoint validation-only',
                  input_mode=args.input_mode, model_name=args.model_name,
                  data_audit=data_report, selected_epoch=int(history[int(np.argmax([x['validation_tuned']['f1'] for x in history]))]['epoch']),
                  threshold_tuned_on_validation=best_threshold,
                  validation_default_05=metric(yval, probs, .5),
                  validation_tuned=metric(yval, probs, best_threshold),
                  training_history=history, seconds=float(time.time()-started),
                  config=vars(args),
                  limitations=['Curated CodeJIT negatives do not represent ordinary CI commits.',
                               'No chronological split; repositories can share code/forks.',
                               'Patch prefix and tokenizer truncation may omit evidence.',
                               'Validation-based checkpoint/threshold selection is exploratory.',
                               'Held-out test cohort never accessed.'])
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mode=args.input_mode, selected_epoch=result['selected_epoch'],
                          validation_default_05=result['validation_default_05'],
                          validation_tuned=result['validation_tuned'], output=str(out)), indent=2))

if __name__ == '__main__':
    main()
