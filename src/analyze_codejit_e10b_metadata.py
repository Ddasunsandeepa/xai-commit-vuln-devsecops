#!/usr/bin/env python3
"""E10-B: frozen GraphCodeBERT message metadata intervention audit (validation only).

Run from repository root; never reads held-out test. Uses original verified
load_records and build_tokens. Intervention removes matched spans from messages
rather than inserting mask tokens; this changes length and is a separate diagnostic.
"""
import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_VAL_SHA = '1e284c9c6647b615fce51a9bb33a9da7112edf0fca33bdbb72d866e818ab15c1'
PATTERNS = {
    'bug_reference': re.compile(r'(?im)^\s*Bug\s*:\s*[^\r\n]*(?:\r?\n)?'),
    'fix_word': re.compile(r'(?i)\bfix(?:es|ed|ing)?\b'),
    'cve_identifier': re.compile(r'(?i)\bCVE-\d{4}-\d{4,}\b'),
    'change_id': re.compile(r'(?im)^\s*Change-Id\s*:\s*[^\r\n]*(?:\r?\n)?'),
    'cherry_pick': re.compile(r'(?im)^\s*\(cherry picked from commit\s+[0-9a-f]+\)\s*(?:\r?\n)?'),
}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--val', default='reports/codejit_train_val_v3/val_manifest.csv')
    p.add_argument('--extraction', default='reports/codejit_extraction_v1')
    p.add_argument('--checkpoint', default='reports/codejit_graphcodebert_v3/message/message/best_model')
    p.add_argument('--predictions', default='reports/codejit_graphcodebert_v3/message/message/validation_predictions.csv')
    p.add_argument('--out', default='reports/codejit_explainability_v1/metadata_audit')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--tolerance', type=float, default=0.002)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    valpath, predpath, checkpoint = Path(args.val), Path(args.predictions), Path(args.checkpoint)
    for f in (valpath, predpath, checkpoint / 'config.json', checkpoint / 'model.safetensors'):
        if not f.is_file():
            raise FileNotFoundError(f)
    if sha(valpath) != EXPECTED_VAL_SHA:
        raise RuntimeError('Frozen validation manifest SHA mismatch: STOP')
    val, frozen = pd.read_csv(valpath), pd.read_csv(predpath)
    if len(val) != 323 or len(frozen) != 323:
        raise RuntimeError('Expected exactly 323 validation rows')
    keys = ['repo', 'commit_id', 'label']
    if val[keys].astype(str).reset_index(drop=True).equals(frozen[keys].astype(str).reset_index(drop=True)) is False:
        raise RuntimeError('Frozen validation/prediction identity mismatch')
    if val[['repo','commit_id']].duplicated().any():
        raise RuntimeError('Duplicate validation identity')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    loader = importlib.import_module('run_codejit_large_baselines')
    trainer = importlib.import_module('train_codejit_codebert')
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle GPU')
    torch.manual_seed(42)
    messages, patches, audit = loader.load_records(val, Path(args.extraction), 32768)
    if len(messages) != 323 or len(patches) != 323:
        raise RuntimeError('Loader length mismatch')
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).cuda().eval()

    def encode(message):
        ids, _, _ = trainer.build_tokens(tok, message, '', 'message', 256, 64)
        return ids

    @torch.inference_mode()
    def predict(sequences):
        out = []
        for start in range(0, len(sequences), args.batch_size):
            batch = tok.pad({'input_ids': sequences[start:start+args.batch_size]}, padding=True, return_tensors='pt')
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(**batch).logits
            out.extend(torch.softmax(logits.float(), dim=-1)[:,1].cpu().tolist())
        return np.asarray(out, dtype=float)

    originals = predict([encode(m) for m in messages])
    stored = frozen.probability_positive.to_numpy(dtype=float)
    replay_error = float(np.max(np.abs(originals - stored)))
    if replay_error > args.tolerance or not np.array_equal((originals >= .5).astype(int), frozen.predicted.to_numpy(dtype=int)):
        raise RuntimeError(f'Frozen prediction replay failed (max error {replay_error:.8f}): STOP')

    rows = []
    for name, regex in PATTERNS.items():
        affected = []
        sequences = []
        for i, message in enumerate(messages):
            if regex.search(message):
                edited = regex.sub('', message)
                if edited != message:
                    affected.append(i)
                    sequences.append(encode(edited))
        if not affected:
            continue
        changed_probs = predict(sequences)
        for i, new_p in zip(affected, changed_probs):
            r = val.iloc[i]
            before = float(originals[i]); after = float(new_p)
            rows.append(dict(pattern=name, repo=r.repo, commit_id=r.commit_id, label=int(r.label),
                             original_probability=before, intervened_probability=after,
                             signed_delta=after-before, absolute_delta=abs(after-before),
                             original_prediction=int(before>=.5), intervened_prediction=int(after>=.5),
                             prediction_flip=bool((before>=.5)!=(after>=.5))))
    cols = ['pattern','repo','commit_id','label','original_probability','intervened_probability',
            'signed_delta','absolute_delta','original_prediction','intervened_prediction','prediction_flip']
    effects = pd.DataFrame(rows, columns=cols)
    summary = []
    for name in PATTERNS:
        group = effects[effects.pattern == name]
        for label in ('all', 0, 1):
            sub = group if label == 'all' else group[group.label == label]
            summary.append(dict(pattern=name, label=str(label), n=int(len(sub)),
                                repos=int(sub.repo.nunique()),
                                mean_signed_delta=float(sub.signed_delta.mean()) if len(sub) else None,
                                mean_absolute_delta=float(sub.absolute_delta.mean()) if len(sub) else None,
                                median_signed_delta=float(sub.signed_delta.median()) if len(sub) else None,
                                prediction_flips=int(sub.prediction_flip.sum()) if len(sub) else 0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    effects.to_csv(out / 'intervention_effects.csv', index=False)
    pd.DataFrame(summary).to_csv(out / 'pattern_summary.csv', index=False)
    provenance = dict(experiment='E10-B exploratory validation-only metadata deletion audit',
                      heldout_test_accessed=False, validation_rows=323, interventions=len(effects),
                      replay_max_abs_error=replay_error, manifest_sha256=sha(valpath),
                      predictions_sha256=sha(predpath), checkpoint_sha256=sha(checkpoint/'model.safetensors'),
                      patterns={k:v.pattern for k,v in PATTERNS.items()},
                      limitations=['Patterns selected after inspecting E10-A validation examples; exploratory.',
                                   'Deleting spans changes message length and token positions; it is not token masking.',
                                   'Only occurrences present in the full original message are tested; matches beyond the 254-token model window may have no effect.',
                                   'Some patterns may occur in very few repositories, so repo and label are confounded.',
                                   'CVE pattern requires full CVE identifier; standalone CVE is not matched.',
                                   'Class 0 is not representative ordinary benign commits.',
                                   'No independent test, causal vulnerability localization, or multiple-testing correction.'])
    (out/'audit.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    print('Prediction replay max absolute error:', replay_error)
    print(pd.DataFrame(summary).to_string(index=False))
    print('Saved:', out.resolve())

if __name__ == '__main__':
    main()
