#!/usr/bin/env python3
"""E10: validation-only, frozen-checkpoint message-token perturbation pilot.

Uses the original verified CodeJIT loader and original build_tokens() function.
No training, no held-out test access. Requires Kaggle GPU and local extraction records.
"""
import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
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
    p.add_argument('--out', default='reports/codejit_explainability_v1/message_pilot')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--max-patch-bytes', type=int, default=32768)
    p.add_argument('--message-budget', type=int, default=64)
    p.add_argument('--tolerance', type=float, default=0.002)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--random-repeats', type=int, default=10)
    p.add_argument('--mask-k', type=int, default=3)
    args = p.parse_args()

    if args.n < 1 or args.batch_size < 1 or args.mask_k < 1 or args.random_repeats < 1:
        p.error('n, batch-size, mask-k, random-repeats must be positive')
    if args.max_length != 256:
        p.error('E10 pilot is locked to the original max_length=256')
    if args.max_patch_bytes != 32768:
        p.error('E10 pilot is locked to the original max_patch_bytes=32768')

    val_path = Path(args.val)
    checkpoint = Path(args.checkpoint)
    pred_path = Path(args.predictions)
    for required in [val_path, pred_path, checkpoint / 'config.json', checkpoint / 'model.safetensors']:
        if not required.is_file():
            raise FileNotFoundError(required)

    expected_val_hash = '1e284c9c6647b615fce51a9bb33a9da7112edf0fca33bdbb72d866e818ab15c1'
    if sha256(val_path) != expected_val_hash:
        raise RuntimeError('Validation manifest hash mismatch: STOP')

    val = pd.read_csv(val_path)
    frozen = pd.read_csv(pred_path)
    if len(val) != 323 or len(frozen) != 323:
        raise RuntimeError('Expected 323 frozen validation rows')
    if val[['repo', 'commit_id']].duplicated().any() or frozen[['repo', 'commit_id']].duplicated().any():
        raise RuntimeError('Duplicate validation identities')
    if not (val[['repo', 'commit_id', 'label']].reset_index(drop=True).astype(str).equals(
            frozen[['repo', 'commit_id', 'label']].reset_index(drop=True).astype(str))):
        raise RuntimeError('Frozen predictions and validation manifest are not identically ordered')

    src = Path(__file__).resolve().parent
    sys.path.insert(0, str(src))
    baseline = importlib.import_module('run_codejit_large_baselines')
    trainer = importlib.import_module('train_codejit_codebert')
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for E10; enable Kaggle GPU')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Original loader verifies extraction records, integrity, and commit identity.
    messages, patches, loader_audit = baseline.load_records(val, args.extraction, args.max_patch_bytes)
    if len(messages) != len(val) or len(patches) != len(val):
        raise RuntimeError('Loader returned wrong number of records')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
    if tokenizer.mask_token_id is None:
        raise RuntimeError('Tokenizer has no mask token')

    sequences = []
    for message, patch in zip(messages, patches):
        ids, _, _ = trainer.build_tokens(tokenizer, message, patch, 'message', args.max_length, args.message_budget)
        if len(ids) > args.max_length:
            raise RuntimeError('Tokenizer exceeded max_length')
        sequences.append(ids)

    @torch.inference_mode()
    def probabilities(batch_sequences):
        values = []
        for start in range(0, len(batch_sequences), args.batch_size):
            batch = tokenizer.pad({'input_ids': batch_sequences[start:start + args.batch_size]},
                                  padding=True, return_tensors='pt')
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(**batch).logits
            values.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        return np.asarray(values, dtype=np.float64)

    # Reproduce all 323 stored validation probabilities before selecting pilot.
    replay = probabilities(sequences)
    saved = frozen['probability_positive'].to_numpy(dtype=np.float64)
    max_difference = float(np.max(np.abs(replay - saved)))
    if max_difference > args.tolerance:
        raise RuntimeError(f'Frozen prediction replay failed: max abs diff {max_difference:.8f} > {args.tolerance}')
    if not np.array_equal((replay >= .5).astype(int), frozen['predicted'].to_numpy(dtype=int)):
        raise RuntimeError('Frozen prediction class mismatch at threshold 0.5')

    rng = random.Random(args.seed)
    # Stratified deterministic sampling by frozen label, without using test.
    zero = val.index[val['label'] == 0].tolist()
    one = val.index[val['label'] == 1].tolist()
    rng.shuffle(zero)
    rng.shuffle(one)
    n_zero = args.n // 2
    n_one = args.n - n_zero
    if len(zero) < n_zero or len(one) < n_one:
        raise RuntimeError('Not enough validation examples for requested pilot')
    chosen = sorted(zero[:n_zero] + one[:n_one])

    token_rows = []
    case_rows = []
    rng_controls = random.Random(args.seed + 1000)
    for index in chosen:
        ids = sequences[index]
        # Exclude special tokens from candidate interventions.
        positions = [j for j in range(1, len(ids) - 1)
                     if ids[j] not in tokenizer.all_special_ids]
        if not positions:
            continue
        original_p = float(replay[index])
        perturbations = []
        for pos in positions:
            altered = ids.copy()
            altered[pos] = tokenizer.mask_token_id
            perturbations.append(altered)
        altered_p = probabilities(perturbations)
        # Positive delta means removing this token lowers class-1 probability.
        importance = original_p - altered_p
        ranking = sorted(range(len(positions)), key=lambda k: (-abs(float(importance[k])), positions[k]))
        k = min(args.mask_k, len(positions))
        top_positions = [positions[j] for j in ranking[:k]]
        top_masked = ids.copy()
        for pos in top_positions:
            top_masked[pos] = tokenizer.mask_token_id
        top_p = float(probabilities([top_masked])[0])
        random_ps = []
        for _ in range(args.random_repeats):
            rand_positions = rng_controls.sample(positions, k)
            random_masked = ids.copy()
            for pos in rand_positions:
                random_masked[pos] = tokenizer.mask_token_id
            random_ps.append(float(probabilities([random_masked])[0]))
        random_effects = [abs(original_p - x) for x in random_ps]
        top_effect = abs(original_p - top_p)
        row = val.iloc[index]
        case_rows.append({
            'repo': row.repo, 'commit_id': row.commit_id, 'label': int(row.label),
            'original_probability': original_p,
            'original_predicted': int(original_p >= .5),
            'candidate_tokens': len(positions), 'mask_k': k,
            'top_mask_probability': top_p,
            'top_abs_probability_change': top_effect,
            'random_mean_abs_probability_change': float(np.mean(random_effects)),
            'random_sd_abs_probability_change': float(np.std(random_effects)),
            'top_minus_random_mean_abs_change': top_effect - float(np.mean(random_effects)),
            'random_repeats': args.random_repeats,
        })
        for pos, masked_prob, delta in zip(positions, altered_p, importance):
            token_rows.append({
                'repo': row.repo, 'commit_id': row.commit_id,
                'token_position': pos,
                'token_id': int(ids[pos]),
                'token': tokenizer.convert_ids_to_tokens(int(ids[pos])),
                'original_probability': original_p,
                'single_mask_probability': float(masked_prob),
                'signed_probability_drop': float(delta),
                'absolute_effect': abs(float(delta)),
                'selected_top_k': pos in top_positions,
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame(case_rows)
    tokens = pd.DataFrame(token_rows)
    cases.to_csv(out / 'pilot_cases.csv', index=False)
    tokens.to_csv(out / 'token_attributions.csv', index=False)
    summary = {
        'experiment': 'E10 message-only leave-one-token-out and masking pilot',
        'validation_only': True, 'heldout_test_accessed': False,
        'model': 'GraphCodeBERT sequence classifier (no explicit graph)',
        'checkpoint': str(checkpoint), 'checkpoint_config_sha256': sha256(checkpoint / 'config.json'),
        'checkpoint_weights_sha256': sha256(checkpoint / 'model.safetensors'),
        'val_manifest_sha256': sha256(val_path), 'frozen_predictions_sha256': sha256(pred_path),
        'n_validation': len(val), 'n_selected': len(chosen), 'n_analyzed': len(cases),
        'seed': args.seed, 'max_length': args.max_length, 'max_patch_bytes': args.max_patch_bytes,
        'mask_k': args.mask_k, 'random_repeats': args.random_repeats,
        'replay_max_abs_probability_error': max_difference,
        'mean_top_abs_change': float(cases['top_abs_probability_change'].mean()) if len(cases) else None,
        'mean_random_abs_change': float(cases['random_mean_abs_probability_change'].mean()) if len(cases) else None,
        'mean_top_minus_random_change': float(cases['top_minus_random_mean_abs_change'].mean()) if len(cases) else None,
        'limitations': [
            'Exploratory validation-only pilot; validation reused for model selection.',
            'Top tokens are selected and evaluated on the same cases; superiority over random is partly by construction.',
            'Single-token mask effects are not additive; simultaneous masking can interact.',
            'Masking creates potentially out-of-distribution inputs.',
            'Attributions are model influence, not proof of vulnerable code or causal software defects.',
            'Code changes are not attributed by this message-only pilot.',
            'The negative CodeJIT label is not a representative sample of ordinary benign commits.',
        ],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Saved:', out / 'pilot_cases.csv', out / 'token_attributions.csv', out / 'summary.json')

if __name__ == '__main__':
    main()
