#!/usr/bin/env python3
"""E10-C: frozen GraphCodeBERT code-only visibility audit and line-masking pilot.

Validation only. No training, test access, or checkpoint changes. Run at repo root.
"""
import argparse
import hashlib
import importlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

VAL_HASH = '1e284c9c6647b615fce51a9bb33a9da7112edf0fca33bdbb72d866e818ab15c1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def diff_lines(patch):
    """Return (line_index, kind, start, end, text) with exact character ranges.

    Classification is deliberately conservative: lines after @@ hunk headers only.
    """
    rows = []
    pos = 0
    in_hunk = False
    for i, line in enumerate(patch.splitlines(keepends=True)):
        stripped = line.rstrip('\r\n')
        if stripped.startswith('diff --git '):
            in_hunk = False
            kind = 'header'
        elif stripped.startswith('@@ '):
            in_hunk = True
            kind = 'hunk_header'
        elif stripped.startswith('@@@ '):
            in_hunk = False  # combined diffs not classified as standard changed lines
            kind = 'header'
        elif in_hunk and stripped.startswith('+') and not stripped.startswith('+++'):
            kind = 'added'
        elif in_hunk and stripped.startswith('-') and not stripped.startswith('---'):
            kind = 'deleted'
        elif in_hunk and stripped.startswith(' '):
            kind = 'context'
        else:
            kind = 'header'
        rows.append((i, kind, pos, pos + len(line), stripped))
        pos += len(line)
    return rows


def line_token_positions(lines, offsets, token_ids):
    """Map each visible patch token to a line via token character start offset.

    Exclude newline-only tokens to avoid assigning newline influence to source code.
    """
    mapping = defaultdict(list)
    li = 0
    for j, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        while li + 1 < len(lines) and start >= lines[li + 1][2]:
            li += 1
        if li >= len(lines):
            continue
        _, _, ls, le, text = lines[li]
        if not (ls <= start < le):
            continue
        # Ignore line ending token(s); a newline token is not a code-content token.
        line_end = ls + len(text)
        if start >= line_end:
            continue
        mapping[li].append(j + 1)  # +1 for leading BOS token
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--val', default='reports/codejit_train_val_v3/val_manifest.csv')
    p.add_argument('--extraction', default='reports/codejit_extraction_v1')
    p.add_argument('--checkpoint', default='reports/codejit_graphcodebert_v3/code/code/best_model')
    p.add_argument('--predictions', default='reports/codejit_graphcodebert_v3/code/code/validation_predictions.csv')
    p.add_argument('--out', default='reports/codejit_explainability_v1/code_pilot')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--max-patch-bytes', type=int, default=32768)
    p.add_argument('--message-budget', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--tolerance', type=float, default=0.002)
    p.add_argument('--random-repeats', type=int, default=10)
    args = p.parse_args()
    if args.max_length != 256 or args.max_patch_bytes != 32768:
        p.error('Frozen model requires max-length=256 and max-patch-bytes=32768')
    if min(args.n, args.batch_size, args.random_repeats) < 1:
        p.error('n, batch-size, random-repeats must be positive')

    val_path, pred_path, ckpt = Path(args.val), Path(args.predictions), Path(args.checkpoint)
    for path in (val_path, pred_path, ckpt / 'config.json', ckpt / 'model.safetensors'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(val_path) != VAL_HASH:
        raise RuntimeError('STOP: validation manifest SHA256 mismatch')
    val, frozen = pd.read_csv(val_path), pd.read_csv(pred_path)
    if len(val) != 323 or len(frozen) != 323:
        raise RuntimeError('Expected exactly 323 validation records')
    cols = ['repo', 'commit_id', 'label']
    if val[cols[:2]].duplicated().any() or frozen[cols[:2]].duplicated().any():
        raise RuntimeError('Duplicate validation identities')
    if not val[cols].astype(str).reset_index(drop=True).equals(frozen[cols].astype(str).reset_index(drop=True)):
        raise RuntimeError('STOP: frozen predictions not aligned with validation manifest')

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    baseline = importlib.import_module('run_codejit_large_baselines')
    trainer = importlib.import_module('train_codejit_codebert')
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU is required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    messages, patches, loader_audit = baseline.load_records(val, Path(args.extraction), args.max_patch_bytes)
    if len(patches) != 323:
        raise RuntimeError('Loader returned incorrect number of patches')
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    if not tok.is_fast or tok.mask_token_id is None:
        raise RuntimeError('Fast tokenizer with mask token required')
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, local_files_only=True).cuda().eval()

    sequences = []
    all_lines = []
    all_maps = []
    visibility_rows = []
    for idx, (message, patch) in enumerate(zip(messages, patches)):
        ids, _, truncated = trainer.build_tokens(tok, message, patch, 'code', args.max_length, args.message_budget)
        encoded = tok(patch, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        raw_ids = encoded['input_ids']
        if ids != [tok.bos_token_id] + raw_ids[:args.max_length - 2] + [tok.eos_token_id]:
            raise RuntimeError(f'Exact token replay mismatch at validation index {idx}')
        offsets = encoded['offset_mapping'][:args.max_length - 2]
        lines = diff_lines(patch)
        mapping = line_token_positions(lines, offsets, raw_ids[:args.max_length - 2])
        counts = {kind: sum(1 for i, k, *_ in lines if k == kind) for kind in ('added', 'deleted', 'context', 'header', 'hunk_header')}
        visible_counts = {kind: sum(1 for i, k, *_ in lines if k == kind and bool(mapping.get(i))) for kind in counts}
        changed = counts['added'] + counts['deleted']
        visible_changed = visible_counts['added'] + visible_counts['deleted']
        row = val.iloc[idx]
        visibility_rows.append({
            'repo': row.repo, 'commit_id': row.commit_id, 'label': int(row.label),
            'patch_tokens_total': len(raw_ids), 'patch_tokens_visible': min(len(raw_ids), 254),
            'patch_truncated': bool(truncated),
            'changed_lines_total': changed, 'changed_lines_visible': visible_changed,
            'changed_lines_invisible': changed - visible_changed,
            'changed_line_visibility_fraction': visible_changed / changed if changed else None,
            'added_lines_total': counts['added'], 'added_lines_visible': visible_counts['added'],
            'deleted_lines_total': counts['deleted'], 'deleted_lines_visible': visible_counts['deleted'],
            'context_lines_visible': visible_counts['context'],
            'header_lines_visible': visible_counts['header'],
            'hunk_headers_visible': visible_counts['hunk_header'],
            'visible_changed_tokens': sum(len(mapping.get(i, [])) for i, k, *_ in lines if k in ('added', 'deleted')),
        })
        sequences.append(ids)
        all_lines.append(lines)
        all_maps.append(mapping)

    @torch.inference_mode()
    def probs(items):
        values = []
        for start in range(0, len(items), args.batch_size):
            batch = tok.pad({'input_ids': items[start:start + args.batch_size]}, padding=True, return_tensors='pt')
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(**batch).logits
            values.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        return np.asarray(values, dtype=float)

    replay = probs(sequences)
    max_error = float(np.max(np.abs(replay - frozen['probability_positive'].to_numpy(dtype=float))))
    if max_error > args.tolerance:
        raise RuntimeError(f'STOP: frozen prediction replay error {max_error:.8f} > {args.tolerance}')
    if not np.array_equal((replay >= .5).astype(int), frozen['predicted'].to_numpy(dtype=int)):
        raise RuntimeError('STOP: frozen predicted classes mismatch')
    print(f'Frozen prediction replay passed: max absolute error {max_error:.8f}')

    # Select stratified examples among those with at least one visible changed line.
    eligible = [i for i, x in enumerate(visibility_rows) if x['changed_lines_visible'] > 0]
    rng = random.Random(args.seed)
    by_label = {label: [i for i in eligible if int(val.iloc[i].label) == label] for label in (0, 1)}
    for group in by_label.values():
        rng.shuffle(group)
    take0 = min(args.n // 2, len(by_label[0]))
    take1 = min(args.n - take0, len(by_label[1]))
    chosen = sorted(by_label[0][:take0] + by_label[1][:take1])
    if len(chosen) < args.n:
        remaining = [i for i in eligible if i not in chosen]
        rng.shuffle(remaining)
        chosen = sorted(chosen + remaining[:args.n - len(chosen)])
    if not chosen:
        raise RuntimeError('No validation cases have visible changed lines')

    attribution_rows, pilot_rows = [], []
    controls_rng = random.Random(args.seed + 1000)
    for idx in chosen:
        lines, mapping, original_ids = all_lines[idx], all_maps[idx], sequences[idx]
        original_p = float(replay[idx])
        changed_lines = [(i, k, positions, content) for i, k, _, _, content in lines
                         if k in ('added', 'deleted') and (positions := mapping.get(i))]
        context_lines = [(i, positions) for i, k, *_ in lines
                         if k == 'context' and (positions := mapping.get(i))]
        # Each line intervention masks only model-visible content tokens; length is unchanged.
        variants = []
        for _, _, positions, _ in changed_lines:
            changed = original_ids.copy()
            for position in positions:
                changed[position] = tok.mask_token_id
            variants.append(changed)
        changed_ps = probs(variants)
        effects = []
        for (line_idx, kind, positions, content), perturbed_p in zip(changed_lines, changed_ps):
            delta = original_p - float(perturbed_p)
            effects.append((line_idx, kind, positions, content, float(perturbed_p), delta))
            attribution_rows.append({
                'repo': val.iloc[idx].repo, 'commit_id': val.iloc[idx].commit_id,
                'label': int(val.iloc[idx].label), 'line_index': line_idx, 'line_kind': kind,
                'line_text': content[:500], 'visible_token_count': len(positions),
                'original_probability': original_p, 'masked_probability': float(perturbed_p),
                'signed_probability_drop': delta, 'absolute_probability_change': abs(delta),
            })
        strongest = max(effects, key=lambda x: abs(x[5]))
        # Equal-token-count context control: sample context tokens (not changed lines),
        # avoiding unequal perturbation sizes where sufficient context is available.
        context_positions = sorted(set(pos for _, positions in context_lines for pos in positions))
        k = len(strongest[2])
        random_changes = []
        if len(context_positions) >= k:
            control_variants = []
            for _ in range(args.random_repeats):
                positions = controls_rng.sample(context_positions, k)
                modified = original_ids.copy()
                for position in positions:
                    modified[position] = tok.mask_token_id
                control_variants.append(modified)
            random_changes = [abs(original_p - float(x)) for x in probs(control_variants)]
        pilot_rows.append({
            'repo': val.iloc[idx].repo, 'commit_id': val.iloc[idx].commit_id,
            'label': int(val.iloc[idx].label), 'original_probability': original_p,
            'visible_changed_lines': len(changed_lines), 'visible_context_tokens': len(context_positions),
            'top_changed_line_index': strongest[0], 'top_changed_line_kind': strongest[1],
            'top_changed_line_visible_tokens': k, 'top_changed_line_text': strongest[3][:500],
            'top_changed_line_masked_probability': strongest[4],
            'top_changed_line_abs_change': abs(strongest[5]),
            'mean_equal_size_context_abs_change': float(np.mean(random_changes)) if random_changes else None,
            'n_context_controls': len(random_changes),
            'top_minus_context_abs_change': (abs(strongest[5]) - float(np.mean(random_changes))) if random_changes else None,
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(visibility_rows).to_csv(out / 'visibility_audit.csv', index=False)
    pd.DataFrame(attribution_rows).to_csv(out / 'line_attributions.csv', index=False)
    pd.DataFrame(pilot_rows).to_csv(out / 'pilot_cases.csv', index=False)
    vis = pd.DataFrame(visibility_rows)
    pilot = pd.DataFrame(pilot_rows)
    summary = {
        'experiment': 'E10-C code-only visibility and changed-line masking pilot',
        'validation_only': True, 'heldout_test_accessed': False,
        'model': 'GraphCodeBERT sequence classifier; no explicit graph/data-flow features',
        'val_manifest_sha256': sha256(val_path),
        'frozen_predictions_sha256': sha256(pred_path),
        'checkpoint_config_sha256': sha256(ckpt / 'config.json'),
        'checkpoint_weights_sha256': sha256(ckpt / 'model.safetensors'),
        'n_validation': len(vis), 'n_pilot': len(pilot), 'seed': args.seed,
        'max_length': args.max_length, 'max_patch_bytes': args.max_patch_bytes,
        'prediction_replay_max_abs_error': max_error,
        'n_truncated': int(vis.patch_truncated.sum()),
        'n_with_any_visible_changed_line': int((vis.changed_lines_visible > 0).sum()),
        'n_with_no_visible_changed_line': int((vis.changed_lines_visible == 0).sum()),
        'total_changed_lines': int(vis.changed_lines_total.sum()),
        'total_visible_changed_lines': int(vis.changed_lines_visible.sum()),
        'micro_changed_line_visibility': (float(vis.changed_lines_visible.sum() / vis.changed_lines_total.sum())
                                          if vis.changed_lines_total.sum() else None),
        'mean_top_changed_line_abs_change': float(pilot.top_changed_line_abs_change.mean()),
        'n_pilot_with_context_controls': int(pilot.n_context_controls.gt(0).sum()),
        'mean_top_minus_context_abs_change_on_comparable_cases': (
            float(pilot.top_minus_context_abs_change.mean())
            if pilot.top_minus_context_abs_change.notna().any() else None),
        'limitations': [
            'Exploratory validation-only sample, not held-out generalization evidence.',
            'CodeJIT labels are commit-level and do not localize vulnerable lines.',
            'Token masking creates out-of-distribution inputs; effects are not causal vulnerability explanations.',
            'The strongest changed line is selected and evaluated on the same examples (selection bias).',
            'Equal-token-count context controls exist only where enough visible context tokens remain.',
            'Line visibility is based on token start offsets; tokens crossing boundaries are assigned by start.',
            'Unified-diff hunk parsing is conservative; binary/combined patches are not line-localized.',
            'Patch text comes from the original verified loader and its byte-limited prefix.',
        ],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Saved:', out)


if __name__ == '__main__':
    main()
