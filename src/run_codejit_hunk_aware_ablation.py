#!/usr/bin/env python3
"""Controlled CodeJIT hunk-aware diff selection, train/validation only.

Reuses the EXISTING src/train_codejit_codebert.py trainer and verified loader.
Changes only the code text supplied to the tokenizer. Does not access test.
Run separate --model-family codebert|graphcodebert and --input-mode code|combined.

Selection is deterministic and label-blind: parse unified-diff hunks, collect
changed +/- lines in original order, with file/hunk markers. If no changed
lines are parsed, fall back to the original verified patch prefix. The
selection is capped at the same 32768 UTF-8 bytes as the baseline; the
transformer still has the same max-length token limit (default 256).

Important: this is a CHANGED-LINE-PRIORITY experiment, not semantic hunk
ranking, not a graph model, and not multi-chunk aggregation.
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path


def record_key(repo, commit):
    return hashlib.sha256(f'{repo}\0{commit}'.encode('utf-8')).hexdigest()[:24]


def selected_patch(raw: bytes, limit: int) -> tuple[str, dict]:
    """Return a label-blind changed-line-focused representation.

    The original raw diff is decoded only after the verified loader has
    checked extraction identity, integrity, and validity. File and hunk
    headers are kept; context lines are excluded. Hunk order is preserved.
    """
    text = raw.decode('utf-8', errors='replace')
    lines = text.splitlines()
    output = []
    file_header = ''
    hunk_header = ''
    file_emitted = False
    hunk_emitted = False
    changed = 0
    files = 0
    hunks = 0
    for line in lines:
        if line.startswith('diff --git '):
            file_header = line
            hunk_header = ''
            file_emitted = False
            hunk_emitted = False
            files += 1
        elif line.startswith('@@ ') or line.startswith('@@-'):
            hunk_header = line
            hunk_emitted = False
            hunks += 1
        elif line.startswith(('+', '-')) and not line.startswith(('+++ ', '--- ')):
            # Exclude diff metadata outside hunks (e.g. ---/+++ headers).
            if not hunk_header:
                continue
            if not file_emitted and file_header:
                output.append(file_header)
                file_emitted = True
            if not hunk_emitted:
                output.append(hunk_header)
                hunk_emitted = True
            output.append(line)
            changed += 1
    if changed == 0:
        # Binary diffs and nonstandard patches have no parseable changed lines.
        result = raw[:limit].decode('utf-8', errors='replace')
        return result, {'fallback': True, 'changed_lines': 0, 'files_seen': files, 'hunks_seen': hunks,
                        'selected_bytes_before_cap': len(raw), 'selected_bytes_after_cap': len(raw[:limit])}
    payload = ('\n'.join(output) + '\n').encode('utf-8')
    cut = payload[:limit]
    return cut.decode('utf-8', errors='replace'), {
        'fallback': False, 'changed_lines': changed, 'files_seen': files,
        'hunks_seen': hunks, 'selected_bytes_before_cap': len(payload),
        'selected_bytes_after_cap': len(cut)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='reports/codejit_train_val_v2/train_manifest.csv')
    p.add_argument('--val', default='reports/codejit_train_val_v2/val_manifest.csv')
    p.add_argument('--extraction', default='reports/codejit_extraction_v1')
    p.add_argument('--model-family', required=True, choices=['codebert', 'graphcodebert'])
    p.add_argument('--input-mode', required=True, choices=['code', 'combined'])
    p.add_argument('--max-patch-bytes', type=int, default=32768)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--message-budget', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='reports/codejit_hunk_aware_v1')
    p.add_argument('--check-data-only', action='store_true')
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()
    if args.max_patch_bytes < 1:
        p.error('--max-patch-bytes must be positive')
    src = Path(__file__).resolve().parent
    sys.path.insert(0, str(src))
    import run_codejit_large_baselines as baseline
    import train_codejit_codebert as trainer
    original_loader = baseline.load_records
    reports = []

    def hunk_loader(frame, extraction, max_patch_bytes):
        # Original loader verifies all identities, patch sizes and gzip integrity.
        messages, prefixes, audit = original_loader(frame, extraction, max_patch_bytes)
        transformed = []
        for row in frame.itertuples(index=False):
            key = record_key(str(row.repo), str(row.commit_id))
            path = Path(extraction) / 'records' / (key + '.diff.gz')
            if not path.is_file():
                raise FileNotFoundError(f'Missing verified patch: {path}')
            with gzip.open(path, 'rb') as handle:
                raw = handle.read()
            patch, stats = selected_patch(raw, max_patch_bytes)
            transformed.append(patch)
            reports.append({'repo': row.repo, 'commit_id': row.commit_id,
                            'label': int(row.label), **stats})
        return messages, transformed, audit

    baseline.load_records = hunk_loader
    family = args.model_family
    model_name = ('microsoft/codebert-base' if family == 'codebert'
                  else 'microsoft/graphcodebert-base')
    output = str(Path(args.out) / family)
    trainer_args = [
        '--train', args.train, '--val', args.val, '--extraction', args.extraction,
        '--out', output, '--input-mode', args.input_mode,
        '--model-name', model_name, '--max-patch-bytes', str(args.max_patch_bytes),
        '--max-length', str(args.max_length), '--message-budget', str(args.message_budget),
        '--batch-size', str(args.batch_size), '--grad-accum', str(args.grad_accum),
        '--epochs', str(args.epochs), '--patience', str(args.patience),
        '--learning-rate', str(args.learning_rate), '--seed', str(args.seed),
    ]
    if args.smoke_test:
        trainer_args.append('--smoke-test')
    if args.check_data_only:
        trainer_args.append('--check-data-only')
    sys.argv = [sys.argv[0], *trainer_args]
    trainer.main()

    # Persist selection provenance. For smoke test, only training itself is
    # reduced; original trainer loads all rows before taking smoke subset.
    dest = Path(output) / args.input_mode
    if args.smoke_test:
        dest /= 'SMOKE_NOT_FOR_REPORTING'
    dest.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    ntrain = len(pd.read_csv(args.train))
    for i, item in enumerate(reports):
        item['subset'] = 'train' if i < ntrain else 'val'
    pd.DataFrame(reports).to_csv(dest / 'hunk_selection_coverage.csv', index=False)
    meta = {'representation': 'changed-line-priority with diff/file/hunk markers',
            'label_blind': True, 'heldout_test_accessed': False,
            'max_patch_bytes': args.max_patch_bytes, 'max_length': args.max_length,
            'model_family': family, 'input_mode': args.input_mode,
            'train_manifest': args.train, 'val_manifest': args.val,
            'selection_rows': len(reports),
            'fallback_rows': sum(x['fallback'] for x in reports),
            'limitations': ['Preserves changed lines in original order; later hunks can still be truncated.',
                            'Diff metadata/header context is reduced.',
                            'Not a causal explanation or explicit graph processing.',
                            'Validation checkpoint selection remains exploratory.']}
    (dest / 'hunk_selection_report.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('HUNK SELECTION:', json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
