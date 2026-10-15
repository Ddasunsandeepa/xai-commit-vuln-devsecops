#!/usr/bin/env python3
"""Train GraphCodeBERT as a sequence classifier on frozen CodeJIT train/val.

This is a controlled pretrained-checkpoint comparison, NOT graph/data-flow
injection. Uses the exact data loader, tokenizer budget, optimizer, metrics,
checkpoint selection and validation-only protocol of train_codejit_codebert.py.

Place both scripts in src/ and run:
 python src/train_codejit_graphcodebert.py --train reports/codejit_train_val_v2/train_manifest.csv --val reports/codejit_train_val_v2/val_manifest.csv --input-mode message --check-data-only
 python src/train_codejit_graphcodebert.py --train reports/codejit_train_val_v2/train_manifest.csv --val reports/codejit_train_val_v2/val_manifest.csv --input-mode message --smoke-test
 python src/train_codejit_graphcodebert.py --train reports/codejit_train_val_v2/train_manifest.csv --val reports/codejit_train_val_v2/val_manifest.csv --input-mode message

Repeat with --input-mode code and combined. Held-out test is never accessed.
"""
import sys
from pathlib import Path

MODEL = 'microsoft/graphcodebert-base'
OUTPUT = 'reports/codejit_graphcodebert_v1'


def main():
    # Import sibling training implementation; do not silently diverge from
    # the CodeBERT implementation in a nine-model comparison.
    src = Path(__file__).resolve().parent
    sys.path.insert(0, str(src))
    try:
        from train_codejit_codebert import main as shared_main
    except ImportError as exc:
        raise SystemExit('Missing src/train_codejit_codebert.py (fixed tokenizer version required)') from exc

    supplied = sys.argv[1:]
    if '--model-name' in supplied or any(a.startswith('--model-name=') for a in supplied):
        raise SystemExit('Do not override --model-name; this experiment must use ' + MODEL)
    if '--out' in supplied or any(a.startswith('--out=') for a in supplied):
        raise SystemExit('Do not override --out; outputs are isolated under ' + OUTPUT)
    sys.argv = [sys.argv[0], '--model-name', MODEL, '--out', OUTPUT, *supplied]
    shared_main()


if __name__ == '__main__':
    main()
