#!/usr/bin/env python3
"""Create a separate E09 hunk-aware runner with a shared source byte window.

Run from the repository root. Does not modify the historical v1 runner.
"""
from pathlib import Path
import ast
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='src/run_codejit_hunk_aware_ablation.py')
    p.add_argument('--target', default='src/run_codejit_hunk_aware_v3.py')
    args = p.parse_args()
    source, target = Path(args.source), Path(args.target)
    if not source.is_file():
        p.error(f'Missing original runner: {source}')
    if source.resolve() == target.resolve():
        p.error('Refusing to overwrite the original runner')
    if target.exists():
        p.error(f'Target already exists; refusing to overwrite: {target}')
    original = source.read_text(encoding='utf-8')
    needle = 'selected_patch(raw, max_patch_bytes)'
    replacement = 'selected_patch(_shared_source_window(raw, max_patch_bytes), max_patch_bytes)'
    if original.count(needle) != 1:
        p.error(f'Expected exactly one call {needle!r}; found {original.count(needle)}. Inspect source manually.')
    helper = '''\n\ndef _shared_source_window(raw, max_patch_bytes):\n    """Limit both representations to the same original UTF-8 byte window.\n\n    Avoid decoding an incomplete multibyte character at the boundary.\n    """\n    if isinstance(raw, bytes):\n        return raw[:max_patch_bytes].decode('utf-8', errors='ignore')\n    if isinstance(raw, str):\n        return raw.encode('utf-8')[:max_patch_bytes].decode('utf-8', errors='ignore')\n    raise TypeError(f'Unexpected raw patch type: {type(raw).__name__}')\n'''
    # Append helper at the end: it is resolved at call time, after module import.
    # For runners using `if __name__ == '__main__': main()`, it must instead be
    # defined before that block, so insert it immediately before the entrypoint.
    patched = original.replace(needle, replacement)
    entrypoint = "if __name__ == '__main__':"
    alt_entrypoint = 'if __name__ == "__main__":'
    if entrypoint in patched:
        patched = patched.replace(entrypoint, helper + '\n' + entrypoint, 1)
    elif alt_entrypoint in patched:
        patched = patched.replace(alt_entrypoint, helper + '\n' + alt_entrypoint, 1)
    else:
        p.error('Could not find a conventional __main__ entrypoint; inspect original runner')
    # Version output default without changing command-line options.
    default_out = "default='reports/codejit_hunk_aware_v1'"
    if patched.count(default_out) != 1:
        p.error('Could not identify unique historical output default; refusing unsafe patch')
    patched = patched.replace(default_out, "default='reports/codejit_hunk_aware_v3'", 1)
    ast.parse(patched, filename=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(patched, encoding='utf-8')
    print(f'Created: {target}')
    print('Original unchanged:', source)
    print('Control: hunk selection sees only the first max_patch_bytes UTF-8 bytes.')
    print('IMPORTANT: verify the original prefix loader uses the same byte-window semantics.')


if __name__ == '__main__':
    main()
