
"""
GraphCodeBERT training wrapper.

Reuses the CodeBERT training implementation to ensure consistent
data loading, preprocessing, training, and evaluation.

Allows versioned output directories without overriding the model.
"""

import sys
from pathlib import Path

MODEL = "microsoft/graphcodebert-base"
DEFAULT_OUTPUT = "reports/codejit_graphcodebert_v1"


def main():
    src = Path(__file__).resolve().parent
    sys.path.insert(0, str(src))

    try:
        from train_codejit_codebert import main as shared_main
    except ImportError as exc:
        raise SystemExit(
            "Missing src/train_codejit_codebert.py"
        ) from exc

    supplied = sys.argv[1:]

    if "--model-name" in supplied or any(
        arg.startswith("--model-name=") for arg in supplied
    ):
        raise SystemExit(
            "Do not override --model-name; this experiment must use "
            + MODEL
        )

    # Preserve the historical default, but permit versioned outputs.
    if "--out" not in supplied and not any(
        arg.startswith("--out=") for arg in supplied
    ):
        supplied = ["--out", DEFAULT_OUTPUT, *supplied]

    sys.argv = [
        sys.argv[0],
        "--model-name",
        MODEL,
        *supplied,
    ]

    shared_main()


if __name__ == "__main__":
    main()
