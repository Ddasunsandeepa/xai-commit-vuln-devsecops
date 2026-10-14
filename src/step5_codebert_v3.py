"""
=============================================================================
STEP 5 - CODEBERT FINE-TUNING v3
Strict Project-Disjoint Train / Validation / Test Evaluation
=============================================================================

Purpose
-------
Evaluate CodeBERT and conventional vulnerability-prediction baselines under
a strict cross-project evaluation protocol.

Key design:
    - Entire projects are assigned to Train / Validation / Test.
    - No project may appear in more than one split.
    - Training data is balanced by majority-class undersampling.
    - Validation and test sets retain their natural class distributions.
    - Model selection and threshold tuning use ONLY the validation set.
    - The test set is used ONLY for final evaluation.
    - All baselines use exactly the same project-disjoint split.
    - Project-level bootstrap confidence intervals are reported for CodeBERT.

Relationship to v2
------------------
v2:
    Project-disjoint Train/Test split, but validation was created using a
    row-level split from the training projects. Therefore, training and
    validation could contain examples from the same projects.

v3:
    Train, Validation and Test are all project-disjoint.

    v3 also retains natural validation/test prevalence and uses balanced
    training through majority-class undersampling.

Core CodeBERT hyperparameters such as learning rate, epoch limit, maximum
sequence length and early-stopping patience remain aligned with v2.

Usage
-----
Full experiment:
    python step5_codebert_v3.py

Baselines only:
    python step5_codebert_v3.py --baselines-only

Alternative project split:
    python step5_codebert_v3.py --split-seed 7

GTX 1650 4 GB:
    BATCH_SIZE=8 GRAD_ACCUM=4 python step5_codebert_v3.py

PowerShell:
    $env:BATCH_SIZE="8"
    $env:GRAD_ACCUM="4"
    python step5_codebert_v3.py

Outputs
-------
    reports/splits_v3.json
    reports/codebert_v3_results.txt
    reports/codebert_v3_predictions.csv
    models/codebert_v3/best_model/
    models/codebert_v3/training_history.json

v2 outputs are NOT overwritten.
=============================================================================
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "data/processed/big_vul_enriched.csv"

MODEL_NAME = "microsoft/codebert-base"

TEXT_COL = "func_before"
LABEL_COL = "label"
PROJECT_COL = "project"

# Core CodeBERT configuration
EPOCHS = 5
LR = 2e-5
MAX_LEN = 256
PATIENCE = 2

WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.10

# Target ROW proportions.
# Assignment itself is performed using WHOLE PROJECTS.
TARGET_FRACS = {
    "train": 0.70,
    "val": 0.10,
    "test": 0.20,
}

MODEL_SEED = 42


# =============================================================================
# OUTPUT PATHS
# =============================================================================

OUT_MODEL_DIR = "models/codebert_v3"

OUT_REPORT = "reports/codebert_v3_results.txt"
OUT_SPLITS = "reports/splits_v3.json"
OUT_PREDS = "reports/codebert_v3_predictions.csv"


# =============================================================================
# FEATURE GROUPS
# =============================================================================

COMPLEXITY_FEATS = [
    "cyclomatic_complexity",
    "nesting_depth",
    "num_parameters",
    "num_function_calls",
    "token_diversity",
]

SIZE_FEATS = [
    "func_before_length",
    "diff_length",
    "lines_added",
    "lines_removed",
    "files_changed",
]

STRUCT_FEATS = (
    COMPLEXITY_FEATS
    + SIZE_FEATS
    + [
        "security_keyword_count",
        "comment_ratio",
        "has_security_terms",
    ]
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed):
    """
    Set random seeds used by Python, NumPy and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except ImportError:
        pass


def banner(title):
    print(
        "\n"
        + "=" * 78
        + f"\n  {title}\n"
        + "=" * 78
    )


# =============================================================================
# 1. PROJECT-DISJOINT SPLITTING
# =============================================================================

def project_disjoint_split(
    df,
    seed=42,
    n_trials=3000,
    fracs=TARGET_FRACS,
    min_projects=10,
    max_share=0.50,
):
    """
    Assign WHOLE PROJECTS to Train / Validation / Test.

    The allocation procedure performs multiple deterministic, seeded candidate
    assignments and selects the best candidate according to several criteria:

        1. Row fractions should be close to the requested 70/10/20 targets.
        2. Validation and test positive rates should remain reasonably close
           to the global positive rate.
        3. Validation and test should contain at least `min_projects`.
        4. A single project should not dominate validation or test.

    IMPORTANT:
        This is an optimized project-group allocation procedure, not a simple
        row-level random split.

    Returns
    -------
    dict
        {
            "train": set(project_names),
            "val": set(project_names),
            "test": set(project_names)
        }
    """

    grouped = (
        df.groupby(PROJECT_COL)[LABEL_COL]
        .agg(["size", "sum"])
    )

    projects = grouped.index.to_numpy()
    sizes = grouped["size"].to_numpy()
    positives = grouped["sum"].to_numpy()

    n_rows = sizes.sum()
    global_positive_rate = positives.sum() / n_rows

    split_names = list(fracs)

    targets = {
        split: fracs[split] * n_rows
        for split in split_names
    }

    rng = np.random.RandomState(seed)

    best_assignment = None
    best_score = np.inf

    for _ in range(n_trials):

        order = rng.permutation(len(projects))

        current_rows = {
            split: 0
            for split in split_names
        }

        current_positives = {
            split: 0
            for split in split_names
        }

        current_project_count = {
            split: 0
            for split in split_names
        }

        largest_project = {
            split: 0
            for split in split_names
        }

        assignment = {}

        for i in order:

            # Select the split currently having the greatest
            # relative row-count deficit.
            selected_split = max(
                split_names,
                key=lambda split:
                    (targets[split] - current_rows[split])
                    / targets[split]
            )

            assignment[i] = selected_split

            current_rows[selected_split] += sizes[i]
            current_positives[selected_split] += positives[i]
            current_project_count[selected_split] += 1

            largest_project[selected_split] = max(
                largest_project[selected_split],
                sizes[i],
            )

        score = 0.0

        for split in split_names:

            # Difference from target row fraction.
            score += abs(
                current_rows[split] / n_rows
                - fracs[split]
            ) * 4

            # Validation and test should contain enough projects.
            if (
                split != "train"
                and current_project_count[split] < min_projects
            ):
                score += 10

            if split != "train":

                split_positive_rate = (
                    current_positives[split]
                    / max(current_rows[split], 1)
                )

                # Keep class prevalence reasonably representative.
                score += abs(
                    split_positive_rate
                    - global_positive_rate
                ) * 3

                # Avoid one project dominating a held-out partition.
                largest_share = (
                    largest_project[split]
                    / max(current_rows[split], 1)
                )

                if largest_share > max_share:
                    score += 5

        if score < best_score:
            best_score = score
            best_assignment = dict(assignment)

    split_projects = {
        split: set()
        for split in split_names
    }

    for project_index, split in best_assignment.items():
        split_projects[split].add(
            projects[project_index]
        )

    return split_projects


def assert_disjoint(split_projects):
    """
    Assert that Train / Validation / Test contain no overlapping projects.
    """

    train_projects = split_projects["train"]
    val_projects = split_projects["val"]
    test_projects = split_projects["test"]

    overlap = {
        "train&val": len(
            train_projects & val_projects
        ),
        "train&test": len(
            train_projects & test_projects
        ),
        "val&test": len(
            val_projects & test_projects
        ),
    }

    for name, count in overlap.items():

        assert count == 0, (
            f"PROJECT OVERLAP detected in {name}: "
            f"{count}"
        )

    return overlap


def undersample_balance(df, seed):
    """
    Balance the TRAINING partition to approximately 50/50 by randomly
    undersampling the majority class.

    Validation and test partitions are NOT balanced.
    """

    positive = df[df[LABEL_COL] == 1]
    negative = df[df[LABEL_COL] == 0]

    n = min(
        len(positive),
        len(negative),
    )

    balanced = pd.concat(
        [
            positive.sample(
                n,
                random_state=seed,
            ),
            negative.sample(
                n,
                random_state=seed,
            ),
        ]
    )

    balanced = balanced.sample(
        frac=1.0,
        random_state=seed,
    )

    return balanced.reset_index(drop=True)


# =============================================================================
# 2. SPLIT STATISTICS
# =============================================================================

def split_statistics(df, projects):
    """
    Produce reproducibility statistics for one partition.
    """

    counts = (
        df[LABEL_COL]
        .value_counts()
        .sort_index()
    )

    project_counts = (
        df[PROJECT_COL]
        .value_counts()
    )

    return {
        "rows": int(len(df)),

        "projects": int(len(projects)),

        "class_counts": {
            "0": int(counts.get(0, 0)),
            "1": int(counts.get(1, 0)),
        },

        "positive_rate": float(
            df[LABEL_COL].mean()
        ),

        "largest_project": (
            str(project_counts.index[0])
            if len(project_counts)
            else None
        ),

        "largest_project_rows": (
            int(project_counts.iloc[0])
            if len(project_counts)
            else 0
        ),

        "largest_project_share": (
            float(
                project_counts.iloc[0] / len(df)
            )
            if len(df)
            else 0.0
        ),
    }


# =============================================================================
# 3. METRICS
# =============================================================================

def metrics_at(y_true, probabilities, threshold):
    """
    Calculate classification metrics at a fixed decision threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "thr": float(threshold),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def tune_threshold(y_true, probabilities):
    """
    Select the F1-maximising threshold using ONLY validation predictions.

    Search grid:
        0.01 ... 0.99
    """

    thresholds = np.linspace(
        0.01,
        0.99,
        99,
    )

    f1_scores = [
        f1_score(
            y_true,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_index = int(
        np.argmax(f1_scores)
    )

    return float(
        thresholds[best_index]
    )


def fmt(name, metrics):
    """
    Pretty-print one evaluation result.
    """

    return (
        f"{name:<42s} "
        f"F1={metrics['f1']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  "
        f"AUC={metrics['roc_auc']:.4f}  "
        f"PR-AUC={metrics['pr_auc']:.4f}  "
        f"thr={metrics['thr']:.3f}  "
        f"(TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TN={metrics['tn']})"
    )


# =============================================================================
# 4. PROJECT-CLUSTER BOOTSTRAP
# =============================================================================

def cluster_bootstrap(
    y_true,
    probabilities,
    threshold,
    groups,
    n_boot=500,
    seed=0,
):
    """
    Estimate 95% confidence intervals by resampling PROJECTS rather than
    individual rows.

    Projects are treated as the resampling unit because this experiment
    evaluates cross-project generalisation.
    """

    rng = np.random.RandomState(seed)

    unique_groups = np.unique(groups)

    indices_by_group = {
        group: np.where(groups == group)[0]
        for group in unique_groups
    }

    f1_values = []
    auc_values = []
    pr_auc_values = []

    for _ in range(n_boot):

        sampled_groups = rng.choice(
            unique_groups,
            len(unique_groups),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                indices_by_group[group]
                for group in sampled_groups
            ]
        )

        y_sample = y_true[sampled_indices]
        p_sample = probabilities[sampled_indices]

        # ROC-AUC is undefined when only one class exists.
        if y_sample.min() == y_sample.max():
            continue

        predictions = (
            p_sample >= threshold
        ).astype(int)

        f1_values.append(
            f1_score(
                y_sample,
                predictions,
                zero_division=0,
            )
        )

        auc_values.append(
            roc_auc_score(
                y_sample,
                p_sample,
            )
        )

        pr_auc_values.append(
            average_precision_score(
                y_sample,
                p_sample,
            )
        )

    def ci(values):

        return (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )

    return {
        "f1": ci(f1_values),
        "roc_auc": ci(auc_values),
        "pr_auc": ci(pr_auc_values),
        "successful_bootstraps": len(f1_values),
    }


# =============================================================================
# 5. BASELINES
# =============================================================================

def run_baselines(train, val, test):
    """
    Train all conventional baselines using exactly the same project-disjoint
    experimental partitions used by CodeBERT.

    Thresholds are selected ONLY using validation data.
    """

    results = {}
    probabilities = {}

    y_train = train[LABEL_COL].to_numpy()
    y_val = val[LABEL_COL].to_numpy()
    y_test = test[LABEL_COL].to_numpy()

    def fit_eval(
        name,
        X_train,
        X_val,
        X_test,
    ):

        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=MODEL_SEED,
        )

        classifier.fit(
            X_train,
            y_train,
        )

        p_val = classifier.predict_proba(
            X_val
        )[:, 1]

        p_test = classifier.predict_proba(
            X_test
        )[:, 1]

        threshold = tune_threshold(
            y_val,
            p_val,
        )

        results[name] = {
            "val": metrics_at(
                y_val,
                p_val,
                threshold,
            ),

            "test": metrics_at(
                y_test,
                p_test,
                threshold,
            ),

            "test_default": metrics_at(
                y_test,
                p_test,
                0.5,
            ),
        }

        probabilities[name] = (
            p_val,
            p_test,
        )

    # -------------------------------------------------------------------------
    # Numerical / structural baselines
    # -------------------------------------------------------------------------

    feature_sets = [
        (
            "Structural LR (all)",
            STRUCT_FEATS,
        ),
        (
            "Complexity-only LR",
            COMPLEXITY_FEATS,
        ),
        (
            "Size-only LR",
            SIZE_FEATS,
        ),
    ]

    for name, features in feature_sets:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            train[features]
        )

        X_val = scaler.transform(
            val[features]
        )

        X_test = scaler.transform(
            test[features]
        )

        fit_eval(
            name,
            X_train,
            X_val,
            X_test,
        )

    # -------------------------------------------------------------------------
    # Character-level TF-IDF baseline
    # -------------------------------------------------------------------------

    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
    )

    X_train = tfidf.fit_transform(
        train[TEXT_COL].astype(str)
    )

    X_val = tfidf.transform(
        val[TEXT_COL].astype(str)
    )

    X_test = tfidf.transform(
        test[TEXT_COL].astype(str)
    )

    fit_eval(
        "Char TF-IDF LR",
        X_train,
        X_val,
        X_test,
    )

    return results, probabilities


# =============================================================================
# 6. CODEBERT
# =============================================================================

def run_codebert(train, val, test):

    import torch

    from torch.utils.data import DataLoader

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device == "cuda":

        vram = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 1e9
        )

        print(
            f"Device: CUDA\n"
            f"GPU   : {torch.cuda.get_device_name(0)}\n"
            f"VRAM  : {vram:.1f} GB"
        )

    else:

        vram = 0

        print("Device: CPU")
        print(
            "WARNING: CodeBERT training on CPU "
            "will be extremely slow."
        )

    # -------------------------------------------------------------------------
    # Batch configuration
    # -------------------------------------------------------------------------

    default_micro_batch = (
        16
        if vram >= 10
        else 8
    )

    micro_batch = int(
        os.environ.get(
            "BATCH_SIZE",
            default_micro_batch,
        )
    )

    grad_accum = int(
        os.environ.get(
            "GRAD_ACCUM",
            max(
                1,
                32 // micro_batch,
            ),
        )
    )

    print(
        f"Micro-batch       : {micro_batch}\n"
        f"Gradient accum.   : {grad_accum}\n"
        f"Effective batch   : "
        f"{micro_batch * grad_accum}"
    )

    # -------------------------------------------------------------------------
    # Tokenizer / model
    # -------------------------------------------------------------------------

    print(
        f"\nLoading tokenizer and model: "
        f"{MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
        .to(device)
    )

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------

    def make_loader(
        dataframe,
        shuffle,
        batch_size,
    ):

        texts = (
            dataframe[TEXT_COL]
            .astype(str)
            .tolist()
        )

        labels = (
            dataframe[LABEL_COL]
            .tolist()
        )

        def collate(batch):

            batch_texts, batch_labels = zip(
                *batch
            )

            encoded = tokenizer(
                list(batch_texts),
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            )

            encoded["labels"] = torch.tensor(
                batch_labels,
                dtype=torch.long,
            )

            return encoded

        return DataLoader(
            list(
                zip(
                    texts,
                    labels,
                )
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    train_loader = make_loader(
        train,
        True,
        micro_batch,
    )

    val_loader = make_loader(
        val,
        False,
        micro_batch * 2,
    )

    test_loader = make_loader(
        test,
        False,
        micro_batch * 2,
    )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def predict(loader):

        model.eval()

        outputs = []

        for batch in loader:

            # Labels are not required during inference.
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            with torch.autocast(
                device_type=device,
                dtype=torch.float16,
                enabled=(device == "cuda"),
            ):

                logits = model(
                    **batch
                ).logits

            probabilities = torch.softmax(
                logits.float(),
                dim=-1,
            )[:, 1]

            outputs.append(
                probabilities
                .cpu()
                .numpy()
            )

        return np.concatenate(outputs)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # IMPORTANT:
    # ceil() is required because the final partial gradient-accumulation group
    # also performs an optimizer step.
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader)
        / grad_accum
    )

    total_optimizer_steps = (
        EPOCHS
        * optimizer_steps_per_epoch
    )

    warmup_steps = int(
        WARMUP_FRAC
        * total_optimizer_steps
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda"),
    )

    print(
        f"Optimizer steps/epoch : "
        f"{optimizer_steps_per_epoch}\n"
        f"Total optimizer steps : "
        f"{total_optimizer_steps}\n"
        f"Warmup steps          : "
        f"{warmup_steps}"
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    y_val = val[LABEL_COL].to_numpy()

    best_f1 = -1.0
    bad_epochs = 0
    history = []

    best_model_dir = os.path.join(
        OUT_MODEL_DIR,
        "best_model",
    )

    os.makedirs(
        best_model_dir,
        exist_ok=True,
    )

    banner("CODEBERT TRAINING")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        n_batches = 0

        optimizer.zero_grad()

        for batch_index, batch in enumerate(
            train_loader,
            1,
        ):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type=device,
                dtype=torch.float16,
                enabled=(device == "cuda"),
            ):

                # Training data is already balanced through
                # majority-class undersampling.
                loss = (
                    model(**batch).loss
                    / grad_accum
                )

            scaler.scale(
                loss
            ).backward()

            running_loss += (
                loss.item()
                * grad_accum
            )

            n_batches += 1

            should_step = (
                batch_index % grad_accum == 0
                or batch_index == len(train_loader)
            )

            if should_step:

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scheduler.step()

                optimizer.zero_grad()

            if (
                batch_index
                % (20 * grad_accum)
                == 0
            ):

                print(
                    f"  Epoch {epoch} "
                    f"step "
                    f"{math.ceil(batch_index / grad_accum)} "
                    f"loss="
                    f"{running_loss / n_batches:.4f}"
                )

        # ---------------------------------------------------------------------
        # Project-disjoint validation
        # ---------------------------------------------------------------------

        val_probabilities = predict(
            val_loader
        )

        threshold = tune_threshold(
            y_val,
            val_probabilities,
        )

        val_metrics = metrics_at(
            y_val,
            val_probabilities,
            threshold,
        )

        avg_loss = (
            running_loss
            / n_batches
        )

        print(
            f"\nEpoch {epoch} "
            f"average loss: "
            f"{avg_loss:.4f}"
        )

        print(
            fmt(
                f"[Val epoch {epoch}] "
                f"(project-disjoint)",
                val_metrics,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                **val_metrics,
            }
        )

        # ---------------------------------------------------------------------
        # Model selection using VALIDATION ONLY
        # ---------------------------------------------------------------------

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]
            bad_epochs = 0

            model.save_pretrained(
                best_model_dir
            )

            tokenizer.save_pretrained(
                best_model_dir
            )

            print(
                f"Best validation F1: "
                f"{best_f1:.4f} "
                f"- model saved "
                f"(threshold="
                f"{threshold:.3f})"
            )

        else:

            bad_epochs += 1

            print(
                f"No improvement "
                f"({bad_epochs}/"
                f"{PATIENCE})"
            )

            if bad_epochs >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------

    with open(
        os.path.join(
            OUT_MODEL_DIR,
            "training_history.json",
        ),
        "w",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Reload BEST model
    # -------------------------------------------------------------------------

    banner("FINAL TEST EVALUATION")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            best_model_dir
        )
        .to(device)
    )

    # Recalculate validation probabilities using the BEST model.
    val_probabilities = predict(
        val_loader
    )

    test_probabilities = predict(
        test_loader
    )

    return (
        val_probabilities,
        test_probabilities,
    )


# =============================================================================
# 7. MAIN EXPERIMENT
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baselines-only",
        action="store_true",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--data",
        default=DATA_PATH,
    )

    args = parser.parse_args()

    set_seed(
        MODEL_SEED
    )

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    os.makedirs(
        OUT_MODEL_DIR,
        exist_ok=True,
    )

    start_time = time.time()

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    banner("LOAD DATA")

    df = pd.read_csv(
        args.data
    )

    df = df.dropna(
        subset=[
            TEXT_COL,
            LABEL_COL,
            PROJECT_COL,
        ]
    )

    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(int)
    )

    print(
        f"Rows          : {len(df):,}\n"
        f"Projects      : "
        f"{df[PROJECT_COL].nunique()}\n"
        f"Positive rate : "
        f"{df[LABEL_COL].mean():.4f}"
    )

    # =========================================================================
    # PROJECT-DISJOINT SPLIT
    # =========================================================================

    banner(
        "PROJECT-DISJOINT "
        "TRAIN / VALIDATION / TEST SPLIT"
    )

    split_projects = (
        project_disjoint_split(
            df,
            seed=args.split_seed,
        )
    )

    overlap = assert_disjoint(
        split_projects
    )

    parts = {
        split: (
            df[
                df[PROJECT_COL]
                .isin(
                    split_projects[split]
                )
            ]
            .reset_index(drop=True)
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    assert (
        sum(
            len(part)
            for part in parts.values()
        )
        == len(df)
    )

    stats = {}

    for split, partition in parts.items():

        stats[split] = split_statistics(
            partition,
            split_projects[split],
        )

        s = stats[split]

        print(
            f"{split.upper():<5s} | "
            f"rows={s['rows']:>5,} | "
            f"projects={s['projects']:>3} | "
            f"safe={s['class_counts']['0']:>5} | "
            f"vulnerable="
            f"{s['class_counts']['1']:>5} | "
            f"pos={s['positive_rate']:.3f}"
        )

        print(
            f"       largest project: "
            f"{s['largest_project']} "
            f"({s['largest_project_rows']} rows, "
            f"{s['largest_project_share']:.1%})"
        )

    print(
        "\nProject overlap:"
    )

    print(
        f"  Train ∩ Val  = "
        f"{overlap['train&val']}"
    )

    print(
        f"  Train ∩ Test = "
        f"{overlap['train&test']}"
    )

    print(
        f"  Val ∩ Test   = "
        f"{overlap['val&test']}"
    )

    print(
        "All project-overlap checks passed."
    )

    # =========================================================================
    # BALANCE TRAINING DATA ONLY
    # =========================================================================

    train_balanced = (
        undersample_balance(
            parts["train"],
            MODEL_SEED,
        )
    )

    print(
        "\nTraining after "
        "majority-class undersampling:"
    )

    print(
        f"  Rows          : "
        f"{len(train_balanced):,}"
    )

    print(
        f"  Positive rate : "
        f"{train_balanced[LABEL_COL].mean():.3f}"
    )

    print(
        "\nValidation and test retain "
        "their NATURAL class distributions."
    )

    # =========================================================================
    # SAVE SPLIT METADATA
    # =========================================================================

    split_metadata = {
        "split_seed": args.split_seed,

        "model_seed": MODEL_SEED,

        "target_row_fractions": TARGET_FRACS,

        "global": {
            "rows": int(len(df)),
            "projects": int(
                df[PROJECT_COL].nunique()
            ),
            "positive_rate": float(
                df[LABEL_COL].mean()
            ),
        },

        "overlap": overlap,

        "partitions": stats,

        "balanced_training": {
            "rows": int(
                len(train_balanced)
            ),
            "class_counts": {
                "0": int(
                    (
                        train_balanced[LABEL_COL]
                        == 0
                    ).sum()
                ),
                "1": int(
                    (
                        train_balanced[LABEL_COL]
                        == 1
                    ).sum()
                ),
            },
            "positive_rate": float(
                train_balanced[LABEL_COL]
                .mean()
            ),
        },

        "projects": {
            split: sorted(
                map(
                    str,
                    split_projects[split],
                )
            )
            for split in split_projects
        },
    }

    with open(
        OUT_SPLITS,
        "w",
    ) as file:

        json.dump(
            split_metadata,
            file,
            indent=2,
        )

    print(
        f"\nSplit metadata -> "
        f"{OUT_SPLITS}"
    )

    # =========================================================================
    # BASELINES
    # =========================================================================

    banner(
        "BASELINES "
        "(SAME PROJECT-DISJOINT SPLIT)"
    )

    baseline_results, baseline_probs = (
        run_baselines(
            train_balanced,
            parts["val"],
            parts["test"],
        )
    )

    for name, result in (
        baseline_results.items()
    ):

        print(
            fmt(
                f"{name} [VAL]",
                result["val"],
            )
        )

        print(
            fmt(
                f"{name} [TEST]",
                result["test"],
            )
        )

        print()

    # =========================================================================
    # BASIC TEST INFORMATION
    # =========================================================================

    y_val = (
        parts["val"][LABEL_COL]
        .to_numpy()
    )

    y_test = (
        parts["test"][LABEL_COL]
        .to_numpy()
    )

    test_projects = (
        parts["test"][PROJECT_COL]
        .to_numpy()
    )

    test_prevalence = (
        y_test.mean()
    )

    # F1 obtained if every example is predicted vulnerable.
    trivial_f1 = (
        2 * test_prevalence
        / (1 + test_prevalence)
    )

    # =========================================================================
    # CODEBERT
    # =========================================================================

    codebert_results = None
    codebert_ci = None

    if not args.baselines_only:

        val_probabilities, test_probabilities = (
            run_codebert(
                train_balanced,
                parts["val"],
                parts["test"],
            )
        )

        # Threshold selection occurs ONLY on validation.
        best_threshold = tune_threshold(
            y_val,
            val_probabilities,
        )

        codebert_results = {

            "val": metrics_at(
                y_val,
                val_probabilities,
                best_threshold,
            ),

            "test_default": metrics_at(
                y_test,
                test_probabilities,
                0.5,
            ),

            "test": metrics_at(
                y_test,
                test_probabilities,
                best_threshold,
            ),
        }

        print(
            "\n"
            + fmt(
                "CodeBERT v3 "
                "[val-tuned threshold] VAL",
                codebert_results["val"],
            )
        )

        print(
            fmt(
                "CodeBERT v3 "
                "[default 0.5] TEST",
                codebert_results[
                    "test_default"
                ],
            )
        )

        print(
            fmt(
                "CodeBERT v3 "
                "[val-tuned threshold] TEST",
                codebert_results["test"],
            )
        )

        # =====================================================================
        # SAVE PREDICTIONS
        # =====================================================================

        val_predictions = pd.DataFrame(
            {
                "split": "val",

                "commit_hash":
                    parts["val"][
                        "commit_hash"
                    ],

                "project":
                    parts["val"][
                        PROJECT_COL
                    ],

                "label":
                    y_val,

                "prob":
                    val_probabilities,
            }
        )

        test_predictions = pd.DataFrame(
            {
                "split": "test",

                "commit_hash":
                    parts["test"][
                        "commit_hash"
                    ],

                "project":
                    parts["test"][
                        PROJECT_COL
                    ],

                "label":
                    y_test,

                "prob":
                    test_probabilities,
            }
        )

        predictions = pd.concat(
            [
                val_predictions,
                test_predictions,
            ],
            ignore_index=True,
        )

        predictions.to_csv(
            OUT_PREDS,
            index=False,
        )

        # =====================================================================
        # PROJECT-CLUSTER BOOTSTRAP CI
        # =====================================================================

        codebert_ci = cluster_bootstrap(
            y_test,
            test_probabilities,
            best_threshold,
            test_projects,
            n_boot=500,
            seed=MODEL_SEED,
        )

    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================

    banner(
        "FINAL COMPARISON "
        "(SAME PROJECT-DISJOINT TEST SET)"
    )

    comparison_rows = [
        (
            "Predict-all-vulnerable (trivial)",
            trivial_f1,
            0.5000,
            test_prevalence,
        )
    ]

    for name, result in (
        baseline_results.items()
    ):

        comparison_rows.append(
            (
                name,
                result["test"]["f1"],
                result["test"]["roc_auc"],
                result["test"]["pr_auc"],
            )
        )

    if codebert_results:

        comparison_rows.append(
            (
                "CodeBERT v3 (val-tuned thr)",
                codebert_results[
                    "test"
                ]["f1"],
                codebert_results[
                    "test"
                ]["roc_auc"],
                codebert_results[
                    "test"
                ]["pr_auc"],
            )
        )

    header = (
        f"{'Approach':<38s}"
        f"{'F1':>10s}"
        f"{'ROC-AUC':>12s}"
        f"{'PR-AUC':>12s}"
    )

    comparison_lines = [
        header,
        "-" * len(header),
    ]

    for (
        name,
        f1_value,
        auc_value,
        pr_auc_value,
    ) in comparison_rows:

        comparison_lines.append(
            f"{name:<38s}"
            f"{f1_value:>10.4f}"
            f"{auc_value:>12.4f}"
            f"{pr_auc_value:>12.4f}"
        )

    print(
        "\n".join(
            comparison_lines
        )
    )

    print(
        f"\nTest prevalence = "
        f"{test_prevalence:.4f}"
    )

    print(
        "PR-AUC of a random ranking "
        f"≈ prevalence = "
        f"{test_prevalence:.4f}"
    )

    # =========================================================================
    # CONFIDENCE INTERVAL
    # =========================================================================

    ci_text = ""

    if (
        codebert_results
        and codebert_ci
    ):

        ci_text = (
            "CodeBERT v3 test 95% "
            "project-cluster bootstrap CI:\n"
            f"  F1      : "
            f"{codebert_ci['f1'][0]:.3f}"
            f" - "
            f"{codebert_ci['f1'][1]:.3f}\n"
            f"  ROC-AUC : "
            f"{codebert_ci['roc_auc'][0]:.3f}"
            f" - "
            f"{codebert_ci['roc_auc'][1]:.3f}\n"
            f"  PR-AUC  : "
            f"{codebert_ci['pr_auc'][0]:.3f}"
            f" - "
            f"{codebert_ci['pr_auc'][1]:.3f}\n"
            f"  Successful bootstrap samples: "
            f"{codebert_ci['successful_bootstraps']}"
        )

        print(
            "\n" + ci_text
        )

        print(
            "\n"
            "v2 -> v3 methodological comparison:"
        )

        print(
            "  v2 validation F1 = 0.9394 "
            "(row-level validation with "
            "shared training projects)"
        )

        print(
            f"  v3 validation F1 = "
            f"{codebert_results['val']['f1']:.4f} "
            "(project-disjoint validation)"
        )

        print(
            "\nIMPORTANT:"
        )

        print(
            "  v2 and v3 test partitions may contain "
            "different projects."
        )

        print(
            "  Therefore their test scores should NOT "
            "be interpreted as a direct controlled "
            "performance comparison."
        )

        print(
            "  The primary v3 comparison is between "
            "models evaluated on the SAME v3 split."
        )

    # =========================================================================
    # SAVE REPORT
    # =========================================================================

    with open(
        OUT_REPORT,
        "w",
    ) as file:

        file.write(
            "CodeBERT v3 - Strict "
            "Project-Disjoint Evaluation\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Split seed: "
            f"{args.split_seed}\n"
        )

        file.write(
            f"Model seed: "
            f"{MODEL_SEED}\n"
        )

        file.write(
            f"Project overlap: "
            f"{overlap}\n\n"
        )

        file.write(
            "DATA PARTITIONS\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        for split in (
            "train",
            "val",
            "test",
        ):

            s = stats[split]

            file.write(
                f"{split}: "
                f"rows={s['rows']} "
                f"projects={s['projects']} "
                f"safe={s['class_counts']['0']} "
                f"vulnerable="
                f"{s['class_counts']['1']} "
                f"positive_rate="
                f"{s['positive_rate']:.4f} "
                f"largest_project="
                f"{s['largest_project']} "
                f"largest_share="
                f"{s['largest_project_share']:.4f}\n"
            )

        file.write(
            "\nBalanced training:\n"
        )

        file.write(
            f"rows={len(train_balanced)} "
            f"positive_rate="
            f"{train_balanced[LABEL_COL].mean():.4f}\n"
        )

        file.write(
            "\nFINAL COMPARISON\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(
            "\n".join(
                comparison_lines
            )
            + "\n"
        )

        file.write(
            f"\nTest prevalence = "
            f"{test_prevalence:.4f}\n"
        )

        file.write(
            f"Random PR-AUC baseline ≈ "
            f"{test_prevalence:.4f}\n\n"
        )

        file.write(
            "BASELINE DETAILS\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        for name, result in (
            baseline_results.items()
        ):

            file.write(
                fmt(
                    f"{name} [VAL]",
                    result["val"],
                )
                + "\n"
            )

            file.write(
                fmt(
                    f"{name} [TEST]",
                    result["test"],
                )
                + "\n"
            )

        if codebert_results:

            file.write(
                "\nCODEBERT DETAILS\n"
            )

            file.write(
                "-" * 70
                + "\n"
            )

            file.write(
                fmt(
                    "CodeBERT v3 VAL",
                    codebert_results["val"],
                )
                + "\n"
            )

            file.write(
                fmt(
                    "CodeBERT v3 TEST "
                    "default",
                    codebert_results[
                        "test_default"
                    ],
                )
                + "\n"
            )

            file.write(
                fmt(
                    "CodeBERT v3 TEST "
                    "val-tuned",
                    codebert_results["test"],
                )
                + "\n\n"
            )

            file.write(
                ci_text
                + "\n"
            )

    # =========================================================================
    # FINISH
    # =========================================================================

    elapsed_minutes = (
        time.time()
        - start_time
    ) / 60

    banner(
        "EXPERIMENT COMPLETE"
    )

    print(
        f"Report      -> "
        f"{OUT_REPORT}"
    )

    print(
        f"Splits      -> "
        f"{OUT_SPLITS}"
    )

    if not args.baselines_only:

        print(
            f"Predictions -> "
            f"{OUT_PREDS}"
        )

        print(
            f"Best model  -> "
            f"{OUT_MODEL_DIR}/best_model"
        )

        print(
            f"History     -> "
            f"{OUT_MODEL_DIR}/training_history.json"
        )

    print(
        f"\nElapsed time: "
        f"{elapsed_minutes:.1f} minutes"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()