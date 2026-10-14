"""
Multi-seed cross-project robustness experiment.

Research question
-----------------
Does CodeBERT consistently generalize to completely unseen software
projects, or is the observed performance dependent on a particular
project split?

Protocol
--------
1. Split by PROJECT, never by individual rows.
2. Train, validation and test projects are mutually exclusive.
3. Preserve natural validation/test class distributions.
4. Undersample ONLY the training partition.
5. Tune thresholds using validation data only.
6. Select CodeBERT checkpoint using validation F1 only.
7. Evaluate the held-out test set after model selection.
8. Repeat the complete experiment over several random seeds.
9. Report per-seed results and mean ± standard deviation.

IMPORTANT:
The test set must never be used for:
- threshold tuning
- checkpoint selection
- hyperparameter tuning
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = Path("data/processed/big_vul_enriched.csv")

REPORT_DIR = Path("reports/multiseed")
MODEL_DIR = Path("models/codebert_multiseed")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Seeds
# -------------------------------------------------------------------------

SEEDS = [42, 7, 21, 84, 123]


# -------------------------------------------------------------------------
# Dataset columns
# -------------------------------------------------------------------------

PROJECT_COLUMN = "project"
LABEL_COLUMN = "label"
TEXT_COLUMN = "func_before"


# -------------------------------------------------------------------------
# Split targets
# -------------------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Avoid a single project dominating validation/test where possible.
MAX_VAL_PROJECT_SHARE = 0.30
MAX_TEST_PROJECT_SHARE = 0.30

# Number of candidate random project assignments attempted for each seed.
SPLIT_SEARCH_ATTEMPTS = 5000


# -------------------------------------------------------------------------
# Training configuration
# -------------------------------------------------------------------------

MODEL_NAME = "microsoft/codebert-base"

MAX_LENGTH = 256

EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

MICRO_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2

PATIENCE = 2

WARMUP_RATIO = 0.10


# -------------------------------------------------------------------------
# Baseline features
# -------------------------------------------------------------------------

STRUCTURAL_FEATURES = [
    "lines_added",
    "lines_removed",
    "files_changed",
    "diff_length",
    "func_before_length",
    "has_security_terms",
    "cyclomatic_complexity",
    "num_parameters",
    "num_function_calls",
    "nesting_depth",
    "token_diversity",
    "security_keyword_count",
    "comment_ratio",
]

COMPLEXITY_FEATURES = [
    "cyclomatic_complexity",
    "num_parameters",
    "num_function_calls",
    "nesting_depth",
    "token_diversity",
    "comment_ratio",
]

SIZE_FEATURES = [
    "lines_added",
    "lines_removed",
    "files_changed",
    "diff_length",
    "func_before_length",
]


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# METRICS
# =============================================================================

def safe_roc_auc(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        return float("nan")

    return roc_auc_score(y_true, probabilities)


def safe_pr_auc(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        return float("nan")

    return average_precision_score(y_true, probabilities)


def evaluate_predictions(
    y_true,
    probabilities,
    threshold: float,
):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = safe_roc_auc(
        y_true,
        probabilities,
    )

    pr_auc = safe_pr_auc(
        y_true,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Select threshold using VALIDATION DATA ONLY.

    Search thresholds from 0.01 to 0.99.
    """

    thresholds = np.arange(
        0.01,
        1.00,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


def print_metrics(
    name: str,
    metrics: dict,
):
    print(
        f"{name:<43} "
        f"F1={metrics['f1']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  "
        f"AUC={metrics['roc_auc']:.4f}  "
        f"PR-AUC={metrics['pr_auc']:.4f}  "
        f"thr={metrics['threshold']:.3f}"
    )


# =============================================================================
# PROJECT-DISJOINT SPLITTING
# =============================================================================

def describe_split(
    name: str,
    df: pd.DataFrame,
):
    counts = df[LABEL_COLUMN].value_counts()

    safe = int(counts.get(0, 0))
    vulnerable = int(counts.get(1, 0))

    project_counts = df[PROJECT_COLUMN].value_counts()

    largest_project = project_counts.index[0]
    largest_count = int(project_counts.iloc[0])
    largest_share = largest_count / len(df)

    print(
        f"{name:<5} | "
        f"rows={len(df):5,d} | "
        f"projects={df[PROJECT_COLUMN].nunique():3d} | "
        f"safe={safe:5d} | "
        f"vulnerable={vulnerable:5d} | "
        f"pos={df[LABEL_COLUMN].mean():.3f}"
    )

    print(
        f"      largest project: "
        f"{largest_project} "
        f"({largest_count} rows, "
        f"{largest_share:.1%})"
    )


def split_score(
    train_df,
    val_df,
    test_df,
    global_positive_rate,
):
    """
    Lower score is better.

    We want:
    - approximately 70/10/20 row distribution
    - similar positive rates
    - no extreme project dominance
    """

    total = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    row_error = (
        abs(len(train_df) / total - TRAIN_RATIO)
        + abs(len(val_df) / total - VAL_RATIO)
        + abs(len(test_df) / total - TEST_RATIO)
    )

    prevalence_error = (
        abs(train_df[LABEL_COLUMN].mean() - global_positive_rate)
        + abs(val_df[LABEL_COLUMN].mean() - global_positive_rate)
        + abs(test_df[LABEL_COLUMN].mean() - global_positive_rate)
    )

    val_largest_share = (
        val_df[PROJECT_COLUMN]
        .value_counts(normalize=True)
        .max()
    )

    test_largest_share = (
        test_df[PROJECT_COLUMN]
        .value_counts(normalize=True)
        .max()
    )

    dominance_penalty = 0.0

    if val_largest_share > MAX_VAL_PROJECT_SHARE:
        dominance_penalty += (
            val_largest_share
            - MAX_VAL_PROJECT_SHARE
        ) * 5

    if test_largest_share > MAX_TEST_PROJECT_SHARE:
        dominance_penalty += (
            test_largest_share
            - MAX_TEST_PROJECT_SHARE
        ) * 5

    return (
        row_error
        + prevalence_error
        + dominance_penalty
    )


def create_project_disjoint_split(
    df: pd.DataFrame,
    seed: int,
):
    """
    Search for a reasonable project-disjoint partition.

    Projects are shuffled, not rows.

    The search attempts to obtain:
        ~70% train rows
        ~10% validation rows
        ~20% test rows

    while keeping class prevalence reasonably similar and
    avoiding one project dominating validation/test.
    """

    projects = df[PROJECT_COLUMN].unique().tolist()

    global_positive_rate = df[LABEL_COLUMN].mean()

    best = None
    best_score = float("inf")

    base_rng = np.random.default_rng(seed)

    for attempt in range(SPLIT_SEARCH_ATTEMPTS):

        # Deterministic sequence derived from the experiment seed.
        attempt_seed = int(
            base_rng.integers(
                0,
                2**31 - 1,
            )
        )

        rng = np.random.default_rng(attempt_seed)

        shuffled = projects.copy()
        rng.shuffle(shuffled)

        n_projects = len(shuffled)

        n_train_projects = int(
            round(n_projects * TRAIN_RATIO)
        )

        n_val_projects = int(
            round(n_projects * VAL_RATIO)
        )

        train_projects = set(
            shuffled[:n_train_projects]
        )

        val_projects = set(
            shuffled[
                n_train_projects:
                n_train_projects + n_val_projects
            ]
        )

        test_projects = set(
            shuffled[
                n_train_projects + n_val_projects:
            ]
        )

        train_df = df[
            df[PROJECT_COLUMN].isin(train_projects)
        ].copy()

        val_df = df[
            df[PROJECT_COLUMN].isin(val_projects)
        ].copy()

        test_df = df[
            df[PROJECT_COLUMN].isin(test_projects)
        ].copy()

        # Every partition must contain both classes.
        if (
            train_df[LABEL_COLUMN].nunique() < 2
            or val_df[LABEL_COLUMN].nunique() < 2
            or test_df[LABEL_COLUMN].nunique() < 2
        ):
            continue

        score = split_score(
            train_df,
            val_df,
            test_df,
            global_positive_rate,
        )

        if score < best_score:
            best_score = score

            best = (
                train_df,
                val_df,
                test_df,
            )

    if best is None:
        raise RuntimeError(
            f"Could not create valid split for seed {seed}"
        )

    train_df, val_df, test_df = best

    train_projects = set(
        train_df[PROJECT_COLUMN].unique()
    )

    val_projects = set(
        val_df[PROJECT_COLUMN].unique()
    )

    test_projects = set(
        test_df[PROJECT_COLUMN].unique()
    )

    assert train_projects.isdisjoint(val_projects)
    assert train_projects.isdisjoint(test_projects)
    assert val_projects.isdisjoint(test_projects)

    return (
        train_df,
        val_df,
        test_df,
    )


# =============================================================================
# TRAINING-ONLY UNDERSAMPLING
# =============================================================================

def balance_training_data(
    train_df: pd.DataFrame,
    seed: int,
):
    """
    Undersample majority class ONLY in training.

    Validation and test remain untouched.
    """

    vulnerable = train_df[
        train_df[LABEL_COLUMN] == 1
    ]

    safe = train_df[
        train_df[LABEL_COLUMN] == 0
    ]

    target = min(
        len(vulnerable),
        len(safe),
    )

    vulnerable_sample = vulnerable.sample(
        n=target,
        random_state=seed,
    )

    safe_sample = safe.sample(
        n=target,
        random_state=seed,
    )

    balanced = pd.concat(
        [
            vulnerable_sample,
            safe_sample,
        ]
    )

    balanced = balanced.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return balanced


# =============================================================================
# BASELINES
# =============================================================================

def prepare_numeric_features(
    df,
    feature_columns,
):
    X = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(float)
    )

    return X


def run_logistic_baseline(
    name,
    feature_columns,
    train_df,
    val_df,
    test_df,
    seed,
):
    X_train = prepare_numeric_features(
        train_df,
        feature_columns,
    )

    X_val = prepare_numeric_features(
        val_df,
        feature_columns,
    )

    X_test = prepare_numeric_features(
        test_df,
        feature_columns,
    )

    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values
    y_test = test_df[LABEL_COLUMN].values

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    val_metrics = evaluate_predictions(
        y_val,
        val_probabilities,
        threshold,
    )

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate_predictions(
        y_test,
        test_probabilities,
        threshold,
    )

    print_metrics(
        f"{name} [VAL]",
        val_metrics,
    )

    print_metrics(
        f"{name} [TEST]",
        test_metrics,
    )

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }


def run_tfidf_baseline(
    train_df,
    val_df,
    test_df,
    seed,
):
    train_text = (
        train_df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

    val_text = (
        val_df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

    test_text = (
        test_df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values
    y_test = test_df[LABEL_COLUMN].values

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(
        train_text
    )

    X_val = vectorizer.transform(
        val_text
    )

    X_test = vectorizer.transform(
        test_text
    )

    classifier = LogisticRegression(
        max_iter=3000,
        random_state=seed,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    val_probabilities = classifier.predict_proba(
        X_val
    )[:, 1]

    threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    val_metrics = evaluate_predictions(
        y_val,
        val_probabilities,
        threshold,
    )

    test_probabilities = classifier.predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate_predictions(
        y_test,
        test_probabilities,
        threshold,
    )

    print_metrics(
        "Char TF-IDF LR [VAL]",
        val_metrics,
    )

    print_metrics(
        "Char TF-IDF LR [TEST]",
        test_metrics,
    )

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }


# =============================================================================
# CODEBERT DATASET
# =============================================================================

class CodeDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
    ):
        self.texts = (
            dataframe[TEXT_COLUMN]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe[LABEL_COLUMN]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids":
                encoded["input_ids"].squeeze(0),

            "attention_mask":
                encoded["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[index],
                    dtype=torch.long,
                ),
        }


# =============================================================================
# CODEBERT EVALUATION
# =============================================================================

@torch.no_grad()
def predict_codebert(
    model,
    loader,
    device,
):
    model.eval()

    probabilities = []
    labels = []

    for batch in loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch[
            "attention_mask"
        ].to(device)

        y = batch["labels"].cpu().numpy()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1,
        )[:, 1]

        probabilities.extend(
            probs.detach().cpu().numpy()
        )

        labels.extend(y)

    return (
        np.asarray(labels),
        np.asarray(probabilities),
    )


# =============================================================================
# CODEBERT TRAINING
# =============================================================================

def train_codebert(
    train_df,
    val_df,
    test_df,
    seed,
):
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU   :",
            torch.cuda.get_device_name(0),
        )

        print(
            "VRAM  :",
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    model.to(device)

    train_dataset = CodeDataset(
        train_df,
        tokenizer,
    )

    val_dataset = CodeDataset(
        val_df,
        tokenizer,
    )

    test_dataset = CodeDataset(
        test_df,
        tokenizer,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader)
        / GRADIENT_ACCUMULATION_STEPS
    )

    total_steps = (
        optimizer_steps_per_epoch
        * EPOCHS
    )

    warmup_steps = int(
        total_steps
        * WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    seed_model_dir = (
        MODEL_DIR
        / f"seed_{seed}"
        / "best_model"
    )

    seed_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_f1 = -1.0
    best_threshold = 0.50
    patience_counter = 0

    history = []

    optimizer.zero_grad(
        set_to_none=True
    )

    print()
    print("=" * 78)
    print(
        f"CODEBERT TRAINING — SEED {seed}"
    )
    print("=" * 78)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = (
                outputs.loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            loss.backward()

            running_loss += (
                loss.item()
                * GRADIENT_ACCUMULATION_STEPS
            )

            should_step = (
                step
                % GRADIENT_ACCUMULATION_STEPS
                == 0
                or step == len(train_loader)
            )

            if should_step:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

        average_loss = (
            running_loss
            / len(train_loader)
        )

        y_val, val_probabilities = (
            predict_codebert(
                model,
                val_loader,
                device,
            )
        )

        threshold = find_best_threshold(
            y_val,
            val_probabilities,
        )

        val_metrics = evaluate_predictions(
            y_val,
            val_probabilities,
            threshold,
        )

        print(
            f"\nEpoch {epoch} "
            f"average loss: "
            f"{average_loss:.4f}"
        )

        print_metrics(
            f"[Val epoch {epoch}]",
            val_metrics,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss":
                    float(average_loss),
                **val_metrics,
            }
        )

        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_threshold = threshold

            patience_counter = 0

            model.save_pretrained(
                seed_model_dir
            )

            tokenizer.save_pretrained(
                seed_model_dir
            )

            with open(
                seed_model_dir
                / "selection.json",
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "validation_f1":
                            best_val_f1,
                        "threshold":
                            best_threshold,
                    },
                    f,
                    indent=2,
                )

            print(
                f"Best validation F1: "
                f"{best_val_f1:.4f} "
                f"- saved"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

    # ---------------------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------------------

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            seed_model_dir
        )
    )

    best_model.to(device)

    # ---------------------------------------------------------------------
    # Validation confirmation
    # ---------------------------------------------------------------------

    y_val, val_probabilities = (
        predict_codebert(
            best_model,
            val_loader,
            device,
        )
    )

    final_val_metrics = (
        evaluate_predictions(
            y_val,
            val_probabilities,
            best_threshold,
        )
    )

    # ---------------------------------------------------------------------
    # TEST — only after model selection
    # ---------------------------------------------------------------------

    y_test, test_probabilities = (
        predict_codebert(
            best_model,
            test_loader,
            device,
        )
    )

    tuned_test_metrics = (
        evaluate_predictions(
            y_test,
            test_probabilities,
            best_threshold,
        )
    )

    default_test_metrics = (
        evaluate_predictions(
            y_test,
            test_probabilities,
            0.50,
        )
    )

    print()
    print_metrics(
        "CodeBERT [VAL selected]",
        final_val_metrics,
    )

    print_metrics(
        "CodeBERT [TEST tuned]",
        tuned_test_metrics,
    )

    print_metrics(
        "CodeBERT [TEST @ 0.5]",
        default_test_metrics,
    )

    predictions_df = pd.DataFrame(
        {
            "project":
                test_df[
                    PROJECT_COLUMN
                ].values,

            "label":
                y_test,

            "probability":
                test_probabilities,

            "prediction_tuned":
                (
                    test_probabilities
                    >= best_threshold
                ).astype(int),

            "prediction_default":
                (
                    test_probabilities
                    >= 0.50
                ).astype(int),
        }
    )

    predictions_df.to_csv(
        REPORT_DIR
        / f"seed_{seed}_predictions.csv",
        index=False,
    )

    with open(
        REPORT_DIR
        / f"seed_{seed}_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    del best_model
    del optimizer
    del scheduler

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "validation":
            final_val_metrics,

        "test_tuned":
            tuned_test_metrics,

        "test_default":
            default_test_metrics,

        "history":
            history,
    }


# =============================================================================
# SAVE SPLIT INFORMATION
# =============================================================================

def save_split_metadata(
    seed,
    train_df,
    val_df,
    test_df,
):

    metadata = {
        "seed": seed,

        "train": {
            "rows":
                len(train_df),

            "projects":
                sorted(
                    train_df[
                        PROJECT_COLUMN
                    ].unique().tolist()
                ),

            "positive_rate":
                float(
                    train_df[
                        LABEL_COLUMN
                    ].mean()
                ),
        },

        "validation": {
            "rows":
                len(val_df),

            "projects":
                sorted(
                    val_df[
                        PROJECT_COLUMN
                    ].unique().tolist()
                ),

            "positive_rate":
                float(
                    val_df[
                        LABEL_COLUMN
                    ].mean()
                ),
        },

        "test": {
            "rows":
                len(test_df),

            "projects":
                sorted(
                    test_df[
                        PROJECT_COLUMN
                    ].unique().tolist()
                ),

            "positive_rate":
                float(
                    test_df[
                        LABEL_COLUMN
                    ].mean()
                ),
        },
    }

    with open(
        REPORT_DIR
        / f"seed_{seed}_split.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


# =============================================================================
# ONE COMPLETE SEED
# =============================================================================

def run_seed(
    df,
    seed,
):

    print("\n\n")
    print("#" * 78)
    print(
        f"ROBUSTNESS EXPERIMENT — SEED {seed}"
    )
    print("#" * 78)

    set_seed(seed)

    (
        train_df,
        val_df,
        test_df,
    ) = create_project_disjoint_split(
        df,
        seed,
    )

    print()
    describe_split(
        "TRAIN",
        train_df,
    )

    describe_split(
        "VAL",
        val_df,
    )

    describe_split(
        "TEST",
        test_df,
    )

    train_projects = set(
        train_df[PROJECT_COLUMN]
    )

    val_projects = set(
        val_df[PROJECT_COLUMN]
    )

    test_projects = set(
        test_df[PROJECT_COLUMN]
    )

    print()
    print(
        "Project overlap:"
    )

    print(
        "Train ∩ Val :",
        len(
            train_projects
            & val_projects
        ),
    )

    print(
        "Train ∩ Test:",
        len(
            train_projects
            & test_projects
        ),
    )

    print(
        "Val ∩ Test  :",
        len(
            val_projects
            & test_projects
        ),
    )

    assert train_projects.isdisjoint(
        val_projects
    )

    assert train_projects.isdisjoint(
        test_projects
    )

    assert val_projects.isdisjoint(
        test_projects
    )

    save_split_metadata(
        seed,
        train_df,
        val_df,
        test_df,
    )

    # ---------------------------------------------------------------------
    # Balance TRAIN only
    # ---------------------------------------------------------------------

    balanced_train = (
        balance_training_data(
            train_df,
            seed,
        )
    )

    print()
    print(
        "Training after undersampling:"
    )

    print(
        "Rows          :",
        len(balanced_train),
    )

    print(
        "Positive rate :",
        f"{balanced_train[LABEL_COLUMN].mean():.3f}",
    )

    print(
        "Validation/test retain natural distributions."
    )

    # ---------------------------------------------------------------------
    # Baselines
    # ---------------------------------------------------------------------

    print()
    print("=" * 78)
    print("BASELINES")
    print("=" * 78)

    structural = (
        run_logistic_baseline(
            "Structural LR",
            STRUCTURAL_FEATURES,
            balanced_train,
            val_df,
            test_df,
            seed,
        )
    )

    complexity = (
        run_logistic_baseline(
            "Complexity-only LR",
            COMPLEXITY_FEATURES,
            balanced_train,
            val_df,
            test_df,
            seed,
        )
    )

    size_only = (
        run_logistic_baseline(
            "Size-only LR",
            SIZE_FEATURES,
            balanced_train,
            val_df,
            test_df,
            seed,
        )
    )

    tfidf = run_tfidf_baseline(
        balanced_train,
        val_df,
        test_df,
        seed,
    )

    # ---------------------------------------------------------------------
    # CodeBERT
    # ---------------------------------------------------------------------

    codebert = train_codebert(
        balanced_train,
        val_df,
        test_df,
        seed,
    )

    return {
        "seed": seed,

        "train_rows":
            len(train_df),

        "validation_rows":
            len(val_df),

        "test_rows":
            len(test_df),

        "train_projects":
            train_df[
                PROJECT_COLUMN
            ].nunique(),

        "validation_projects":
            val_df[
                PROJECT_COLUMN
            ].nunique(),

        "test_projects":
            test_df[
                PROJECT_COLUMN
            ].nunique(),

        "structural":
            structural["test"],

        "complexity":
            complexity["test"],

        "size":
            size_only["test"],

        "tfidf":
            tfidf["test"],

        "codebert":
            codebert["test_tuned"],

        "codebert_default":
            codebert["test_default"],
    }


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_results(
    all_results,
):

    rows = []

    model_names = [
        "structural",
        "complexity",
        "size",
        "tfidf",
        "codebert",
    ]

    for result in all_results:

        seed = result["seed"]

        for model_name in model_names:

            metrics = result[
                model_name
            ]

            rows.append(
                {
                    "seed": seed,
                    "model":
                        model_name,

                    "f1":
                        metrics["f1"],

                    "precision":
                        metrics[
                            "precision"
                        ],

                    "recall":
                        metrics[
                            "recall"
                        ],

                    "roc_auc":
                        metrics[
                            "roc_auc"
                        ],

                    "pr_auc":
                        metrics[
                            "pr_auc"
                        ],

                    "threshold":
                        metrics[
                            "threshold"
                        ],
                }
            )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        REPORT_DIR
        / "all_seed_results.csv",
        index=False,
    )

    summary = (
        results_df
        .groupby("model")
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),

            precision_mean=(
                "precision",
                "mean",
            ),

            recall_mean=(
                "recall",
                "mean",
            ),

            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),

            roc_auc_std=(
                "roc_auc",
                "std",
            ),

            pr_auc_mean=(
                "pr_auc",
                "mean",
            ),

            pr_auc_std=(
                "pr_auc",
                "std",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        REPORT_DIR
        / "multiseed_summary.csv",
        index=False,
    )

    print("\n\n")
    print("=" * 90)
    print(
        "MULTI-SEED CROSS-PROJECT ROBUSTNESS SUMMARY"
    )
    print("=" * 90)

    for _, row in summary.iterrows():

        print(
            f"{row['model']:<20} "
            f"F1={row['f1_mean']:.4f}"
            f" ± {row['f1_std']:.4f}   "
            f"ROC-AUC="
            f"{row['roc_auc_mean']:.4f}"
            f" ± {row['roc_auc_std']:.4f}   "
            f"PR-AUC="
            f"{row['pr_auc_mean']:.4f}"
            f" ± {row['pr_auc_std']:.4f}"
        )

    print()
    print(
        "Detailed results ->",
        REPORT_DIR
        / "all_seed_results.csv",
    )

    print(
        "Summary ->",
        REPORT_DIR
        / "multiseed_summary.csv",
    )

    return (
        results_df,
        summary,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    start_time = time.time()

    print("=" * 78)
    print(
        "MULTI-SEED CROSS-PROJECT ROBUSTNESS EXPERIMENT"
    )
    print("=" * 78)

    df = pd.read_csv(
        DATA_PATH
    )

    required_columns = {
        PROJECT_COLUMN,
        LABEL_COLUMN,
        TEXT_COLUMN,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.dropna(
        subset=[
            PROJECT_COLUMN,
            LABEL_COLUMN,
            TEXT_COLUMN,
        ]
    ).copy()

    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(int)
    )

    print(
        f"Rows          : {len(df):,}"
    )

    print(
        f"Projects      : "
        f"{df[PROJECT_COLUMN].nunique():,}"
    )

    print(
        f"Positive rate : "
        f"{df[LABEL_COLUMN].mean():.4f}"
    )

    print(
        f"Seeds         : {SEEDS}"
    )

    all_results = []

    for seed in SEEDS:

        result = run_seed(
            df,
            seed,
        )

        all_results.append(
            result
        )

        # Save immediately so an interrupted later run
        # does not destroy completed results.
        with open(
            REPORT_DIR
            / "completed_runs.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                all_results,
                f,
                indent=2,
            )

    summarize_results(
        all_results
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 78)
    print("EXPERIMENT COMPLETE")
    print("=" * 78)

    print(
        f"Elapsed time: "
        f"{elapsed / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()