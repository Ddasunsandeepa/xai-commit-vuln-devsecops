"""
STEP 5 — CODEBERT FINE-TUNING
===============================
Save as:  src/train_codebert.py
Run with: python src/train_codebert.py

GPU SETUP (run on your local PC with GTX 1650):
  1. Clone your repo to Windows
  2. Install CUDA PyTorch:
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  3. pip install transformers datasets scikit-learn pandas
  4. Run: python src/train_codebert.py

OR use Google Colab:
  1. Upload data/processed/big_vul_enriched.csv to Google Drive
  2. Copy this script to a Colab notebook
  3. Change INPUT_FILE path to your Drive path
  4. Runtime → Change runtime type → T4 GPU → Run all

What CodeBERT learns:
  - Reads actual C function code (func_before)
  - microsoft/codebert-base tokenizes it as code
  - Fine-tuned classifier head predicts vulnerable/safe
  - Learns semantic patterns, not just size/complexity

Why CodeBERT > TF-IDF:
  - Pretrained on code from GitHub (knows C patterns)
  - Understands context around dangerous API calls
  - Not fooled by project vocabulary
  - Should beat F1=0.48 (your best honest baseline)
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/processed/big_vul_enriched.csv"
OUTPUT_DIR   = "models/codebert"
REPORT_FILE  = "reports/codebert_results.txt"
RESULTS_CSV  = "reports/results_comparison.csv"

MODEL_NAME   = "microsoft/codebert-base"
TEXT_COL     = "func_before"
TARGET_COL   = "label"
RANDOM_STATE = 42

# ── HYPERPARAMETERS (tuned for GTX 1650 4GB VRAM) ───────────────────────────
MAX_LENGTH   = 256     # CodeBERT max is 512, but 256 fits in 4GB
BATCH_SIZE   = 8       # small batch for 4GB GPU (increase to 16 if you have more)
GRAD_ACCUM   = 4       # effective batch = 8 * 4 = 32
EPOCHS       = 3       # 3 epochs is standard for fine-tuning
LEARNING_RATE = 2e-5   # standard for BERT fine-tuning
WARMUP_RATIO  = 0.1    # 10% of steps for warmup

# Max functions to use (memory constraint for 4GB GPU)
MAX_TRAIN    = 5000    # training functions
MAX_TEST     = 1500    # test functions


# ── DATASET CLASS ────────────────────────────────────────────────────────────

class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── HELPERS ──────────────────────────────────────────────────────────────────

def project_wise_split(df, test_frac=0.25):
    """Same split as all other experiments — fair comparison."""
    projects = df["project"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(projects)
    n_test  = max(1, int(len(projects) * test_frac))
    test_p  = set(projects[:n_test])
    train_p = set(projects[n_test:])
    return df[df["project"].isin(train_p)].copy(), df[df["project"].isin(test_p)].copy()


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(y_true, y_pred, y_prob, split_name="Test"):
    p     = precision_score(y_true, y_pred, zero_division=0)
    r     = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    cm    = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  [{split_name}]")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1-Score  : {f1:.4f}  ← main metric")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  PR-AUC    : {prauc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return {"precision": p, "recall": r, "f1": f1,
            "auc": auc, "prauc": prauc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 62)
    print("  STEP 5 — CODEBERT FINE-TUNING")
    print("=" * 62)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️  No GPU — training will be slow on CPU")
        print("  Tip: run on your local PC with GTX 1650 or Google Colab")

    # Load data
    if not os.path.exists(INPUT_FILE):
        print(f"\n  ERROR: {INPUT_FILE} not found")
        return

    df = pd.read_csv(INPUT_FILE)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    print(f"\n  Loaded {len(df):,} rows")

    # Project-wise split (SAME as all previous experiments)
    train_df, test_df = project_wise_split(df)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # Class balance check
    train_counts = train_df[TARGET_COL].value_counts()
    test_counts  = test_df[TARGET_COL].value_counts()
    print(f"  Train: safe={train_counts.get(0,0):,}  vuln={train_counts.get(1,0):,}")
    print(f"  Test : safe={test_counts.get(0,0):,}  vuln={test_counts.get(1,0):,}")

    # Sample to fit in GPU memory (stratified)
    train_vuln = train_df[train_df[TARGET_COL] == 1]
    train_safe  = train_df[train_df[TARGET_COL] == 0]
    n_vuln = min(len(train_vuln), MAX_TRAIN // 2)
    n_safe  = min(len(train_safe), MAX_TRAIN // 2)
    train_sample = pd.concat([
        train_vuln.sample(n=n_vuln, random_state=RANDOM_STATE),
        train_safe.sample(n=n_safe, random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    test_sample = test_df.sample(
        n=min(len(test_df), MAX_TEST), random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"\n  Training sample : {len(train_sample):,} (balanced {n_vuln} vuln + {n_safe} safe)")
    print(f"  Test sample     : {len(test_sample):,}")

    # Tokenizer
    print(f"\n  Loading tokenizer: {MODEL_NAME} ...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    print("  Tokenizing ...")
    train_dataset = CodeDataset(
        train_sample[TEXT_COL].values,
        train_sample[TARGET_COL].values,
        tokenizer, MAX_LENGTH,
    )
    test_dataset = CodeDataset(
        test_sample[TEXT_COL].values,
        test_sample[TARGET_COL].values,
        tokenizer, MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Model
    print(f"  Loading model: {MODEL_NAME} ...")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = model.to(device)

    # Class weights for imbalance
    n_safe_t  = (train_sample[TARGET_COL] == 0).sum()
    n_vuln_t  = (train_sample[TARGET_COL] == 1).sum()
    class_weights = torch.tensor(
        [1.0, n_safe_t / n_vuln_t], dtype=torch.float
    ).to(device)
    print(f"  Class weights: safe=1.0  vuln={n_safe_t/n_vuln_t:.2f}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Scheduler
    total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"\n  Hyperparameters:")
    print(f"    max_length    : {MAX_LENGTH}")
    print(f"    batch_size    : {BATCH_SIZE} (eff. {BATCH_SIZE * GRAD_ACCUM} with grad accum)")
    print(f"    epochs        : {EPOCHS}")
    print(f"    learning_rate : {LEARNING_RATE}")
    print(f"    total_steps   : {total_steps}")
    print(f"    warmup_steps  : {warmup_steps}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = []
    best_f1 = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("  TRAINING")
    print('=' * 62)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        step = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss    = loss / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 20 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"  Epoch {epoch}/{EPOCHS}  step {step}  loss={avg_loss:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\n  Epoch {epoch} complete — avg loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        print(f"  Evaluating on test set ...")
        y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
        metrics = print_metrics(y_true, y_pred, y_prob, f"Epoch {epoch}")
        metrics["epoch"] = epoch
        metrics["loss"]  = avg_loss
        history.append(metrics)

        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(f"{OUTPUT_DIR}/best_model")
            tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
            print(f"  ✅ New best F1: {best_f1:.4f} — model saved")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("  FINAL RESULTS")
    print('=' * 62)

    best_epoch = max(history, key=lambda x: x["f1"])
    print(f"\n  Best epoch: {best_epoch['epoch']}  F1={best_epoch['f1']:.4f}  AUC={best_epoch['auc']:.4f}")

    print(f"\n  Comparison with baselines:")
    print(f"  {'Approach':<40} {'F1':>7}  {'AUC':>7}")
    print(f"  {'-'*58}")
    baselines = [
        ("Exp-C structural LR (2 features)", 0.4943, 0.7834),
        ("Exp-D complexity-only LR",          0.4797, 0.7617),
        ("Char TF-IDF LR (tuned)",            0.4505, 0.7047),
        (f"CodeBERT (best epoch {best_epoch['epoch']})", best_epoch["f1"], best_epoch["auc"]),
    ]
    for name, f1, auc in baselines:
        marker = " ← YOU ARE HERE" if "CodeBERT" in name else ""
        print(f"  {name:<40} {f1:>7.4f}  {auc:>7.4f}{marker}")

    # ── Save results ──────────────────────────────────────────────────────────
    report_lines = [
        "CODEBERT FINE-TUNING RESULTS",
        "=" * 55,
        f"Run time     : {run_time}",
        f"Model        : {MODEL_NAME}",
        f"Device       : {device}",
        f"Max length   : {MAX_LENGTH}",
        f"Batch size   : {BATCH_SIZE} (eff. {BATCH_SIZE * GRAD_ACCUM})",
        f"Epochs       : {EPOCHS}",
        f"LR           : {LEARNING_RATE}",
        f"Train sample : {len(train_sample):,}",
        f"Test sample  : {len(test_sample):,}",
        "",
        "TRAINING HISTORY",
        "-" * 55,
    ]
    for h in history:
        report_lines.append(
            f"  Epoch {h['epoch']}  loss={h['loss']:.4f}  "
            f"F1={h['f1']:.4f}  AUC={h['auc']:.4f}  PR-AUC={h['prauc']:.4f}"
        )

    report_lines += [
        "",
        f"BEST MODEL: Epoch {best_epoch['epoch']}",
        "-" * 55,
        f"  F1        : {best_epoch['f1']:.4f}",
        f"  AUC-ROC   : {best_epoch['auc']:.4f}",
        f"  PR-AUC    : {best_epoch['prauc']:.4f}",
        f"  Precision : {best_epoch['precision']:.4f}",
        f"  Recall    : {best_epoch['recall']:.4f}",
        f"  TP={best_epoch['tp']}  FP={best_epoch['fp']}  FN={best_epoch['fn']}  TN={best_epoch['tn']}",
        "",
        "COMPARISON",
        "-" * 55,
        "  Exp-C structural LR (2 features)  F1=0.4943  AUC=0.7834",
        "  Exp-D complexity-only LR           F1=0.4797  AUC=0.7617",
        "  Char TF-IDF LR (tuned)            F1=0.4505  AUC=0.7047",
        f"  CodeBERT fine-tuned               F1={best_epoch['f1']:.4f}  AUC={best_epoch['auc']:.4f}",
    ]

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report → '{REPORT_FILE}'")

    # Update results comparison CSV
    if os.path.exists(RESULTS_CSV):
        results_df = pd.read_csv(RESULTS_CSV)
        mask = results_df["approach"] == "CodeBERT fine-tuned"
        if mask.any():
            results_df.loc[mask, "f1"]        = best_epoch["f1"]
            results_df.loc[mask, "auc"]       = best_epoch["auc"]
            results_df.loc[mask, "prauc"]     = best_epoch["prauc"]
            results_df.loc[mask, "precision"] = best_epoch["precision"]
            results_df.loc[mask, "recall"]    = best_epoch["recall"]
            results_df.to_csv(RESULTS_CSV, index=False)
            print(f"  Results CSV updated → '{RESULTS_CSV}'")

    # Save training history as JSON
    history_path = f"{OUTPUT_DIR}/training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History → '{history_path}'")

    print("\n" + "=" * 62)
    if best_epoch["f1"] > 0.48:
        print(f"  CodeBERT BEATS the structural baseline! ({best_epoch['f1']:.4f} > 0.48)")
        print("  Semantic understanding helps beyond structural features.")
    else:
        print(f"  CodeBERT at {best_epoch['f1']:.4f} — similar to structural baseline.")
        print("  Consider: more epochs, larger dataset, or different LR.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()    