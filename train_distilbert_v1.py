"""Fine-tune DistilBERT for 4-class ImpersonaBench (train + val only).

Test is never used for early stopping or model selection.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from explore import FOCAL_CLASS_ID, ID_TO_LABEL, LABEL_TO_ID

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "data" / "splits" / "train_v1.csv"
VAL_PATH = ROOT / "data" / "splits" / "val_v1.csv"
OUT_DIR = ROOT / "models" / "distilbert_v1"
REPORT_PATH = ROOT / "reports" / "distilbert_train_v1.md"
METRICS_PATH = ROOT / "metrics" / "distilbert_v1.json"

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 128
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 8
SEED = 42
LEGIT_IDS = {LABEL_TO_ID["LEGIT_PERSONAL"], LABEL_TO_ID["LEGIT_TRANSACTIONAL"]}
SCAM_IDS = {LABEL_TO_ID["IMPERSONATION_SCAM"], LABEL_TO_ID["OTHER_SCAM"]}


class MessageDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_split(path: Path):
    df = pd.read_csv(path)
    texts = df["message"].astype(str).tolist()
    labels = df["label"].map(LABEL_TO_ID).astype(int).tolist()
    return texts, labels, df


def compute_fpr(y_true, y_pred):
    legit_mask = [yt in LEGIT_IDS for yt in y_true]
    denom = sum(legit_mask)
    if denom == 0:
        return 0.0, 0.0
    strict = sum(
        1
        for yt, yp, ok in zip(y_true, y_pred, legit_mask)
        if ok and yp == FOCAL_CLASS_ID
    )
    any_scam = sum(
        1 for yt, yp, ok in zip(y_true, y_pred, legit_mask) if ok and yp in SCAM_IDS
    )
    return strict / denom, any_scam / denom


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    imp_f1 = f1_score(labels, preds, labels=[FOCAL_CLASS_ID], average="macro", zero_division=0)
    # labels=[FOCAL_CLASS_ID] with average=macro is just that one class F1
    imp_f1 = float(
        f1_score(labels, preds, labels=[FOCAL_CLASS_ID], average=None, zero_division=0)[0]
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "impersonation_f1": imp_f1,
    }


def score_split(model, tokenizer, texts, labels, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
    y_true = labels
    y_pred = preds
    strict_fpr, any_fpr = compute_fpr(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "impersonation_f1": float(
            f1_score(y_true, y_pred, labels=[FOCAL_CLASS_ID], average=None, zero_division=0)[0]
        ),
        "strict_fpr_legit": float(strict_fpr),
        "any_scam_fpr_legit": float(any_fpr),
    }


def predict_one(model, tokenizer, text: str, device) -> str:
    model.eval()
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        pred_id = int(model(**enc).logits.argmax(dim=-1).item())
    return ID_TO_LABEL[pred_id]


def main():
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("checkpoint:", CHECKPOINT)

    train_texts, train_labels, _ = load_split(TRAIN_PATH)
    val_texts, val_labels, _ = load_split(VAL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=4,
        id2label={i: name for i, name in ID_TO_LABEL.items()},
        label2id=LABEL_TO_ID,
    )

    train_enc = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    val_enc = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    train_ds = MessageDataset(train_enc, train_labels)
    val_ds = MessageDataset(val_enc, val_labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "runs"),
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        logging_steps=20,
        save_total_limit=2,
        report_to=[],
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    # reload best for scoring
    best = AutoModelForSequenceClassification.from_pretrained(str(OUT_DIR))
    best.to(device)

    train_metrics = score_split(best, tokenizer, train_texts, train_labels, device)
    val_metrics = score_split(best, tokenizer, val_texts, val_labels, device)

    sample = "Bhai phone toot gaya, 5000 is UPI pe abhi bhej de 9876512340@ybl. Call mat karna."
    sample_pred = predict_one(best, tokenizer, sample, device)

    gap_macro = train_metrics["macro_f1"] - val_metrics["macro_f1"]
    gap_imp = train_metrics["impersonation_f1"] - val_metrics["impersonation_f1"]

    baseline = json.loads((ROOT / "metrics" / "baseline_v1.json").read_text(encoding="utf-8"))
    fpr_vs_baseline = (
        "better"
        if val_metrics["strict_fpr_legit"] < baseline["val"]["strict_fpr_legit"]
        else (
            "same"
            if abs(val_metrics["strict_fpr_legit"] - baseline["val"]["strict_fpr_legit"]) < 1e-9
            else "worse"
        )
    )

    # best epoch from trainer state if available
    best_metric = trainer.state.best_metric
    best_epoch = None
    if trainer.state.log_history:
        for row in trainer.state.log_history:
            if row.get("eval_macro_f1") == best_metric or (
                "eval_macro_f1" in row and best_metric is not None
                and abs(row["eval_macro_f1"] - best_metric) < 1e-9
            ):
                best_epoch = row.get("epoch")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(
        [
            "# DistilBERT train v1",
            "",
            "## Setup",
            f"- checkpoint: `{CHECKPOINT}`",
            f"- max_length: {MAX_LENGTH}",
            f"- epochs (max): {EPOCHS}",
            f"- learning_rate: {LR}",
            f"- batch_size: {BATCH_SIZE}",
            f"- seed: {SEED}",
            f"- device: {device}",
            "- early stopping: patience=2 on val `macro_f1`",
            "- selection: best val checkpoint (`load_best_model_at_end=True`), not last epoch",
            "- test set: not used for training or early stopping",
            "",
            "## Why these choices",
            "- DistilBERT is smaller/faster than BERT; enough for a portfolio baseline comparison.",
            "- Messages are short Hinglish/English SMS-like text → max_length=128 is enough.",
            "- Fine-tuning uses a small LR (~2e-5) so we adapt the head without destroying pretrained weights.",
            "- With only 319 train rows, more than ~4 epochs usually memorizes templates.",
            "",
            "## Results (same metric definitions as baseline eval)",
            "",
            "### Train",
            f"- accuracy: {train_metrics['accuracy']:.4f}",
            f"- macro-F1: {train_metrics['macro_f1']:.4f}",
            f"- IMPERSONATION_SCAM F1: {train_metrics['impersonation_f1']:.4f}",
            f"- strict FPR on legit: {train_metrics['strict_fpr_legit']:.4f}",
            "",
            "### Val",
            f"- accuracy: {val_metrics['accuracy']:.4f}",
            f"- macro-F1: {val_metrics['macro_f1']:.4f}",
            f"- IMPERSONATION_SCAM F1: {val_metrics['impersonation_f1']:.4f}",
            f"- strict FPR on legit: {val_metrics['strict_fpr_legit']:.4f}",
            f"- any-scam FPR on legit: {val_metrics['any_scam_fpr_legit']:.4f}",
            "",
            "### Train vs val gap",
            f"- macro-F1 gap (train - val): {gap_macro:.4f}",
            f"- impersonation F1 gap (train - val): {gap_imp:.4f}",
            "",
            "### vs TF-IDF baseline (val)",
            f"- baseline val macro-F1: {baseline['val']['macro_f1']:.4f}",
            f"- DistilBERT val macro-F1: {val_metrics['macro_f1']:.4f}",
            f"- baseline val impersonation F1: {baseline['val']['impersonation_f1']:.4f}",
            f"- DistilBERT val impersonation F1: {val_metrics['impersonation_f1']:.4f}",
            f"- baseline val strict FPR: {baseline['val']['strict_fpr_legit']:.4f}",
            f"- DistilBERT val strict FPR: {val_metrics['strict_fpr_legit']:.4f} ({fpr_vs_baseline})",
            "",
            f"## Best epoch / metric",
            f"- best_metric (val macro_f1 tracked by Trainer): {best_metric}",
            f"- best_epoch (approx from logs): {best_epoch}",
            "",
            "## Sanity prediction",
            f"- sample: `{sample}`",
            f"- predicted: `{sample_pred}`",
            "",
            "## Honest notes",
            "- Test was not used to pick this model.",
            "- If train is near-perfect and val is only slightly better/worse than TF-IDF, DistilBERT may not be worth the complexity yet.",
            "- Do not declare DistilBERT the winner until Step 10 (side-by-side comparison).",
            "",
            f"## Saved model",
            f"- `{OUT_DIR}`",
        ]
    )
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "model": "distilbert_v1",
                "checkpoint": CHECKPOINT,
                "max_length": MAX_LENGTH,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "best_metric_name": "macro_f1",
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "train": train_metrics,
                "val": val_metrics,
                "gaps": {
                    "macro_f1_train_minus_val": gap_macro,
                    "impersonation_f1_train_minus_val": gap_imp,
                },
                "val_strict_fpr_vs_tfidf_baseline": fpr_vs_baseline,
                "sample_prediction": {"text": sample, "pred": sample_pred},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("val impersonation F1:", round(val_metrics["impersonation_f1"], 4))
    print("val macro-F1:", round(val_metrics["macro_f1"], 4))
    print("train-val macro gap:", round(gap_macro, 4))
    print("train-val imp gap:", round(gap_imp, 4))
    print("val strict FPR vs baseline:", fpr_vs_baseline)
    print("sample pred:", sample_pred)
    print("wrote", REPORT_PATH)
    print("wrote", METRICS_PATH)
    print("saved", OUT_DIR)


if __name__ == "__main__":
    main()
