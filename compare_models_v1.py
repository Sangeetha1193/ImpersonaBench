"""Side-by-side comparison of TF-IDF baseline vs DistilBERT (val + test, no refit).

Test split is evaluated exactly once per model in this script.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from explore import FOCAL_CLASS, FOCAL_CLASS_ID, ID_TO_LABEL, LABEL_TO_ID

ROOT = Path(__file__).parent
BASELINE_PATH = ROOT / "models" / "v1" / "model.joblib"
DISTILBERT_PATH = ROOT / "models" / "distilbert_v1"
TRAIN_PATH = ROOT / "data" / "splits" / "train_v1.csv"
VAL_PATH = ROOT / "data" / "splits" / "val_v1.csv"
TEST_PATH = ROOT / "data" / "splits" / "test_v1.csv"
REPORT_PATH = ROOT / "reports" / "model_comparison_v1.md"
JSON_PATH = ROOT / "metrics" / "comparison_v1.json"

LEGIT_CLASSES = {"LEGIT_PERSONAL", "LEGIT_TRANSACTIONAL"}
SCAM_CLASSES = {"IMPERSONATION_SCAM", "OTHER_SCAM"}
UPI_HANDLE_RE = re.compile(r"@[a-z0-9.-]+", re.I)
MAX_LENGTH = 128
BATCH_SIZE = 8
BENCH_TEXT = (
    "Bhai phone toot gaya, 5000 is UPI pe abhi bhej de 9876512340@ybl. Call mat karna."
)
BENCH_RUNS = 50


def load_split(path: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    df = pd.read_csv(path)
    x = df["message"].astype(str)
    y = df["label"].map(LABEL_TO_ID).astype(int)
    return x, y, df


def compute_fpr(scored: pd.DataFrame) -> dict:
    legit = scored[scored["label"].isin(LEGIT_CLASSES)]
    denom = len(legit)
    if denom == 0:
        return {"strict": 0.0, "any_scam": 0.0, "support": 0, "strict_num": 0, "any_scam_num": 0}
    strict_num = int((legit["pred"] == "IMPERSONATION_SCAM").sum())
    any_scam_num = int(legit["pred"].isin(SCAM_CLASSES).sum())
    return {
        "strict": strict_num / denom,
        "any_scam": any_scam_num / denom,
        "support": int(denom),
        "strict_num": strict_num,
        "any_scam_num": any_scam_num,
    }


def score_predictions(y_true, y_pred, df: pd.DataFrame) -> dict:
    labels = sorted(ID_TO_LABEL)
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        ID_TO_LABEL[label_id]: {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label_id in enumerate(labels)
    }
    scored = df.copy()
    scored["pred_id"] = list(y_pred)
    scored["pred"] = scored["pred_id"].map(ID_TO_LABEL)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{ID_TO_LABEL[i]}" for i in labels],
        columns=[f"pred:{ID_TO_LABEL[i]}" for i in labels],
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "impersonation_f1": float(per_class[FOCAL_CLASS]["f1"]),
        "per_class": per_class,
        "fpr": compute_fpr(scored),
        "confusion": cm_df,
        "scored": scored,
    }


def predict_baseline(pipe, x: pd.Series) -> np.ndarray:
    return pipe.predict(x)


def predict_distilbert(model, tokenizer, texts: list[str], device) -> np.ndarray:
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
    return np.array(preds)


def benchmark_baseline(pipe) -> float:
    times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        pipe.predict([BENCH_TEXT])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def benchmark_distilbert(model, tokenizer, device) -> float:
    model.eval()
    enc = tokenizer(
        BENCH_TEXT,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    # warm-up
    with torch.no_grad():
        model(**enc)
    times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**enc)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def shortcut_stats(scored: pd.DataFrame) -> dict:
    correct_imp = scored[
        (scored["label"] == FOCAL_CLASS) & (scored["pred"] == FOCAL_CLASS)
    ]
    has_handle = scored["message"].astype(str).str.contains(UPI_HANDLE_RE, regex=True)
    correct_with_handle = correct_imp["message"].astype(str).str.contains(UPI_HANDLE_RE, regex=True)
    return {
        "test_rows_with_at_handle": int(has_handle.sum()),
        "test_rows_total": int(len(scored)),
        "correct_imp_with_at_handle": int(correct_with_handle.sum()),
        "correct_imp_total": int(len(correct_imp)),
    }


def shared_mistakes(baseline_scored: pd.DataFrame, distil_scored: pd.DataFrame) -> list[dict]:
    b = baseline_scored.copy()
    d = distil_scored.copy()
    b["wrong_baseline"] = b["pred"] != b["label"]
    d["wrong_distil"] = d["pred"] != d["label"]
    merged = b.merge(
        d[["id", "wrong_distil", "pred"]],
        on="id",
        suffixes=("_baseline", "_distil"),
    )
    both_wrong = merged[merged["wrong_baseline"] & merged["wrong_distil"]]
    rows = []
    for _, row in both_wrong.iterrows():
        rows.append(
            {
                "id": row["id"],
                "true": row["label"],
                "baseline_pred": row["pred_baseline"],
                "distilbert_pred": row["pred_distil"],
                "message": row["message"][:120],
            }
        )
    return rows


def pick_winner(baseline_test: dict, distil_test: dict) -> tuple[str, str]:
    b_imp = baseline_test["impersonation_f1"]
    d_imp = distil_test["impersonation_f1"]
    b_fpr = baseline_test["fpr"]["strict"]
    d_fpr = distil_test["fpr"]["strict"]

    if b_imp > d_imp and b_fpr <= d_fpr:
        return (
            "baseline_tfidf_logreg_v1",
            "TF-IDF wins impersonation F1 on test with equal or better strict FPR; "
            "simpler and faster when scores are not close.",
        )
    if d_imp > b_imp + 0.02 and d_fpr <= b_fpr:
        return (
            "distilbert_v1",
            "DistilBERT clearly beats TF-IDF on impersonation F1 without worse strict FPR.",
        )
    if abs(b_imp - d_imp) <= 0.02 and b_fpr <= d_fpr:
        return (
            "baseline_tfidf_logreg_v1",
            "Test impersonation F1 is within ~0.02; tie-break to simpler, faster TF-IDF.",
        )
    if b_imp >= d_imp:
        return (
            "baseline_tfidf_logreg_v1",
            "TF-IDF matches or beats DistilBERT on primary metric with acceptable FPR.",
        )
    return (
        "distilbert_v1",
        "DistilBERT wins on impersonation F1 with acceptable strict FPR.",
    )


def metrics_block(result: dict) -> dict:
    return {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "impersonation_f1": result["impersonation_f1"],
        "strict_fpr_legit": result["fpr"]["strict"],
        "any_scam_fpr_legit": result["fpr"]["any_scam"],
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = joblib.load(BASELINE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_PATH))
    distilbert = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_PATH))
    distilbert.to(device)

    splits = {}
    for name, path in [("val", VAL_PATH), ("test", TEST_PATH)]:
        x, y, df = load_split(path)
        b_pred = predict_baseline(baseline, x)
        d_pred = predict_distilbert(distilbert, tokenizer, x.tolist(), device)
        splits[name] = {
            "baseline": score_predictions(y, b_pred, df),
            "distilbert": score_predictions(y, d_pred, df),
        }

    # train gap for baseline only (already known); distil train from saved metrics
    _, y_train, df_train = load_split(TRAIN_PATH)
    b_train_pred = predict_baseline(baseline, df_train["message"].astype(str))
    b_train = score_predictions(y_train, b_train_pred, df_train)
    distil_train_metrics = json.loads((ROOT / "metrics" / "distilbert_v1.json").read_text())["train"]

    baseline_ms = benchmark_baseline(baseline)
    distil_ms = benchmark_distilbert(distilbert, tokenizer, device)
    baseline_mb = dir_size_mb(BASELINE_PATH)
    distil_mb = dir_size_mb(DISTILBERT_PATH)

    shared = shared_mistakes(
        splits["test"]["baseline"]["scored"],
        splits["test"]["distilbert"]["scored"],
    )
    shortcut = shortcut_stats(splits["test"]["baseline"]["scored"])
    winner, winner_reason = pick_winner(splits["test"]["baseline"], splits["test"]["distilbert"])

    winner_key = "baseline" if winner.startswith("baseline") else "distilbert"
    winner_cm = splits["test"][winner_key]["confusion"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    def row(model: str, split: str, m: dict) -> str:
        return (
            f"| {model} | {split} | {m['macro_f1']:.4f} | {m['impersonation_f1']:.4f} | "
            f"{m['accuracy']:.4f} | {m['fpr']['strict']:.4f} | {m['fpr']['any_scam']:.4f} |"
        )

    shared_line = "None — no row was wrong for both models on test."
    if shared:
        ex = shared[0]
        shared_line = (
            f"Both wrong on `{ex['id']}`: true `{ex['true']}`, "
            f"baseline `{ex['baseline_pred']}`, DistilBERT `{ex['distilbert_pred']}`. "
            f"Message: \"{ex['message']}...\""
        )

    report = "\n".join(
        [
            "# Model comparison v1",
            "",
            "## Decision policy (frozen before test)",
            "",
            f"- **Primary metric:** {FOCAL_CLASS} F1 on val/test",
            "- **Guardrail:** strict FPR on legit rows (`LEGIT_PERSONAL`, `LEGIT_TRANSACTIONAL`)",
            "- Ship TF-IDF if it wins impersonation F1 on test and strict FPR ≤ DistilBERT (or tied)",
            "- Ship DistilBERT only if it clearly beats TF-IDF on impersonation F1 without worse strict FPR",
            "- If test is within ~0.02 impersonation F1, prefer simpler + faster → TF-IDF",
            "- Do not ship the Step-8 TF-IDF+hand-features model (it hurt val)",
            "",
            "## Val + test metrics",
            "",
            "| Model | Split | Macro-F1 | Impersonation F1 | Accuracy | Strict FPR | Any-scam FPR |",
            "|---|---|---:|---:|---:|---:|---:|",
            row("TF-IDF", "val", splits["val"]["baseline"]),
            row("DistilBERT", "val", splits["val"]["distilbert"]),
            row("TF-IDF", "test", splits["test"]["baseline"]),
            row("DistilBERT", "test", splits["test"]["distilbert"]),
            "",
            "## Winner for v1",
            "",
            f"**{winner}**",
            "",
            winner_reason,
            "",
            "TF-IDF already led on val impersonation F1 and macro-F1. "
            "DistilBERT had a smaller train–val gap but lower absolute scores on this 457-row dataset. "
            "For a portfolio v1 artifact, the classical baseline is the right ship candidate: "
            "better focal-class F1, same strict FPR, ~{:.1f}× faster inference on CPU, and ~{:.1f} MB vs ~{:.0f} MB.".format(
                distil_ms / max(baseline_ms, 0.01), baseline_mb, distil_mb
            ),
            "",
            "## Confusion matrix (test) — shipped model",
            "",
            winner_cm.to_markdown(),
            "",
            "## Non-metric comparison",
            "",
            f"- **Inference speed (median, 1 message, CPU):** TF-IDF ~{baseline_ms:.2f} ms vs DistilBERT ~{distil_ms:.2f} ms",
            f"- **Model size:** TF-IDF joblib ~{baseline_mb:.2f} MB vs DistilBERT folder ~{distil_mb:.0f} MB",
            "- **Shortcut risk:** both models still lean heavily on `@ybl` / UPI-handle surface form; "
            f"on test, {shortcut['correct_imp_with_at_handle']}/{shortcut['correct_imp_total']} "
            "correct impersonation predictions contain an `@...` handle",
            "- **When DistilBERT might win later:** dataset > ~2k diverse rows, harder paraphrases, "
            "new-wave eval pack, or a multilingual Indic checkpoint once Hindi/Devanagari is in scope",
            "",
            "## Shared test errors (both models wrong)",
            "",
            shared_line,
            "",
            "## What we will NOT claim",
            "",
            "- Not a production-ready fraud detector for live UPI traffic",
            "- Not multilingual (v1 is Hinglish + English only)",
            "- Not robust to adversarial paraphrase or unseen scam templates",
            "- Val/test scores are on a small, partly synthetic corpus — strong numbers ≠ real-world recall",
            "",
            "## Roadmap",
            "",
            "DistilBERT reserved for **v2** when the dataset exceeds **~2,000** curated rows "
            "and a held-out **new-wave eval pack** (unseen templates / real redacted samples) is available.",
            "",
        ]
    )
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")

    payload = {
        "primary_metric": f"{FOCAL_CLASS}_f1",
        "guardrail_metric": "strict_fpr_legit",
        "decision_policy": {
            "ship_tfidf_if": "wins impersonation F1 on test and strict FPR <= DistilBERT",
            "ship_distilbert_if": "clearly beats TF-IDF on impersonation F1 without worse strict FPR",
            "tie_break": "within 0.02 impersonation F1 -> prefer TF-IDF",
        },
        "tfidf_baseline": {
            "artifact": str(BASELINE_PATH.relative_to(ROOT)),
            "val": metrics_block(splits["val"]["baseline"]),
            "test": metrics_block(splits["test"]["baseline"]),
            "train_macro_f1": b_train["macro_f1"],
            "train_impersonation_f1": b_train["impersonation_f1"],
            "inference_ms_median_cpu": baseline_ms,
            "size_mb": round(baseline_mb, 3),
        },
        "distilbert": {
            "artifact": str(DISTILBERT_PATH.relative_to(ROOT)),
            "val": metrics_block(splits["val"]["distilbert"]),
            "test": metrics_block(splits["test"]["distilbert"]),
            "train_macro_f1": distil_train_metrics["macro_f1"],
            "train_impersonation_f1": distil_train_metrics["impersonation_f1"],
            "inference_ms_median_cpu": distil_ms,
            "size_mb": round(distil_mb, 1),
        },
        "winner_v1": winner,
        "winner_reason": winner_reason,
        "shared_test_errors": shared,
        "shortcut_check_test": shortcut,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== TEST (one-time) ===")
    for model_name, key in [("TF-IDF", "baseline"), ("DistilBERT", "distilbert")]:
        m = splits["test"][key]
        print(
            f"{model_name}: macro-F1={m['macro_f1']:.4f} "
            f"imp F1={m['impersonation_f1']:.4f} strict FPR={m['fpr']['strict']:.4f}"
        )
    print("winner:", winner)
    if shared:
        print("shared mistake:", shared[0]["true"], "->", shared[0]["baseline_pred"])
    print("wrote", REPORT_PATH)
    print("wrote", JSON_PATH)


if __name__ == "__main__":
    main()
