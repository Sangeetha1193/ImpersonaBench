"""Evaluate the saved baseline on train / val / test without refitting.

This step is a camera, not a workshop:
- loads the already-fitted pipeline
- computes metrics separately on train, val, and test
- writes a markdown report and JSON metrics for later CI comparison
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from explore import FOCAL_CLASS, ID_TO_LABEL, LABEL_TO_ID, PRIMARY_METRIC

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "v1" / "model.joblib"
TRAIN_PATH = ROOT / "data" / "splits" / "train_v1.csv"
VAL_PATH = ROOT / "data" / "splits" / "val_v1.csv"
TEST_PATH = ROOT / "data" / "splits" / "test_v1.csv"
REPORT_PATH = ROOT / "reports" / "eval_baseline_v1.md"
JSON_PATH = ROOT / "metrics" / "baseline_v1.json"

LEGIT_CLASSES = {"LEGIT_PERSONAL", "LEGIT_TRANSACTIONAL"}
SCAM_CLASSES = {"IMPERSONATION_SCAM", "OTHER_SCAM"}
UPI_HANDLE_RE = re.compile(r"@[a-z0-9.-]+", re.I)


def load_split(path: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    df = pd.read_csv(path)
    x = df["message"].astype(str)
    y = df["label"].map(LABEL_TO_ID).astype(int)
    return x, y, df


def confusion_table(y_true, y_pred) -> pd.DataFrame:
    ids = sorted(ID_TO_LABEL)
    labels = [ID_TO_LABEL[i] for i in ids]
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    return pd.DataFrame(cm, index=[f"true:{x}" for x in labels], columns=[f"pred:{x}" for x in labels])


def add_predictions(df: pd.DataFrame, y_pred) -> pd.DataFrame:
    out = df.copy()
    out["pred_id"] = list(y_pred)
    out["pred"] = out["pred_id"].map(ID_TO_LABEL)
    return out


def compute_fpr_tables(scored: pd.DataFrame) -> dict:
    legit = scored[scored["label"].isin(LEGIT_CLASSES)].copy()
    denom = len(legit)
    if denom == 0:
        return {"strict": 0.0, "any_scam": 0.0, "support": 0}

    strict_num = ((legit["pred"] == "IMPERSONATION_SCAM")).sum()
    any_scam_num = legit["pred"].isin(SCAM_CLASSES).sum()
    return {
        "strict": strict_num / denom,
        "any_scam": any_scam_num / denom,
        "support": denom,
        "strict_num": int(strict_num),
        "any_scam_num": int(any_scam_num),
    }


def handle_shortcut_check(scored: pd.DataFrame) -> dict:
    legit = scored[scored["label"].isin(LEGIT_CLASSES)]
    val_mistakes = scored[scored["pred"] != scored["label"]]
    correct_imp = scored[
        (scored["label"] == "IMPERSONATION_SCAM") & (scored["pred"] == "IMPERSONATION_SCAM")
    ]

    def rate(frame: pd.DataFrame) -> dict:
        n = len(frame)
        has_handle = frame["message"].astype(str).str.contains(UPI_HANDLE_RE, regex=True)
        return {
            "rows": int(n),
            "with_at_handle": int(has_handle.sum()),
            "rate": float(has_handle.mean()) if n else 0.0,
        }

    return {
        "val_mistakes": rate(val_mistakes),
        "correct_impersonation": rate(correct_imp),
        "legit_rows": rate(legit),
    }


def score_split(name: str, pipe, path: Path) -> dict:
    x, y, df = load_split(path)
    y_pred = pipe.predict(x)
    scored = add_predictions(df, y_pred)

    labels = sorted(ID_TO_LABEL)
    pr, rc, f1, support = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for idx, label_id in enumerate(labels):
        per_class[ID_TO_LABEL[label_id]] = {
            "precision": float(pr[idx]),
            "recall": float(rc[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "name": name,
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "impersonation_f1": float(per_class[FOCAL_CLASS]["f1"]),
        "per_class": per_class,
        "fpr": compute_fpr_tables(scored),
        "confusion": confusion_table(y, y_pred),
        "scored": scored,
    }


def markdown_per_class(result: dict) -> list[str]:
    lines = [
        f"### {result['name'].title()}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in result["per_class"].items():
        lines.append(
            f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |"
        )
    lines += [
        "",
        f"- Accuracy: {result['accuracy']:.4f}",
        f"- Macro-F1: {result['macro_f1']:.4f}",
        f"- {FOCAL_CLASS} F1: {result['impersonation_f1']:.4f}",
        f"- Strict FPR on legit rows: {result['fpr']['strict']:.4f} "
        f"({result['fpr']['strict_num']}/{result['fpr']['support']})",
        f"- Any-scam FPR on legit rows: {result['fpr']['any_scam']:.4f} "
        f"({result['fpr']['any_scam_num']}/{result['fpr']['support']})",
        "",
    ]
    return lines


def main() -> None:
    pipe = joblib.load(MODEL_PATH)

    train = score_split("train", pipe, TRAIN_PATH)
    val = score_split("val", pipe, VAL_PATH)
    test = score_split("test", pipe, TEST_PATH)

    shortcut = handle_shortcut_check(val["scored"])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Baseline evaluation v1",
        "",
        "## Metric definitions (written before reading the scores)",
        "",
        f"- Primary metric: **F1 of {FOCAL_CLASS}**",
        f"- Also report: **{PRIMARY_METRIC}** and **accuracy**",
        "- Strict FPR on legit rows = `pred == IMPERSONATION_SCAM` among true `{LEGIT_PERSONAL, LEGIT_TRANSACTIONAL}`",
        "- Any-scam FPR on legit rows = `pred in {IMPERSONATION_SCAM, OTHER_SCAM}` among the same legit rows",
        "- Train, val, and test are reported separately. They must never be mixed.",
        "- The support for impersonation is small (~19-20 rows in val/test), so one extra error moves F1 noticeably.",
        "",
        "## Model being evaluated",
        "",
        "- Artifact: `models/v1/model.joblib`",
        "- This script does not refit the model and does not tune any hyperparameters.",
        "- Test split is used here for evaluation only, after the baseline was already frozen.",
        "",
    ]

    report_lines += markdown_per_class(val)
    report_lines += markdown_per_class(test)

    report_lines += [
        "## Train vs val sanity check",
        "",
        f"- Train macro-F1: {train['macro_f1']:.4f}",
        f"- Val macro-F1: {val['macro_f1']:.4f}",
        f"- Train {FOCAL_CLASS} F1: {train['impersonation_f1']:.4f}",
        f"- Val {FOCAL_CLASS} F1: {val['impersonation_f1']:.4f}",
        "- If train is near 1.0 and val is lower, the model is fitting templates or dataset artifacts. Note it; do not 'fix' it in the evaluation step.",
        "",
        "## Confusion matrix (val)",
        "",
        val["confusion"].to_markdown(),
        "",
        "## Confusion matrix (test)",
        "",
        test["confusion"].to_markdown(),
        "",
        "## Shortcut check (@handle / UPI-handle signal)",
        "",
        f"- Among val mistakes: {shortcut['val_mistakes']['with_at_handle']}/{shortcut['val_mistakes']['rows']} "
        f"= {shortcut['val_mistakes']['rate']:.3f} contain an `@...` handle",
        f"- Among correct val IMPERSONATION_SCAM predictions: {shortcut['correct_impersonation']['with_at_handle']}/"
        f"{shortcut['correct_impersonation']['rows']} = {shortcut['correct_impersonation']['rate']:.3f} contain an `@...` handle",
        f"- Among all legit val rows: {shortcut['legit_rows']['with_at_handle']}/{shortcut['legit_rows']['rows']} "
        f"= {shortcut['legit_rows']['rate']:.3f} contain an `@...` handle",
        "- This is a shortcut check, not new feature engineering. If the scam classes are mostly 'has @ybl', say so honestly.",
        "",
        "## Plain-English takeaways",
        "",
        "1. The confusion matrix cell `true:LEGIT_TRANSACTIONAL → pred:OTHER_SCAM` means the model saw a real alert but treated it like a scam. In plain English: it got too suspicious.",
        "2. A strong val score can still be partly template-driven. Train vs val tells us whether the model is mostly memorizing patterns from synthetic rows.",
        "3. What we must not do next: tune hyperparameters on the test set. Test is the final exam, not a practice worksheet.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    headline = {
        "model": "impersonabench-tfidf-logreg-v1",
        "primary_metric": f"{FOCAL_CLASS}_f1",
        "val": {
            "accuracy": val["accuracy"],
            "macro_f1": val["macro_f1"],
            "impersonation_f1": val["impersonation_f1"],
            "strict_fpr_legit": val["fpr"]["strict"],
            "any_scam_fpr_legit": val["fpr"]["any_scam"],
        },
        "test": {
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "impersonation_f1": test["impersonation_f1"],
            "strict_fpr_legit": test["fpr"]["strict"],
            "any_scam_fpr_legit": test["fpr"]["any_scam"],
        },
        "train": {
            "accuracy": train["accuracy"],
            "macro_f1": train["macro_f1"],
            "impersonation_f1": train["impersonation_f1"],
        },
    }
    JSON_PATH.write_text(json.dumps(headline, indent=2), encoding="utf-8")

    print(f"train macro-F1: {train['macro_f1']:.4f}")
    print(f"val macro-F1:   {val['macro_f1']:.4f}")
    print(f"test macro-F1:  {test['macro_f1']:.4f}")
    print(f"val {FOCAL_CLASS} F1:  {val['impersonation_f1']:.4f}")
    print(f"test {FOCAL_CLASS} F1: {test['impersonation_f1']:.4f}")
    print(f"val strict FPR legit:  {val['fpr']['strict']:.4f}")
    print(f"test strict FPR legit: {test['fpr']['strict']:.4f}")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
