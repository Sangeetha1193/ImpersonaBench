"""Compare TF-IDF baseline vs TF-IDF + hand-made features on train/val.

This step changes the representation, not the split and not the evaluation rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from explore import FOCAL_CLASS, FOCAL_CLASS_ID, ID_TO_LABEL, LABEL_TO_ID
from features_v1 import (
    LATER_EXPLAIN_PHRASES,
    SECRECY_PHRASES,
    URGENCY_WORDS,
    VERIFY_BLOCK_PHRASES,
    build_feature_frame,
)

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "data" / "splits" / "train_v1.csv"
VAL_PATH = ROOT / "data" / "splits" / "val_v1.csv"
BASELINE_JSON = ROOT / "metrics" / "baseline_v1.json"
REPORT_PATH = ROOT / "reports" / "feature_run_v1.md"
MODEL_PATH = ROOT / "archive" / "experiments" / "baseline_tfidf_plus_features_v1.joblib"

TOKEN_PATTERN = r"(?u)\b\w[\w@.-]*\b"


def load_xy(path: Path):
    df = pd.read_csv(path)
    x = df["message"].astype(str)
    y = df["label"].map(LABEL_TO_ID).astype(int)
    return x, y, df


def build_matrices(x_train: pd.Series, x_val: pd.Series):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        token_pattern=TOKEN_PATTERN,
    )
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_val = vectorizer.transform(x_val)

    feat_train = build_feature_frame(x_train)
    feat_val = build_feature_frame(x_val)

    x_train_all = hstack([tfidf_train, csr_matrix(feat_train.to_numpy())], format="csr")
    x_val_all = hstack([tfidf_val, csr_matrix(feat_val.to_numpy())], format="csr")
    return vectorizer, feat_train, feat_val, x_train_all, x_val_all


def train_and_score(x_train, y_train, x_val, y_val):
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    clf.fit(x_train, y_train)
    train_pred = clf.predict(x_train)
    val_pred = clf.predict(x_val)
    return clf, {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "val_accuracy": accuracy_score(y_val, val_pred),
        "train_macro_f1": f1_score(y_train, train_pred, average="macro"),
        "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
        "train_imp_f1": f1_score(y_train, train_pred, average=None, labels=[FOCAL_CLASS_ID])[0],
        "val_imp_f1": f1_score(y_val, val_pred, average=None, labels=[FOCAL_CLASS_ID])[0],
    }


def one_shortcut_sentence() -> str:
    return "The biggest shortcut the model still has is that a visible UPI handle like @ybl strongly pushes it toward a scam prediction."


def main():
    x_train, y_train, _ = load_xy(TRAIN_PATH)
    x_val, y_val, _ = load_xy(VAL_PATH)

    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    vectorizer, feat_train, feat_val, x_train_all, x_val_all = build_matrices(x_train, x_val)
    clf, scores = train_and_score(x_train_all, y_train, x_val_all, y_val)

    # tiny ablation: remove one engineered feature at a time, keep tf-idf fixed
    tfidf_train = vectorizer.transform(x_train)
    tfidf_val = vectorizer.transform(x_val)
    hurting_features = []
    for col in feat_train.columns:
        kept_train = feat_train.drop(columns=[col])
        kept_val = feat_val.drop(columns=[col])
        train_mat = hstack([tfidf_train, csr_matrix(kept_train.to_numpy())], format="csr")
        val_mat = hstack([tfidf_val, csr_matrix(kept_val.to_numpy())], format="csr")
        _, ablation_scores = train_and_score(train_mat, y_train, val_mat, y_val)
        if ablation_scores["val_macro_f1"] > scores["val_macro_f1"]:
            hurting_features.append(
                (col, scores["val_macro_f1"], ablation_scores["val_macro_f1"])
            )

    # save a simple artifact for later inference
    import joblib

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": clf,
            "feature_columns": list(feat_train.columns),
            "label_to_id": LABEL_TO_ID,
        },
        MODEL_PATH,
    )

    old_val_macro = baseline["val"]["macro_f1"]
    old_val_imp = baseline["val"]["impersonation_f1"]
    old_train_macro = baseline["train"]["macro_f1"]
    old_gap = old_train_macro - old_val_macro
    new_gap = scores["train_macro_f1"] - scores["val_macro_f1"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature run v1",
        "",
        "## Feature list",
        "",
        "- `has_upi_handle`: direct payee handle like `@ybl`; scam asks often point to a target UPI",
        "- `phone_number_present`: raw 10-digit number; new contact number is an impersonation cue",
        "- `urgency_count`: words like abhi / jaldi / urgent; measures panic pressure",
        "- `secrecy_count`: phrases like mat batana / don't tell; classic social engineering",
        "- `verify_block_count`: don't call / phone dead / new number; blocks normal verification",
        "- `money_amount_present`: ₹ / rs / large numbers; scam asks often contain a concrete demand",
        "- `message_length_chars`: longer story-like messages can look more scammy",
        "- `exclamation_count`: panic tone marker",
        "- `caps_ratio`: shouting / emphasis marker",
        "- `later_explain_count`: baad mein / later I'll explain; delay-accountability cue",
        "",
        "## Word lists used",
        "",
        f"- Urgency words: {', '.join(URGENCY_WORDS)}",
        f"- Secrecy phrases: {', '.join(SECRECY_PHRASES)}",
        f"- Verify-block phrases: {', '.join(VERIFY_BLOCK_PHRASES)}",
        f"- Later-explain phrases: {', '.join(LATER_EXPLAIN_PHRASES)}",
        "",
        "## How features were combined",
        "",
        "- TF-IDF stayed sparse",
        "- Engineered features are dense numeric columns",
        "- Combined with `scipy.sparse.hstack` into one train matrix",
        "- Only TF-IDF was fit on train; hand-made features were static rules on both train and val",
        "",
        "## Old vs new (train / val)",
        "",
        "| Model | Train macro-F1 | Val macro-F1 | Train impersonation F1 | Val impersonation F1 |",
        "|---|---:|---:|---:|---:|",
        f"| Old TF-IDF baseline | {old_train_macro:.4f} | {old_val_macro:.4f} | {baseline['train']['impersonation_f1']:.4f} | {old_val_imp:.4f} |",
        f"| TF-IDF + features | {scores['train_macro_f1']:.4f} | {scores['val_macro_f1']:.4f} | {scores['train_imp_f1']:.4f} | {scores['val_imp_f1']:.4f} |",
        "",
        f"- Old train-val macro-F1 gap: {old_gap:.4f}",
        f"- New train-val macro-F1 gap: {new_gap:.4f}",
        f"- Gap narrowed: {'yes' if new_gap < old_gap else 'no'}",
        "",
        "## Which features hurt?",
        "",
    ]
    if hurting_features:
        for name, full_score, drop_score in hurting_features:
            lines.append(
                f"- `{name}` hurt a bit: full val macro-F1 = {full_score:.4f}, removing it = {drop_score:.4f}"
            )
    else:
        lines.append("- None of the individual hand-made features clearly hurt val macro-F1 in this small ablation.")

    lines += [
        "",
        "## Shortcut still present",
        "",
        one_shortcut_sentence(),
        "",
        "## Verdict",
        "",
        "Small improvements count here. With only 319 training rows, a +0.01 or +0.02 macro-F1 gain is real.",
        "The goal is not to beat transformers yet; it is to learn how to add features carefully and measure whether they help.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("train matrix with features:", x_train_all.shape)
    print("old val macro-F1:", round(old_val_macro, 4))
    print("new val macro-F1:", round(scores["val_macro_f1"], 4))
    print("old val impersonation F1:", round(old_val_imp, 4))
    print("new val impersonation F1:", round(scores["val_imp_f1"], 4))
    print("old gap:", round(old_gap, 4))
    print("new gap:", round(new_gap, 4))
    print("wrote", REPORT_PATH)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
