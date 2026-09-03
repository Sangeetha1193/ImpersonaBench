"""First baseline: TF-IDF + multinomial logistic regression.

X = message only. y = LABEL_TO_ID from explore.py.
Fits on train, scores val, never loads test.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from explore import ID_TO_LABEL, LABEL_TO_ID

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "data" / "splits" / "train_v1.csv"
VAL_PATH = ROOT / "data" / "splits" / "val_v1.csv"
MODEL_PATH = ROOT / "models" / "v1" / "model.joblib"
REPORT_PATH = ROOT / "reports" / "baseline_run_v1.md"

# Keep digits and @handles (e.g. 98765@ybl) as tokens. Default \w would split on @.
TOKEN_PATTERN = r"(?u)\b\w[\w@.-]*\b"


def load_xy(path):
    df = pd.read_csv(path)
    unknown = set(df["label"]) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"Unknown labels in {path}: {unknown}")
    X = df["message"].astype(str)
    y = df["label"].map(LABEL_TO_ID).astype(int)
    return X, y, df


def main():
    X_train, y_train, _ = load_xy(TRAIN_PATH)
    X_val, y_val, val_df = load_xy(VAL_PATH)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        token_pattern=TOKEN_PATTERN,
    )
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("logreg", clf),
    ])
    pipe.fit(X_train, y_train)

    train_matrix = pipe.named_steps["tfidf"].transform(X_train)
    print("train TF-IDF shape:", train_matrix.shape)

    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    print("val accuracy:", round(acc, 4))
    print("val macro-F1:", round(macro_f1, 4))
    print()
    print(classification_report(
        y_val,
        y_pred,
        target_names=[ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)],
        digits=3,
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print("saved", MODEL_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join([
            "# Baseline run v1",
            "",
            "## What this is",
            "First supervised model. Message text only. Test set was not loaded.",
            "",
            "## Data",
            f"- train: `{TRAIN_PATH.name}` ({len(X_train)} rows)",
            f"- val: `{VAL_PATH.name}` ({len(X_val)} rows)",
            "- test: not used",
            "- X = `message`",
            "- y = `LABEL_TO_ID` from `explore.py` (0 impersonation, 1 legit personal, 2 other scam, 3 transactional)",
            "",
            "## Vectorizer (TfidfVectorizer)",
            "- lowercase=True (Hinglish casing is noisy; same setting must be used at inference)",
            "- ngram_range=(1, 2) so phrases like `mat call` / `bhej de` can fire",
            "- min_df=2 (drop one-off typos)",
            "- max_features=10000",
            "- token_pattern keeps digits and @handles (`98765@ybl` stays one token)",
            f"- train matrix shape: {train_matrix.shape}  (rows = train size, cols = vocab)",
            "",
            "## Classifier",
            "- LogisticRegression, multinomial (sklearn default for >2 classes with lbfgs)",
            "- class_weight=balanced (LEGIT_TRANSACTIONAL is the small class)",
            "- max_iter=2000",
            "- random_state=42",
            "",
            "## Val sanity check (not tuned)",
            f"- val accuracy: {acc:.4f}",
            f"- val macro-F1: {macro_f1:.4f}",
            f"- val predictions: {len(y_pred)} rows",
            "",
            "A very high score would be suspicious (leakage or too-easy templates).",
            "This run is a reproducible floor, not a production model.",
            "",
            "## Saved artifact",
            f"- `{MODEL_PATH.relative_to(ROOT)}`  (Pipeline: vectorizer + classifier together)",
            "",
            "## Not done",
            "- no hyperparameter search",
            "- test_v1.csv not loaded",
            "- split CSVs not modified",
        ]) + "\n",
        encoding="utf-8",
    )
    print("wrote", REPORT_PATH)

    # keep a small prediction table for later error analysis, not for scoring test
    out = val_df[["id", "message", "label"]].copy()
    out["pred_id"] = y_pred
    out["pred"] = out["pred_id"].map(ID_TO_LABEL)
    pred_path = ROOT / "reports" / "baseline_val_preds_v1.csv"
    out.to_csv(pred_path, index=False)
    print("wrote", pred_path)


if __name__ == "__main__":
    main()
