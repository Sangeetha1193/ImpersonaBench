"""Package the v1 champion into models/v1/ — weights, schema, metadata, model card.

No retraining. Copies the fitted sklearn Pipeline and writes documentation artifacts.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from explore import ID_TO_LABEL, LABEL_TO_ID

ROOT = Path(__file__).parent
BUNDLE_DIR = ROOT / "models" / "v1"
MODEL_PATH = BUNDLE_DIR / "model.joblib"
LABEL_SCHEMA_PATH = BUNDLE_DIR / "label_schema.json"
METADATA_PATH = BUNDLE_DIR / "metadata.json"
MODEL_CARD_PATH = BUNDLE_DIR / "MODEL_CARD.md"
COMPARISON_PATH = ROOT / "metrics" / "comparison_v1.json"
TRAIN_PATH = ROOT / "data" / "splits" / "train_v1.csv"
TEST_PATH = ROOT / "data" / "splits" / "test_v1.csv"

# Fixed rows for reload verification (correct impersonation predictions on test).
VERIFY_IDS = ["ib_v1_000005", "ib_v1_000031"]

TOKEN_PATTERN = r"(?u)\b\w[\w@.-]*\b"


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_label_schema() -> dict:
    return {
        "version": "v1",
        "description": "4-class label schema for ImpersonaBench message classification",
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "focal_class": "IMPERSONATION_SCAM",
        "focal_class_id": LABEL_TO_ID["IMPERSONATION_SCAM"],
        "classes": [
            {
                "id": LABEL_TO_ID[name],
                "name": name,
                "description": desc,
            }
            for name, desc in [
                ("IMPERSONATION_SCAM", "Trusted-contact impersonation asking for UPI/payment"),
                ("LEGIT_PERSONAL", "Benign personal/family/colleague chat"),
                ("OTHER_SCAM", "Non-impersonation scams (KYC, OTP, lottery, brand fraud)"),
                ("LEGIT_TRANSACTIONAL", "Real bank, telco, or merchant alerts"),
            ]
        ],
    }


def build_metadata(comparison: dict, verify: list[dict]) -> dict:
    m = comparison["tfidf_baseline"]
    return {
        "model_name": "impersonabench-tfidf-logreg-v1",
        "model_type": "sklearn_tfidf_logistic_regression",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "sklearn_version": sklearn.__version__,
        "python_dependency_note": "Load with the same sklearn major.minor as training when possible.",
        "training_rows": 319,
        "train_file": "data/splits/train_v1.csv",
        "val_file": "data/splits/val_v1.csv",
        "test_file": "data/splits/test_v1.csv",
        "split_seed": 42,
        "split_report": "reports/split_v1.md",
        "artifact": "models/v1/model.joblib",
        "artifact_format": "sklearn.pipeline.Pipeline (TfidfVectorizer + LogisticRegression)",
        "vectorizer": {
            "class": "TfidfVectorizer",
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 10000,
            "token_pattern": TOKEN_PATTERN,
            "token_pattern_note": "Keeps digits and @handles as single tokens (e.g. 98765@ybl)",
        },
        "classifier": {
            "class": "LogisticRegression",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42,
        },
        "label_schema_path": "models/v1/label_schema.json",
        "primary_metric": "IMPERSONATION_SCAM F1",
        "guardrail_metric": "strict_fpr_legit",
        "metrics_val": {
            "impersonation_f1": m["val"]["impersonation_f1"],
            "macro_f1": m["val"]["macro_f1"],
            "strict_fpr_legit": m["val"]["strict_fpr_legit"],
            "accuracy": m["val"]["accuracy"],
        },
        "metrics_test": {
            "impersonation_f1": m["test"]["impersonation_f1"],
            "macro_f1": m["test"]["macro_f1"],
            "strict_fpr_legit": m["test"]["strict_fpr_legit"],
            "accuracy": m["test"]["accuracy"],
        },
        "champion_reason": comparison["winner_reason"],
        "champion_selected_over": "distilbert_v1",
        "comparison_report": "reports/model_comparison_v1.md",
        "error_analysis_report": "reports/error_analysis_v1.md",
        "known_limitations": [
            "Real HDFC/BSNL/Flipkart transactional SMS can be flagged as OTHER_SCAM (brand-alert confusion).",
            "Matched-pair secrecy/refund wording can false-alarm on legit personal messages (twin template overlap).",
            "UPI Circle–style scams may land in OTHER_SCAM instead of IMPERSONATION_SCAM; small 69-row test set.",
        ],
        "reload_verification": verify,
        "not_included": [
            "distilbert_v1 (experiment only)",
            "baseline_tfidf_plus_features_v1 (hurt val)",
            "val/test CSV copies",
            "secrets or API keys",
        ],
    }


def build_model_card(metadata: dict) -> str:
    test = metadata["metrics_test"]
    val = metadata["metrics_val"]
    limits = metadata["known_limitations"]
    return "\n".join(
        [
            "# Model Card — impersonabench-tfidf-logreg-v1",
            "",
            "## Model description",
            "",
            "4-class Hinglish/English SMS-style message classifier built on TF-IDF features",
            "and multinomial logistic regression. Primary focus: detecting **IMPERSONATION_SCAM**",
            "(trusted-contact UPI impersonation) as a decision-support aid, not an autonomous blocker.",
            "",
            "## Intended use",
            "",
            "- Research and portfolio demonstration of an end-to-end ML workflow",
            "- Pre-payment “second opinion” on suspicious chat messages",
            "- **Not** a bank product, not licensed fraud prevention, not for autonomous blocking",
            "",
            "## Training data",
            "",
            "- **Dataset:** ImpersonaBench v1 (`data/dataset_v1.csv`)",
            "- **Size:** 457 rows (partly synthetic Hinglish matched pairs)",
            "- **Train split:** 319 rows (`data/splits/train_v1.csv`)",
            "- **Splitting:** pair-aware group split, seed 42 — see `reports/split_v1.md`",
            "- **Features:** `message` text only (no metadata columns)",
            "",
            "## Evaluation (frozen test split, one-time)",
            "",
            "| Split | Impersonation F1 | Macro-F1 | Strict FPR (legit) |",
            "|---|---:|---:|---:|",
            f"| Val | {val['impersonation_f1']:.4f} | {val['macro_f1']:.4f} | {val['strict_fpr_legit']:.4f} |",
            f"| Test | {test['impersonation_f1']:.4f} | {test['macro_f1']:.4f} | {test['strict_fpr_legit']:.4f} |",
            "",
            "Selected over DistilBERT v1 on test impersonation F1 and strict FPR with lower latency and size.",
            "",
            "## Limitations",
            "",
        ]
        + [f"- {item}" for item in limits]
        + [
            "- Strong reliance on `@ybl`-style UPI handles for catching impersonation; scams without visible UPI may slip.",
            "- Evaluated on a small, partly synthetic corpus — metrics are not proof of real-world recall.",
            "",
            "## Ethical note",
            "",
            "This model is a **decision aid only**. Users should verify requests through a separate",
            "channel before sending money. For confirmed fraud in India, call **1930** (Cyber Crime Helpline).",
            "No guarantee of detection; false alarms on legitimate bank SMS and family chat are known.",
            "",
            "## How to load",
            "",
            "```python",
            "import joblib",
            "from pathlib import Path",
            "",
            "bundle = Path('models/v1')",
            "pipe = joblib.load(bundle / 'model.joblib')",
            "schema = json.loads((bundle / 'label_schema.json').read_text())",
            "id_to_label = {int(k): v for k, v in schema['id_to_label'].items()}",
            "",
            "message = 'Bhai phone toot gaya, 5000 UPI pe abhi bhej de 9876512340@ybl.'",
            "pred_id = int(pipe.predict([message])[0])",
            "label = id_to_label[pred_id]",
            "```",
            "",
            f"- **Artifact:** `models/v1/model.joblib`",
            f"- **sklearn version at package time:** {metadata['sklearn_version']}",
            "- **Other deps:** pandas, joblib (see project environment)",
            "",
        ]
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Champion not found: {MODEL_PATH}. Run baseline.py first."
        )

    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    test_df = pd.read_csv(TEST_PATH)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(MODEL_PATH)
    verify = []
    for row_id in VERIFY_IDS:
        row = test_df.loc[test_df["id"] == row_id].iloc[0]
        msg = str(row["message"])
        pred_id = int(pipe.predict([msg])[0])
        verify.append(
            {
                "id": row_id,
                "true_label": row["label"],
                "pred_id": pred_id,
                "pred_label": ID_TO_LABEL[pred_id],
                "correct": ID_TO_LABEL[pred_id] == row["label"],
            }
        )

    label_schema = build_label_schema()
    LABEL_SCHEMA_PATH.write_text(json.dumps(label_schema, indent=2), encoding="utf-8")

    metadata = build_metadata(comparison, verify)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    MODEL_CARD_PATH.write_text(build_model_card(metadata), encoding="utf-8")

    print("packaged:", BUNDLE_DIR)
    print("sklearn:", metadata["sklearn_version"])
    print("reload verify:", verify)
    print("wrote", MODEL_PATH)
    print("wrote", LABEL_SCHEMA_PATH)
    print("wrote", METADATA_PATH)
    print("wrote", MODEL_CARD_PATH)


if __name__ == "__main__":
    main()
