"""Inference for ImpersonaBench v1 champion (TF-IDF + LogisticRegression).

Usage:
    python -m src.predict --text "Bhai abhi 4500 bhej de..."
    from src.predict import predict
    result = predict("...")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

from features_v1 import (
    PHONE_RE,
    SECRECY_PHRASES,
    URGENCY_WORDS,
    UPI_HANDLE_RE,
    VERIFY_BLOCK_PHRASES,
    _count_matches,
)

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT / "models" / "v1"
MODEL_PATH = BUNDLE_DIR / "model.joblib"
SCHEMA_PATH = BUNDLE_DIR / "label_schema.json"
METADATA_PATH = BUNDLE_DIR / "metadata.json"

SCAM_LABELS = {"IMPERSONATION_SCAM", "OTHER_SCAM"}
BRAND_WORDS = [
    "hdfc", "sbi", "icici", "axis", "kotak", "bsnl", "jio", "airtel",
    "flipkart", "amazon", "paytm", "phonepe", "gpay", "npci",
]

DISCLAIMER = (
    "Decision aid only - not guaranteed fraud detection. "
    "Verify urgent payment requests through a known contact channel before sending money. "
    "For confirmed cyber fraud in India, call 1930."
)

_predictor: "Predictor | None" = None


class Predictor:
    """Loads models/v1/ once; safe to reuse across many predictions."""

    def __init__(self, bundle_dir: Path | None = None) -> None:
        bundle = Path(bundle_dir) if bundle_dir else BUNDLE_DIR
        model_path = bundle / "model.joblib"
        schema_path = bundle / "label_schema.json"
        metadata_path = bundle / "metadata.json"

        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not schema_path.is_file():
            raise FileNotFoundError(f"Label schema not found: {schema_path}")

        self.bundle_dir = bundle
        self.pipe = joblib.load(model_path)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.id_to_label = {int(k): v for k, v in schema["id_to_label"].items()}
        self.label_to_id = {k: int(v) for k, v in schema["label_to_id"].items()}

        if metadata_path.is_file():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.model_version = meta.get("model_name", "v1")
        else:
            self.model_version = schema.get("version", "v1")

    def predict(self, message: str) -> dict:
        return predict(message, predictor=self)


def load_model(bundle_dir: Path | str | None = None) -> Predictor:
    global _predictor
    if bundle_dir is not None:
        return Predictor(bundle_dir)
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def _matched_terms(text: str, phrases: list[str]) -> list[str]:
    lowered = text.lower()
    return [p for p in phrases if p in lowered]


def _risk_level(label: str, confidence: float) -> str:
    if label in SCAM_LABELS and confidence > 0.5:
        return "HIGH"
    if label in SCAM_LABELS:
        return "MEDIUM"
    if confidence < 0.5:
        return "MEDIUM"
    return "LOW"


def _heuristic_reasons(text: str, predicted_label: str) -> list[str]:
    """Explain the prediction only — never changes the model output."""
    reasons: list[str] = []
    lowered = text.lower()

    if UPI_HANDLE_RE.search(text):
        reasons.append("Message contains UPI handle (@...)")

    urgency_hits = _matched_terms(text, URGENCY_WORDS)
    if urgency_hits:
        reasons.append(f"Urgency words detected: {', '.join(urgency_hits[:5])}")

    secrecy_hits = _matched_terms(text, SECRECY_PHRASES)
    if secrecy_hits:
        reasons.append(f"Secrecy phrases detected: {', '.join(secrecy_hits[:3])}")

    verify_hits = _matched_terms(text, VERIFY_BLOCK_PHRASES)
    if verify_hits:
        reasons.append(f"Verification-block phrases detected: {', '.join(verify_hits[:3])}")

    if PHONE_RE.search(text):
        reasons.append("10-digit phone number present")

    brand_hits = [b for b in BRAND_WORDS if b in lowered]
    if brand_hits:
        reasons.append(
            f"Brand/institution words present: {', '.join(brand_hits[:4])}"
            + (
                " (model may confuse real alerts with scams)"
                if predicted_label == "OTHER_SCAM"
                else ""
            )
        )

    return reasons[:5]


def predict(message: str, predictor: Predictor | None = None) -> dict:
    """Classify one message. Heuristics explain only; they do not override the model."""
    if message is None:
        return {
            "error": "message must be a non-empty string",
            "disclaimer": DISCLAIMER,
        }

    text = str(message).strip()
    if not text:
        return {
            "error": "message must be a non-empty string",
            "disclaimer": DISCLAIMER,
        }

    model = predictor or load_model()

    pred_id = int(model.pipe.predict([text])[0])
    proba = model.pipe.predict_proba([text])[0]
    label_names = [model.id_to_label[i] for i in sorted(model.id_to_label)]
    prob_map = {name: round(float(proba[model.label_to_id[name]]), 4) for name in label_names}
    confidence = round(float(max(proba)), 4)
    predicted_label = model.id_to_label[pred_id]

    reasons = [
        f"Model predicted {predicted_label} with {confidence:.2f} confidence",
    ]
    top3 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:3]
    reasons.append(
        "Top classes: " + ", ".join(f"{lbl} ({p:.2f})" for lbl, p in top3)
    )
    reasons.extend(_heuristic_reasons(text, predicted_label))

    if predicted_label == "OTHER_SCAM" and any(b in text.lower() for b in ("hdfc", "sbi", "bsnl")):
        reasons.append(
            "Known limitation: legit bank/telco alerts are sometimes flagged as OTHER_SCAM"
        )

    return {
        "model_version": model.model_version,
        "predicted_label": predicted_label,
        "predicted_id": pred_id,
        "confidence": confidence,
        "probabilities": prob_map,
        "risk_level": _risk_level(predicted_label, confidence),
        "reasons": reasons[:5],
        "disclaimer": DISCLAIMER,
        "heuristics_override_prediction": False,
    }


def _format_output(result: dict, as_json: bool) -> str:
    if as_json:
        return json.dumps(result, indent=2, ensure_ascii=False)
    if "error" in result:
        return f"Error: {result['error']}\n{result.get('disclaimer', '')}"
    lines = [
        f"model_version: {result['model_version']}",
        f"predicted_label: {result['predicted_label']}",
        f"predicted_id: {result['predicted_id']}",
        f"confidence: {result['confidence']}",
        f"risk_level: {result['risk_level']}",
        "probabilities:",
    ]
    for lbl, p in result["probabilities"].items():
        lines.append(f"  {lbl}: {p}")
    lines.append("reasons:")
    for r in result["reasons"]:
        lines.append(f"  - {r}")
    lines.append(f"disclaimer: {result['disclaimer']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ImpersonaBench v1 message classifier")
    parser.add_argument("--text", type=str, help="Message text to classify")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    if not args.text:
        parser.error("provide --text")

    try:
        result = predict(args.text)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(_format_output(result, args.json))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
