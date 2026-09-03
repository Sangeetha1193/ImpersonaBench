# Inference v1

Step 13: load the packaged champion from `models/v1/` and classify a single message — no notebook, no train CSV.

---

## Interface (chosen: CLI + importable function)

**CLI** (demo-friendly):

```bash
python -m src.predict --text "Bhai abhi 4500 bhej de 9876512340@ybl"
python -m src.predict --text "..." --json
```

**Python function** (for FastAPI / Docker later):

```python
from src.predict import predict

result = predict("Bhai abhi 4500 bhej de...")
```

Both call the same code path. Heuristics **explain** the model output only; they never change `predicted_label`.

---

## Output schema (stable keys)

| Key | Type | Description |
|---|---|---|
| `model_version` | str | From `models/v1/metadata.json` (`impersonabench-tfidf-logreg-v1`) |
| `predicted_label` | str | One of four class names |
| `predicted_id` | int | 0–3 per `label_schema.json` |
| `confidence` | float | Max class probability (rounded 4 dp) |
| `probabilities` | dict | All class label → probability |
| `risk_level` | str | `HIGH` if scam class and confidence > 0.5; `MEDIUM` if scam with lower confidence or uncertain; `LOW` for legit top class |
| `reasons` | list[str] | Model summary + optional heuristic cues (max 5) |
| `disclaimer` | str | Fixed product-honesty string (1930 helpline) |
| `heuristics_override_prediction` | bool | Always `false` in v1 |

On invalid input (`""` or whitespace only):

```json
{
  "error": "message must be a non-empty string",
  "disclaimer": "..."
}
```

---

## Load path discipline

- Project root resolved as `Path(__file__).resolve().parent.parent` inside `src/predict.py`
- Artifacts: `models/v1/model.joblib`, `label_schema.json`, `metadata.json`
- `load_model()` caches one `Predictor` instance per process (no reload per message)
- Raw message string goes straight to `pipeline.predict` / `predict_proba` — vectorizer handles `lowercase=True`; do not strip UPI handles in Python

---

## Reasons (two layers)

1. **Model:** predicted class, confidence, top-3 class probabilities  
2. **Heuristics** (from `features_v1.py` word lists + brand tokens): UPI handle, urgency, secrecy, verify-block, phone, brand words  

Heuristics are **not** fed into the classifier. When the model says `OTHER_SCAM` on an HDFC alert, reasons include a known-limitation note instead of pretending certainty.

---

## Smoke tests (manual, from project root)

| Case | Source | Expected |
|---|---|---|
| Impersonation | `ib_v1_000005` text | `IMPERSONATION_SCAM`, HIGH risk |
| Legit personal | `ib_v1_000006` text | `LEGIT_PERSONAL`, LOW risk |
| HDFC RTGS | `ib_v1_000301` text | `OTHER_SCAM` (known limitation) |
| Empty string | `predict("")` | Error dict, no crash |

Reload parity: `ib_v1_000005` and `ib_v1_000031` both return `IMPERSONATION_SCAM` (id 0) — matches Step 12 `package_model_v1.py` verification.

---

### Example 1 — impersonation (`ib_v1_000005`)

**Command:**

```bash
python -m src.predict --text "Yaar mera handset drown ho gaya bathroom mein. Temporary SIM se msg hai. 3000 bhej de 7007007000@oksbi pe, landlord rent maang raha hai aaj. Call mat karna yeh number unknown hai." --json
```

**Output:**

```json
{
  "model_version": "impersonabench-tfidf-logreg-v1",
  "predicted_label": "IMPERSONATION_SCAM",
  "predicted_id": 0,
  "confidence": 0.7396,
  "probabilities": {
    "IMPERSONATION_SCAM": 0.7396,
    "LEGIT_PERSONAL": 0.1398,
    "OTHER_SCAM": 0.0547,
    "LEGIT_TRANSACTIONAL": 0.0659
  },
  "risk_level": "HIGH",
  "reasons": [
    "Model predicted IMPERSONATION_SCAM with 0.74 confidence",
    "Top classes: IMPERSONATION_SCAM (0.74), LEGIT_PERSONAL (0.14), LEGIT_TRANSACTIONAL (0.07)",
    "Message contains UPI handle (@...)",
    "Urgency words detected: aaj, now",
    "Verification-block phrases detected: call mat karna"
  ],
  "disclaimer": "Decision aid only - not guaranteed fraud detection. Verify urgent payment requests through a known contact channel before sending money. For confirmed cyber fraud in India, call 1930.",
  "heuristics_override_prediction": false
}
```

---

### Example 2 — legit personal (`ib_v1_000006`)

```json
{
  "model_version": "impersonabench-tfidf-logreg-v1",
  "predicted_label": "LEGIT_PERSONAL",
  "predicted_id": 1,
  "confidence": 0.6671,
  "probabilities": {
    "IMPERSONATION_SCAM": 0.1643,
    "LEGIT_PERSONAL": 0.6671,
    "OTHER_SCAM": 0.092,
    "LEGIT_TRANSACTIONAL": 0.0766
  },
  "risk_level": "LOW",
  "reasons": [
    "Model predicted LEGIT_PERSONAL with 0.67 confidence",
    "Top classes: LEGIT_PERSONAL (0.67), IMPERSONATION_SCAM (0.16), OTHER_SCAM (0.09)"
  ],
  "disclaimer": "Decision aid only - not guaranteed fraud detection. Verify urgent payment requests through a known contact channel before sending money. For confirmed cyber fraud in India, call 1930.",
  "heuristics_override_prediction": false
}
```

---

### Example 3 — HDFC RTGS known limitation (`ib_v1_000301`)

```json
{
  "model_version": "impersonabench-tfidf-logreg-v1",
  "predicted_label": "OTHER_SCAM",
  "predicted_id": 2,
  "confidence": 0.4315,
  "probabilities": {
    "IMPERSONATION_SCAM": 0.1335,
    "LEGIT_PERSONAL": 0.1369,
    "OTHER_SCAM": 0.4315,
    "LEGIT_TRANSACTIONAL": 0.2982
  },
  "risk_level": "MEDIUM",
  "reasons": [
    "Model predicted OTHER_SCAM with 0.43 confidence",
    "Top classes: OTHER_SCAM (0.43), LEGIT_TRANSACTIONAL (0.30), LEGIT_PERSONAL (0.14)",
    "Brand/institution words present: hdfc (model may confuse real alerts with scams)",
    "Known limitation: legit bank/telco alerts are sometimes flagged as OTHER_SCAM"
  ],
  "disclaimer": "Decision aid only - not guaranteed fraud detection. Verify urgent payment requests through a known contact channel before sending money. For confirmed cyber fraud in India, call 1930.",
  "heuristics_override_prediction": false
}
```

Note: confidence is only 0.43 — reasons do not claim high certainty.

---

### Example 4 — empty string

**Command:**

```python
python -c "from src.predict import predict; import json; print(json.dumps(predict(''), indent=2))"
```

**Output:**

```json
{
  "error": "message must be a non-empty string",
  "disclaimer": "Decision aid only - not guaranteed fraud detection. Verify urgent payment requests through a known contact channel before sending money. For confirmed cyber fraud in India, call 1930."
}
```

No exception raised; CLI exit code 0 for API use. (Passing `--text` with no value on the shell is a argparse error — use the function for empty-input tests.)

---

## Files

| Path | Role |
|---|---|
| `src/predict.py` | `load_model()`, `predict()`, CLI |
| `models/v1/model.joblib` | Fitted sklearn Pipeline |
| `models/v1/label_schema.json` | id ↔ label mapping |
| `features_v1.py` | Heuristic word lists only (not used in model scoring) |

---

## Next step (Step 14)

Wrap `predict()` in FastAPI (`POST /classify`) and Dockerfile — inference module is ready.
