# Model Card — impersonabench-tfidf-logreg-v1

## Model description

4-class Hinglish/English SMS-style message classifier built on TF-IDF features
and multinomial logistic regression. Primary focus: detecting **IMPERSONATION_SCAM**
(trusted-contact UPI impersonation) as a decision-support aid, not an autonomous blocker.

## Intended use

- Research and portfolio demonstration of an end-to-end ML workflow
- Pre-payment “second opinion” on suspicious chat messages
- **Not** a bank product, not licensed fraud prevention, not for autonomous blocking

## Training data

- **Dataset:** ImpersonaBench v1 (`data/dataset_v1.csv`)
- **Size:** 457 rows (partly synthetic Hinglish matched pairs)
- **Train split:** 319 rows (`data/splits/train_v1.csv`)
- **Splitting:** pair-aware group split, seed 42 — see `reports/split_v1.md`
- **Features:** `message` text only (no metadata columns)

## Evaluation (frozen test split, one-time)

| Split | Impersonation F1 | Macro-F1 | Strict FPR (legit) |
|---|---:|---:|---:|
| Val | 0.9474 | 0.9197 | 0.0333 |
| Test | 0.9500 | 0.9175 | 0.0333 |

Selected over DistilBERT v1 on test impersonation F1 and strict FPR with lower latency and size.

## Limitations

- Real HDFC/BSNL/Flipkart transactional SMS can be flagged as OTHER_SCAM (brand-alert confusion).
- Matched-pair secrecy/refund wording can false-alarm on legit personal messages (twin template overlap).
- UPI Circle–style scams may land in OTHER_SCAM instead of IMPERSONATION_SCAM; small 69-row test set.
- Strong reliance on `@ybl`-style UPI handles for catching impersonation; scams without visible UPI may slip.
- Evaluated on a small, partly synthetic corpus — metrics are not proof of real-world recall.

## Ethical note

This model is a **decision aid only**. Users should verify requests through a separate
channel before sending money. For confirmed fraud in India, call **1930** (Cyber Crime Helpline).
No guarantee of detection; false alarms on legitimate bank SMS and family chat are known.

## How to load

```python
import joblib
from pathlib import Path

bundle = Path('models/v1')
pipe = joblib.load(bundle / 'model.joblib')
schema = json.loads((bundle / 'label_schema.json').read_text())
id_to_label = {int(k): v for k, v in schema['id_to_label'].items()}

message = 'Bhai phone toot gaya, 5000 UPI pe abhi bhej de 9876512340@ybl.'
pred_id = int(pipe.predict([message])[0])
label = id_to_label[pred_id]
```

- **Artifact:** `models/v1/model.joblib`
- **sklearn version at package time:** 1.7.2
- **Other deps:** pandas, joblib (see project environment)
