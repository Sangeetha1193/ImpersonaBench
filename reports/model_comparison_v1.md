# Model comparison v1

## Decision policy (frozen before test)

- **Primary metric:** IMPERSONATION_SCAM F1 on val/test
- **Guardrail:** strict FPR on legit rows (`LEGIT_PERSONAL`, `LEGIT_TRANSACTIONAL`)
- Ship TF-IDF if it wins impersonation F1 on test and strict FPR ≤ DistilBERT (or tied)
- Ship DistilBERT only if it clearly beats TF-IDF on impersonation F1 without worse strict FPR
- If test is within ~0.02 impersonation F1, prefer simpler + faster → TF-IDF
- Do not ship the Step-8 TF-IDF+hand-features model (it hurt val)

## Val + test metrics

| Model | Split | Macro-F1 | Impersonation F1 | Accuracy | Strict FPR | Any-scam FPR |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF | val | 0.9197 | 0.9474 | 0.9275 | 0.0333 | 0.0667 |
| DistilBERT | val | 0.7623 | 0.8947 | 0.8261 | 0.0333 | 0.2333 |
| TF-IDF | test | 0.9175 | 0.9500 | 0.9275 | 0.0333 | 0.1333 |
| DistilBERT | test | 0.7270 | 0.8571 | 0.8116 | 0.0667 | 0.2667 |

## Winner for v1

**baseline_tfidf_logreg_v1**

TF-IDF wins impersonation F1 on test with equal or better strict FPR; simpler and faster when scores are not close.

TF-IDF already led on val impersonation F1 and macro-F1. DistilBERT had a smaller train–val gap but lower absolute scores on this 457-row dataset. For a portfolio v1 artifact, the classical baseline is the right ship candidate: better focal-class F1, same strict FPR, ~36.4× faster inference on CPU, and ~0.1 MB vs ~1789 MB.

## Confusion matrix (test) — shipped model

|                          |   pred:IMPERSONATION_SCAM |   pred:LEGIT_PERSONAL |   pred:OTHER_SCAM |   pred:LEGIT_TRANSACTIONAL |
|:-------------------------|--------------------------:|----------------------:|------------------:|---------------------------:|
| true:IMPERSONATION_SCAM  |                        19 |                     0 |                 1 |                          0 |
| true:LEGIT_PERSONAL      |                         1 |                    18 |                 0 |                          0 |
| true:OTHER_SCAM          |                         0 |                     0 |                19 |                          0 |
| true:LEGIT_TRANSACTIONAL |                         0 |                     0 |                 3 |                          8 |

## Non-metric comparison

- **Inference speed (median, 1 message, CPU):** TF-IDF ~0.69 ms vs DistilBERT ~25.14 ms
- **Model size:** TF-IDF joblib ~0.10 MB vs DistilBERT folder ~1789 MB
- **Shortcut risk:** both models still lean heavily on `@ybl` / UPI-handle surface form; on test, 18/19 correct impersonation predictions contain an `@...` handle
- **When DistilBERT might win later:** dataset > ~2k diverse rows, harder paraphrases, new-wave eval pack, or a multilingual Indic checkpoint once Hindi/Devanagari is in scope

## Shared test errors (both models wrong)

Both wrong on `ib_v1_000301`: true `LEGIT_TRANSACTIONAL`, baseline `OTHER_SCAM`, DistilBERT `OTHER_SCAM`. Message: "HDFC: no pending RTGS on your account. Official helpline is 1860-266-0333 only...."

## What we will NOT claim

- Not a production-ready fraud detector for live UPI traffic
- Not multilingual (v1 is Hinglish + English only)
- Not robust to adversarial paraphrase or unseen scam templates
- Val/test scores are on a small, partly synthetic corpus — strong numbers ≠ real-world recall

## Roadmap

DistilBERT reserved for **v2** when the dataset exceeds **~2,000** curated rows and a held-out **new-wave eval pack** (unseen templates / real redacted samples) is available.

