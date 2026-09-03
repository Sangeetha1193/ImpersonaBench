# Baseline evaluation v1

## Metric definitions (written before reading the scores)

- Primary metric: **F1 of IMPERSONATION_SCAM**
- Also report: **macro_f1** and **accuracy**
- Strict FPR on legit rows = `pred == IMPERSONATION_SCAM` among true `{LEGIT_PERSONAL, LEGIT_TRANSACTIONAL}`
- Any-scam FPR on legit rows = `pred in {IMPERSONATION_SCAM, OTHER_SCAM}` among the same legit rows
- Train, val, and test are reported separately. They must never be mixed.
- The support for impersonation is small (~19-20 rows in val/test), so one extra error moves F1 noticeably.

## Model being evaluated

- Artifact: `models/baseline_tfidf_logreg_v1.joblib`
- This script does not refit the model and does not tune any hyperparameters.
- Test split is used here for evaluation only, after the baseline was already frozen.

### Val

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| IMPERSONATION_SCAM | 0.947 | 0.947 | 0.947 | 19 |
| LEGIT_PERSONAL | 0.947 | 0.947 | 0.947 | 19 |
| OTHER_SCAM | 0.905 | 0.950 | 0.927 | 20 |
| LEGIT_TRANSACTIONAL | 0.900 | 0.818 | 0.857 | 11 |

- Accuracy: 0.9275
- Macro-F1: 0.9197
- IMPERSONATION_SCAM F1: 0.9474
- Strict FPR on legit rows: 0.0333 (1/30)
- Any-scam FPR on legit rows: 0.0667 (2/30)

### Test

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| IMPERSONATION_SCAM | 0.950 | 0.950 | 0.950 | 20 |
| LEGIT_PERSONAL | 1.000 | 0.947 | 0.973 | 19 |
| OTHER_SCAM | 0.826 | 1.000 | 0.905 | 19 |
| LEGIT_TRANSACTIONAL | 1.000 | 0.727 | 0.842 | 11 |

- Accuracy: 0.9275
- Macro-F1: 0.9175
- IMPERSONATION_SCAM F1: 0.9500
- Strict FPR on legit rows: 0.0333 (1/30)
- Any-scam FPR on legit rows: 0.1333 (4/30)

## Train vs val sanity check

- Train macro-F1: 0.9972
- Val macro-F1: 0.9197
- Train IMPERSONATION_SCAM F1: 1.0000
- Val IMPERSONATION_SCAM F1: 0.9474
- If train is near 1.0 and val is lower, the model is fitting templates or dataset artifacts. Note it; do not 'fix' it in the evaluation step.

## Confusion matrix (val)

|                          |   pred:IMPERSONATION_SCAM |   pred:LEGIT_PERSONAL |   pred:OTHER_SCAM |   pred:LEGIT_TRANSACTIONAL |
|:-------------------------|--------------------------:|----------------------:|------------------:|---------------------------:|
| true:IMPERSONATION_SCAM  |                        18 |                     0 |                 1 |                          0 |
| true:LEGIT_PERSONAL      |                         0 |                    18 |                 1 |                          0 |
| true:OTHER_SCAM          |                         0 |                     0 |                19 |                          1 |
| true:LEGIT_TRANSACTIONAL |                         1 |                     1 |                 0 |                          9 |

## Confusion matrix (test)

|                          |   pred:IMPERSONATION_SCAM |   pred:LEGIT_PERSONAL |   pred:OTHER_SCAM |   pred:LEGIT_TRANSACTIONAL |
|:-------------------------|--------------------------:|----------------------:|------------------:|---------------------------:|
| true:IMPERSONATION_SCAM  |                        19 |                     0 |                 1 |                          0 |
| true:LEGIT_PERSONAL      |                         1 |                    18 |                 0 |                          0 |
| true:OTHER_SCAM          |                         0 |                     0 |                19 |                          0 |
| true:LEGIT_TRANSACTIONAL |                         0 |                     0 |                 3 |                          8 |

## Shortcut check (@handle / UPI-handle signal)

- Among val mistakes: 1/5 = 0.200 contain an `@...` handle
- Among correct val IMPERSONATION_SCAM predictions: 17/18 = 0.944 contain an `@...` handle
- Among all legit val rows: 0/30 = 0.000 contain an `@...` handle
- This is a shortcut check, not new feature engineering. If the scam classes are mostly 'has @ybl', say so honestly.

## Plain-English takeaways

1. The confusion matrix cell `true:LEGIT_TRANSACTIONAL → pred:OTHER_SCAM` means the model saw a real alert but treated it like a scam. In plain English: it got too suspicious.
2. A strong val score can still be partly template-driven. Train vs val tells us whether the model is mostly memorizing patterns from synthetic rows.
3. What we must not do next: tune hyperparameters on the test set. Test is the final exam, not a practice worksheet.

