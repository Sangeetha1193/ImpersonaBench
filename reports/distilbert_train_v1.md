# DistilBERT train v1

## Setup
- checkpoint: `distilbert-base-uncased`
- max_length: 128
- epochs (max): 4
- learning_rate: 2e-05
- batch_size: 8
- seed: 42
- device: cpu
- early stopping: patience=2 on val `macro_f1`
- selection: best val checkpoint (`load_best_model_at_end=True`), not last epoch
- test set: not used for training or early stopping

## Why these choices
- DistilBERT is smaller/faster than BERT; enough for a portfolio baseline comparison.
- Messages are short Hinglish/English SMS-like text → max_length=128 is enough.
- Fine-tuning uses a small LR (~2e-5) so we adapt the head without destroying pretrained weights.
- With only 319 train rows, more than ~4 epochs usually memorizes templates.

## Results (same metric definitions as baseline eval)

### Train
- accuracy: 0.8558
- macro-F1: 0.7968
- IMPERSONATION_SCAM F1: 0.9486
- strict FPR on legit: 0.0000

### Val
- accuracy: 0.8261
- macro-F1: 0.7623
- IMPERSONATION_SCAM F1: 0.8947
- strict FPR on legit: 0.0333
- any-scam FPR on legit: 0.2333

### Train vs val gap
- macro-F1 gap (train - val): 0.0345
- impersonation F1 gap (train - val): 0.0538

### vs TF-IDF baseline (val)
- baseline val macro-F1: 0.9197
- DistilBERT val macro-F1: 0.7623
- baseline val impersonation F1: 0.9474
- DistilBERT val impersonation F1: 0.8947
- baseline val strict FPR: 0.0333
- DistilBERT val strict FPR: 0.0333 (same)

## Best epoch / metric
- best_metric (val macro_f1 tracked by Trainer): 0.7623488067996077
- best_epoch (approx from logs): 4.0

## Sanity prediction
- sample: `Bhai phone toot gaya, 5000 is UPI pe abhi bhej de 9876512340@ybl. Call mat karna.`
- predicted: `IMPERSONATION_SCAM`

## Honest notes
- Test was not used to pick this model.
- If train is near-perfect and val is only slightly better/worse than TF-IDF, DistilBERT may not be worth the complexity yet.
- Do not declare DistilBERT the winner until Step 10 (side-by-side comparison).

## Saved model
- `C:\Users\sange\Work\ImpersonaBench\models\distilbert_v1`
