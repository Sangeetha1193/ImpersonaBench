# Baseline run v1

## What this is
First supervised model. Message text only. Test set was not loaded.

## Data
- train: `train_v1.csv` (319 rows)
- val: `val_v1.csv` (69 rows)
- test: not used
- X = `message`
- y = `LABEL_TO_ID` from `explore.py` (0 impersonation, 1 legit personal, 2 other scam, 3 transactional)

## Vectorizer (TfidfVectorizer)
- lowercase=True (Hinglish casing is noisy; same setting must be used at inference)
- ngram_range=(1, 2) so phrases like `mat call` / `bhej de` can fire
- min_df=2 (drop one-off typos)
- max_features=10000
- token_pattern keeps digits and @handles (`98765@ybl` stays one token)
- train matrix shape: (319, 1543)  (rows = train size, cols = vocab)

## Classifier
- LogisticRegression, multinomial (sklearn default for >2 classes with lbfgs)
- class_weight=balanced (LEGIT_TRANSACTIONAL is the small class)
- max_iter=2000
- random_state=42

## Val sanity check (not tuned)
- val accuracy: 0.9275
- val macro-F1: 0.9197
- val predictions: 69 rows

A very high score would be suspicious (leakage or too-easy templates).
This run is a reproducible floor, not a production model.

## Saved artifact
- `models\baseline_tfidf_logreg_v1.joblib`  (Pipeline: vectorizer + classifier together)

## Not done
- no hyperparameter search
- test_v1.csv not loaded
- split CSVs not modified
