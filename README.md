# ImpersonaBench

4-class Hinglish message classifier for **trusted-contact UPI impersonation** detection.  
Portfolio project demonstrating a full ML workflow: data QC → EDA → baseline → experiment → evaluation → error analysis → model packaging.

**Shipped model (v1):** `models/v1/` — TF-IDF + LogisticRegression (~0.1 MB).

**Inference:** `python -m src.predict --text "..."` — see `reports/inference_v1.md`.

---

## Is the project “done”?

| Phase | Status | What you have |
|---|---|---|
| **ML pipeline (Steps 1–12)** | Done | Dataset, splits, baseline, DistilBERT experiment, comparison, error analysis, model bundle |
| **Inference API (Step 13)** | **You do next** | `predict.py` or FastAPI wrapper around `models/v1/model.joblib` |
| **Docker (Step 14)** | **You do next** | `Dockerfile` + `docker build` / `docker run` |
| **CI/CD (Step 15)** | **You do next** | GitHub Actions: lint, metric gate, reload test |
| **Cloud deploy (Step 16)** | **You do next** | Push image to ECR/GCR → run on Cloud Run / App Runner / ECS |

The **modeling story is complete**. What remains is **MLOps / deployment** — intentionally left for you to do manually so you can explain every step in interviews.

---

## Repo layout (interview map)

```
ImpersonaBench/
├── README.md                 # you are here
├── requirements.txt          # sklearn stack (torch commented out)
├── .gitignore                # excludes ~1.8 GB DistilBERT weights
│
├── explore.py                # EDA + LABEL_TO_ID + metric definitions
├── build_dataset_v1.py       # build data/dataset_v1.csv from archive/raw
├── split_v1.py               # pair-aware train/val/test split
│
├── baseline.py               # train champion → models/v1/model.joblib
├── eval_baseline.py          # evaluate champion (no refit)
├── features_v1.py            # hand-crafted features (module)
├── feature_run_v1.py         # negative experiment: features hurt val
├── train_distilbert_v1.py    # DL experiment (weights gitignored)
├── compare_models_v1.py      # TF-IDF vs DistilBERT → pick champion
├── error_analysis_v1.py      # failure modes on val+test
├── package_model_v1.py       # refresh metadata.json + MODEL_CARD.md
│
├── data/
│   ├── dataset_v1.csv        # frozen 457-row corpus
│   └── splits/               # train_v1, val_v1, test_v1
│
├── models/v1/                # PRODUCTION BUNDLE (push to GitHub)
│   ├── model.joblib
│   ├── label_schema.json
│   ├── metadata.json
│   └── MODEL_CARD.md
│
├── reports/                  # human-readable evidence (push to GitHub)
├── metrics/                  # JSON numbers for CI gates (push to GitHub)
└── archive/                  # legacy raw data + old pipeline
```

---

## Quick start (reproduce v1 numbers)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python explore.py               # EDA
python split_v1.py              # splits (if regenerating)
python baseline.py              # trains → models/v1/model.joblib
python eval_baseline.py         # val + test metrics
python package_model_v1.py      # metadata + model card
```

Champion test metrics (frozen): **impersonation F1 = 0.95**, **strict FPR = 0.0333** — see `models/v1/MODEL_CARD.md`.

---

## What to push to GitHub (manual)

### 1. Create repo and first push

```bash
git add .
git status                      # confirm no distilbert_v1/ (~1.8 GB)
git commit -m "ImpersonaBench v1: ML pipeline + packaged TF-IDF champion"
git remote add origin https://github.com/YOUR_USER/ImpersonaBench.git
git push -u origin main
```

**Do push:** code, `data/`, `models/v1/`, `reports/`, `metrics/`, `archive/`  
**Do not push:** `models/distilbert_v1/` (in `.gitignore`), `__pycache__/`, `.venv/`

### 2. CI/CD (GitHub Actions — you write this)

Create `.github/workflows/ci.yml` with roughly:

1. `pip install -r requirements.txt`
2. `python -c "import joblib; joblib.load('models/v1/model.joblib')"`  — reload test
3. Compare `metrics/comparison_v1.json` test impersonation F1 ≥ floor (e.g. 0.90)
4. Optional: run `error_analysis_v1.py` and fail if error count spikes

This teaches **metric gates** — the same idea banks use before promoting a model.

### 3. Docker (you write this)

Minimal path:

```
Dockerfile
├── FROM python:3.11-slim
├── COPY requirements.txt + models/v1/ + predict.py
├── EXPOSE 8000
└── CMD uvicorn app:app --host 0.0.0.0 --port 8000
```

Files to add yourself:

- `predict.py` — `load()` + `predict(message) -> label, proba`
- `app.py` — FastAPI `POST /classify` with JSON `{"message": "..."}`
- `Dockerfile`
- `.dockerignore` — exclude `archive/`, `data/splits/`, distilbert

```bash
docker build -t impersonabench:v1 .
docker run -p 8000:8000 impersonabench:v1
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d "{\"message\":\"Bhai 5000 bhej de @ybl\"}"
```

### 4. Cloud deploy (pick one, do manually)

| Platform | Rough flow |
|---|---|
| **AWS** | ECR push image → App Runner or ECS Fargate |
| **GCP** | Artifact Registry → Cloud Run |
| **Azure** | ACR → Container Apps |
| **Railway / Render** | Connect GitHub repo → deploy Dockerfile (easiest for portfolio) |

For interviews: *“I containerized a sklearn pipeline, health-checked model reload in CI, and deployed a stateless inference API — the model artifact is versioned in `models/v1/metadata.json`.”*

---

## Suggested order (your homework)

1. **Write `predict.py`** — 30 lines, load `models/v1/`, return label + confidence  
2. **Push to GitHub** — clean repo, good README, no giant files  
3. **Add GitHub Actions CI** — reload + metric floor  
4. **Add FastAPI + Dockerfile** — local `docker run` demo  
5. **Deploy to Railway/Render** — free tier, share public URL in README  
6. *(Optional v2)* More data, DistilBERT revisit, drift monitoring (Evidently)

---

## Key interview talking points

- **Problem:** 4-class Hinglish impersonation vs legit personal vs other scams vs bank SMS  
- **Primary metric:** `IMPERSONATION_SCAM` F1; **guardrail:** strict FPR on legit rows  
- **Why TF-IDF won:** better test F1 + FPR, 36× faster, 0.1 MB vs 1.8 GB  
- **Known failure:** HDFC/BSNL legit SMS → `OTHER_SCAM`; twin-pair secrecy false alarms  
- **Not claiming:** production fraud detector — decision aid only; call **1930** for confirmed fraud

---

## License

Portfolio / educational use. Not financial or security advice.
