# Archive

Legacy and intermediate files kept for reproducibility, not used in the v1 inference path.

| Path | What it is |
|---|---|
| `legacy/data_pipeline.py` | Old binary scam/benign pipeline (pre–schema v1) |
| `raw/impersonabench_dataset.csv` | Original messy 330-row dump |
| `raw/gold100_pairs.csv` | Intermediate Gold-100 pair file |

To rebuild `data/dataset_v1.csv` from scratch: `python build_dataset_v1.py` (reads from `archive/raw/`).
