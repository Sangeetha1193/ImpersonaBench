"""Split dataset_v1.csv into train / val / test by group (pair_id), not by row.

Run:  python split_v1.py

Rule: if p_001 has a scam row and a legit row, BOTH stay in the same split.
Splitting one twin into train and the other into test is leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "dataset_v1.csv"
OUT_DIR = ROOT / "data" / "splits"
REPORT = ROOT / "reports" / "split_v1.md"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15   # of total
TEST_RATIO = 0.15  # of total


def build_groups(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group. Unpaired rows use their own id as the group key."""
    rows = []
    has_pair = df["pair_id"].fillna("").astype(str).str.match(r"^p_")

    for pair_id, g in df[has_pair].groupby("pair_id"):
        rows.append({
            "group_id": pair_id,
            "strata": "pair_impersonation",
            "n_rows": len(g),
            "labels": sorted(g["label"].unique().tolist()),
        })

    for _, r in df[~has_pair].iterrows():
        rows.append({
            "group_id": r["id"],
            "strata": r["label"],
            "n_rows": 1,
            "labels": [r["label"]],
        })

    return pd.DataFrame(rows)


def assign_split(groups: pd.DataFrame) -> pd.DataFrame:
    """Split groups 70 / 15 / 15, stratified by strata key."""
    g = groups.copy()

    # first cut: 70% train, 30% temp (val + test)
    train_ids, temp_ids = train_test_split(
        g["group_id"],
        test_size=VAL_RATIO + TEST_RATIO,
        stratify=g["strata"],
        random_state=SEED,
    )

    temp = g[g["group_id"].isin(temp_ids)]
    # second cut: split temp 50/50 → 15% val, 15% test of original
    val_ids, test_ids = train_test_split(
        temp["group_id"],
        test_size=0.5,
        stratify=temp["strata"],
        random_state=SEED,
    )

    split_map = {}
    for gid in train_ids:
        split_map[gid] = "train"
    for gid in val_ids:
        split_map[gid] = "val"
    for gid in test_ids:
        split_map[gid] = "test"

    g["split"] = g["group_id"].map(split_map)
    return g


def row_split(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Map group-level split back onto every row."""
    has_pair = df["pair_id"].fillna("").astype(str).str.match(r"^p_")
    df = df.copy()
    df["group_id"] = df["id"]
    df.loc[has_pair, "group_id"] = df.loc[has_pair, "pair_id"]

    split_map = dict(zip(groups["group_id"], groups["split"]))
    df["split"] = df["group_id"].map(split_map)
    if df["split"].isna().any():
        raise ValueError("Some rows did not get a split assignment")
    return df


def leakage_checks(df: pd.DataFrame) -> dict:
    """Run the checks we promised in the report."""
    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    issues = []

    # pair_id overlap across splits
    train_pairs = set(splits["train"]["group_id"])
    val_pairs = set(splits["val"]["group_id"])
    test_pairs = set(splits["test"]["group_id"])
    pair_overlap = (train_pairs & val_pairs) | (train_pairs & test_pairs) | (val_pairs & test_pairs)
    if pair_overlap:
        issues.append(f"group_id overlap: {len(pair_overlap)}")

    # exact message overlap
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(splits[a]["message"]) & set(splits[b]["message"])
        if overlap:
            issues.append(f"message overlap {a}/{b}: {len(overlap)}")

    # missing classes
    all_labels = set(df["label"])
    for name, part in splits.items():
        missing = all_labels - set(part["label"])
        if missing:
            issues.append(f"{name} missing labels: {missing}")

    return {
        "group_id_overlap": len(pair_overlap),
        "message_overlap": issues,
        "issues": issues,
        "rows": {s: len(splits[s]) for s in splits},
        "groups": {s: splits[s]["group_id"].nunique() for s in splits},
        "labels": {s: splits[s]["label"].value_counts().to_dict() for s in splits},
        "patterns": {
            s: splits[s]["pattern"].value_counts().head(5).to_dict()
            for s in splits
        },
    }


def write_report(stats: dict, groups: pd.DataFrame) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    strata_rule = (
        "Paired rows (120 groups with pair_id p_*) share strata pair_impersonation; "
        "each unpaired row is its own group with strata equal to its label."
    )

    lines = [
        "# Split v1 report",
        "",
        "## Rules",
        "",
        "1. **Split unit = group_id**, not row. Paired twins always stay together.",
        "2. **Ratios:** train 70% / val 15% / test 15%, seed=42.",
        "3. **Strata rule:** " + strata_rule,
        "4. **Smallest class note:** LEGIT_TRANSACTIONAL (~72 rows) → test may have ~10–15. F1 will be noisy.",
        "5. **Leakage:** group_id overlap must be 0; message overlap must be 0.",
        "",
        "## Group inventory (before split)",
        "",
        f"- Total rows: 457",
        f"- Paired groups (size 2): 120 → 240 rows",
        f"- Unpaired groups (size 1): 217 → 217 rows",
        f"- Total groups: {len(groups)}",
        "",
        "### Unpaired row breakdown",
        "- OTHER_SCAM: 130",
        "- LEGIT_TRANSACTIONAL: 72",
        "- IMPERSONATION_SCAM: 10",
        "- LEGIT_PERSONAL: 5",
        "",
        "## Results",
        "",
        f"| Split | Rows | Groups |",
        f"|-------|-----:|-------:|",
    ]
    for s in ("train", "val", "test"):
        lines.append(f"| {s} | {stats['rows'][s]} | {stats['groups'][s]} |")

    lines += [
        "",
        f"**group_id overlap across splits:** {stats['group_id_overlap']} (must be 0)",
        "",
        "### Label counts per split",
        "",
    ]
    for s in ("train", "val", "test"):
        lines.append(f"**{s}:**")
        for lbl, cnt in sorted(stats["labels"][s].items()):
            lines.append(f"- {lbl}: {cnt}")
        lines.append("")

    lines += ["### Top patterns per split", ""]
    for s in ("train", "val", "test"):
        lines.append(f"**{s}:** {stats['patterns'][s]}")
        lines.append("")

    if stats["issues"]:
        lines += ["### Warnings", ""]
        for issue in stats["issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("### Leakage checks: all passed")

    lines += [
        "",
        "## Strata rule (one sentence)",
        "",
        strata_rule,
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} rows from {DATA}")

    groups = build_groups(df)
    print(f"Built {len(groups)} groups "
          f"({(groups['strata']=='pair_impersonation').sum()} paired, "
          f"{(groups['strata']!='pair_impersonation').sum()} unpaired)")

    groups = assign_split(groups)
    df = row_split(df, groups)

    stats = leakage_checks(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        part = df[df["split"] == split_name].drop(columns=["group_id", "split"])
        path = OUT_DIR / f"{split_name}_v1.csv"
        part.to_csv(path, index=False)
        print(f"Wrote {path} ({len(part)} rows)")

    write_report(stats, groups)

    # summary for the user
    print("\n--- split summary ---")
    print("rows:", stats["rows"])
    print("groups:", stats["groups"])
    print("group_id overlap:", stats["group_id_overlap"])
    print("labels:")
    for s in ("train", "val", "test"):
        print(f"  {s}:", stats["labels"][s])

    if stats["issues"]:
        print("WARNINGS:", stats["issues"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
