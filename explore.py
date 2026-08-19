
from __future__ import annotations
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "dataset_v1.csv"

EXPECTED_LABELS = {
    "IMPERSONATION_SCAM",
    "LEGIT_PERSONAL",
    "OTHER_SCAM",
    "LEGIT_TRANSACTIONAL",
}

HINGLISH_HINTS = re.compile(
    r"\b(hai|hain|hun|hoon|bhej|bhejo|kal|abhi|yaar|bhai|didi|papa|mummy|"
    r"mat|nahi|pe|ghar|paisa|jaldi|dunga|karo|mein|ka|ke|ki)\b",
    re.I,
)


LABEL_TO_ID = {
    "IMPERSONATION_SCAM":  0,   
    "LEGIT_PERSONAL":      1,   
    "OTHER_SCAM":          2,   
    "LEGIT_TRANSACTIONAL": 3,  
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


PRIMARY_METRIC  = "macro_f1"
FOCAL_CLASS     = "IMPERSONATION_SCAM"   # the one CI watches
FOCAL_CLASS_ID  = LABEL_TO_ID[FOCAL_CLASS]
CI_F1_FLOOR     = 0.70   


def banner(title: str) -> None:
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)


def counts_with_pct(series: pd.Series) -> pd.DataFrame:
    c = series.value_counts(dropna=False)
    return pd.DataFrame({"count": c, "pct": (c / c.sum() * 100).round(2)})


def normalize(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"₹\s*\d[\d,]*", "AMT", t)
    t = re.sub(r"\b\d[\d,]*\b", "AMT", t)
    t = re.sub(r"[a-z0-9._-]+@[a-z0-9.-]+", "UPI", t)
    t = re.sub(r"[^a-z0-9\u0900-\u097f ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def word_tokens(text: str) -> list[str]:
   
    return re.findall(r"[A-Za-z\u0900-\u097F]{2,}", str(text).lower())


def token_set(text: str) -> set[str]:
    return {t for t in normalize(text).split() if len(t) > 2}


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def top_words(texts: pd.Series, n: int = 15) -> list[tuple[str, int]]:
    stop = {
        "the", "and", "for", "you", "your", "this", "that", "with", "from",
        "hai", "hain", "ke", "ki", "ka", "ko", "se", "pe", "me", "mein",
        "not", "will", "was", "are",
    }
    bag = [w for msg in texts for w in word_tokens(msg) if w not in stop]
    return Counter(bag).most_common(n)


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
   
    unknown = set(df["label"]) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"Labels not in the map: {unknown}")
    df = df.copy()
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    return df


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data["message"] = data["message"].astype(str)
    data["chars"]   = data["message"].str.len()
    data["words"]   = data["message"].str.split().str.len()
    data["norm"]    = data["message"].map(normalize)

    # ── 1. schema 
    banner("1. Load + schema")
    print("path  :", DATA_PATH)
    print("rows  :", len(data), "  cols:", len(data.columns))
    print("cols  :", data.columns.tolist())
    print("\nnulls:\n", data.isnull().sum().to_string())
    print("empty messages      :", (data["message"].str.strip() == "").sum())
    print("exact row dups      :", data.duplicated().sum())
    print("exact message dups  :", data["message"].duplicated().sum())
    bad_labels = set(data["label"]) - EXPECTED_LABELS
    print("unexpected labels   :", bad_labels or "none")

    # ── 2. class / language / split mix 
    banner("2. Class / language / split / source mix")
    for col in ["label", "language", "split_role", "source", "channel", "role"]:
        print(f"\n--- {col} ---")
        print(counts_with_pct(data[col]).to_string())

    # ── 3. crosstabs 
    banner("3. Crosstabs — where leakage often hides")
    print("\nlanguage x label\n",   pd.crosstab(data["language"],   data["label"], margins=True))
    print("\nsplit_role x label\n", pd.crosstab(data["split_role"], data["label"], margins=True))
    print("\nsource x label\n",     pd.crosstab(data["source"],     data["label"], margins=True))
    print("\nchannel x label\n",    pd.crosstab(data["channel"],    data["label"], margins=True))
    print("\nrole x label\n",       pd.crosstab(data["role"],       data["label"], margins=True))

    # channel and role almost perfectly predict label → never use as model features
    print("\nsource x label (row %)\n",
          (pd.crosstab(data["source"], data["label"], normalize="index") * 100).round(1))

    # ── 4. message length 
    banner("4. Message length — by label and by source")
    print("\nchars by label\n",  data.groupby("label")["chars"].describe().round(1))
    print("\nchars by source\n", data.groupby("source")["chars"].describe().round(1))
    print("\nshortest 5")
    print(data.nsmallest(5, "chars")[["id", "label", "chars", "message"]].to_string(index=False))
    print("\nlongest 5")
    print(data.nlargest(5, "chars")[["id", "label", "chars", "message"]].to_string(index=False))

    # ── 5. pair coverage 
    banner("5. Pattern / persona coverage (A/B matched pairs)")
    paired = data[data["pair_id"].fillna("").astype(str).str.match(r"^p_")]
    print("paired rows:", len(paired), "  unique pairs:", paired["pair_id"].nunique())
    print("\npersona (pairs)\n",
          counts_with_pct(paired.drop_duplicates("pair_id")["persona"]).to_string())
    print("\npattern (pairs)\n",
          counts_with_pct(paired.drop_duplicates("pair_id")["pattern"]).to_string())

    # ── 6. pair consistency 
    banner("6. Matched-pair consistency")
    problems = []
    for pid, g in paired.groupby("pair_id"):
        if len(g) != 2:
            problems.append(f"{pid}: {len(g)} rows")
        elif set(g["label"]) != {"IMPERSONATION_SCAM", "LEGIT_PERSONAL"}:
            problems.append(f"{pid}: labels {sorted(g['label'])}")
        elif g["split_role"].nunique() != 1:
            problems.append(f"{pid}: split mismatch")
        elif g["persona"].nunique() != 1 or g["pattern"].nunique() != 1:
            problems.append(f"{pid}: persona/pattern mismatch")
    print("pair problems:", problems or "none")

    # ── 7. template collapse + near-dups 
    banner("7. Template collapse + near-duplicates")
    vc = data["norm"].value_counts()
    collapsed = vc[vc > 1]
    print("unique normalised messages:", data["norm"].nunique(), "of", len(data))
    print("normalised clones:", int(collapsed.sum()) if len(collapsed) else 0)
    if len(collapsed):
        print(collapsed.head(8))

    near = []
    for label, g in data.groupby("label"):
        idx = list(g.index)[:80]
        for i, j in combinations(idx, 2):
            score = jaccard(data.at[i, "message"], data.at[j, "message"])
            if score >= 0.8 and data.at[i, "norm"] != data.at[j, "norm"]:
                near.append((round(score, 2), data.at[i, "id"], data.at[j, "id"], label))
    print("near-dup pairs (Jaccard>=0.8, first 80/class):", len(near))
    for item in near[:8]:
        print(" ", item)

    # ── 8. train vs holdout leakage 
    banner("8. Train vs holdout leakage")
    train_norms = set(data.loc[data["split_role"] == "train", "norm"])
    hold_rows   = data[data["split_role"] == "eval_holdout"]
    exact_leak  = sum(1 for n in hold_rows["norm"] if n in train_norms)
    print("exact normalised overlap train∩holdout:", exact_leak)
    near_leak = 0
    for _, h in hold_rows.iterrows():
        for _, t in data[data["split_role"] == "train"].iterrows():
            if h.get("pair_id") and h["pair_id"] == t["pair_id"]:
                continue
            if jaccard(h["message"], t["message"]) >= 0.8:
                near_leak += 1
                if near_leak <= 5:
                    print(" near leak:", h["id"], "↔", t["id"], h["label"])
                break
    print("holdout rows with Jaccard>=0.8 vs any train row:", near_leak)

    # ── 9. language tag vs script 
    banner("9. Language tag vs script")
    weak = []
    for _, r in data.iterrows():
        msg = r["message"]
        has_hi  = bool(HINGLISH_HINTS.search(msg))
        has_dev = bool(re.search(r"[\u0900-\u097F]", msg))
        if r["language"] == "Hinglish" and not has_hi:
            weak.append((r["id"], r["label"], msg[:80]))
        if r["language"] == "English" and has_dev:
            weak.append((r["id"], "english_but_devanagari", msg[:80]))
    print("weak / mismatched language tags:", len(weak))
    for item in weak[:10]:
        print(" ", item)

    # ── 10. samples per class 
    banner("10. Samples per class (read with the label book)")
    for lbl in sorted(EXPECTED_LABELS):
        print(f"\n--- {lbl} ---")
        sample = data[data["label"] == lbl].sample(
            n=min(3, (data["label"] == lbl).sum()), random_state=42
        )
        for _, r in sample.iterrows():
            print(f"  [{r['id']} | {r['language']} | {r['source']}]\n  {r['message']}")

    # ── 11. top tokens per class 
    banner("11. Top tokens per class")
    for lbl in sorted(EXPECTED_LABELS):
        print(f"\n{lbl}:", top_words(data.loc[data["label"] == lbl, "message"]))

    # ── 12. hard boundary: A vs B 
    banner("12. Hard boundary — IMPERSONATION_SCAM vs LEGIT_PERSONAL")
    a_tok = Counter(word_tokens(" ".join(data.loc[data["label"] == "IMPERSONATION_SCAM", "message"])))
    b_tok = Counter(word_tokens(" ".join(data.loc[data["label"] == "LEGIT_PERSONAL",     "message"])))
    shared = (
        {w for w, _ in a_tok.most_common(40)} &
        {w for w, _ in b_tok.most_common(40)}
    )
    print("tokens in both classes' top-40:", sorted(shared))
    print(
        "\nIf this list is mostly bhai/upi/pe: matched-pair design is working.\n"
        "Keywords overlap between classes — a TF-IDF baseline needs the CUES, not kinship."
    )

    # ── 13. EDA takeaways 
    banner("13. EDA takeaways")
    n = len(data)
    maj = data["label"].value_counts().max() / n
    print(f"  N = {n}")
    print(f"  Dummy majority-class accuracy = {maj:.1%}  ← beat this, not 50%")
    print("  channel / role / source / pattern almost perfectly predict label.")
    print("  → never use these as model features (data leakage).")
    print("  Length by label ≈ length by source — artifact, not signal.")
    print("  Next: TF-IDF + LinearSVC baseline on split_role train/holdout.")

    # ── 14. label ID mapping 
    banner("14. Label ID mapping")
   
    data = encode_labels(data)
    print("Mapping (fixed — never shuffle):")
    for name, num in LABEL_TO_ID.items():
        cnt = (data["label"] == name).sum()
        print(f"  {num}  →  {name:<24}  ({cnt} rows)")
    print("\n5-row sample showing label + label_id:")
    print(data[["message", "label", "label_id"]].sample(5, random_state=7)
          .to_string(index=False))

    # ── 15. primary metric 
    banner("15. Primary metric + CI floor")
    print(f"  Primary metric  : {PRIMARY_METRIC}")
    print(f"  Focal class     : {FOCAL_CLASS}  (id = {FOCAL_CLASS_ID})")
    print(f"  CI F1 floor     : {CI_F1_FLOOR}  (impersonation F1 must stay above this)")
    print()
    print(f"  Dummy majority-class accuracy  : {maj:.1%}")
    


if __name__ == "__main__":
    main()
