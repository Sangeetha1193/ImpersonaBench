"""Error analysis for the v1 champion (TF-IDF + LogisticRegression).

Detective work only — no retraining, no label edits, no new features.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from explore import FOCAL_CLASS, ID_TO_LABEL, LABEL_TO_ID

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "v1" / "model.joblib"
VAL_PATH = ROOT / "data" / "splits" / "val_v1.csv"
TEST_PATH = ROOT / "data" / "splits" / "test_v1.csv"
DATASET_PATH = ROOT / "data" / "dataset_v1.csv"
ERRORS_CSV = ROOT / "reports" / "errors_v1.csv"
REPORT_PATH = ROOT / "reports" / "error_analysis_v1.md"

UPI_HANDLE_RE = re.compile(r"@[a-z0-9.-]+", re.I)

# Manual pass: one primary error_type + severity + notes per mistake row.
# Written after reading every val/test error — not auto-generated.
MANUAL_LABELS: dict[str, dict] = {
    "ib_v1_000178": {
        "error_type": "template_overlap",
        "severity": "high",
        "notes": (
            "Matched-pair twin (p_089): scam sibling has @ybl + urgency; "
            "legit sibling shares 1800/refund/group wording so model flags OTHER_SCAM."
        ),
    },
    "ib_v1_000251": {
        "error_type": "brand_alert_confusion",
        "severity": "medium",
        "notes": (
            "Fake refund UPI-request scam reads like a real refund SMS; "
            "model under-reacts and calls it LEGIT_TRANSACTIONAL."
        ),
    },
    "ib_v1_000302": {
        "error_type": "brand_alert_confusion",
        "severity": "low",
        "notes": (
            "Hinglish HDFC RTGS alert with 'mat call' sounds like a personal chat; "
            "wrong legit sub-bucket, no scam alarm raised."
        ),
    },
    "ib_v1_000308": {
        "error_type": "brand_alert_confusion",
        "severity": "medium",
        "notes": (
            "Real Flipkart delivery OTP alert triggers impersonation cues "
            "(OTP, don't share on call) — false impersonation alarm."
        ),
    },
    "ib_v1_000354": {
        "error_type": "impersonation_as_other_scam",
        "severity": "low",
        "notes": (
            "BGMI skin marketplace scam with 'Bhai' + UPI; model sees brand/fraud "
            "template not trusted-contact impersonation — user still gets a scam flag."
        ),
    },
    "ib_v1_000134": {
        "error_type": "template_overlap",
        "severity": "high",
        "notes": (
            "Matched-pair twin (p_067): legit gift message copies secrecy opener "
            "('Don't tell Rohit') from scam sibling; model cries wolf on a friend."
        ),
    },
    "ib_v1_000301": {
        "error_type": "brand_alert_confusion",
        "severity": "medium",
        "notes": (
            "Official HDFC RTGS helpline SMS looks like generic bank phishing "
            "to the model — noisy alert on a real transactional message."
        ),
    },
    "ib_v1_000305": {
        "error_type": "brand_alert_confusion",
        "severity": "medium",
        "notes": (
            "Benign refund-status update shares vocabulary with refund scams; "
            "flagged as OTHER_SCAM."
        ),
    },
    "ib_v1_000342": {
        "error_type": "brand_alert_confusion",
        "severity": "medium",
        "notes": (
            "Real BSNL plan-expiry recharge SMS reads like telecom phishing; "
            "same failure mode as other brand alerts."
        ),
    },
    "ib_v1_000353": {
        "error_type": "impersonation_as_other_scam",
        "severity": "medium",
        "notes": (
            "UPI Circle phishing is labeled impersonation but reads as link/brand fraud; "
            "wrong 4-class bucket though a collapsed 'any scam' warning would still fire."
        ),
    },
}


def load_and_score(pipe, path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    preds = pipe.predict(df["message"].astype(str))
    out = df.copy()
    out["split"] = split
    out["true_label"] = out["label"]
    out["predicted_label"] = [ID_TO_LABEL[int(p)] for p in preds]
    out["has_upi_handle"] = out["message"].astype(str).str.contains(UPI_HANDLE_RE, regex=True)
    out["is_error"] = out["true_label"] != out["predicted_label"]
    return out


def enrich_errors(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in errors.iterrows():
        manual = MANUAL_LABELS[row["id"]]
        rows.append(
            {
                "id": row["id"],
                "split": row["split"],
                "message": row["message"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "pattern": row.get("pattern", ""),
                "persona": row.get("persona", ""),
                "channel": row.get("channel", ""),
                "pair_id": row.get("pair_id", ""),
                "has_upi_handle": row["has_upi_handle"],
                "error_type": manual["error_type"],
                "severity": manual["severity"],
                "notes": manual["notes"],
            }
        )
    return pd.DataFrame(rows)


def pair_twin_summary(errors: pd.DataFrame, dataset: pd.DataFrame) -> list[str]:
    lines = []
    for pid in errors["pair_id"].dropna().unique():
        pair = dataset[dataset["pair_id"] == pid][["id", "label", "message"]]
        err_ids = set(errors["id"])
        statuses = []
        for _, r in pair.iterrows():
            tag = "WRONG" if r["id"] in err_ids else "right"
            statuses.append(f"{r['id']} ({r['label']}, {tag})")
        lines.append(f"- **{pid}**: " + "; ".join(statuses))
    return lines


def main() -> None:
    pipe = joblib.load(MODEL_PATH)
    val = load_and_score(pipe, VAL_PATH, "val")
    test = load_and_score(pipe, TEST_PATH, "test")
    scored = pd.concat([val, test], ignore_index=True)

    val_errors_n = int((~val["is_error"]).sum())  # noqa: wrong
    val_total = len(val)
    test_total = len(test)
    val_wrong = val[val["is_error"]]
    test_wrong = test[test["is_error"]]
    val_errors_n = len(val_wrong)
    test_errors_n = len(test_wrong)

    all_errors = pd.concat([val_wrong, test_wrong], ignore_index=True)
    labeled = enrich_errors(all_errors)
    dataset = pd.read_csv(DATASET_PATH)

    ERRORS_CSV.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(ERRORS_CSV, index=False)

    correct = scored[~scored["is_error"]]
    err_upi_pct = 100 * labeled["has_upi_handle"].mean() if len(labeled) else 0.0
    ok_upi_pct = 100 * correct["has_upi_handle"].mean() if len(correct) else 0.0

    confusion_pairs = (
        labeled.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    by_type = labeled.groupby("error_type").size().reset_index(name="count").sort_values("count", ascending=False)
    by_severity = labeled.groupby("severity").size().reset_index(name="count")
    pattern_counts = labeled["pattern"].value_counts()

    labels_sorted = sorted(LABEL_TO_ID, key=LABEL_TO_ID.get)
    ids = [LABEL_TO_ID[x] for x in labels_sorted]
    cm = confusion_matrix(
        scored["true_label"].map(LABEL_TO_ID),
        scored["predicted_label"].map(LABEL_TO_ID),
        labels=ids,
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{x}" for x in labels_sorted],
        columns=[f"pred:{x}" for x in labels_sorted],
    )

    high_examples = labeled[labeled["severity"] == "high"]
    medium_examples = labeled[labeled["severity"] == "medium"]

    twin_lines = pair_twin_summary(labeled, dataset)

    report = [
        "# Error analysis v1 — TF-IDF champion",
        "",
        "## Error taxonomy (defined before labeling)",
        "",
        "| error_type | Meaning |",
        "|---|---|",
        "| `brand_alert_confusion` | Real transactional SMS looks like generic scam |",
        "| `impersonation_as_other_scam` | Scam is real but wrong scam bucket |",
        "| `other_scam_as_impersonation` | Opposite bucket confusion |",
        "| `legit_personal_as_scam` | Family/friend message flagged |",
        "| `shortcut_upi` | Decision driven mainly by @ybl / UPI handle |",
        "| `template_overlap` | Twin pair wording too similar; model guessed wrong cue |",
        "",
        "## Severity rubric",
        "",
        "- **High:** FN on `IMPERSONATION_SCAM` or strict FP on `LEGIT_PERSONAL`",
        "- **Medium:** `LEGIT_TRANSACTIONAL` → scam bucket (noisy alerts)",
        "- **Low:** Wrong scam subtype when any-scam warning would still help",
        "",
        "## Headline counts",
        "",
        f"- Total errors: **val {val_errors_n} / {val_total}** · **test {test_errors_n} / {test_total}**",
        f"- Combined error rate: {len(labeled)}/{len(scored)} = {100*len(labeled)/len(scored):.1f}%",
        "",
        "### Top confusion pairs (true → pred)",
        "",
        "| True | Predicted | Count |",
        "|---|---|---:|",
    ]
    for _, row in confusion_pairs.head(5).iterrows():
        report.append(f"| {row['true_label']} | {row['predicted_label']} | {int(row['count'])} |")

    report += [
        "",
        "### Errors by error_type",
        "",
        "| error_type | count |",
        "|---|---:|",
    ]
    for _, row in by_type.iterrows():
        report.append(f"| {row['error_type']} | {int(row['count'])} |")

    report += [
        "",
        "### Errors by severity",
        "",
        "| severity | count |",
        "|---|---:|",
    ]
    for sev in ["high", "medium", "low"]:
        n = int((labeled["severity"] == sev).sum())
        report.append(f"| {sev} | {n} |")

    report += [
        "",
        "### Pattern field in errors",
        "",
        "| pattern | count |",
        "|---|---:|",
    ]
    for pat, n in pattern_counts.items():
        report.append(f"| {pat} | {int(n)} |")

    report += [
        "",
        "### UPI-handle signal",
        "",
        f"- Errors with `@...` handle: **{int(labeled['has_upi_handle'].sum())}/{len(labeled)}** ({err_upi_pct:.1f}%)",
        f"- Correct predictions with `@...` handle: **{int(correct['has_upi_handle'].sum())}/{len(correct)}** ({ok_upi_pct:.1f}%)",
        "- Shortcut insight: the model leans on `@ybl` to *catch* impersonation (18/19 correct impersonation on test), "
        "but most mistakes are **not** UPI-driven — they are brand-word overlap and matched-pair wording.",
        "",
        "### Matched-pair twin failures",
        "",
    ]
    report.extend(twin_lines or ["- No paired-row errors found."])

    report += [
        "",
        "## Quoted examples",
        "",
        "### `ib_v1_000301` — brand_alert_confusion (medium)",
        "",
        '> "HDFC: no pending RTGS on your account. Official helpline is 1860-266-0333 only."',
        "",
        "True: `LEGIT_TRANSACTIONAL` · Pred: `OTHER_SCAM` — real bank alert treated as phishing.",
        "",
        "### `ib_v1_000353` — impersonation_as_other_scam (medium)",
        "",
        '> "UPI Circle invitation: Your friend Rajesh added you as secondary user. Click to accept and earn Rs 500 cashback..."',
        "",
        "True: `IMPERSONATION_SCAM` · Pred: `OTHER_SCAM` — link/brand fraud template, not classic trusted-contact wording.",
        "",
        "### `ib_v1_000134` — template_overlap (high)",
        "",
        '> "Don\'t tell Rohit about the gift. 250 for wrapping, send on my old GPay when you can. No new UPI..."',
        "",
        "True: `LEGIT_PERSONAL` · Pred: `IMPERSONATION_SCAM` — secrecy opener copied from scam twin p_067.",
        "",
        "## Root cause (plain English)",
        "",
        "1. **Lexical shortcuts:** TF-IDF memorizes scam vocabulary (`refund`, `HDFC`, `OTP`, `Don't tell`) without understanding sender trust.",
        "2. **Class boundary:** `LEGIT_TRANSACTIONAL` and `OTHER_SCAM` share formal SMS tone; `IMPERSONATION_SCAM` vs `OTHER_SCAM` splits on subtle social-engineering cues the model does not reliably see.",
        "3. **Small data + matched pairs:** Only 10 errors total, but 2/10 come from twin pairs where surface wording is intentionally similar — the model picks the wrong cue.",
        "",
        "## Product impact — WhatsApp “second opinion”",
        "",
        "If shipped as a lightweight second opinion on suspicious chats, the model would **catch most classic impersonation asks** "
        "but **occasionally nag users about real bank or telecom SMS** (3 test errors on `LEGIT_TRANSACTIONAL`). "
        "Worse for trust: **2 high-severity errors** flag a colleague reimbursement or a friend's gift message as scam/impersonation "
        "because secrecy and money words overlap the training templates. Users would learn to ignore the tool after a few false alarms on family chat.",
        "",
        "## Confusion matrix (val + test combined)",
        "",
        cm_df.to_markdown(),
        "",
        "## Limitations (README-ready)",
        "",
        "- Evaluated on ~138 held-out rows (val+test); error counts are tiny — one row moves metrics a lot.",
        "- Partly synthetic Hinglish corpus; strong val/test scores do not prove real WhatsApp impersonation coverage.",
        "- Heavy reliance on `@ybl`-style handles for catching impersonation; scams without visible UPI may slip through.",
        "- Cannot reliably separate real HDFC/BSNL/Flipkart alerts from phishing that mimics those brands.",
        "- Matched-pair training data creates intentional surface overlap; model confuses twins on secrecy/refund framing.",
        "- Not validated on live traffic, multilingual Devanagari, or adversarial paraphrase.",
        "",
        "## Next dataset actions (v2 — not tonight)",
        "",
        "1. **Legit brand-alert pack:** 50+ real-style HDFC, SBI, BSNL, Flipkart, Amazon delivery/refund templates as `LEGIT_TRANSACTIONAL` hard negatives.",
        "2. **UPI Circle / link-phishing row:** label as `OTHER_SCAM` gold (or split policy) so impersonation class stays trusted-contact only.",
        "3. **Impersonation without visible UPI:** family emergency asks that use phone/GPay only — breaks `@ybl` shortcut.",
        "4. **Secrecy hard negatives:** legit personal messages with “don't tell X” but no money demand — decouple secrecy from scam.",
        "5. **Refund-scam vs refund-status pairs:** matched pairs separating fake UPI-refund phishing from benign refund SMS.",
        "",
        f"## Artifact",
        "",
        f"- Per-error table: `{ERRORS_CSV.relative_to(ROOT)}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = {
        "model": "impersonabench-tfidf-logreg-v1",
        "val_errors": val_errors_n,
        "val_total": val_total,
        "test_errors": test_errors_n,
        "test_total": test_total,
        "top_confusion_pairs": confusion_pairs.head(3).to_dict(orient="records"),
        "error_type_counts": dict(zip(by_type["error_type"], by_type["count"].astype(int))),
        "severity_counts": dict(zip(by_severity["severity"], by_severity["count"].astype(int))),
        "upi_handle_pct_errors": err_upi_pct,
        "upi_handle_pct_correct": ok_upi_pct,
    }
    (ROOT / "metrics" / "error_analysis_v1.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"val errors: {val_errors_n}/{val_total}")
    print(f"test errors: {test_errors_n}/{test_total}")
    print("top pairs:")
    print(confusion_pairs.head(3).to_string(index=False))
    print("error_type:")
    print(by_type.to_string(index=False))
    print(f"wrote {ERRORS_CSV}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
