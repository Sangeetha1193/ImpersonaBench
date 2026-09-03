# Error analysis v1 — TF-IDF champion

## Error taxonomy (defined before labeling)

| error_type | Meaning |
|---|---|
| `brand_alert_confusion` | Real transactional SMS looks like generic scam |
| `impersonation_as_other_scam` | Scam is real but wrong scam bucket |
| `other_scam_as_impersonation` | Opposite bucket confusion |
| `legit_personal_as_scam` | Family/friend message flagged |
| `shortcut_upi` | Decision driven mainly by @ybl / UPI handle |
| `template_overlap` | Twin pair wording too similar; model guessed wrong cue |

## Severity rubric

- **High:** FN on `IMPERSONATION_SCAM` or strict FP on `LEGIT_PERSONAL`
- **Medium:** `LEGIT_TRANSACTIONAL` → scam bucket (noisy alerts)
- **Low:** Wrong scam subtype when any-scam warning would still help

## Headline counts

- Total errors: **val 5 / 69** · **test 5 / 69**
- Combined error rate: 10/138 = 7.2%

### Top confusion pairs (true → pred)

| True | Predicted | Count |
|---|---|---:|
| LEGIT_TRANSACTIONAL | OTHER_SCAM | 3 |
| IMPERSONATION_SCAM | OTHER_SCAM | 2 |
| LEGIT_PERSONAL | IMPERSONATION_SCAM | 1 |
| LEGIT_PERSONAL | OTHER_SCAM | 1 |
| LEGIT_TRANSACTIONAL | IMPERSONATION_SCAM | 1 |

### Errors by error_type

| error_type | count |
|---|---:|
| brand_alert_confusion | 6 |
| impersonation_as_other_scam | 2 |
| template_overlap | 2 |

### Errors by severity

| severity | count |
|---|---:|
| high | 2 |
| medium | 6 |
| low | 2 |

### Pattern field in errors

| pattern | count |
|---|---:|
| transactional | 5 |
| trusted_contact | 2 |
| return_tomorrow | 1 |
| refund | 1 |
| secrecy | 1 |

### UPI-handle signal

- Errors with `@...` handle: **1/10** (10.0%)
- Correct predictions with `@...` handle: **43/128** (33.6%)
- Shortcut insight: the model leans on `@ybl` to *catch* impersonation (18/19 correct impersonation on test), but most mistakes are **not** UPI-driven — they are brand-word overlap and matched-pair wording.

### Matched-pair twin failures

- **p_089**: ib_v1_000177 (IMPERSONATION_SCAM, right); ib_v1_000178 (LEGIT_PERSONAL, WRONG)
- **p_067**: ib_v1_000133 (IMPERSONATION_SCAM, right); ib_v1_000134 (LEGIT_PERSONAL, WRONG)

## Quoted examples

### `ib_v1_000301` — brand_alert_confusion (medium)

> "HDFC: no pending RTGS on your account. Official helpline is 1860-266-0333 only."

True: `LEGIT_TRANSACTIONAL` · Pred: `OTHER_SCAM` — real bank alert treated as phishing.

### `ib_v1_000353` — impersonation_as_other_scam (medium)

> "UPI Circle invitation: Your friend Rajesh added you as secondary user. Click to accept and earn Rs 500 cashback..."

True: `IMPERSONATION_SCAM` · Pred: `OTHER_SCAM` — link/brand fraud template, not classic trusted-contact wording.

### `ib_v1_000134` — template_overlap (high)

> "Don't tell Rohit about the gift. 250 for wrapping, send on my old GPay when you can. No new UPI..."

True: `LEGIT_PERSONAL` · Pred: `IMPERSONATION_SCAM` — secrecy opener copied from scam twin p_067.

## Root cause (plain English)

1. **Lexical shortcuts:** TF-IDF memorizes scam vocabulary (`refund`, `HDFC`, `OTP`, `Don't tell`) without understanding sender trust.
2. **Class boundary:** `LEGIT_TRANSACTIONAL` and `OTHER_SCAM` share formal SMS tone; `IMPERSONATION_SCAM` vs `OTHER_SCAM` splits on subtle social-engineering cues the model does not reliably see.
3. **Small data + matched pairs:** Only 10 errors total, but 2/10 come from twin pairs where surface wording is intentionally similar — the model picks the wrong cue.

## Product impact — WhatsApp “second opinion”

If shipped as a lightweight second opinion on suspicious chats, the model would **catch most classic impersonation asks** but **occasionally nag users about real bank or telecom SMS** (3 test errors on `LEGIT_TRANSACTIONAL`). Worse for trust: **2 high-severity errors** flag a colleague reimbursement or a friend's gift message as scam/impersonation because secrecy and money words overlap the training templates. Users would learn to ignore the tool after a few false alarms on family chat.

## Confusion matrix (val + test combined)

|                          |   pred:IMPERSONATION_SCAM |   pred:LEGIT_PERSONAL |   pred:OTHER_SCAM |   pred:LEGIT_TRANSACTIONAL |
|:-------------------------|--------------------------:|----------------------:|------------------:|---------------------------:|
| true:IMPERSONATION_SCAM  |                        37 |                     0 |                 2 |                          0 |
| true:LEGIT_PERSONAL      |                         1 |                    36 |                 1 |                          0 |
| true:OTHER_SCAM          |                         0 |                     0 |                38 |                          1 |
| true:LEGIT_TRANSACTIONAL |                         1 |                     1 |                 3 |                         17 |

## Limitations (README-ready)

- Evaluated on ~138 held-out rows (val+test); error counts are tiny — one row moves metrics a lot.
- Partly synthetic Hinglish corpus; strong val/test scores do not prove real WhatsApp impersonation coverage.
- Heavy reliance on `@ybl`-style handles for catching impersonation; scams without visible UPI may slip through.
- Cannot reliably separate real HDFC/BSNL/Flipkart alerts from phishing that mimics those brands.
- Matched-pair training data creates intentional surface overlap; model confuses twins on secrecy/refund framing.
- Not validated on live traffic, multilingual Devanagari, or adversarial paraphrase.

## Next dataset actions (v2 — not tonight)

1. **Legit brand-alert pack:** 50+ real-style HDFC, SBI, BSNL, Flipkart, Amazon delivery/refund templates as `LEGIT_TRANSACTIONAL` hard negatives.
2. **UPI Circle / link-phishing row:** label as `OTHER_SCAM` gold (or split policy) so impersonation class stays trusted-contact only.
3. **Impersonation without visible UPI:** family emergency asks that use phone/GPay only — breaks `@ybl` shortcut.
4. **Secrecy hard negatives:** legit personal messages with “don't tell X” but no money demand — decouple secrecy from scam.
5. **Refund-scam vs refund-status pairs:** matched pairs separating fake UPI-refund phishing from benign refund SMS.

## Artifact

- Per-error table: `reports\errors_v1.csv`

