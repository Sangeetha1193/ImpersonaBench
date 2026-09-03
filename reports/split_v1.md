# Split v1 report

## Rules

1. **Split unit = group_id**, not row. Paired twins always stay together.
2. **Ratios:** train 70% / val 15% / test 15%, seed=42.
3. **Strata rule:** Paired rows (120 groups with pair_id p_*) share strata pair_impersonation; each unpaired row is its own group with strata equal to its label.
4. **Smallest class note:** LEGIT_TRANSACTIONAL (~72 rows) → test may have ~10–15. F1 will be noisy.
5. **Leakage:** group_id overlap must be 0; message overlap must be 0.

## Group inventory (before split)

- Total rows: 457
- Paired groups (size 2): 120 → 240 rows
- Unpaired groups (size 1): 217 → 217 rows
- Total groups: 337

### Unpaired row breakdown
- OTHER_SCAM: 130
- LEGIT_TRANSACTIONAL: 72
- IMPERSONATION_SCAM: 10
- LEGIT_PERSONAL: 5

## Results

| Split | Rows | Groups |
|-------|-----:|-------:|
| train | 319 | 235 |
| val | 69 | 51 |
| test | 69 | 51 |

**group_id overlap across splits:** 0 (must be 0)

### Label counts per split

**train:**
- IMPERSONATION_SCAM: 91
- LEGIT_PERSONAL: 87
- LEGIT_TRANSACTIONAL: 50
- OTHER_SCAM: 91

**val:**
- IMPERSONATION_SCAM: 19
- LEGIT_PERSONAL: 19
- LEGIT_TRANSACTIONAL: 11
- OTHER_SCAM: 20

**test:**
- IMPERSONATION_SCAM: 20
- LEGIT_PERSONAL: 19
- LEGIT_TRANSACTIONAL: 11
- OTHER_SCAM: 19

### Top patterns per split

**train:** {'transactional': 50, 'medical': 32, 'phone_broken': 30, 'borrowed_phone': 26, 'fees': 24}

**val:** {'transactional': 11, 'phone_broken': 8, 'secrecy': 8, 'borrowed_phone': 6, 'stranded_friend': 4}

**test:** {'return_tomorrow': 12, 'transactional': 11, 'phone_broken': 8, 'fees': 4, 'stranded_friend': 4}

### Leakage checks: all passed

## Strata rule (one sentence)

Paired rows (120 groups with pair_id p_*) share strata pair_impersonation; each unpaired row is its own group with strata equal to its label.