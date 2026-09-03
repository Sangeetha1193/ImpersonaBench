# Feature run v1

## Feature list

- `has_upi_handle`: direct payee handle like `@ybl`; scam asks often point to a target UPI
- `phone_number_present`: raw 10-digit number; new contact number is an impersonation cue
- `urgency_count`: words like abhi / jaldi / urgent; measures panic pressure
- `secrecy_count`: phrases like mat batana / don't tell; classic social engineering
- `verify_block_count`: don't call / phone dead / new number; blocks normal verification
- `money_amount_present`: ₹ / rs / large numbers; scam asks often contain a concrete demand
- `message_length_chars`: longer story-like messages can look more scammy
- `exclamation_count`: panic tone marker
- `caps_ratio`: shouting / emphasis marker
- `later_explain_count`: baad mein / later I'll explain; delay-accountability cue

## Word lists used

- Urgency words: abhi, jaldi, turant, urgent, urgently, immediately, aaj, now, asap, quick, quickly, last date, last hour
- Secrecy phrases: mat batana, kisi ko mat, don't tell, dont tell, delete this chat, group mein mat, screenshot mat
- Verify-block phrases: call mat karna, mat call, don't call, dont call, battery nahi, phone toot, screen toot, phone dead, new number, naya number, borrowed phone, temporary number
- Later-explain phrases: baad mein, later i'll explain, later i will explain, samjhaunga, samjhaungi, call later, baad me bataunga

## How features were combined

- TF-IDF stayed sparse
- Engineered features are dense numeric columns
- Combined with `scipy.sparse.hstack` into one train matrix
- Only TF-IDF was fit on train; hand-made features were static rules on both train and val

## Old vs new (train / val)

| Model | Train macro-F1 | Val macro-F1 | Train impersonation F1 | Val impersonation F1 |
|---|---:|---:|---:|---:|
| Old TF-IDF baseline | 0.9972 | 0.9197 | 1.0000 | 0.9474 |
| TF-IDF + features | 0.9426 | 0.8767 | 0.9677 | 0.9048 |

- Old train-val macro-F1 gap: 0.0775
- New train-val macro-F1 gap: 0.0659
- Gap narrowed: yes

## Which features hurt?

- `exclamation_count` hurt a bit: full val macro-F1 = 0.8767, removing it = 0.8952
- `caps_ratio` hurt a bit: full val macro-F1 = 0.8767, removing it = 0.8952

## Shortcut still present

The biggest shortcut the model still has is that a visible UPI handle like @ybl strongly pushes it toward a scam prediction.

## Verdict

Small improvements count here. With only 319 training rows, a +0.01 or +0.02 macro-F1 gain is real.
The goal is not to beat transformers yet; it is to learn how to add features carefully and measure whether they help.
