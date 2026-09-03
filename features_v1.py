"""Hand-made text features for ImpersonaBench v1.

All feature functions take one message string and return numbers only.
Nothing here is fitted on train/val/test. The word lists and regexes are static.
That makes the logic reusable later at inference time.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

UPI_HANDLE_RE = re.compile(r"@[a-z0-9.-]+", re.I)
PHONE_RE = re.compile(r"\b\d{10}\b")
AMOUNT_RE = re.compile(r"(₹|\brs\b|\brs\.\b|\b\d{2,}00\b|\b\d{3,}\b)", re.I)

URGENCY_WORDS = [
    "abhi", "jaldi", "turant", "urgent", "urgently", "immediately",
    "aaj", "now", "asap", "quick", "quickly", "last date", "last hour",
]
SECRECY_PHRASES = [
    "mat batana", "kisi ko mat", "don't tell", "dont tell",
    "delete this chat", "group mein mat", "screenshot mat",
]
VERIFY_BLOCK_PHRASES = [
    "call mat karna", "mat call", "don't call", "dont call",
    "battery nahi", "phone toot", "screen toot", "phone dead",
    "new number", "naya number", "borrowed phone", "temporary number",
]
LATER_EXPLAIN_PHRASES = [
    "baad mein", "later i'll explain", "later i will explain",
    "samjhaunga", "samjhaungi", "call later", "baad me bataunga",
]


def _count_matches(text: str, phrases: Iterable[str]) -> int:
    lowered = str(text).lower()
    return sum(1 for phrase in phrases if phrase in lowered)


def _caps_ratio(text: str) -> float:
    letters = [ch for ch in str(text) if ch.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for ch in letters if ch.isupper())
    return caps / len(letters)


def extract_feature_row(text: str) -> dict[str, float]:
    text = str(text)
    return {
        "has_upi_handle": float(bool(UPI_HANDLE_RE.search(text))),
        "phone_number_present": float(bool(PHONE_RE.search(text))),
        "urgency_count": float(_count_matches(text, URGENCY_WORDS)),
        "secrecy_count": float(_count_matches(text, SECRECY_PHRASES)),
        "verify_block_count": float(_count_matches(text, VERIFY_BLOCK_PHRASES)),
        "money_amount_present": float(bool(AMOUNT_RE.search(text))),
        "message_length_chars": float(len(text)),
        "exclamation_count": float(text.count("!")),
        "caps_ratio": float(_caps_ratio(text)),
        "later_explain_count": float(_count_matches(text, LATER_EXPLAIN_PHRASES)),
    }


def build_feature_frame(messages: pd.Series) -> pd.DataFrame:
    rows = [extract_feature_row(msg) for msg in messages.astype(str)]
    return pd.DataFrame(rows, index=messages.index)
