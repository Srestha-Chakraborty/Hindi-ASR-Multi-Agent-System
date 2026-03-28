"""Unicode-safe transcript normalization for Hindi ASR."""

from __future__ import annotations

import re
import unicodedata


def normalize_transcript(text: str) -> str:
    """
    NFC normalization + whitespace collapse + Devanagari-safe character filter.
    Mirrors production expectations used elsewhere in this repo.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^ऀ-ॿ0-9\s\.,!\?।]", "", text)
    return text.strip()
