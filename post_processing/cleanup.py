"""Production-ready text cleanup pipeline for Hindi ASR output."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml

from data_pipeline.logging_config import setup_logging
from paths import ensure_repo_on_syspath

ensure_repo_on_syspath()

log = logging.getLogger(__name__)

NUMBER_WORDS: Dict[str, int] = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13,
    "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18,
    "उन्नीस": 19, "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेइस": 23, "तेईस": 23,
    "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28,
    "उनतीस": 29, "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38,
    "उनतालीस": 39, "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43,
    "चवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48,
    "उनचास": 49, "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66,
    "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70, "इकहत्तर": 71,
    "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75, "छिहत्तर": 76,
    "सतहत्तर": 77, "अठहत्तर": 78, "उन्यासी": 79, "अस्सी": 80, "इक्यासी": 81,
    "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86,
    "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90, "इक्यानवे": 91,
    "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95, "छियानवे": 96,
    "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}
SCALE_WORDS: Dict[str, int] = {
    "सौ": 100,
    "हजार": 1000,
    "हज़ार": 1000,
    "लाख": 100000,
    "करोड़": 10000000,
}
KNOWN_ENGLISH_WORDS = {
    "interview", "job", "computer", "mobile", "video", "office", "train", "bus",
    "ticket", "form", "update", "download", "class", "school", "college", "manager",
    "इंटरव्यू", "जॉब", "कंप्यूटर", "मोबाइल", "वीडियो", "ऑफिस", "ट्रेन", "बस",
    "टिकट", "फॉर्म", "अपडेट", "डाउनलोड", "क्लास", "स्कूल", "कॉलेज", "मैनेजर",
}
DEFAULT_IDIOMS = {
    "दो-चार", "चार चाँद", "सात समंदर", "तीन तेरह", "दो टूक", "चार सौ बीस",
    "एक दो", "दो चार", "सात खून माफ", "नौ दो ग्यारह", "तीन पाँच",
}


@dataclass(frozen=True)
class CleanupConfig:
    idioms: Sequence[str] = field(default_factory=lambda: sorted(DEFAULT_IDIOMS))
    known_english_words: Sequence[str] = field(default_factory=lambda: sorted(KNOWN_ENGLISH_WORDS))
    tag_format: str = "[EN]{token}[/EN]"


def load_config(config_path: Path) -> CleanupConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return CleanupConfig(
        idioms=raw.get("idioms", sorted(DEFAULT_IDIOMS)),
        known_english_words=raw.get("known_english_words", sorted(KNOWN_ENGLISH_WORDS)),
        tag_format=raw.get("tag_format", "[EN]{token}[/EN]"),
    )


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _tokenize_preserving_separators(text: str) -> List[str]:
    return re.split(r"(\s+)", text)


def _split_punctuation(token: str) -> Tuple[str, str, str]:
    match = re.match(r"^([^\w\u0900-\u097F]*)([\w\u0900-\u097F-]+)([^\w\u0900-\u097F]*)$", token)
    if not match:
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


def _looks_like_number_token(token: str) -> bool:
    return token in NUMBER_WORDS or token in SCALE_WORDS or token.isdigit()


def _match_idiom(tokens: Sequence[str], start_idx: int, idioms: Iterable[str]) -> int | None:
    normalized_tokens = [_split_punctuation(token)[1] for token in tokens]
    for end_idx in range(min(len(tokens), start_idx + 4), start_idx, -1):
        span = " ".join(normalized_tokens[start_idx:end_idx]).strip()
        if span in idioms:
            return end_idx
    return None


def _parse_number_phrase(tokens: Sequence[str], start_idx: int) -> Tuple[int | None, int]:
    total = 0
    current = 0
    idx = start_idx
    matched = False

    while idx < len(tokens):
        token = tokens[idx]
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            matched = True
            idx += 1
            continue
        if token == "और":
            idx += 1
            continue
        if token in SCALE_WORDS:
            matched = True
            scale = SCALE_WORDS[token]
            if scale == 100:
                current = max(1, current) * scale
            else:
                total += max(1, current) * scale
                current = 0
            idx += 1
            continue
        break

    if not matched:
        return None, start_idx
    return total + current, idx


def normalize_numbers(
    text: str,
    config: CleanupConfig | None = None,
    *,
    return_metadata: bool = False,
) -> str | Tuple[str, List[Dict[str, str]]]:
    config = config or CleanupConfig()
    normalized_text = _normalize_whitespace(text)
    if not normalized_text:
        return ("", []) if return_metadata else ""
    parts = _tokenize_preserving_separators(normalized_text)
    word_tokens = [token for token in parts if token and not token.isspace()]
    metadata: List[Dict[str, str]] = []
    rebuilt_words: List[str] = []
    idx = 0
    while idx < len(word_tokens):
        idiom_end = _match_idiom(word_tokens, idx, config.idioms)
        if idiom_end is not None:
            rebuilt_words.extend(word_tokens[idx:idiom_end])
            idx = idiom_end
            continue

        original_token = word_tokens[idx]
        prefix, token, suffix = _split_punctuation(original_token)
        if "-" in token or not _looks_like_number_token(token):
            rebuilt_words.append(original_token)
            idx += 1
            continue

        core_tokens: List[str] = []
        end_scan = idx
        while end_scan < len(word_tokens):
            inner_prefix, inner_token, inner_suffix = _split_punctuation(word_tokens[end_scan])
            if inner_prefix or inner_suffix:
                if inner_token:
                    core_tokens.append(inner_token)
                break
            core_tokens.append(inner_token)
            end_scan += 1

        number_value, end_idx_offset = _parse_number_phrase(core_tokens, 0)
        end_idx = idx + end_idx_offset
        if number_value is None or end_idx == idx:
            rebuilt_words.append(original_token)
            idx += 1
            continue

        source_phrase = " ".join(word_tokens[idx:end_idx])
        if source_phrase in config.idioms:
            rebuilt_words.append(source_phrase)
            idx = end_idx
            continue
        rebuilt_words.append(f"{prefix}{number_value}{suffix}")
        metadata.append({"source": source_phrase, "normalized": str(number_value)})
        idx = end_idx

    normalized = " ".join(rebuilt_words)
    return (normalized, metadata) if return_metadata else normalized


def detect_english_words(
    text: str,
    config: CleanupConfig | None = None,
    *,
    return_metadata: bool = False,
) -> str | Tuple[str, List[str]]:
    config = config or CleanupConfig()
    known_words = set(config.known_english_words)
    tokens = text.split()
    tagged_tokens: List[str] = []
    tagged_words: List[str] = []

    for token in tokens:
        stripped = token.strip()
        is_english = bool(re.search(r"[A-Za-z]", stripped)) or stripped in known_words
        if is_english:
            tagged_tokens.append(config.tag_format.format(token=stripped))
            tagged_words.append(stripped)
        else:
            tagged_tokens.append(stripped)

    tagged_text = " ".join(tagged_tokens)
    return (tagged_text, tagged_words) if return_metadata else tagged_text


def clean_text(
    text: str,
    config: CleanupConfig | None = None,
    *,
    return_metadata: bool = False,
) -> str | Dict[str, object]:
    config = config or CleanupConfig()
    whitespace_normalized = _normalize_whitespace(text)
    number_normalized, number_metadata = normalize_numbers(
        whitespace_normalized,
        config,
        return_metadata=True,
    )
    english_tagged, english_words = detect_english_words(
        number_normalized,
        config,
        return_metadata=True,
    )
    if not return_metadata:
        return english_tagged
    return {
        "input_text": text,
        "normalized_text": number_normalized,
        "cleaned_text": english_tagged,
        "number_conversions": number_metadata,
        "english_words": english_words,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hindi ASR text cleanup")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/post_processing.yaml"))
    parser.add_argument("--json", action="store_true", help="Print full metadata as JSON")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config) if args.config.exists() else CleanupConfig()
    result = clean_text(args.text, config, return_metadata=args.json)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
