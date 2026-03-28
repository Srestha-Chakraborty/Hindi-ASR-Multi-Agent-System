"""Structured ASR error analysis with alignment, categorization, and sampling."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from asr_multiagent.tools.text_metrics import AlignmentResult, AlignmentStep, align_sequences, tokenize_words
from data_pipeline.logging_config import setup_logging
from paths import ensure_repo_on_syspath

ensure_repo_on_syspath()

log = logging.getLogger(__name__)

HINDI_NUMBER_WORDS = {
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "पन्द्रह", "सोलह", "सत्रह",
    "अठारह", "उन्नीस", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी",
    "नब्बे", "सौ", "हजार", "हज़ार", "लाख", "करोड़",
}
KNOWN_CODE_MIX_WORDS = {
    "इंटरव्यू", "जॉब", "कंप्यूटर", "मोबाइल", "वीडियो", "ऑफिस", "ट्रेन", "बस",
    "टिकट", "फॉर्म", "अपडेट", "डाउनलोड", "क्लास", "स्कूल", "कॉलेज", "मैनेजर",
}


@dataclass(frozen=True)
class SamplingConfig:
    worst_n: int
    random_n: int
    random_seed: int


@dataclass(frozen=True)
class OutputConfig:
    output_path: str


@dataclass(frozen=True)
class AnalysisConfig:
    predictions_path: str
    reference_key: str
    prediction_key: str
    model_name: Optional[str]
    sampling: SamplingConfig
    output: OutputConfig


def load_config(config_path: Path) -> AnalysisConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AnalysisConfig(
        predictions_path=raw["predictions_path"],
        reference_key=raw.get("reference_key", "reference"),
        prediction_key=raw.get("prediction_key", "prediction"),
        model_name=raw.get("model_name"),
        sampling=SamplingConfig(**raw["sampling"]),
        output=OutputConfig(**raw["output"]),
    )


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def _load_prediction_rows(config: AnalysisConfig) -> List[Dict[str, Any]]:
    predictions_path = _resolve_path(config.predictions_path)
    with predictions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if config.model_name:
            if config.model_name not in payload:
                raise KeyError(f"Model '{config.model_name}' not found in predictions file.")
            rows = payload[config.model_name]
        elif len(payload) == 1:
            rows = next(iter(payload.values()))
        else:
            raise ValueError("Predictions file contains multiple models. Set model_name in config.")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("Predictions payload must be a list or object.")

    normalized_rows = []
    for idx, row in enumerate(rows):
        normalized_rows.append(
            {
                "row_index": row.get("row_index", idx),
                "reference": str(row[config.reference_key]).strip(),
                "prediction": str(row[config.prediction_key]).strip(),
            }
        )
    return normalized_rows


def _contains_number(token: str) -> bool:
    return any(ch.isdigit() for ch in token) or token in HINDI_NUMBER_WORDS


def _is_code_mix(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token)) or token in KNOWN_CODE_MIX_WORDS


def _categorize_step(step: AlignmentStep) -> List[str]:
    categories = [step.operation]
    if _contains_number(step.reference_token) or _contains_number(step.hypothesis_token):
        categories.append("number_error")
    if _is_code_mix(step.reference_token) or _is_code_mix(step.hypothesis_token):
        categories.append("english_code_mix_error")
    return categories


def _serialize_alignment(alignment: AlignmentResult) -> List[Dict[str, Any]]:
    return [
        {
            "operation": step.operation,
            "reference_token": step.reference_token,
            "hypothesis_token": step.hypothesis_token,
            "reference_index": step.reference_index,
            "hypothesis_index": step.hypothesis_index,
            "categories": _categorize_step(step),
        }
        for step in alignment.steps
        if step.operation != "equal"
    ]


def _analyze_row(row: Dict[str, Any]) -> Dict[str, Any]:
    alignment = align_sequences(
        tokenize_words(row["reference"]),
        tokenize_words(row["prediction"]),
    )
    errors = _serialize_alignment(alignment)
    category_counter: Counter[str] = Counter()
    for error in errors:
        category_counter.update(error["categories"])

    return {
        "row_index": row["row_index"],
        "reference": row["reference"],
        "prediction": row["prediction"],
        "wer": round(alignment.word_error_rate, 6),
        "total_errors": alignment.total_errors,
        "substitutions": alignment.substitutions,
        "deletions": alignment.deletions,
        "insertions": alignment.insertions,
        "categories": dict(category_counter),
        "errors": errors,
    }


def _sample_rows(
    analyzed_rows: List[Dict[str, Any]],
    sampling: SamplingConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    ranked = sorted(
        analyzed_rows,
        key=lambda row: (row["wer"], row["total_errors"], len(row["reference"])),
        reverse=True,
    )
    worst_rows = ranked[: sampling.worst_n]

    rng = random.Random(sampling.random_seed)
    pool = [row for row in analyzed_rows if row not in worst_rows]
    random_rows = rng.sample(pool, k=min(sampling.random_n, len(pool))) if pool else []
    return {"worst_errors": worst_rows, "random_samples": random_rows}


def analyze_errors(config: AnalysisConfig) -> Dict[str, Any]:
    rows = _load_prediction_rows(config)
    analyzed_rows = [_analyze_row(row) for row in rows]
    samples = _sample_rows(analyzed_rows, config.sampling)

    aggregate = Counter()
    operation_counts = Counter()
    for row in analyzed_rows:
        aggregate.update(row["categories"])
        operation_counts.update(
            {
                "substitution": row["substitutions"],
                "deletion": row["deletions"],
                "insertion": row["insertions"],
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_predictions_path": str(_resolve_path(config.predictions_path)),
        "model_name": config.model_name,
        "num_rows": len(analyzed_rows),
        "summary": {
            "operation_counts": dict(operation_counts),
            "category_counts": dict(aggregate),
        },
        "samples": samples,
        "utterances": analyzed_rows,
    }

    output_path = _resolve_path(config.output.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    log.info("Saved error analysis to %s", output_path)
    return payload


def _render_summary(summary: Dict[str, Any]) -> str:
    operation_counts = summary["operation_counts"]
    category_counts = summary["category_counts"]
    lines = [
        "Error Summary",
        f"  substitutions: {operation_counts.get('substitution', 0)}",
        f"  deletions: {operation_counts.get('deletion', 0)}",
        f"  insertions: {operation_counts.get('insertion', 0)}",
        f"  number_error: {category_counts.get('number_error', 0)}",
        f"  english_code_mix_error: {category_counts.get('english_code_mix_error', 0)}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ASR prediction errors")
    parser.add_argument("--config", type=Path, default=Path("configs/error_analysis.yaml"))
    parser.add_argument("--predictions_path", type=str, default=None, help="Optional override for predictions JSON path")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    if args.predictions_path:
        config = AnalysisConfig(
            predictions_path=args.predictions_path,
            reference_key=config.reference_key,
            prediction_key=config.prediction_key,
            model_name=config.model_name,
            sampling=config.sampling,
            output=config.output,
        )
    payload = analyze_errors(config)
    print(_render_summary(payload["summary"]))


if __name__ == "__main__":
    main()
