"""Error-driven post-processing fix loop for ASR predictions."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from asr_multiagent.tools.text_metrics import compute_wer
from data_pipeline.logging_config import setup_logging
from paths import ensure_repo_on_syspath
from post_processing.cleanup import CleanupConfig, clean_text, load_config as load_cleanup_config

ensure_repo_on_syspath()

log = logging.getLogger(__name__)

DEFAULT_SUBSTITUTION_MAP = {
    "हॉस्पिटल": "अस्पताल",
    "स्कूल": "विद्यालय",
    "बिज़नेस": "व्यवसाय",
    "लाइब्रेरी": "पुस्तकालय",
    "बुक्स": "किताबें",
    "स्पीच": "भाषण",
    "पीएम": "प्रधानमंत्री",
    "प्रॉब्लम": "समस्या",
    "मोबाइल": "फोन",
    "रुपए": "रुपये",
    "यहां": "यहाँ",
    "हु": "हूँ",
    "हे": "है",
}

DEFAULT_FILLER_WORDS = {"अ", "अह", "उह", "मतलब"}


@dataclass(frozen=True)
class FixRulesConfig:
    substitution_map: Dict[str, str]
    deletion_suffix_candidates: List[str]
    filler_words: List[str]


@dataclass(frozen=True)
class FixPipelineConfig:
    predictions_path: str
    error_analysis_path: str
    model_name: Optional[str]
    output_path: str
    cleanup_config_path: str
    rules: FixRulesConfig


def load_config(config_path: Path) -> FixPipelineConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    rules = raw.get("rules", {})
    return FixPipelineConfig(
        predictions_path=raw["predictions_path"],
        error_analysis_path=raw["error_analysis_path"],
        model_name=raw.get("model_name"),
        output_path=raw["output_path"],
        cleanup_config_path=raw.get("cleanup_config_path", "configs/post_processing.yaml"),
        rules=FixRulesConfig(
            substitution_map=rules.get("substitution_map", DEFAULT_SUBSTITUTION_MAP),
            deletion_suffix_candidates=rules.get("deletion_suffix_candidates", ["है", "हैं"]),
            filler_words=rules.get("filler_words", sorted(DEFAULT_FILLER_WORDS)),
        ),
    )


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def _load_predictions(path: Path, model_name: Optional[str]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if model_name:
            return list(payload[model_name])
        if len(payload) == 1:
            return list(next(iter(payload.values())))
        raise ValueError("Predictions file contains multiple models. Specify model_name.")
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("Predictions file must be a list or dict.")


def _load_error_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_error_key(key: str) -> str:
    mapping = {
        "substitute": "substitution",
        "delete": "deletion",
        "insert": "insertion",
        "english_code_mix_error": "english_code_mix_error",
        "number_error": "number_error",
    }
    return mapping.get(key, key)


def identify_most_frequent_error_type(error_payload: Dict[str, Any]) -> str:
    category_counts = {
        _normalize_error_key(key): int(value)
        for key, value in error_payload.get("summary", {}).get("category_counts", {}).items()
    }
    operation_counts = {
        _normalize_error_key(key): int(value)
        for key, value in error_payload.get("summary", {}).get("operation_counts", {}).items()
    }
    merged: Dict[str, int] = {}
    for source in (operation_counts, category_counts):
        for key, value in source.items():
            merged[key] = merged.get(key, 0) + value
    if not merged:
        return "substitution"
    dominant = max(merged.items(), key=lambda item: (item[1], item[0]))[0]
    return dominant


def _apply_substitution_fix(text: str, rules: FixRulesConfig) -> str:
    tokens = text.split()
    fixed_tokens = [rules.substitution_map.get(token, token) for token in tokens]
    fixed = " ".join(fixed_tokens)
    fixed = unicodedata.normalize("NFC", fixed)
    fixed = re.sub(r"\s+", " ", fixed).strip()
    return fixed


def _apply_deletion_fix(text: str, rules: FixRulesConfig) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return normalized
    if normalized.endswith(tuple(rules.deletion_suffix_candidates)):
        return normalized
    return f"{normalized} {rules.deletion_suffix_candidates[0]}"


def _apply_insertion_fix(text: str, rules: FixRulesConfig) -> str:
    deduped: List[str] = []
    previous = None
    filler_words = set(rules.filler_words)
    for token in text.split():
        if token == previous:
            continue
        if token in filler_words:
            continue
        deduped.append(token)
        previous = token
    return " ".join(deduped)


def _apply_number_fix(text: str, cleanup_config: CleanupConfig) -> str:
    cleaned = clean_text(text, cleanup_config, return_metadata=True)
    return str(cleaned["normalized_text"])


def _apply_code_mix_fix(text: str, rules: FixRulesConfig) -> str:
    return _apply_substitution_fix(text, rules)


def apply_targeted_fix(
    prediction: str,
    error_type: str,
    rules: FixRulesConfig,
    cleanup_config: CleanupConfig,
) -> str:
    normalized_error_type = _normalize_error_key(error_type)
    if normalized_error_type == "number_error":
        return _apply_number_fix(prediction, cleanup_config)
    if normalized_error_type == "english_code_mix_error":
        return _apply_code_mix_fix(prediction, rules)
    if normalized_error_type == "deletion":
        return _apply_deletion_fix(prediction, rules)
    if normalized_error_type == "insertion":
        return _apply_insertion_fix(prediction, rules)
    return _apply_substitution_fix(prediction, rules)


def run_fix_pipeline(config: FixPipelineConfig) -> Dict[str, Any]:
    predictions_path = _resolve_path(config.predictions_path)
    analysis_path = _resolve_path(config.error_analysis_path)
    output_path = _resolve_path(config.output_path)
    cleanup_cfg_path = _resolve_path(config.cleanup_config_path)

    predictions = _load_predictions(predictions_path, config.model_name)
    error_payload = _load_error_summary(analysis_path)
    dominant_error_type = identify_most_frequent_error_type(error_payload)
    cleanup_config = load_cleanup_config(cleanup_cfg_path) if cleanup_cfg_path.exists() else CleanupConfig()

    before_refs = [str(row["reference"]).strip() for row in predictions]
    before_hyps = [str(row["prediction"]).strip() for row in predictions]
    after_rows: List[Dict[str, Any]] = []

    for row in predictions:
        before_prediction = str(row["prediction"]).strip()
        after_prediction = apply_targeted_fix(
            before_prediction,
            dominant_error_type,
            config.rules,
            cleanup_config,
        )
        after_rows.append(
            {
                "row_index": row.get("row_index"),
                "reference": str(row["reference"]).strip(),
                "prediction_before": before_prediction,
                "prediction_after": after_prediction,
                "changed": before_prediction != after_prediction,
            }
        )

    after_hyps = [row["prediction_after"] for row in after_rows]
    before_wer = round(compute_wer(before_refs, before_hyps), 6)
    after_wer = round(compute_wer(before_refs, after_hyps), 6)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": config.model_name,
        "dominant_error_type": dominant_error_type,
        "summary": {
            "before_wer": before_wer,
            "after_wer": after_wer,
            "delta": round(before_wer - after_wer, 6),
            "changed_predictions": sum(1 for row in after_rows if row["changed"]),
            "num_predictions": len(after_rows),
        },
        "predictions": after_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    log.info("Saved fix-loop comparison to %s", output_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error-driven fix loop on ASR predictions")
    parser.add_argument("--config", type=Path, default=Path("configs/fix_pipeline.yaml"))
    args = parser.parse_args()
    setup_logging()
    payload = run_fix_pipeline(load_config(args.config))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
