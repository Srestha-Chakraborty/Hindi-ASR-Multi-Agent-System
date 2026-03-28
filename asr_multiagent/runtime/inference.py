"""Production inference runtime with caching, cleanup, and confidence scoring."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import yaml

from analysis.fix_pipeline import (
    DEFAULT_SUBSTITUTION_MAP,
    FixRulesConfig,
    _apply_code_mix_fix,
    _apply_deletion_fix,
    _apply_insertion_fix,
    _apply_substitution_fix,
    identify_most_frequent_error_type,
)
from asr_multiagent.runtime.cache import DiskJSONCache, hash_bytes, hash_json
from data_pipeline.logging_config import setup_logging
from paths import ensure_repo_on_syspath
from post_processing.cleanup import CleanupConfig, clean_text, load_config as load_cleanup_config

ensure_repo_on_syspath()

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    model_name_or_path: str
    language: str
    task: str
    cache_dir: str
    cleanup_config_path: str
    error_analysis_path: Optional[str]
    enable_cache: bool
    enable_cleanup: bool
    enable_error_aware_postprocessing: bool
    compute_confidence: bool
    device: str


def load_config(config_path: Path) -> InferenceConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return InferenceConfig(
        model_name_or_path=raw.get("model_name_or_path", "openai/whisper-small"),
        language=raw.get("language", "hi"),
        task=raw.get("task", "transcribe"),
        cache_dir=raw.get("cache_dir", "outputs/cache/inference"),
        cleanup_config_path=raw.get("cleanup_config_path", "configs/post_processing.yaml"),
        error_analysis_path=raw.get("error_analysis_path"),
        enable_cache=bool(raw.get("enable_cache", True)),
        enable_cleanup=bool(raw.get("enable_cleanup", True)),
        enable_error_aware_postprocessing=bool(raw.get("enable_error_aware_postprocessing", True)),
        compute_confidence=bool(raw.get("compute_confidence", True)),
        device=str(raw.get("device", "auto")),
    )


def _resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@lru_cache(maxsize=4)
def _load_model_bundle(model_name_or_path: str, language: str, task: str, device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    resolved = _resolve_path(model_name_or_path)
    model_path = str(resolved) if resolved and resolved.exists() else model_name_or_path
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    torch_device = _resolve_device(device)
    model.to(torch_device)
    model.eval()
    return processor, model, torch_device, model_path


def _resample_audio(audio: np.ndarray, original_sr: int, target_sr: int = 16000) -> np.ndarray:
    if original_sr == target_sr:
        return audio.astype(np.float32)
    import librosa

    return librosa.resample(audio.astype(np.float32), orig_sr=original_sr, target_sr=target_sr)


def _load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = _resample_audio(np.asarray(audio, dtype=np.float32), sample_rate, target_sr=16000)
    return audio, 16000


def _compute_confidence(generation_output) -> Optional[float]:
    scores = getattr(generation_output, "scores", None)
    if not scores:
        return None
    confidences = []
    for score_tensor in scores:
        probabilities = torch.softmax(score_tensor, dim=-1)
        confidences.append(float(torch.max(probabilities, dim=-1).values.mean().item()))
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 6)


def _load_dominant_error_type(error_analysis_path: Optional[str]) -> Optional[str]:
    resolved = _resolve_path(error_analysis_path)
    if not resolved or not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return identify_most_frequent_error_type(payload)


def _apply_error_aware_postprocessing(
    text: str,
    dominant_error_type: Optional[str],
    cleanup_config: CleanupConfig,
) -> str:
    if not dominant_error_type:
        return text
    rules = FixRulesConfig(
        substitution_map=DEFAULT_SUBSTITUTION_MAP,
        deletion_suffix_candidates=["है", "हैं"],
        filler_words=["अ", "अह", "उह", "मतलब"],
    )
    normalized_type = dominant_error_type.lower()
    if normalized_type == "english_code_mix_error":
        fixed = _apply_code_mix_fix(text, rules)
    elif normalized_type == "number_error":
        fixed = str(clean_text(text, cleanup_config, return_metadata=True)["normalized_text"])
    elif normalized_type == "deletion":
        fixed = _apply_deletion_fix(text, rules)
    elif normalized_type == "insertion":
        fixed = _apply_insertion_fix(text, rules)
    else:
        fixed = _apply_substitution_fix(text, rules)
    return fixed


def transcribe_audio_bytes(audio_bytes: bytes, config: InferenceConfig) -> Dict[str, Any]:
    setup_logging(log_file="outputs/logs/inference.log")
    cache = DiskJSONCache(config.cache_dir)
    cleanup_config_path = _resolve_path(config.cleanup_config_path)
    cleanup_config = load_cleanup_config(cleanup_config_path) if cleanup_config_path and cleanup_config_path.exists() else CleanupConfig()
    dominant_error_type = _load_dominant_error_type(config.error_analysis_path)

    cache_key = hash_json(
        {
            "audio_sha": hash_bytes(audio_bytes),
            "model_name_or_path": config.model_name_or_path,
            "enable_cleanup": config.enable_cleanup,
            "enable_error_aware_postprocessing": config.enable_error_aware_postprocessing,
            "dominant_error_type": dominant_error_type,
        }
    )
    if config.enable_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            log.info("Inference cache hit for key=%s", cache_key)
            return cached

    audio_array, sample_rate = _load_audio_from_bytes(audio_bytes)
    processor, model, torch_device, resolved_model_path = _load_model_bundle(
        config.model_name_or_path,
        config.language,
        config.task,
        config.device,
    )
    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(torch_device)

    with torch.inference_mode():
        generated = model.generate(
            input_features,
            language=config.language,
            task=config.task,
            return_dict_in_generate=True,
            output_scores=config.compute_confidence,
        )
    raw_text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
    confidence = _compute_confidence(generated) if config.compute_confidence else None

    cleaned_text = raw_text
    cleanup_metadata: Dict[str, Any] = {
        "normalized_text": raw_text,
        "cleaned_text": raw_text,
        "english_words": [],
        "number_conversions": [],
    }
    if config.enable_cleanup:
        cleanup_metadata = dict(clean_text(raw_text, cleanup_config, return_metadata=True))
        cleaned_text = str(cleanup_metadata["cleaned_text"])
    if config.enable_error_aware_postprocessing:
        cleaned_text = _apply_error_aware_postprocessing(cleaned_text, dominant_error_type, cleanup_config)

    result = {
        "model_name_or_path": resolved_model_path,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "normalized_text": cleanup_metadata["normalized_text"],
        "english_words": cleanup_metadata["english_words"],
        "number_conversions": cleanup_metadata["number_conversions"],
        "confidence": confidence,
        "cache_key": cache_key,
        "dominant_error_type": dominant_error_type,
    }
    if config.enable_cache:
        cache.set(cache_key, result)
    log.info(
        "Completed inference | confidence=%s | cleaned_changed=%s",
        confidence,
        cleaned_text != raw_text,
    )
    return result


def transcribe_audio_file(audio_path: Path, config: InferenceConfig) -> Dict[str, Any]:
    return transcribe_audio_bytes(audio_path.read_bytes(), config)
