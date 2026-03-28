"""Audio validation and conversion to 16 kHz mono WAV."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


def decode_resample_to_mono(
    audio_bytes: bytes,
    *,
    target_sr: int,
) -> Tuple[np.ndarray, int]:
    """Decode arbitrary input bytes to float32 mono at target_sr."""
    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def validate_audio_array(
    audio: np.ndarray,
    *,
    sample_rate: int,
    min_duration_sec: float,
    max_duration_sec: float,
) -> Tuple[bool, str]:
    if audio.size == 0:
        return False, "empty_audio"
    if not np.isfinite(audio).all():
        return False, "non_finite_samples"
    dur = float(len(audio)) / float(sample_rate)
    if dur < min_duration_sec:
        return False, f"too_short:{dur:.3f}s"
    if dur > max_duration_sec:
        return False, f"too_long:{dur:.3f}s"
    return True, "ok"


def write_wav_16bit_mono(path: Path, audio_f32: np.ndarray, sample_rate: int) -> None:
    """Write PCM_16 mono WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    sf.write(str(path), pcm, sample_rate, subtype="PCM_16")


def load_audio_dict_for_hf(wav_path: Path, target_sr: int) -> Dict[str, Any]:
    """Structure compatible with datasets.Audio casting."""
    audio, sr = sf.read(str(wav_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return {"array": audio, "sampling_rate": sr}
