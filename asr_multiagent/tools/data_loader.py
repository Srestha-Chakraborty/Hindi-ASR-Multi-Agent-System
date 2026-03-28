import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import librosa
import requests
import soundfile as sf


def _request_with_retry(url: str, timeout: int = 30, retries: int = 3) -> requests.Response:
    last_error: Optional[Exception] = None
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if i < retries - 1:
                time.sleep(2**i)
    raise RuntimeError(f"Failed to fetch URL after {retries} retries: {url}") from last_error


def load_manifest(manifest_path_or_url: str) -> List[Dict[str, Any]]:
    if manifest_path_or_url.startswith("http://") or manifest_path_or_url.startswith("https://"):
        return _request_with_retry(manifest_path_or_url).json()
    with open(manifest_path_or_url, "r", encoding="utf-8") as f:
        return json.load(f)


def transcription_url_for_entry(entry: Dict[str, Any]) -> str:
    if entry.get("transcription_url"):
        return entry["transcription_url"]
    return (
        f"https://storage.googleapis.com/upload_goai/"
        f"{entry['user_id']}/{entry['recording_id']}_transcription.json"
    )


def download_json(url: str) -> Dict[str, Any]:
    return _request_with_retry(url).json()


def sanitize_hindi_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^ऀ-ॿ0-9\s\.,!\?।]", "", text)
    return text.strip()


def download_and_resample_audio_to_array(audio_url: str, target_sr: int = 16000) -> Dict[str, Any]:
    resp = _request_with_retry(audio_url)
    audio_bytes = io.BytesIO(resp.content)
    audio, sr = sf.read(audio_bytes, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
    return {"array": audio, "sampling_rate": target_sr}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
