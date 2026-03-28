"""Manifest loading and URL resolution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from data_pipeline.http_utils import fetch_json

log = logging.getLogger(__name__)


def transcription_url_for_entry(entry: Dict[str, Any]) -> str:
    if entry.get("transcription_url"):
        return str(entry["transcription_url"])
    return (
        f"https://storage.googleapis.com/upload_goai/"
        f"{entry['user_id']}/{entry['recording_id']}_transcription.json"
    )


def load_manifest(manifest_path_or_url: Union[str, Path]) -> List[Dict[str, Any]]:
    path_str = str(manifest_path_or_url)
    if path_str.startswith("http://") or path_str.startswith("https://"):
        log.info("Loading manifest from URL")
        data = fetch_json(path_str, timeout=60.0, max_retries=5, backoff_sec=2.0)
        if not isinstance(data, list):
            raise ValueError("Remote manifest must be a JSON array")
        return data
    p = Path(manifest_path_or_url)
    log.info("Loading manifest from file: %s", p)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON array")
    return data
