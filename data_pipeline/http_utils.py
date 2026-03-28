"""HTTP helpers with retries (used by preprocess workers)."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)


def fetch_json(url: str, *, timeout: float, max_retries: int, backoff_sec: float) -> Dict[str, Any]:
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("fetch_json failed attempt %s/%s url=%s err=%s", attempt + 1, max_retries, url, exc)
            if attempt < max_retries - 1:
                time.sleep(backoff_sec * (2**attempt))
    raise RuntimeError(f"Failed to fetch JSON after {max_retries} retries: {url}") from last


def fetch_bytes(url: str, *, timeout: float, max_retries: int, backoff_sec: float) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("fetch_bytes failed attempt %s/%s url=%s err=%s", attempt + 1, max_retries, url, exc)
            if attempt < max_retries - 1:
                time.sleep(backoff_sec * (2**attempt))
    raise RuntimeError(f"Failed to fetch bytes after {max_retries} retries: {url}") from last
