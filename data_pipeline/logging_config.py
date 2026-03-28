"""Central logging setup for pipeline and training CLIs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    name: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure root logging once per process. Safe to call from worker processes.
    """
    log_level = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        log = logging.getLogger(name or __name__)
        log.setLevel(log_level)
        if log_file:
            _attach_file_handler(root, log_file, log_level)
        return log

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(handler)
    if log_file:
        _attach_file_handler(root, log_file, log_level)
    root.setLevel(log_level)
    log = logging.getLogger(name or __name__)
    log.setLevel(log_level)
    return log


def _attach_file_handler(root: logging.Logger, log_file: str, log_level: int) -> None:
    target_path = Path(log_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(target_path.resolve())
    for existing in root.handlers:
        if isinstance(existing, logging.FileHandler) and getattr(existing, "baseFilename", None) == resolved:
            return
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    file_handler = logging.FileHandler(resolved, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    file_handler.setLevel(log_level)
    root.addHandler(file_handler)
