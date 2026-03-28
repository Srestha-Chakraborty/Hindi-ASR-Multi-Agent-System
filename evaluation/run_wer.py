"""
Compute corpus-level WER from a JSONL file with `reference` and `hypothesis` keys per line.

Run:
  python -m evaluation.run_wer --input refs_hyps.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import List

from jiwer import wer

from data_pipeline.logging_config import setup_logging

log = logging.getLogger(__name__)


def load_pairs(path: Path) -> tuple[List[str], List[str]]:
    refs: List[str] = []
    hyps: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                refs.append(str(obj["reference"]))
                hyps.append(str(obj["hypothesis"]))
            except (json.JSONDecodeError, KeyError) as exc:
                log.error("Line %s: %s", line_no, exc)
                raise
    return refs, hyps


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Compute WER from JSONL (reference, hypothesis)")
    parser.add_argument("--input", type=Path, required=True, help="JSONL path")
    args = parser.parse_args()
    refs, hyps = load_pairs(args.input)
    if not refs:
        log.error("No pairs found")
        sys.exit(1)
    score = float(wer(refs, hyps))
    log.info("Utterances: %s | WER: %.4f", len(refs), score)
    print(f"WER\t{score:.4f}\tcount\t{len(refs)}")


if __name__ == "__main__":
    main()
