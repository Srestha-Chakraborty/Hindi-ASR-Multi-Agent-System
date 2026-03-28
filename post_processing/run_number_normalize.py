"""
Normalize Hindi numerals in text (CLI wrapper around existing normalizer).

Run:
  python -m post_processing.run_number_normalize --text "मैंने दो किताबें खरीदीं"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.logging_config import setup_logging
from asr_multiagent.tools.number_normalizer import normalize_numbers_in_text

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    args = p.parse_args()
    norm, reasons = normalize_numbers_in_text(args.text)
    log.info("normalized=%r reasons=%s", norm, reasons)
    print(norm)


if __name__ == "__main__":
    main()
