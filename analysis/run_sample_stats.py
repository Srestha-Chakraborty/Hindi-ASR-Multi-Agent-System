"""
Summarize a processed_manifest.json (sentence length, duration).

Run:
  python -m analysis.run_sample_stats --manifest data/processed/processed_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.logging_config import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not rows:
        log.warning("Empty manifest")
        return
    durs = [float(r["duration_sec"]) for r in rows]
    lens = [len(r.get("sentence", "")) for r in rows]
    log.info(
        "n=%s duration_sec mean=%.3f stdev=%.3f char_len mean=%.1f",
        len(rows),
        statistics.mean(durs),
        statistics.pstdev(durs) if len(durs) > 1 else 0.0,
        statistics.mean(lens),
    )


if __name__ == "__main__":
    main()
