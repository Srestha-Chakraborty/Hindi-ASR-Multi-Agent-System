"""
Parallel manifest-driven download, validation, and 16 kHz mono WAV export.

Run:
  python -m data_pipeline.preprocess --config configs/preprocess.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from data_pipeline.audio_io import (
    decode_resample_to_mono,
    validate_audio_array,
    write_wav_16bit_mono,
)
from data_pipeline.http_utils import fetch_bytes, fetch_json
from data_pipeline.logging_config import setup_logging
from data_pipeline.manifest_io import load_manifest, transcription_url_for_entry
from data_pipeline.text_normalize import normalize_transcript

log = logging.getLogger(__name__)


def _language_ok(lang: str, allowed: List[str]) -> bool:
    return lang.strip().lower() in {a.lower() for a in allowed}


def process_manifest_entry(payload: Tuple[Dict[str, Any], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Worker: download transcript + audio, validate, write WAV.
    Returns a manifest row dict or None on skip/failure.
    """
    entry, cfg = payload
    setup_logging(cfg.get("log_level", "INFO"))
    wlog = logging.getLogger(__name__)

    rid = str(entry.get("recording_id", "unknown"))
    try:
        dur = float(entry.get("duration", 0) or 0)
        if dur < cfg["min_duration_sec"] or dur > cfg["max_duration_sec"]:
            wlog.debug("skip %s duration filter %s", rid, dur)
            return None

        lang = str(entry.get("language", ""))
        if not _language_ok(lang, cfg["languages"]):
            wlog.debug("skip %s language %s", rid, lang)
            return None

        turl = transcription_url_for_entry(entry)
        tjson = fetch_json(
            turl,
            timeout=cfg["timeout_sec"],
            max_retries=cfg["max_retries"],
            backoff_sec=cfg["backoff_sec"],
        )
        raw_text = tjson.get("transcription", tjson.get("text", ""))
        sentence = normalize_transcript(str(raw_text))
        if not sentence:
            wlog.warning("skip %s empty transcript", rid)
            return None

        audio_url = str(entry["rec_url_gcp"])
        audio_bytes = fetch_bytes(
            audio_url,
            timeout=cfg["timeout_sec"],
            max_retries=cfg["max_retries"],
            backoff_sec=cfg["backoff_sec"],
        )
        audio, sr = decode_resample_to_mono(audio_bytes, target_sr=cfg["target_sr"])
        ok, reason = validate_audio_array(
            audio,
            sample_rate=sr,
            min_duration_sec=cfg["min_duration_sec"],
            max_duration_sec=cfg["max_duration_sec"],
        )
        if not ok:
            wlog.warning("skip %s audio invalid: %s", rid, reason)
            return None

        out_dir = Path(cfg["output_dir"])
        wav_rel = f"audio/{rid}.wav"
        wav_path = out_dir / wav_rel
        write_wav_16bit_mono(wav_path, audio, sr)
        dur_written = len(audio) / float(sr)

        return {
            "recording_id": rid,
            "user_id": str(entry.get("user_id", "")),
            "language": lang,
            "sentence": sentence,
            "audio_relpath": wav_rel,
            "sampling_rate": sr,
            "duration_sec": round(dur_written, 4),
        }
    except Exception as exc:  # noqa: BLE001
        wlog.exception("failed entry %s: %s", rid, exc)
        return None


def _flatten_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    dl = raw.get("download", {})
    aud = raw.get("audio", {})
    flt = raw.get("filters", {})
    return {
        "output_dir": str(raw["output_dir"]),
        "min_duration_sec": float(aud.get("min_duration_sec", 1.0)),
        "max_duration_sec": float(aud.get("max_duration_sec", 30.0)),
        "target_sr": int(aud.get("target_sample_rate", 16000)),
        "languages": list(flt.get("languages", ["hi", "hindi", "hi_in"])),
        "max_retries": int(dl.get("max_retries", 5)),
        "timeout_sec": float(dl.get("timeout_sec", 60.0)),
        "backoff_sec": float(dl.get("backoff_sec", 2.0)),
        "log_level": str(raw.get("log_level", "INFO")),
    }


def run_preprocess(config_path: Path) -> Path:
    setup_logging()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping")

    manifest_path = Path(raw["manifest_path"])
    cfg = _flatten_config(raw)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    log.info("Loaded %s manifest entries", len(entries))

    num_workers = max(1, int(raw.get("num_workers", mp.cpu_count() or 1)))
    tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [(e, cfg) for e in entries]

    results: List[Dict[str, Any]] = []
    if num_workers == 1:
        for t in tasks:
            r = process_manifest_entry(t)
            if r:
                results.append(r)
    else:
        log.info("Starting pool with %s workers", num_workers)
        with mp.Pool(processes=num_workers) as pool:
            for r in pool.imap_unordered(process_manifest_entry, tasks, chunksize=1):
                if r:
                    results.append(r)

    manifest_out = out_dir / "processed_manifest.json"
    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s rows to %s", len(results), manifest_out)
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess manifest → 16 kHz mono WAV + normalized transcripts")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"), help="YAML config path")
    args = parser.parse_args()
    run_preprocess(args.config)


if __name__ == "__main__":
    main()
