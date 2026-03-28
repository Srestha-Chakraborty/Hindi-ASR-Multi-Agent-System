"""
Build a HuggingFace Dataset from processed_manifest.json and persist to disk.

Run:
  python -m data_pipeline.dataset_loader --config configs/dataset_loader.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import Audio, Dataset, DatasetDict, load_from_disk

from data_pipeline.audio_io import load_audio_dict_for_hf
from data_pipeline.logging_config import setup_logging

log = logging.getLogger(__name__)


def _row_to_example(row: Dict[str, Any], base_dir: Path, target_sr: int) -> Dict[str, Any]:
    wav_path = base_dir / row["audio_relpath"]
    if not wav_path.is_file():
        raise FileNotFoundError(f"Missing wav: {wav_path}")
    audio = load_audio_dict_for_hf(wav_path, target_sr=target_sr)
    return {
        "audio": audio,
        "sentence": row["sentence"],
        "recording_id": row["recording_id"],
    }


def build_dataset_from_processed_manifest(
    processed_manifest_path: Path,
    *,
    target_sr: int = 16000,
    test_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    setup_logging()
    base_dir = processed_manifest_path.parent
    with processed_manifest_path.open("r", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)
    if not rows:
        log.warning("No rows in processed manifest; returning empty splits")
        empty = Dataset.from_list([])
        return DatasetDict({"train": empty, "test": empty})

    examples = []
    for row in rows:
        examples.append(_row_to_example(row, base_dir, target_sr))

    ds = Dataset.from_list(examples)
    ds = ds.cast_column("audio", Audio(sampling_rate=target_sr))
    if len(ds) <= 1:
        return DatasetDict({"train": ds, "test": ds})
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def save_dataset_dict(dsd: DatasetDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(path))
    log.info("Saved DatasetDict to %s", path)


def load_dataset_dict(path: Path) -> DatasetDict:
    log.info("Loading DatasetDict from %s", path)
    return load_from_disk(str(path))


def run_from_config(config_path: Path) -> Path:
    setup_logging()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    processed = Path(cfg["processed_manifest_path"])
    out = Path(cfg["hf_dataset_output_dir"])
    target_sr = int(cfg.get("target_sample_rate", 16000))
    test_size = float(cfg.get("test_size", 0.1))
    seed = int(cfg.get("seed", 42))

    dsd = build_dataset_from_processed_manifest(
        processed,
        target_sr=target_sr,
        test_size=test_size,
        seed=seed,
    )
    save_dataset_dict(dsd, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HF Dataset from processed_manifest.json")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_loader.yaml"))
    args = parser.parse_args()
    out = run_from_config(args.config)
    log.info("Done: %s", out)


if __name__ == "__main__":
    main()
