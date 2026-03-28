"""Evaluate baseline and fine-tuned Hindi ASR models on FLEURS."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from paths import ensure_repo_on_syspath

ensure_repo_on_syspath()

from asr_multiagent.tools.text_metrics import compute_cer, compute_wer
from data_pipeline.logging_config import setup_logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: str


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    config_name: str
    split: str
    text_column: str
    audio_column: str
    local_path: Optional[str]
    max_samples: Optional[int]


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str
    results_filename: str
    predictions_filename: str
    include_predictions: bool


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    device: int
    compute_cer: bool
    local_files_only: bool


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: DatasetConfig
    models: List[ModelConfig]
    output: OutputConfig
    runtime: RuntimeConfig


def _resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def load_config(config_path: Path) -> EvaluationConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    models = [ModelConfig(**model_cfg) for model_cfg in raw["models"]]
    dataset_cfg = DatasetConfig(**raw["dataset"])
    output_cfg = OutputConfig(**raw["output"])
    runtime_cfg = RuntimeConfig(**raw["runtime"])
    return EvaluationConfig(
        dataset=dataset_cfg,
        models=models,
        output=output_cfg,
        runtime=runtime_cfg,
    )


def _load_dataset_from_config(config: DatasetConfig):
    from datasets import Audio, load_dataset, load_from_disk

    local_path = _resolve_path(config.local_path)
    if local_path and local_path.exists():
        log.info("Loading dataset from local path: %s", local_path)
        ds = load_from_disk(str(local_path))
        if hasattr(ds, "keys"):
            if config.split not in ds:
                raise KeyError(f"Split '{config.split}' not found in local dataset. Available: {list(ds.keys())}")
            data = ds[config.split]
        else:
            data = ds
    else:
        log.info(
            "Loading dataset from hub: source=%s config=%s split=%s",
            config.source,
            config.config_name,
            config.split,
        )
        data = load_dataset(config.source, config.config_name, split=config.split)

    if config.max_samples is not None:
        sample_count = min(len(data), int(config.max_samples))
        log.info("Selecting first %s examples from dataset", sample_count)
        data = data.select(range(sample_count))

    data = data.cast_column(config.audio_column, Audio(sampling_rate=16000))
    return data


def _build_asr_pipeline(model_path: str, device: int, local_files_only: bool):
    import torch
    from transformers import pipeline

    resolved = _resolve_path(model_path)
    candidate = str(resolved) if resolved and resolved.exists() else model_path
    if device < 0 and torch.cuda.is_available():
        device = 0
    if device >= 0 and not torch.cuda.is_available():
        log.warning("CUDA device requested but CUDA is unavailable. Falling back to CPU.")
        device = -1

    log.info("Loading ASR pipeline from %s", candidate)
    return pipeline(
        task="automatic-speech-recognition",
        model=candidate,
        device=device,
        local_files_only=local_files_only,
        generate_kwargs={"language": "hi", "task": "transcribe"},
    )


def _predict_dataset(
    asr_pipeline,
    dataset,
    *,
    audio_column: str,
    text_column: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for start_idx in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start_idx, min(len(dataset), start_idx + batch_size)))
        for offset, example in enumerate(batch):
            audio = example[audio_column]
            prediction = asr_pipeline(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
            )
            hypothesis = str(prediction.get("text", "")).strip()
            reference = str(example[text_column]).strip()
            rows.append(
                {
                    "row_index": start_idx + offset,
                    "reference": reference,
                    "prediction": hypothesis,
                }
            )
    return rows


def _summarize_model(
    model_name: str,
    predictions: List[Dict[str, Any]],
    *,
    compute_cer_enabled: bool,
) -> Dict[str, Any]:
    references = [row["reference"] for row in predictions]
    hypotheses = [row["prediction"] for row in predictions]
    summary = {
        "model": model_name,
        "num_samples": len(predictions),
        "wer": round(compute_wer(references, hypotheses), 6),
    }
    if compute_cer_enabled:
        summary["cer"] = round(compute_cer(references, hypotheses), 6)
    return summary


def _render_table(rows: Iterable[Dict[str, Any]]) -> str:
    materialized = list(rows)
    headers = ["Model", "Samples", "WER", "CER"]
    values = [
        [
            str(row["model"]),
            str(row["num_samples"]),
            f"{row['wer']:.4f}",
            f"{row['cer']:.4f}" if "cer" in row else "n/a",
        ]
        for row in materialized
    ]
    widths = [
        max(len(headers[idx]), *(len(line[idx]) for line in values)) if values else len(headers[idx])
        for idx in range(len(headers))
    ]

    def _format_line(parts: List[str]) -> str:
        return " | ".join(part.ljust(widths[idx]) for idx, part in enumerate(parts))

    divider = "-+-".join("-" * width for width in widths)
    lines = [_format_line(headers), divider]
    lines.extend(_format_line(line) for line in values)
    return "\n".join(lines)


def evaluate(config: EvaluationConfig) -> Dict[str, Any]:
    dataset = _load_dataset_from_config(config.dataset)
    output_dir = _resolve_path(config.output.output_dir)
    if output_dir is None:
        raise ValueError("Output directory must be configured")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_summaries: List[Dict[str, Any]] = []
    model_predictions: Dict[str, List[Dict[str, Any]]] = {}

    for model_cfg in config.models:
        asr = _build_asr_pipeline(
            model_cfg.path,
            device=config.runtime.device,
            local_files_only=config.runtime.local_files_only,
        )
        predictions = _predict_dataset(
            asr,
            dataset,
            audio_column=config.dataset.audio_column,
            text_column=config.dataset.text_column,
            batch_size=max(1, config.runtime.batch_size),
        )
        summary = _summarize_model(
            model_cfg.name,
            predictions,
            compute_cer_enabled=config.runtime.compute_cer,
        )
        model_summaries.append(summary)
        if config.output.include_predictions:
            model_predictions[model_cfg.name] = predictions
        log.info(
            "Finished evaluation for %s | samples=%s | WER=%.4f%s",
            model_cfg.name,
            summary["num_samples"],
            summary["wer"],
            f" | CER={summary['cer']:.4f}" if "cer" in summary else "",
        )

    results = {
        "dataset": asdict(config.dataset),
        "runtime": asdict(config.runtime),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": model_summaries,
    }

    results_path = output_dir / config.output.results_filename
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    log.info("Saved summary results to %s", results_path)

    if config.output.include_predictions:
        predictions_path = output_dir / config.output.predictions_filename
        with predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(model_predictions, handle, ensure_ascii=False, indent=2)
        log.info("Saved prediction details to %s", predictions_path)

    print(_render_table(model_summaries))
    return results


def _default_config_path() -> Path:
    return Path("configs/evaluation.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Whisper Hindi models on FLEURS")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional override for output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional override for number of evaluation examples")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and exit without model or dataset loading")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    if args.output_dir is not None:
        config = EvaluationConfig(
            dataset=config.dataset,
            models=config.models,
            output=OutputConfig(
                output_dir=str(args.output_dir),
                results_filename=config.output.results_filename,
                predictions_filename=config.output.predictions_filename,
                include_predictions=config.output.include_predictions,
            ),
            runtime=config.runtime,
        )
    if args.max_samples is not None:
        config = EvaluationConfig(
            dataset=DatasetConfig(
                source=config.dataset.source,
                config_name=config.dataset.config_name,
                split=config.dataset.split,
                text_column=config.dataset.text_column,
                audio_column=config.dataset.audio_column,
                local_path=config.dataset.local_path,
                max_samples=args.max_samples,
            ),
            models=config.models,
            output=config.output,
            runtime=config.runtime,
        )

    if args.dry_run:
        log.info("Dry run successful for config=%s", args.config)
        log.info("Configured models: %s", [model.name for model in config.models])
        log.info("Configured dataset: %s/%s split=%s", config.dataset.source, config.dataset.config_name, config.dataset.split)
        return

    evaluate(config)


if __name__ == "__main__":
    main()
