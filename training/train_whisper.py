"""
Fine-tune Whisper-small on a disk-persisted HuggingFace Dataset (Hindi ASR).

Uses WhisperProcessor, log-mel features, tokenized labels, Seq2SeqTrainer.
Checkpoints and final model paths come from config YAML.

Run:
  python training/train_whisper.py --config configs/whisper_train.yaml
  python training/train_whisper.py --config configs/whisper_train.yaml --dry_run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Allow `python training/train_whisper.py` from any CWD
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_from_disk
from jiwer import wer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data_pipeline.logging_config import setup_logging

log = logging.getLogger(__name__)


class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad input features and labels using the Whisper processor."""

    def __init__(self, processor: WhisperProcessor) -> None:
        self.processor = processor

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.size(1) > self.processor.model_max_length:
            labels = labels[:, : self.processor.model_max_length]
        batch["labels"] = labels
        return batch


def _resolve_path(p: str) -> Path:
    root = Path(os.getenv("ASR_PROJECT_ROOT", ".")).resolve()
    path = Path(p)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Training config must be a YAML mapping")
    return data


def _prepare_dataset_batch(batch: Dict[str, Any], processor: WhisperProcessor) -> Dict[str, Any]:
    """Batched prepare: log-mel input_features (one 2D array per row) + tokenizer labels."""
    audios = batch["audio"]
    arrays = [np.asarray(x["array"], dtype=np.float32) for x in audios]
    sr = int(audios[0]["sampling_rate"])
    fe = processor.feature_extractor(arrays, sampling_rate=sr)
    arr = np.asarray(fe["input_features"])
    if arr.ndim == 3:
        feats = [arr[i].copy() for i in range(arr.shape[0])]
    else:
        feats = [arr]
    labels = processor.tokenizer(batch["sentence"])["input_ids"]
    return {"input_features": feats, "labels": labels}


def _build_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        score = float(wer(label_str, pred_str))
        return {"wer": score}

    return compute_metrics


def train(config_path: Path, *, dry_run: bool = False) -> None:
    setup_logging()
    cfg = _load_yaml(config_path)
    log.info("Loaded training config from %s", config_path)

    model_id = cfg["model_name_or_path"]
    lang = cfg.get("language", "hi")
    task = cfg.get("task", "transcribe")
    ds_path = _resolve_path(cfg["dataset"]["path"])
    train_split = cfg["dataset"]["train_split"]
    eval_split = cfg["dataset"]["eval_split"]

    out_root = _resolve_path(cfg["output"]["root"])
    ckpt_dir = out_root / cfg["output"]["checkpoints_subdir"]
    final_dir = out_root / cfg["output"]["final_model_subdir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info("DRY RUN: model=%s dataset=%s splits=%s/%s out=%s", model_id, ds_path, train_split, eval_split, out_root)
        return

    if not ds_path.is_dir():
        raise FileNotFoundError(f"Dataset not found at {ds_path}. Run data_pipeline.dataset_loader first.")

    log.info("Loading dataset from disk: %s", ds_path)
    dsd: DatasetDict = load_from_disk(str(ds_path))
    if train_split not in dsd or eval_split not in dsd:
        raise KeyError(f"Missing splits; have {list(dsd.keys())}")

    log.info("Loading processor and model: %s", model_id)
    processor = WhisperProcessor.from_pretrained(model_id, language=lang, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if cfg["training"].get("gradient_checkpointing"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    log.info("Mapping dataset to log-mel + tokenized labels")
    train_ds = dsd[train_split].map(
        lambda b: _prepare_dataset_batch(b, processor),
        remove_columns=dsd[train_split].column_names,
        batched=True,
        batch_size=8,
        num_proc=1,
    )
    eval_ds = dsd[eval_split].map(
        lambda b: _prepare_dataset_batch(b, processor),
        remove_columns=dsd[eval_split].column_names,
        batched=True,
        batch_size=8,
        num_proc=1,
    )

    t = cfg["training"]
    fp16_setting = t.get("fp16", False)
    if fp16_setting == "auto":
        use_fp16 = torch.cuda.is_available()
    else:
        use_fp16 = bool(fp16_setting)
    use_bf16 = bool(t.get("bf16", False)) and torch.cuda.is_available()

    max_steps = int(t.get("max_steps", -1))
    gen_max = t.get("generation_max_length")
    gen_max_int = int(gen_max) if gen_max is not None else None

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=float(t["num_train_epochs"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        warmup_steps=int(t["warmup_steps"]),
        logging_steps=int(t["logging_steps"]),
        max_steps=max_steps if max_steps > 0 else -1,
        eval_strategy=str(t.get("eval_strategy", "steps")),
        eval_steps=int(t["eval_steps"]),
        save_strategy=str(t.get("save_strategy", "steps")),
        save_steps=int(t["save_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        load_best_model_at_end=bool(t.get("load_best_model_at_end", False)),
        metric_for_best_model=str(t.get("metric_for_best_model", "wer")),
        greater_is_better=bool(t.get("greater_is_better", False)),
        fp16=use_fp16,
        bf16=use_bf16,
        seed=int(t.get("seed", 42)),
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        predict_with_generate=bool(t.get("predict_with_generate", True)),
        generation_max_length=gen_max_int,
        report_to=[],
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    compute_metrics = _build_compute_metrics(processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt: Optional[str] = get_last_checkpoint(str(ckpt_dir))
    if last_ckpt:
        log.info("Resuming from checkpoint: %s", last_ckpt)
    else:
        log.info("Starting training from pretrained weights")

    trainer.train(resume_from_checkpoint=last_ckpt)

    log.info("Saving final model to %s", final_dir)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    log.info("Training complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper (Hindi) with Seq2SeqTrainer")
    parser.add_argument("--config", type=Path, default=Path("configs/whisper_train.yaml"))
    parser.add_argument("--dry_run", action="store_true", help="Validate config and paths only")
    args = parser.parse_args()
    train(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
