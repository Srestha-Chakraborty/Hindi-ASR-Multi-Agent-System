"""Unified CLI for Hindi ASR production workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from asr_multiagent.runtime.env import configure_ml_runtime_env
from data_pipeline.logging_config import setup_logging


def main() -> None:
    configure_ml_runtime_env()
    parser = argparse.ArgumentParser(description="Hindi ASR production CLI")
    parser.add_argument("--mode", choices=["train", "eval", "infer", "pipeline"], required=True)
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--audio", type=Path, default=None, help="Audio file path for infer mode")
    parser.add_argument("--dry_run", action="store_true", help="Validate config without running heavy work when supported")
    args = parser.parse_args()

    if args.mode == "train":
        config_path = args.config or Path("configs/whisper_train.yaml")
        setup_logging(log_file="outputs/logs/training.log")
        if args.dry_run:
            config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {
                        "validated_config": str(config_path),
                        "model_name_or_path": config_payload.get("model_name_or_path"),
                        "dataset_path": config_payload.get("dataset", {}).get("path"),
                        "output_root": config_payload.get("output", {}).get("root"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        from training.train_whisper import train

        train(config_path, dry_run=args.dry_run)
        return

    if args.mode == "eval":
        from evaluation.evaluate import evaluate, load_config as load_eval_config

        config_path = args.config or Path("configs/evaluation.yaml")
        setup_logging(log_file="outputs/logs/evaluation.log")
        if args.dry_run:
            eval_config = load_eval_config(config_path)
            print(json.dumps({"models": [model.name for model in eval_config.models]}, ensure_ascii=False, indent=2))
            return
        result = evaluate(load_eval_config(config_path))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mode == "infer":
        from asr_multiagent.runtime.inference import (
            load_config as load_inference_config,
            transcribe_audio_file,
        )

        if args.audio is None:
            raise SystemExit("--audio is required for infer mode")
        config_path = args.config or Path("configs/inference.yaml")
        setup_logging(log_file="outputs/logs/inference.log")
        result = transcribe_audio_file(args.audio, load_inference_config(config_path))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    from asr_multiagent.agents.production_agents import run_production_pipeline

    config_path = args.config or Path("configs/production_pipeline.yaml")
    setup_logging(log_file="outputs/logs/pipeline.log")
    result = run_production_pipeline(str(config_path))
    print(json.dumps(result.get("artifacts", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
