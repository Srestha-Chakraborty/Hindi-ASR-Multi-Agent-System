"""Production LangGraph agents for train/eval/cleanup/analysis/feedback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph

from analysis.error_analysis import analyze_errors, load_config as load_error_analysis_config
from analysis.fix_pipeline import load_config as load_fix_pipeline_config, run_fix_pipeline
from data_pipeline.logging_config import setup_logging
from post_processing.cleanup import CleanupConfig, clean_text, load_config as load_cleanup_config

log = logging.getLogger(__name__)


class ProductionState(TypedDict, total=False):
    config_path: str
    config: Dict[str, Any]
    artifacts: Dict[str, Any]
    training_result: Dict[str, Any]
    evaluation_result: Dict[str, Any]
    cleanup_result: Dict[str, Any]
    error_analysis_result: Dict[str, Any]
    feedback_result: Dict[str, Any]


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def _load_pipeline_config(config_path: str) -> Dict[str, Any]:
    with _resolve_path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_cleanup_cfg(path_str: str) -> CleanupConfig:
    path = _resolve_path(path_str)
    return load_cleanup_config(path) if path.exists() else CleanupConfig()


def training_agent_node(state: ProductionState) -> ProductionState:
    cfg = state["config"]["training"]
    if not cfg.get("enabled", False):
        state["training_result"] = {"status": "skipped", "reason": "disabled"}
        return state
    from training.train_whisper import train

    train_config_path = _resolve_path(cfg["config_path"])
    dry_run = bool(cfg.get("dry_run", False))
    log.info("TrainingAgent starting | config=%s | dry_run=%s", train_config_path, dry_run)
    train(train_config_path, dry_run=dry_run)
    state["training_result"] = {
        "status": "completed" if not dry_run else "validated",
        "config_path": str(train_config_path),
    }
    return state


def evaluation_agent_node(state: ProductionState) -> ProductionState:
    from evaluation.evaluate import evaluate, load_config as load_eval_config

    cfg = state["config"]["evaluation"]
    if not cfg.get("enabled", True):
        state["evaluation_result"] = {"status": "skipped", "reason": "disabled"}
        existing_predictions = cfg.get("existing_predictions_path")
        existing_results = cfg.get("existing_results_path")
        if existing_predictions:
            state["artifacts"]["predictions_path"] = str(_resolve_path(existing_predictions))
        if existing_results:
            state["artifacts"]["evaluation_results_path"] = str(_resolve_path(existing_results))
        return state
    eval_config = load_eval_config(_resolve_path(cfg["config_path"]))
    log.info("EvaluationAgent starting | config=%s", cfg["config_path"])
    result = evaluate(eval_config)
    state["evaluation_result"] = result
    state["artifacts"]["predictions_path"] = str(
        _resolve_path(eval_config.output.output_dir) / eval_config.output.predictions_filename
    )
    state["artifacts"]["evaluation_results_path"] = str(
        _resolve_path(eval_config.output.output_dir) / eval_config.output.results_filename
    )
    return state


def cleanup_agent_node(state: ProductionState) -> ProductionState:
    cfg = state["config"]["cleanup"]
    if not cfg.get("enabled", True):
        state["cleanup_result"] = {"status": "skipped", "reason": "disabled"}
        return state
    predictions_path = _resolve_path(state["artifacts"]["predictions_path"])
    output_path = _resolve_path(cfg["output_path"])
    cleanup_config = _load_cleanup_cfg(cfg["config_path"])
    predictions_payload = json.loads(predictions_path.read_text(encoding="utf-8"))

    cleaned_payload: Dict[str, List[Dict[str, Any]]] = {}
    for model_name, rows in predictions_payload.items():
        cleaned_payload[model_name] = []
        for row in rows:
            cleaned = clean_text(str(row["prediction"]), cleanup_config, return_metadata=True)
            cleaned_payload[model_name].append(
                {
                    **row,
                    "normalized_prediction": cleaned["normalized_text"],
                    "cleaned_prediction": cleaned["cleaned_text"],
                    "english_words": cleaned["english_words"],
                    "number_conversions": cleaned["number_conversions"],
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cleaned_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("CleanupAgent saved cleaned predictions to %s", output_path)
    state["cleanup_result"] = {"output_path": str(output_path), "models": list(cleaned_payload.keys())}
    state["artifacts"]["cleaned_predictions_path"] = str(output_path)
    return state


def error_analysis_agent_node(state: ProductionState) -> ProductionState:
    cfg = state["config"]["error_analysis"]
    if not cfg.get("enabled", True):
        state["error_analysis_result"] = {"status": "skipped", "reason": "disabled"}
        existing_path = cfg.get("existing_output_path")
        if existing_path:
            state["artifacts"]["error_analysis_path"] = str(_resolve_path(existing_path))
        return state
    analysis_config = load_error_analysis_config(_resolve_path(cfg["config_path"]))
    if cfg.get("predictions_path_override"):
        analysis_config = analysis_config.__class__(
            predictions_path=state["artifacts"]["predictions_path"],
            reference_key=analysis_config.reference_key,
            prediction_key=analysis_config.prediction_key,
            model_name=analysis_config.model_name,
            sampling=analysis_config.sampling,
            output=analysis_config.output,
        )
    result = analyze_errors(analysis_config)
    state["error_analysis_result"] = result
    state["artifacts"]["error_analysis_path"] = str(_resolve_path(analysis_config.output.output_path))
    return state


def feedback_agent_node(state: ProductionState) -> ProductionState:
    cfg = state["config"]["feedback"]
    if not cfg.get("enabled", True):
        state["feedback_result"] = {"status": "skipped", "reason": "disabled"}
        return state
    fix_config = load_fix_pipeline_config(_resolve_path(cfg["config_path"]))
    if cfg.get("use_latest_artifacts", True):
        fix_config = fix_config.__class__(
            predictions_path=state["artifacts"]["predictions_path"],
            error_analysis_path=state["artifacts"]["error_analysis_path"],
            model_name=fix_config.model_name,
            output_path=fix_config.output_path,
            cleanup_config_path=fix_config.cleanup_config_path,
            rules=fix_config.rules,
        )
    result = run_fix_pipeline(fix_config)
    state["feedback_result"] = result
    state["artifacts"]["before_vs_after_path"] = str(_resolve_path(fix_config.output_path))
    return state


def build_production_graph():
    graph = StateGraph(ProductionState)
    graph.add_node("TrainingAgent", training_agent_node)
    graph.add_node("EvaluationAgent", evaluation_agent_node)
    graph.add_node("CleanupAgent", cleanup_agent_node)
    graph.add_node("ErrorAnalysisAgent", error_analysis_agent_node)
    graph.add_node("FeedbackAgent", feedback_agent_node)
    graph.add_edge(START, "TrainingAgent")
    graph.add_edge("TrainingAgent", "EvaluationAgent")
    graph.add_edge("EvaluationAgent", "CleanupAgent")
    graph.add_edge("CleanupAgent", "ErrorAnalysisAgent")
    graph.add_edge("ErrorAnalysisAgent", "FeedbackAgent")
    graph.add_edge("FeedbackAgent", END)
    return graph.compile()


def run_production_pipeline(config_path: str) -> Dict[str, Any]:
    setup_logging(log_file="outputs/logs/pipeline.log")
    config = _load_pipeline_config(config_path)
    state: ProductionState = {"config_path": config_path, "config": config, "artifacts": {}}
    result = build_production_graph().invoke(state)
    output_path = _resolve_path(config.get("output_path", "outputs/production_pipeline.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("Production pipeline completed | output=%s", output_path)
    return result
