from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr_multiagent.runtime.env import configure_ml_runtime_env
from data_pipeline.logging_config import setup_logging

configure_ml_runtime_env()
load_dotenv()
setup_logging(log_file="outputs/logs/streamlit.log")

st.set_page_config(page_title="Hindi ASR Production Console", layout="wide")
st.title("Hindi ASR Production Console")
st.caption("Upload audio, run Hindi ASR, inspect cleanup output, and optionally trigger the production pipeline.")

if "inference_result" not in st.session_state:
    st.session_state["inference_result"] = None
if "pipeline_result" not in st.session_state:
    st.session_state["pipeline_result"] = None

with st.sidebar:
    st.header("Inference Settings")
    inference_config_path = st.text_input("Inference config", value="configs/inference.yaml")
    pipeline_config_path = st.text_input("Pipeline config", value="configs/production_pipeline.yaml")
    show_confidence = st.checkbox("Show confidence", value=True)
    run_error_aware = st.checkbox("Enable error-aware post-processing", value=True)
    use_cache = st.checkbox("Use inference cache", value=True)

uploaded_audio = st.file_uploader(
    "Upload Hindi audio",
    type=["wav", "flac", "ogg", "mp3", "m4a"],
    accept_multiple_files=False,
)

left, right = st.columns([2, 1])
with left:
    run_asr = st.button("Run ASR", type="primary", use_container_width=True)
with right:
    run_pipeline = st.button("Run Production Pipeline", use_container_width=True)

if uploaded_audio is not None:
    st.audio(uploaded_audio.getvalue())

if run_asr:
    if uploaded_audio is None:
        st.warning("Upload an audio file before running ASR.")
    else:
        from asr_multiagent.runtime.inference import load_config, transcribe_audio_bytes

        cfg = load_config(Path(inference_config_path))
        cfg = cfg.__class__(
            model_name_or_path=cfg.model_name_or_path,
            language=cfg.language,
            task=cfg.task,
            cache_dir=cfg.cache_dir,
            cleanup_config_path=cfg.cleanup_config_path,
            error_analysis_path=cfg.error_analysis_path,
            enable_cache=use_cache,
            enable_cleanup=cfg.enable_cleanup,
            enable_error_aware_postprocessing=run_error_aware,
            compute_confidence=show_confidence,
            device=cfg.device,
        )
        with st.spinner("Running Whisper inference..."):
            st.session_state["inference_result"] = transcribe_audio_bytes(uploaded_audio.getvalue(), cfg)

if run_pipeline:
    from asr_multiagent.agents.production_agents import run_production_pipeline

    with st.spinner("Running production multi-agent pipeline..."):
        st.session_state["pipeline_result"] = run_production_pipeline(pipeline_config_path)

result = st.session_state["inference_result"]
if result:
    st.subheader("Inference Output")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Cache Key", result["cache_key"][:12])
    metric_cols[1].metric("Confidence", f"{result['confidence']:.3f}" if result.get("confidence") is not None else "n/a")
    metric_cols[2].metric("Dominant Error Type", result.get("dominant_error_type") or "n/a")

    raw_col, clean_col = st.columns(2)
    with raw_col:
        st.markdown("**Raw Output**")
        st.code(result["raw_text"], language="text")
        st.markdown("**Normalized Output**")
        st.code(result["normalized_text"], language="text")
    with clean_col:
        st.markdown("**Cleaned Output**")
        st.code(result["cleaned_text"], language="text")
        st.markdown("**English Tagging**")
        st.json(result["english_words"])

    if result.get("number_conversions"):
        st.markdown("**Number Conversions**")
        st.json(result["number_conversions"])

    st.download_button(
        "Download inference JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="inference_result.json",
        mime="application/json",
    )

pipeline_result = st.session_state["pipeline_result"]
if pipeline_result:
    st.subheader("Production Pipeline Artifacts")
    st.json(pipeline_result.get("artifacts", {}))

    output_path = Path("outputs/production_pipeline.json")
    if output_path.exists():
        st.download_button(
            "Download pipeline JSON",
            data=output_path.read_bytes(),
            file_name="production_pipeline.json",
            mime="application/json",
        )

if not result and not pipeline_result:
    st.info("Upload audio and run ASR, or run the production pipeline from the sidebar configuration.")
