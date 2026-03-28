import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Q1 — ASR Fine-tuning")
results = st.session_state.get("results", {}).get("q1", {})

if not results:
    st.info("Run Q1 from the main page first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.metric("Baseline WER (FLEURS)", f"{results.get('baseline_wer', 0):.3f}")
    st.metric("Fine-tuned WER (FLEURS)", f"{results.get('finetuned_wer', 0):.3f}")
with col2:
    st.metric("Baseline WER (Held-out)", f"{results.get('heldout_baseline_wer', 0):.3f}")
    st.metric("Fine-tuned WER (Held-out)", f"{results.get('heldout_finetuned_wer', 0):.3f}")

df = pd.DataFrame(
    [
        {"split": "FLEURS", "Baseline": results.get("baseline_wer", 0), "Fine-tuned": results.get("finetuned_wer", 0)},
        {"split": "Held-out", "Baseline": results.get("heldout_baseline_wer", 0), "Fine-tuned": results.get("heldout_finetuned_wer", 0)},
    ]
)
st.plotly_chart(px.bar(df.melt(id_vars=["split"]), x="split", y="value", color="variable", barmode="group"), use_container_width=True)

samples = pd.DataFrame(results.get("error_samples", []))
if not samples.empty:
    st.subheader("Error Samples")
    st.dataframe(samples, use_container_width=True)

st.subheader("Error Taxonomy")
taxonomy = results.get("error_taxonomy", {})
if isinstance(taxonomy, dict):
    for c in taxonomy.get("categories", []):
        with st.expander(c.get("name", "Category")):
            st.write(c.get("description", ""))
            st.write(c.get("examples", []))

st.subheader("Implemented Fix Results")
st.json(results.get("fix_results", {}))

path = Path("outputs/q1_results.json")
if path.exists():
    st.download_button("Download Q1 JSON", path.read_bytes(), file_name="q1_results.json")
