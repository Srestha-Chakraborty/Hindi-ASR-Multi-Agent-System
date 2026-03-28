import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Q4 — Lattice WER")
results = st.session_state.get("results", {}).get("q4", {})
if not results:
    st.info("Run Q4 from the main page first.")
    st.stop()

table = pd.DataFrame(results.get("comparison_table", []))
if not table.empty:
    def color_delta(val):
        if val > 0:
            return "background-color: #d4edda; color: #155724"
        elif val < 0:
            return "background-color: #f8d7da; color: #721c24"
        return ""
    styled = table.style.applymap(color_delta, subset=["delta"])
    st.dataframe(styled, use_container_width=True)

st.subheader("🔷 Lattice Visualization — Sample Utterance")
lattice = results.get("lattice", [])
if lattice:
    sample_bins = lattice[0]
    n_bins = len(sample_bins)
    display_bins = sample_bins[:10]
    cols = st.columns(len(display_bins))
    for col, bin_words in zip(cols, display_bins):
        with col:
            st.markdown(
                f"""
                <div style='
                    border: 2px solid #4A90D9;
                    border-radius: 8px;
                    padding: 8px;
                    min-height: 80px;
                    background: #F0F7FF;
                    font-size: 13px;
                    text-align: center;
                '>
                {"<br>".join(f"<b>{w}</b>" if i == 0 else w for i, w in enumerate(bin_words[:4]))}
                </div>
                """,
                unsafe_allow_html=True,
            )
    if n_bins > 10:
        st.caption(f"Showing first 10 of {n_bins} bins. Full lattice in downloaded JSON.")
    else:
        st.caption(
            "Each box = one alignment position. Top word = reference. "
            "Other words = valid alternatives from model outputs."
        )

st.subheader("Example walkthrough")
st.write(
    "When a model uses an alternative token present in the same lattice bin, "
    "the lattice metric counts it as valid instead of penalizing it as strict substitution."
)

path = Path("outputs/q4_results.json")
if path.exists():
    st.download_button("Download Q4 JSON", path.read_bytes(), file_name="q4_results.json")
