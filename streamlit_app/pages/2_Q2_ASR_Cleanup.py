import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path

import pandas as pd
import streamlit as st

from asr_multiagent.tools.english_detector import tag_english_words
from asr_multiagent.tools.number_normalizer import normalize_numbers_in_text

st.title("Q2 — ASR Cleanup")
results = st.session_state.get("results", {}).get("q2", {})
if not results:
    st.info("Run Q2 from the main page first.")
    st.stop()

data = pd.DataFrame(results.get("english_tagged_transcripts", []))
if not data.empty:
    left, right = st.columns(2)
    with left:
        st.subheader("Raw ASR")
        st.dataframe(data[["recording_id", "raw_asr"]], use_container_width=True)
    with right:
        st.subheader("Normalized + Tagged")
        st.dataframe(data[["recording_id", "normalized", "tagged_text"]], use_container_width=True)

st.subheader("Normalization Examples")
examples = results.get("cleanup_examples", {})
st.write("Correct conversions")
st.json(examples.get("correct_conversions", []))
st.write("Edge cases")
st.json(examples.get("edge_cases", []))

st.subheader("Try custom text")
txt = st.text_input("Hindi text")
if txt:
    norm, reasons = normalize_numbers_in_text(txt)
    tagged = tag_english_words(norm)
    st.write({"normalized": norm, "reasons": reasons, "tagged": tagged})

path = Path("outputs/q2_results.json")
if path.exists():
    st.download_button("Download Q2 JSON", path.read_bytes(), file_name="q2_results.json")
