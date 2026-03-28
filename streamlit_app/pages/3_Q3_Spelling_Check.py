import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Q3 — Spelling Check")
results = st.session_state.get("results", {}).get("q3", {})
if not results:
    st.info("Run Q3 from the main page first.")
    st.stop()

stats = results.get("stats", {})
c1, c2, c3 = st.columns(3)
c1.metric("Total", stats.get("total", 0))
c2.metric("Correct", stats.get("correct", 0))
c3.metric("Incorrect", stats.get("incorrect", 0))

conf = stats.get("confidence_breakdown", {})
if conf:
    conf_df = pd.DataFrame({"confidence": list(conf.keys()), "count": list(conf.values())})
    st.plotly_chart(px.pie(conf_df, values="count", names="confidence"), use_container_width=True)

df = pd.DataFrame(results.get("classified_words", []))
if not df.empty:
    q = st.text_input("Search word")
    if q:
        df = df[df["word"].str.contains(q, na=False)]
    st.dataframe(df, use_container_width=True)

st.subheader("Low Confidence Review")
st.json(results.get("low_confidence_review", []))
st.subheader("Unreliable Categories")
st.json(results.get("unreliable_categories", []))

path = Path("outputs/q3_results.json")
if path.exists():
    st.download_button("Download Q3 JSON", path.read_bytes(), file_name="q3_results.json")
if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download classified_words.csv", csv, file_name="classified_words.csv", mime="text/csv")
