import json
import os
import sys
from pathlib import Path
import csv

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asr_multiagent.agents.orchestrator import run_suite

load_dotenv()

st.set_page_config(page_title="Hindi ASR Research Suite — Josh Talks", layout="wide")
st.title("Hindi ASR Research Suite — Josh Talks")

if "results" not in st.session_state:
    st.session_state["results"] = {}

with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input("GROQ_API_KEY", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    selected = st.multiselect(
        "Select questions",
        options=["q1", "q2", "q3", "q4"],
        default=["q1", "q2", "q3", "q4"],
    )
    manifest_url = st.text_input("Manifest URL or local path", value="data/sample_manifest.json")
    demo_mode = st.checkbox("DEMO MODE (skip heavy Q1 training)", value=True)
    parallel_mode = st.checkbox("Parallel mode (Q2-Q4)", value=True)
    run = st.button("Run")

if run:
    with st.status("Running multi-agent pipeline...", expanded=True) as status:
        st.write("Executing LangGraph orchestrator...")
        results = run_suite(selected, manifest_url, demo_mode=demo_mode, parallel_mode=parallel_mode)
        st.session_state["results"] = results
        for q, payload in results.items():
            out_path = Path("outputs") / f"{q}_results.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        q1 = results.get("q1", {})
        if q1:
            tax_path = Path("outputs/q1_error_taxonomy.json")
            tax_path.write_text(json.dumps(q1.get("error_taxonomy", {}), ensure_ascii=False, indent=2), encoding="utf-8")
            with open("outputs/q1_wer_table.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["split", "baseline_wer", "finetuned_wer"])
                writer.writeheader()
                writer.writerow({"split": "FLEURS", "baseline_wer": q1.get("baseline_wer", 0), "finetuned_wer": q1.get("finetuned_wer", 0)})
                writer.writerow({"split": "Held-out", "baseline_wer": q1.get("heldout_baseline_wer", 0), "finetuned_wer": q1.get("heldout_finetuned_wer", 0)})
        q3 = results.get("q3", {})
        if q3:
            with open("outputs/classified_words.csv", "w", encoding="utf-8", newline="") as f:
                if q3.get("classified_words"):
                    writer = csv.DictWriter(f, fieldnames=list(q3["classified_words"][0].keys()))
                    writer.writeheader()
                    writer.writerows(q3["classified_words"])
        q4 = results.get("q4", {})
        if q4:
            with open("outputs/q4_wer_comparison.csv", "w", encoding="utf-8", newline="") as f:
                rows = q4.get("comparison_table", [])
                if rows:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
        status.update(label="Completed", state="complete")

st.subheader("Final Summary")
if st.session_state["results"]:
    st.json(st.session_state["results"])
    final_report = Path("outputs/final_report.json")
    if final_report.exists():
        st.download_button(
            "Download Final Report",
            final_report.read_bytes(),
            file_name="final_report.json",
            mime="application/json",
        )
else:
    st.info("Run the pipeline to see outputs.")
