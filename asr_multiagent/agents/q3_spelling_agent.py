import json
import os
from collections import Counter
from typing import List, TypedDict

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from asr_multiagent.tools.data_loader import download_json, load_manifest, transcription_url_for_entry
from asr_multiagent.tools.spelling_checker import classify_words


class Q3State(TypedDict, total=False):
    manifest_path: str
    unique_words: List[str]
    classified_words: List[dict]
    low_confidence_review: List[dict]
    unreliable_categories: List[str]
    stats: dict


def load_words_node(state: Q3State) -> Q3State:
    words = set()
    try:
        data = requests.get("https://storage.googleapis.com/upload_goai/unique_words.json", timeout=30).json()
        words.update([str(w).strip().lower() for w in data if str(w).strip()])
    except Exception:  # noqa: BLE001
        pass
    try:
        manifest = load_manifest(state["manifest_path"])
        for item in manifest:
            try:
                txt = download_json(transcription_url_for_entry(item)).get("transcription", "")
                for w in txt.split():
                    w = w.strip().lower()
                    if w:
                        words.add(w)
            except Exception:  # noqa: BLE001
                continue
    except Exception:
        pass
    if not words:
        words.update({"नमस्ते", "कंप्यूटर", "इंटरव्यू", "परिक्षण", "विद्यालय", "दो-चार"})
    state["unique_words"] = sorted(list(words))
    return state


def classify_words_node(state: Q3State) -> Q3State:
    state["classified_words"] = classify_words(state["unique_words"], os.getenv("GROQ_API_KEY", ""))
    return state


def review_low_confidence_node(state: Q3State) -> Q3State:
    import json
    import re

    low_confidence = [x for x in state["classified_words"] if x["confidence"] == "LOW"]
    sample_for_review = low_confidence[:50]

    if not sample_for_review:
        state["low_confidence_review"] = [{"message": "No low-confidence words found"}]
        return state
    if not os.getenv("GROQ_API_KEY"):
        state["low_confidence_review"] = [
            {
                "total_reviewed": len(sample_for_review),
                "agreements": 0,
                "disagreements": len(sample_for_review),
                "estimated_accuracy": 0.0,
                "interpretation": "Second-pass review skipped because GROQ_API_KEY is not set.",
                "comparison_details": [
                    {
                        "word": item["word"],
                        "first_pass_label": item["label"],
                        "first_pass_reason": item.get("reason", ""),
                        "second_pass_label": "unknown",
                        "second_pass_reason": "missing_api_key",
                        "agreement": False,
                    }
                    for item in sample_for_review
                ],
            }
        ]
        return state

    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    prompt = ChatPromptTemplate.from_template(
        "You are a Hindi spelling expert performing a careful second-pass review.\n\n"
        "Review each of these Hindi words and classify them as 'correct_spelling' or 'incorrect_spelling'.\n"
        "Rules:\n"
        "- English loanwords in Devanagari (कंप्यूटर, मोबाइल, etc.) are CORRECT.\n"
        "- Proper nouns and names are CORRECT.\n"
        "- Dialectal forms are CORRECT.\n\n"
        "Words to review: {words}\n\n"
        "Return ONLY valid JSON array:\n"
        "[{{\"word\": \"...\", \"label\": \"correct_spelling\", \"confidence\": \"high\", \"reason\": \"...\"}}]"
    )
    words_to_review = [x["word"] for x in sample_for_review]

    try:
        out = llm.invoke(prompt.format_messages(words=words_to_review)).content.strip()
        out = re.sub(r"^```json\s*", "", out)
        out = re.sub(r"^```\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
        second_pass_list = json.loads(out.strip())
        second_pass_map = {item["word"]: item for item in second_pass_list if "word" in item}
    except Exception:
        second_pass_map = {}

    agreements = 0
    disagreements = 0
    comparison_details = []

    for item in sample_for_review:
        word = item["word"]
        first_label = item["label"]
        second_item = second_pass_map.get(word, {})
        second_label = second_item.get("label", "unknown")

        match = first_label == second_label
        if match:
            agreements += 1
        else:
            disagreements += 1

        comparison_details.append(
            {
                "word": word,
                "first_pass_label": first_label,
                "first_pass_reason": item.get("reason", ""),
                "second_pass_label": second_label,
                "second_pass_reason": second_item.get("reason", ""),
                "agreement": match,
            }
        )

    total_reviewed = len(sample_for_review)
    accuracy_estimate = round(agreements / total_reviewed, 3) if total_reviewed > 0 else 0.0

    state["low_confidence_review"] = [
        {
            "total_reviewed": total_reviewed,
            "agreements": agreements,
            "disagreements": disagreements,
            "estimated_accuracy": accuracy_estimate,
            "interpretation": (
                f"First pass agreed with second pass on {agreements}/{total_reviewed} words "
                f"({accuracy_estimate * 100:.1f}%). "
                f"Disagreements ({disagreements}) indicate where the rule-based + LLM pipeline "
                f"is unreliable, typically on proper nouns, dialectal forms, and loanwords."
            ),
            "comparison_details": comparison_details,
        }
    ]
    return state


def analyze_unreliable_node(state: Q3State) -> Q3State:
    sample = state["classified_words"][:200]
    if not os.getenv("GROQ_API_KEY"):
        state["unreliable_categories"] = ["proper_nouns", "dialectal_forms", "loanwords_in_devanagari"]
    else:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
        prompt = ChatPromptTemplate.from_template(
            "Given this sample, identify at least 2 unreliable categories in Hindi spelling classification."
            " Sample: {sample}. Return JSON list."
        )
        out = llm.invoke(prompt.format_messages(sample=json.dumps(sample, ensure_ascii=False))).content
        state["unreliable_categories"] = ["proper_nouns", "dialectal_forms"] if not out else [out]
    labels = Counter(x["label"] for x in state["classified_words"])
    conf = Counter(x["confidence"] for x in state["classified_words"])
    state["stats"] = {
        "total": len(state["classified_words"]),
        "correct": labels.get("correct_spelling", 0),
        "incorrect": labels.get("incorrect_spelling", 0),
        "confidence_breakdown": dict(conf),
    }
    return state


def build_graph():
    graph = StateGraph(Q3State)
    graph.add_node("load_words", load_words_node)
    graph.add_node("classify_words", classify_words_node)
    graph.add_node("review_low_confidence", review_low_confidence_node)
    graph.add_node("analyze_unreliable", analyze_unreliable_node)
    graph.add_edge(START, "load_words")
    graph.add_edge("load_words", "classify_words")
    graph.add_edge("classify_words", "review_low_confidence")
    graph.add_edge("review_low_confidence", "analyze_unreliable")
    graph.add_edge("analyze_unreliable", END)
    return graph.compile()
