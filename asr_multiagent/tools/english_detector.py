import os
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


LOANWORDS = {
    "इंटरव्यू", "जॉब", "कंप्यूटर", "मोबाइल", "वीडियो", "ऑफिस", "ट्रेन", "बस",
    "टिकट", "फॉर्म", "अपडेट", "डाउनलोड", "क्लास", "स्कूल", "कॉलेज", "मैनेजर",
}


def _llm_ambiguous_tag(words: List[str]) -> List[str]:
    if not words or not os.getenv("GROQ_API_KEY"):
        return []
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    prompt = ChatPromptTemplate.from_template(
        "Mark which words are English-origin loanwords in Hindi script.\n"
        "Words: {words}\nReturn JSON list of selected words only."
    )
    msg = prompt.format_messages(words=words)
    out = llm.invoke(msg).content
    selected = [w for w in words if w in str(out)]
    return selected


def tag_english_words(text: str) -> str:
    tokens = text.split()
    ambiguous = []
    for t in tokens:
        if re.search(r"[A-Za-z]", t):
            continue
        if t in LOANWORDS:
            continue
        if len(t) >= 5 and t.endswith(("र", "ल", "ट")):
            ambiguous.append(t)
    llm_selected = set(_llm_ambiguous_tag(ambiguous))
    tagged = []
    for t in tokens:
        if re.search(r"[A-Za-z]", t) or t in LOANWORDS or t in llm_selected:
            tagged.append(f"[EN]{t}[/EN]")
        else:
            tagged.append(t)
    return " ".join(tagged)
