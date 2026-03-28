import json
import re
import time
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


DEV_RE = re.compile(r"^[\u0900-\u097F0-9]+$")


def _rule_check(word: str) -> Dict[str, str]:
    """Layer 1 & 2: Rule-based checks. Returns label+confidence or 'uncertain'."""
    if not word:
        return {"label": "incorrect_spelling", "confidence": "HIGH", "reason": "empty_word"}
    if not DEV_RE.match(word):
        return {"label": "incorrect_spelling", "confidence": "HIGH", "reason": "non_devanagari_chars"}
    if re.search(r"(.)\1\1", word):
        return {"label": "incorrect_spelling", "confidence": "MEDIUM", "reason": "triple_repeated_char"}
    if word.endswith("्"):
        return {"label": "incorrect_spelling", "confidence": "MEDIUM", "reason": "lone_halant_at_end"}
    if word and "\u093E" <= word[0] <= "\u094F":
        return {"label": "incorrect_spelling", "confidence": "MEDIUM", "reason": "orphan_matra"}
    if len(word) == 1 and word not in "०१२३४५६७८९0123456789":
        return {"label": "uncertain", "confidence": "MEDIUM", "reason": "single_char_word"}
    return {"label": "uncertain", "confidence": "LOW", "reason": "needs_llm_review"}


def _llm_classify_batch(words: List[str], api_key: str) -> List[Dict[str, str]]:
    """Layer 3: LLM batch classification with proper JSON parsing."""
    if not words or not api_key:
        return [
            {"word": w, "label": "correct_spelling", "confidence": "LOW", "reason": "no_api_key_fallback"}
            for w in words
        ]

    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)
    prompt = ChatPromptTemplate.from_template(
        "You are a Hindi linguist and spelling expert.\n"
        "Classify each word below as 'correct_spelling' or 'incorrect_spelling'.\n\n"
        "CRITICAL RULES:\n"
        "1. English loanwords written in Devanagari script (e.g., कंप्यूटर, मोबाइल, इंटरव्यू) are CORRECT.\n"
        "2. Proper nouns and names are CORRECT even if uncommon.\n"
        "3. Dialectal Hindi variations are CORRECT.\n"
        "4. Only mark as incorrect if it has clear spelling mistakes.\n\n"
        "Words to classify: {words}\n\n"
        "Return ONLY a valid JSON array. No explanation, no markdown, no extra text.\n"
        "Format: [{{\"word\": \"...\", \"label\": \"correct_spelling\", \"reason\": \"...\"}}]"
    )

    try:
        out = llm.invoke(prompt.format_messages(words=words)).content.strip()
        out = re.sub(r"^```json\s*", "", out)
        out = re.sub(r"^```\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
        out = out.strip()
        parsed_list = json.loads(out)

        word_to_result = {}
        for item in parsed_list:
            if isinstance(item, dict) and "word" in item:
                word_to_result[item["word"]] = item

        results = []
        for w in words:
            if w in word_to_result:
                item = word_to_result[w]
                label = item.get("label", "correct_spelling")
                if label not in ("correct_spelling", "incorrect_spelling"):
                    label = "correct_spelling"
                results.append(
                    {
                        "word": w,
                        "label": label,
                        "confidence": "LOW",
                        "reason": item.get("reason", "llm_classified"),
                    }
                )
            else:
                results.append(
                    {
                        "word": w,
                        "label": "correct_spelling",
                        "confidence": "LOW",
                        "reason": "not_returned_by_llm",
                    }
                )
        return results
    except (json.JSONDecodeError, Exception) as e:
        return [
            {"word": w, "label": "correct_spelling", "confidence": "LOW", "reason": f"parse_error:{str(e)[:40]}"}
            for w in words
        ]


def classify_words(words: List[str], api_key: str) -> List[Dict[str, str]]:
    """Classify all words using rule-based + LLM fallback pipeline."""
    results: List[Dict[str, str]] = []
    uncertain_words: List[str] = []

    for w in words:
        base = _rule_check(w)
        if base["label"] == "uncertain":
            uncertain_words.append(w)
        else:
            results.append({"word": w, **{k: v for k, v in base.items() if k != "label"}, "label": base["label"]})

    for i in range(0, len(uncertain_words), 50):
        batch = uncertain_words[i: i + 50]
        batch_results = _llm_classify_batch(batch, api_key)
        results.extend(batch_results)
        if i + 50 < len(uncertain_words):
            time.sleep(0.6)

    word_order = {w: idx for idx, w in enumerate(words)}
    results.sort(key=lambda x: word_order.get(x["word"], 9999))
    return results
