"""Lattice-aware WER for multi-hypothesis ASR comparison."""

from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from asr_multiagent.tools.text_metrics import align_sequences, compute_wer, tokenize_words
from data_pipeline.logging_config import setup_logging
from paths import ensure_repo_on_syspath

ensure_repo_on_syspath()

log = logging.getLogger(__name__)

NUMBER_VARIANTS = {
    "एक": "1",
    "दो": "2",
    "तीन": "3",
    "चार": "4",
    "पांच": "5",
    "पाँच": "5",
    "छह": "6",
    "सात": "7",
    "आठ": "8",
    "नौ": "9",
    "दस": "10",
    "हजार": "1000",
    "हज़ार": "1000",
}


@dataclass
class LatticeSlot:
    slot_type: str
    reference_index: int
    alternatives: List[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class WordLattice:
    reference_tokens: List[str]
    slots: List[LatticeSlot]

    def to_serializable(self) -> List[Dict[str, object]]:
        return [asdict(slot) for slot in self.slots]


def _word_variants(word: str) -> List[str]:
    if not word:
        return []
    variants = {
        word,
        unicodedata.normalize("NFC", word),
        unicodedata.normalize("NFD", word),
    }
    if word in NUMBER_VARIANTS:
        variants.add(NUMBER_VARIANTS[word])
    return sorted(variant for variant in variants if variant)


def align_words(reference_words: List[str], hypothesis_words: List[str]) -> List[tuple[str, str, str]]:
    alignment = align_sequences(reference_words, hypothesis_words)
    rows: List[tuple[str, str, str]] = []
    for step in alignment.steps:
        operation = step.operation
        if operation == "equal":
            operation = "match"
        elif operation == "substitute":
            operation = "substitution"
        elif operation == "delete":
            operation = "deletion"
        elif operation == "insert":
            operation = "insertion"
        rows.append((step.reference_token or "∅", step.hypothesis_token or "∅", operation))
    return rows


def build_lattice_for_utterance(reference: str, model_hypotheses: Sequence[str]) -> WordLattice:
    reference_tokens = tokenize_words(reference)
    insertion_bins: List[set[str]] = [set() for _ in range(len(reference_tokens) + 1)]
    reference_bins: List[set[str]] = [set(_word_variants(token)) for token in reference_tokens]
    optional_reference_positions = [False] * len(reference_tokens)

    for hypothesis in model_hypotheses:
        alignment = align_sequences(reference_tokens, tokenize_words(hypothesis))
        for step in alignment.steps:
            if step.operation == "equal":
                reference_bins[step.reference_index].update(_word_variants(step.hypothesis_token))
            elif step.operation == "substitute":
                reference_bins[step.reference_index].update(_word_variants(step.hypothesis_token))
            elif step.operation == "delete":
                optional_reference_positions[step.reference_index] = True
            elif step.operation == "insert":
                insertion_bins[step.reference_index].update(_word_variants(step.hypothesis_token))

    slots: List[LatticeSlot] = []
    for index in range(len(reference_tokens) + 1):
        if insertion_bins[index]:
            slots.append(
                LatticeSlot(
                    slot_type="insertion",
                    reference_index=index,
                    alternatives=sorted(insertion_bins[index]),
                    optional=True,
                )
            )
        if index < len(reference_tokens):
            slots.append(
                LatticeSlot(
                    slot_type="reference",
                    reference_index=index,
                    alternatives=sorted(reference_bins[index]),
                    optional=optional_reference_positions[index],
                )
            )
    return WordLattice(reference_tokens=reference_tokens, slots=slots)


def lattice_wer_for_model(reference: str, hypothesis: str, lattice: WordLattice | List[List[str]] | List[Dict[str, object]]) -> float:
    if not isinstance(lattice, WordLattice):
        if lattice and isinstance(lattice[0], dict):
            lattice = WordLattice(
                reference_tokens=tokenize_words(reference),
                slots=[LatticeSlot(**slot) for slot in lattice],  # type: ignore[arg-type]
            )
        else:
            slots = []
            for index, alternatives in enumerate(lattice):  # type: ignore[assignment]
                slots.append(
                    LatticeSlot(
                        slot_type="reference",
                        reference_index=index,
                        alternatives=list(alternatives),
                        optional=False,
                    )
                )
            lattice = WordLattice(reference_tokens=tokenize_words(reference), slots=slots)

    hypothesis_tokens = tokenize_words(hypothesis)
    num_slots = len(lattice.slots)
    num_words = len(hypothesis_tokens)
    inf = 10 ** 9
    dp = [[inf] * (num_words + 1) for _ in range(num_slots + 1)]
    dp[0][0] = 0

    for slot_idx in range(num_slots + 1):
        for word_idx in range(num_words + 1):
            current = dp[slot_idx][word_idx]
            if current >= inf:
                continue

            if slot_idx == num_slots:
                if word_idx < num_words:
                    dp[slot_idx][word_idx + 1] = min(dp[slot_idx][word_idx + 1], current + 1)
                continue

            slot = lattice.slots[slot_idx]
            alternatives = set(slot.alternatives)

            if slot.optional:
                dp[slot_idx + 1][word_idx] = min(dp[slot_idx + 1][word_idx], current)

            if word_idx < num_words:
                token = hypothesis_tokens[word_idx]
                token_variants = set(_word_variants(token))
                substitution_cost = 0 if token_variants & alternatives else 1
                dp[slot_idx + 1][word_idx + 1] = min(
                    dp[slot_idx + 1][word_idx + 1],
                    current + substitution_cost,
                )

                insertion_cost = 0 if slot.slot_type == "insertion" and token_variants & alternatives else 1
                dp[slot_idx][word_idx + 1] = min(dp[slot_idx][word_idx + 1], current + insertion_cost)

            deletion_cost = 0 if slot.optional else 1
            dp[slot_idx + 1][word_idx] = min(dp[slot_idx + 1][word_idx], current + deletion_cost)

    reference_length = max(1, len(lattice.reference_tokens))
    return dp[num_slots][num_words] / reference_length


def standard_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    return float(compute_wer(list(references), list(hypotheses)))


def comparison_rows(standard: Dict[str, float], lattice: Dict[str, float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name, baseline in standard.items():
        lattice_score = lattice.get(model_name, baseline)
        rows.append(
            {
                "model": model_name,
                "standard_WER": round(baseline, 4),
                "lattice_WER": round(lattice_score, 4),
                "delta": round(baseline - lattice_score, 4),
                "improved": lattice_score < baseline,
            }
        )
    return rows


def leave_one_out_lattice_score(
    reference: str,
    model_name: str,
    model_hypotheses: Dict[str, str],
) -> float:
    peer_hypotheses = [
        hypothesis
        for peer_name, hypothesis in model_hypotheses.items()
        if peer_name != model_name
    ]
    if not peer_hypotheses:
        peer_hypotheses = [model_hypotheses[model_name]]
    lattice = build_lattice_for_utterance(reference, peer_hypotheses)
    return lattice_wer_for_model(reference, model_hypotheses[model_name], lattice)


def _build_demo_payload() -> Dict[str, object]:
    references = [
        "यह वाक्य संख्या 1 है",
        "यह वाक्य संख्या 2 है",
        "उसने दो किताबें खरीदीं",
    ]
    models = {
        "baseline": [
            "यह वाक्य नंबर 1 है",
            "यह वाक्य संख्या 2",
            "उसने 2 किताबें खरीदी",
        ],
        "candidate": [
            "यह वाक्य संख्या 1 है",
            "यह वाक्य संख्या 2 है",
            "उसने दो किताबें खरीदीं",
        ],
        "alt_model": [
            "यह वाक्य 1 है",
            "यह वाक्य नंबर 2 है",
            "उसने दो पुस्तकें खरीदीं",
        ],
    }

    standard_scores = {
        model_name: standard_wer(references, hyps) for model_name, hyps in models.items()
    }
    lattice_scores = {}
    for model_name, hyps in models.items():
        utterance_scores = [
            leave_one_out_lattice_score(
                references[idx],
                model_name,
                {peer_name: peer_hyps[idx] for peer_name, peer_hyps in models.items()},
            )
            for idx in range(len(references))
        ]
        lattice_scores[model_name] = sum(utterance_scores) / len(utterance_scores)

    return {
        "references": references,
        "models": models,
        "standard_scores": standard_scores,
        "lattice_scores": lattice_scores,
        "comparison_table": comparison_rows(standard_scores, lattice_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lattice-aware WER demo")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    setup_logging()
    payload = _build_demo_payload()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Saved lattice demo payload to %s", args.output)
    print(json.dumps(payload["comparison_table"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
