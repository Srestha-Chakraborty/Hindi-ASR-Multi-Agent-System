from typing import List

from jiwer import wer


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    if not references:
        return 0.0
    return float(wer(references, hypotheses))
