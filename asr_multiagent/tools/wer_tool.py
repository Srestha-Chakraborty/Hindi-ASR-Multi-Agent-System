from typing import List

from asr_multiagent.tools.text_metrics import compute_wer as compute_wer_local


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    if not references:
        return 0.0
    return float(compute_wer_local(references, hypotheses))
