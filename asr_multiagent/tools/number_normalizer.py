from typing import List, Tuple

from post_processing.cleanup import CleanupConfig, normalize_numbers


def normalize_numbers_in_text(text: str) -> Tuple[str, List[str]]:
    normalized, metadata = normalize_numbers(text, CleanupConfig(), return_metadata=True)
    reasons = [f"{item['source']}->{item['normalized']}" for item in metadata]
    return normalized, reasons
