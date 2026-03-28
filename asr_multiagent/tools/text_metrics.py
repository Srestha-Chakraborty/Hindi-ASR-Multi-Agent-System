"""Shared text metrics and alignments for ASR evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class AlignmentStep:
    operation: str
    reference_token: str
    hypothesis_token: str
    reference_index: int
    hypothesis_index: int


@dataclass(frozen=True)
class AlignmentResult:
    reference_tokens: List[str]
    hypothesis_tokens: List[str]
    steps: List[AlignmentStep]
    hits: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        return len(self.reference_tokens)

    @property
    def word_error_rate(self) -> float:
        if not self.reference_tokens:
            return 0.0 if not self.hypothesis_tokens else 1.0
        return self.total_errors / len(self.reference_tokens)


def tokenize_words(text: str) -> List[str]:
    return str(text).strip().split()


def tokenize_characters(text: str) -> List[str]:
    return list(str(text).strip())


def align_sequences(
    reference_tokens: Sequence[str],
    hypothesis_tokens: Sequence[str],
) -> AlignmentResult:
    ref = list(reference_tokens)
    hyp = list(hypothesis_tokens)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    backpointers = [[("", -1, -1)] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
        backpointers[i][0] = ("delete", i - 1, 0)
    for j in range(1, cols):
        dp[0][j] = j
        backpointers[0][j] = ("insert", 0, j - 1)

    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                backpointers[i][j] = ("equal", i - 1, j - 1)
                continue

            substitute_cost = dp[i - 1][j - 1] + 1
            delete_cost = dp[i - 1][j] + 1
            insert_cost = dp[i][j - 1] + 1
            best_cost = min(substitute_cost, delete_cost, insert_cost)
            dp[i][j] = best_cost
            if best_cost == substitute_cost:
                backpointers[i][j] = ("substitute", i - 1, j - 1)
            elif best_cost == delete_cost:
                backpointers[i][j] = ("delete", i - 1, j)
            else:
                backpointers[i][j] = ("insert", i, j - 1)

    steps: List[AlignmentStep] = []
    hits = substitutions = deletions = insertions = 0
    i = len(ref)
    j = len(hyp)
    while i > 0 or j > 0:
        operation, prev_i, prev_j = backpointers[i][j]
        if operation == "equal":
            hits += 1
            steps.append(
                AlignmentStep("equal", ref[i - 1], hyp[j - 1], i - 1, j - 1)
            )
            i -= 1
            j -= 1
        elif operation == "substitute":
            substitutions += 1
            steps.append(
                AlignmentStep("substitute", ref[i - 1], hyp[j - 1], i - 1, j - 1)
            )
            i -= 1
            j -= 1
        elif operation == "delete":
            deletions += 1
            steps.append(AlignmentStep("delete", ref[i - 1], "", i - 1, j))
            i -= 1
        elif operation == "insert":
            insertions += 1
            steps.append(AlignmentStep("insert", "", hyp[j - 1], i, j - 1))
            j -= 1
        else:
            break

    steps.reverse()
    return AlignmentResult(
        reference_tokens=ref,
        hypothesis_tokens=hyp,
        steps=steps,
        hits=hits,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


def compute_wer(reference_texts: Sequence[str], hypothesis_texts: Sequence[str]) -> float:
    total_reference_words = 0
    total_errors = 0
    for reference, hypothesis in zip(reference_texts, hypothesis_texts):
        alignment = align_sequences(tokenize_words(reference), tokenize_words(hypothesis))
        total_reference_words += alignment.reference_length
        total_errors += alignment.total_errors

    if total_reference_words == 0:
        return 0.0 if not list(hypothesis_texts) else 1.0
    return total_errors / total_reference_words


def compute_cer(reference_texts: Sequence[str], hypothesis_texts: Sequence[str]) -> float:
    total_reference_chars = 0
    total_errors = 0
    for reference, hypothesis in zip(reference_texts, hypothesis_texts):
        alignment = align_sequences(
            tokenize_characters(reference),
            tokenize_characters(hypothesis),
        )
        total_reference_chars += alignment.reference_length
        total_errors += alignment.total_errors

    if total_reference_chars == 0:
        return 0.0 if not list(hypothesis_texts) else 1.0
    return total_errors / total_reference_chars
