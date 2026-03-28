import unicodedata
from typing import Dict, List, Tuple


def align_words(ref: List[str], hyp: List[str]) -> List[Tuple[str, str, str]]:
    """
    Align two word sequences using dynamic programming.
    Returns list of (ref_word, hyp_word, operation) tuples.
    operation ∈ {match, substitution, deletion, insertion}
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "deletion"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "insertion"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            opts = [
                (dp[i - 1][j - 1] + cost, "match" if cost == 0 else "substitution"),
                (dp[i - 1][j] + 1, "deletion"),
                (dp[i][j - 1] + 1, "insertion"),
            ]
            dp[i][j], bt[i][j] = min(opts, key=lambda x: x[0])

    out = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in {"match", "substitution"}:
            out.append((ref[i - 1], hyp[j - 1], op))
            i -= 1
            j -= 1
        elif op == "deletion":
            out.append((ref[i - 1], "∅", "deletion"))
            i -= 1
        else:
            out.append(("∅", hyp[j - 1], "insertion"))
            j -= 1
    return list(reversed(out))


_NUM_VARIANTS = {
    "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5",
    "छह": "6", "सात": "7", "आठ": "8", "नौ": "9", "दस": "10",
    "चौदह": "14", "बीस": "20", "पच्चीस": "25", "तीस": "30",
    "पचास": "50", "सौ": "100", "हज़ार": "1000", "हजार": "1000",
}


def _word_variants(word: str) -> List[str]:
    variants = {
        word,
        unicodedata.normalize("NFC", word),
        unicodedata.normalize("NFD", word),
    }
    if word in _NUM_VARIANTS:
        variants.add(_NUM_VARIANTS[word])
    return [v for v in variants if v]


def build_lattice_for_utterance(reference: str, model_hyps: List[str]) -> List[List[str]]:
    ref_words = reference.split()
    bins: List[set] = [set(_word_variants(w)) for w in ref_words]

    for hyp in model_hyps:
        hyp_words = hyp.split() if hyp.strip() else []
        aligned = align_words(ref_words, hyp_words)

        ref_idx = 0
        insertion_buf = []

        for _, hw, op in aligned:
            if op == "insertion":
                insertion_buf.append(hw)
                continue

            if ref_idx >= len(bins):
                bins.append(set())

            if op in ("match", "substitution"):
                bins[ref_idx].update(_word_variants(hw))

            ref_idx += 1

        for hw in insertion_buf:
            bins.append(set(_word_variants(hw)))

    return [sorted(list(b)) for b in bins]


def lattice_wer_for_model(reference: str, hypothesis: str, lattice_bins: List[List[str]]) -> float:
    ref_len = max(1, len(reference.split()))
    hyp_words = hypothesis.split() if hypothesis.strip() else []
    errors = 0

    for i in range(max(len(hyp_words), len(lattice_bins))):
        hw = hyp_words[i] if i < len(hyp_words) else "∅"
        if i >= len(lattice_bins):
            errors += 1
            continue
        bin_set = set(lattice_bins[i])
        hw_variants = _word_variants(hw)
        if not any(v in bin_set for v in hw_variants):
            errors += 1

    return errors / ref_len


def comparison_rows(standard: Dict[str, float], lattice: Dict[str, float]) -> List[Dict]:
    rows = []
    for k, v in standard.items():
        lv = lattice.get(k, v)
        rows.append(
            {
                "model": k,
                "standard_WER": round(v, 4),
                "lattice_WER": round(lv, 4),
                "delta": round(v - lv, 4),
                "improved": v > lv,
            }
        )
    return rows
