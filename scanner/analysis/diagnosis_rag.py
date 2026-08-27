"""
AI Diagnosis Module (Step 7) -- "RAG-lite"

Kid explanation:
    We have a little medical textbook of known machine sicknesses (the
    fault knowledge base). Each page says roughly "this frequency +
    amplitude pattern often means X problem." This module takes what we
    actually measured (Step 5's frequency + amplitude) and finds the
    page of the textbook that sounds most like it.

    Two clues are combined:

    1. NUMBER MATCHING (rule-based, exact):
       Does the measured frequency literally fall inside a textbook
       entry's frequency range? Does the amplitude level match?
       This is precise and easy to verify/debug.

    2. MEANING MATCHING (embedding similarity -- the "RAG-lite" part):
       We turn our measurement into a short sentence (e.g. "high
       frequency, low amplitude vibration") and use a small pretrained
       AI model to check which textbook entry's DESCRIPTION talks about
       something similar in MEANING, not just matching numbers. This
       catches borderline cases that don't cleanly fall inside any one
       numeric bucket -- like a doctor saying "doesn't perfectly match
       any textbook case, but sounds closest to bearing wear."

    We combine both clues into one score per fault pattern and return
    the best match. Rules are weighted more heavily than the embedding
    score, because a tiny language model can't reliably reason about
    numeric ranges ("is 87 between 80 and 200?") -- it's good at fuzzy
    MEANING comparison, not arithmetic. The embedding score mainly acts
    as a tie-breaker and a fallback for ambiguous cases.

Model used for embeddings:
    sentence-transformers' "all-MiniLM-L6-v2" -- a small (~90MB),
    CPU-friendly model. It downloads once (needs internet the first
    time only) then runs fully offline. If it isn't installed/available,
    this module automatically falls back to a simple, dependency-free
    word-overlap similarity so the app degrades gracefully instead of
    crashing.
"""
from dataclasses import dataclass, field

import numpy as np

# --- Heuristic amplitude thresholds -----------------------------------
# IMPORTANT: our amplitude numbers are in arbitrary pixel-motion units
# (Step 3/5's output), NOT calibrated physical units like mm/s or g's.
# A real product would calibrate this using the camera-to-subject
# distance and known industry standards (e.g. ISO 10816). These
# thresholds are reasonable demo defaults, easy to retune later.
LOW_AMPLITUDE_MAX = 2.0
MEDIUM_AMPLITUDE_MAX = 6.0


def classify_amplitude(amplitude: float) -> str:
    """Turns a raw amplitude number into 'low' / 'medium' / 'high'."""
    if amplitude <= LOW_AMPLITUDE_MAX:
        return "low"
    elif amplitude <= MEDIUM_AMPLITUDE_MAX:
        return "medium"
    return "high"


def _frequency_band_words(freq_hz: float) -> str:
    """Turns a raw frequency number into words, similar to how our
    knowledge base descriptions talk about frequency, so the embedding
    model has matching vocabulary to compare against."""
    if freq_hz < 15:
        return "very low frequency"
    elif freq_hz < 40:
        return "low frequency"
    elif freq_hz < 80:
        return "medium frequency"
    else:
        return "high frequency"


def build_query_text(frequency_hz, amplitude, secondary_peaks_hz=None) -> str:
    """
    Turns raw measured numbers into a short natural-language sentence,
    so we can compare its MEANING against each fault pattern's
    description using embeddings.
    """
    amp_level = classify_amplitude(amplitude)
    freq_words = _frequency_band_words(frequency_hz)
    text = f"{freq_words} vibration around {frequency_hz:.1f} Hz with {amp_level} amplitude."
    if secondary_peaks_hz:
        peaks_str = ", ".join(f"{p:.1f} Hz" for p in secondary_peaks_hz[:2])
        text += f" Secondary vibration also detected near {peaks_str}."
    return text


# --- Embedding backend, with a safe no-dependency fallback -------------

_embedder = None
_embedder_checked = False


def _get_embedder():
    """
    Lazily loads the sentence-transformers model once and reuses it.
    Returns None if the library isn't installed/available, so callers
    can fall back gracefully instead of crashing.
    """
    global _embedder, _embedder_checked
    if _embedder_checked:
        return _embedder
    _embedder_checked = True
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _embedder = None
    return _embedder


def _fallback_similarity(text_a: str, text_b: str) -> float:
    """
    Dependency-free backup similarity measure, used only if
    sentence-transformers isn't installed/available. Simple word-overlap
    (Jaccard similarity) -- much cruder than real embeddings, but keeps
    the app functional and keeps the SAME scoring interface, so nothing
    downstream needs to know which backend produced the score.
    """
    words_a = set(text_a.lower().replace(".", "").replace(",", "").split())
    words_b = set(text_b.lower().replace(".", "").replace(",", "").split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union


def _cosine_similarity(vec_a, vec_b) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Returns a 0-1 similarity score between two pieces of text,
    using real embeddings if available, or the fallback otherwise."""
    embedder = _get_embedder()
    if embedder is None:
        return _fallback_similarity(text_a, text_b)
    vec_a, vec_b = embedder.encode([text_a, text_b])
    # Cosine similarity is usually 0-1 for related sentences with this
    # model; clip defensively in case of small negative noise.
    return max(0.0, _cosine_similarity(vec_a, vec_b))


# --- Main diagnosis logic ----------------------------------------------

@dataclass
class DiagnosisResult:
    matched_pattern_name: str
    description: str
    recommendation: str
    health_status: str
    confidence: float           # 0-1, how confident the match is
    rule_score: float           # 0-1, numeric range match strength
    embedding_score: float      # 0-1, semantic similarity strength
    query_text: str
    all_scores: list = field(default_factory=list)  # every pattern's scores, for debugging/transparency
    success: bool = True
    error_message: str = ""


def _rule_score(frequency_hz, amplitude, pattern) -> float:
    """
    Kid explanation: worth 1.0 if the measurement perfectly fits this
    textbook entry's numeric range, 0.0 if it's completely outside it,
    and something in between if it's just outside (so a near-miss still
    counts for a bit -- e.g. 78 Hz vs. an 80-200 Hz bucket shouldn't be
    treated the same as 5 Hz vs. that same bucket).
    """
    freq_min, freq_max = pattern["freq_min_hz"], pattern["freq_max_hz"]
    if freq_min <= frequency_hz <= freq_max:
        freq_score = 1.0
    else:
        # Distance-based partial credit, decaying to 0 over a 20 Hz margin
        distance = min(abs(frequency_hz - freq_min), abs(frequency_hz - freq_max))
        freq_score = max(0.0, 1.0 - distance / 20.0)

    measured_amp_level = classify_amplitude(amplitude)
    amp_score = 1.0 if measured_amp_level == pattern["amplitude_level"] else 0.0

    # Frequency match matters more: a fault's frequency range is usually
    # its most physically distinctive signature.
    return 0.7 * freq_score + 0.3 * amp_score


def diagnose(frequency_hz, amplitude, fault_patterns, secondary_peaks_hz=None,
             rule_weight=0.65) -> DiagnosisResult:
    """
    Main entry point for Step 7.

    frequency_hz, amplitude: from Step 5's VibrationFeatures.
    fault_patterns: a list of dicts, each with keys matching the
      FaultPattern model (name, freq_min_hz, freq_max_hz, amplitude_level,
      description, recommendation, health_status). Accepting plain dicts
      (rather than requiring Django model instances) keeps this testable
      standalone, and Step 8 can pass in `FaultPattern.objects.values()`
      directly from the real database.
    rule_weight: how much to trust exact numeric matching vs. semantic
      similarity when combining scores (0-1). Weighted toward rules by
      default, since numeric ranges are ground truth and embeddings are
      a fuzzy assist, not a replacement for arithmetic.
    """
    if not fault_patterns:
        return DiagnosisResult(
            matched_pattern_name="", description="", recommendation="",
            health_status="watch", confidence=0, rule_score=0, embedding_score=0,
            query_text="", success=False,
            error_message="No fault patterns available to match against.",
        )

    query_text = build_query_text(frequency_hz, amplitude, secondary_peaks_hz)

    scored = []
    for pattern in fault_patterns:
        pattern_text = (
            f"{pattern['name']}. {pattern['description']} "
            f"Typical amplitude: {pattern['amplitude_level']}."
        )
        r_score = _rule_score(frequency_hz, amplitude, pattern)
        e_score = semantic_similarity(query_text, pattern_text)
        combined = rule_weight * r_score + (1 - rule_weight) * e_score
        scored.append({
            "name": pattern["name"],
            "rule_score": round(r_score, 3),
            "embedding_score": round(e_score, 3),
            "combined_score": round(combined, 3),
            "pattern": pattern,
        })

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    best = scored[0]
    best_pattern = best["pattern"]

    return DiagnosisResult(
        matched_pattern_name=best_pattern["name"],
        description=best_pattern["description"],
        recommendation=best_pattern["recommendation"],
        health_status=best_pattern.get("health_status", "watch"),
        confidence=best["combined_score"],
        rule_score=best["rule_score"],
        embedding_score=best["embedding_score"],
        query_text=query_text,
        all_scores=[{k: v for k, v in s.items() if k != "pattern"} for s in scored],
        success=True,
    )