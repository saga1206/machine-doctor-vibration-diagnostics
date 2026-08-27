"""
Diagnosis Generation Module (Step 7b) -- completes the "RAG" in RAG-lite

Kid explanation:
    diagnosis_rag.py already does the "R" in RAG -- it RETRIEVES the
    most relevant page from our fault knowledge base using embedding
    similarity. But retrieving a fact isn't the same as explaining it
    well. This file is the "G" -- GENERATION: it hands the retrieved
    facts to a real AI language model (Google's Gemini) and asks it to
    WRITE a natural, easy-to-read diagnosis paragraph using those facts
    -- not just copy-paste a pre-written sentence from the JSON file.

    Full RAG pattern:
        measured numbers + retrieved knowledge-base entry
            --> [prompt] --> Gemini API --> natural diagnosis paragraph

Why call an API instead of running a local model:
    Writing genuinely natural-sounding English well needs a real LLM.
    Running one locally needs a GPU (which our target machine doesn't
    have) or a large, slow CPU model. The standard real-world answer is
    to keep retrieval local/free (our embedding search) and only send
    the small "please write this up nicely" step to a hosted API.
    Gemini's Flash-tier models have a genuinely free tier suitable for
    a student project (no credit card needed) -- get a key at
    https://aistudio.google.com/app/apikey and set it as the
    GEMINI_API_KEY environment variable.

Graceful degradation:
    If no API key is set, the API call fails, or the `google-genai`
    package isn't installed, this module automatically falls back to
    the same template-based sentence used before -- consistent with
    every other module in this app treating "AI feature unavailable"
    as a non-fatal, recoverable situation, not a crash.
"""
import os
from dataclasses import dataclass

# Model choice: Gemini's free tier currently covers "Flash" models.
# Google renames/replaces specific model versions fairly often -- if
# this exact name is ever retired, check https://ai.google.dev/gemini-api/docs/models
# for the current recommended free-tier Flash model name and swap it in.
GENERATION_MODEL = "gemini-3.6-flash"


@dataclass
class GenerationResult:
    text: str
    used_llm: bool       # True if Gemini actually generated this, False if fallback template was used
    success: bool = True
    error_message: str = ""


def build_prompt(frequency_hz, amplitude, matched_pattern_name, description,
                  recommendation, confidence, secondary_peaks_hz=None) -> str:
    """
    Kid explanation: this assembles the "here are the facts, please
    write this up" message we send to the AI. Keeping the retrieved
    facts explicit and separate from the instruction is what makes this
    RAG rather than just asking the AI to guess -- the AI is grounded in
    real numbers and a real knowledge-base match, not making things up.
    """
    secondary_text = (
        ", ".join(f"{p:.1f} Hz" for p in secondary_peaks_hz[:2])
        if secondary_peaks_hz else "none detected"
    )
    return f"""You are explaining a non-contact vibration diagnostic result to a
non-technical equipment owner, in plain, friendly, concise English.

MEASURED DATA (from analyzing their video):
- Dominant vibration frequency: {frequency_hz:.2f} Hz
- Amplitude: {amplitude:.3f} (relative units)
- Secondary frequency peaks: {secondary_text}

CLOSEST MATCHING KNOWN FAULT PATTERN (retrieved from our reference database):
- Name: {matched_pattern_name}
- Typical description: {description}
- Recommended action: {recommendation}
- Match confidence: {confidence:.0%}

Write a short diagnosis paragraph (3-5 sentences) for the equipment owner.
Naturally reference the specific measured numbers above. If the match
confidence is below 70%, clearly acknowledge the result is uncertain/
borderline rather than sounding falsely confident. Do not invent any
facts beyond what is given above."""


def _fallback_text(frequency_hz, matched_pattern_name, description, recommendation, confidence) -> str:
    """The old template-based sentence, used whenever the real LLM call
    isn't available -- keeps the app fully functional offline/free."""
    return (
        f"{matched_pattern_name}: {description} "
        f"Recommendation: {recommendation} "
        f"(confidence: {confidence:.0%})"
    )


def generate_diagnosis_narrative(frequency_hz, amplitude, matched_pattern_name,
                                  description, recommendation, confidence,
                                  secondary_peaks_hz=None) -> GenerationResult:
    """
    Main entry point. Tries real Gemini generation first; falls back to
    a template sentence if the API key/package/network isn't available.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    fallback = _fallback_text(frequency_hz, matched_pattern_name, description, recommendation, confidence)

    if not api_key:
        return GenerationResult(
            text=fallback, used_llm=False, success=True,
            error_message="GEMINI_API_KEY not set -- using template text instead of real generation.",
        )

    try:
        from google import genai
    except ImportError:
        return GenerationResult(
            text=fallback, used_llm=False, success=True,
            error_message="google-genai package not installed -- using template text instead.",
        )

    try:
        prompt = build_prompt(
            frequency_hz, amplitude, matched_pattern_name, description,
            recommendation, confidence, secondary_peaks_hz,
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        generated_text = response.text.strip()

        if not generated_text:
            raise ValueError("Gemini returned an empty response")

        return GenerationResult(text=generated_text, used_llm=True, success=True)

    except Exception as e:
        # Any failure (bad key, rate limit, network down, etc.) -> fall
        # back gracefully instead of breaking the whole scan pipeline.
        return GenerationResult(
            text=fallback, used_llm=False, success=True,
            error_message=f"Gemini generation failed, used fallback text: {e}",
        )