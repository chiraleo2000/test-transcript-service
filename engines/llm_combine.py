"""LLM-based transcript combination and correction via Azure OpenAI GPT-5.4-nano.

Workflow
--------
1. Parse each selected transcript into {start_sec, end_sec, speaker, text} dicts.
2. Align all parsed sources to the Azure segment list (Azure is always the
   canonical time-grid baseline).
3. Send every aligned segment in a SINGLE streaming call to GPT-5.4-nano
   (400 K-token context / 128 K-token output — enough for ~1 hour of audio).
4. Stream tokens back, updating the Gradio progress bar as they arrive.
5. Return the corrected transcript in Azure's native
   ``[HH:MM:SS → HH:MM:SS] [SPEAKER_XX]: text`` format.
"""

import logging
import os
import re
import time
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Azure OpenAI constants
# ---------------------------------------------------------------------------

_AZURE_API_VERSION = "2024-12-01-preview"

# Rough estimate: number of output tokens per aligned segment.
# Used only for progress-bar scaling; accuracy is not critical.
_TOKENS_PER_SEGMENT = 35

# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

# Matches:  [HH:MM:SS → HH:MM:SS] [SPEAKER_XX]: text
#       or  [HH:MM:SS → HH:MM:SS] text
_LINE_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2})\s*\u2192\s*(\d{2}:\d{2}:\d{2})\]\s*"
    r"(?:\[([^\]]+)\]:\s*)?(.*)"
)


def _ts_to_sec(ts: str) -> float:
    """Convert ``HH:MM:SS`` string to float seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_transcript(text: str) -> list[dict]:
    """Parse an Azure-format transcript into a list of segment dicts.

    Each dict has keys:
        start_sec, end_sec   – float seconds
        start_str, end_str   – original ``HH:MM:SS`` strings (preserved for output)
        speaker              – str or ``None``
        text                 – transcribed text
    """
    segments: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        start_str, end_str, speaker, content = m.groups()
        segments.append({
            "start_sec": _ts_to_sec(start_str),
            "end_sec":   _ts_to_sec(end_str),
            "start_str": start_str,
            "end_str":   end_str,
            "speaker":   speaker,
            "text":      content.strip(),
        })
    return segments


# ---------------------------------------------------------------------------
# Segment alignment
# ---------------------------------------------------------------------------

def align_to_azure(
    azure_segs: list[dict],
    other_segs: dict[str, list[dict]],
) -> list[dict]:
    """Align other transcript sources to the Azure canonical time grid.

    For each Azure segment ``[s, e]``, collect every overlapping segment from
    each other source and concatenate their texts.  The overlap condition is
    ``max(s, other_s) < min(e, other_e)`` (strictly positive overlap).

    Returns
    -------
    list[dict]
        Each dict contains all Azure segment keys plus ``{source}_text`` for
        every source in *other_segs* (``None`` when no overlap found).
    """
    aligned: list[dict] = []
    for az in azure_segs:
        entry: dict = {
            "start_sec": az["start_sec"],
            "end_sec":   az["end_sec"],
            "start_str": az["start_str"],
            "end_str":   az["end_str"],
            "speaker":   az["speaker"],
            "azure_text": az["text"],
        }
        for source_name, segs in other_segs.items():
            pieces: list[str] = []
            for seg in segs:
                overlap = (
                    min(az["end_sec"], seg["end_sec"])
                    - max(az["start_sec"], seg["start_sec"])
                )
                if overlap > 0:
                    pieces.append(seg["text"])
            entry[f"{source_name}_text"] = " ".join(pieces) if pieces else None
        aligned.append(entry)
    return aligned


# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------

def get_openai_client():
    """Create an ``AzureOpenAI`` client using credentials from env."""
    from openai import AzureOpenAI  # lazy import — openai may not always be installed

    api_key = os.getenv("AZURE_OPENAI_KEY", "").strip()
    endpoint = os.getenv("AZURE_ENDPOINT", "").strip()

    if not api_key:
        raise ValueError(
            "AZURE_OPENAI_KEY not found in environment. "
            "Please add AZURE_OPENAI_KEY=<your-key> to your .env file and restart the app."
        )
    if not endpoint:
        raise ValueError(
            "AZURE_ENDPOINT not found in environment. "
            "Please add AZURE_ENDPOINT=<your-endpoint-url> to your .env file and restart the app."
        )
    return AzureOpenAI(
        api_version=_AZURE_API_VERSION,
        azure_endpoint=endpoint,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return (
        "You are an expert transcript editor specialising in Thai and multilingual speech.\n\n"
        "You will receive a numbered list of speech segments. "
        "Each segment has a timestamp, an optional speaker label, "
        "and transcriptions from one or more automatic speech recognition (ASR) engines "
        "(Azure Speech, Typhoon Whisper, and/or Thonburian Whisper).\n\n"
        "Your task:\n"
        "1. For EVERY segment produce exactly ONE corrected output line. "
        "Cross-reference all provided source texts to fix errors, "
        "missing words, wrong characters, and mis-transcribed proper nouns.\n"
        "2. Output lines MUST follow this exact format:\n"
        "   [HH:MM:SS \u2192 HH:MM:SS] [SPEAKER_XX]: corrected text\n"
        "   If no speaker label is available:\n"
        "   [HH:MM:SS \u2192 HH:MM:SS] corrected text\n"
        "3. Keep the EXACT timestamps and speaker labels from the input — do NOT alter them.\n"
        "4. Do NOT add, remove, merge, split, or reorder segments.\n"
        "5. Output ONLY the corrected lines — no headers, no explanations, "
        "no segment numbers, no blank lines between segments.\n"
        "6. Preserve the original spoken language (Thai, English, mixed, etc.) exactly."
    )


def _build_user_prompt(aligned: list[dict], other_source_names: list[str]) -> str:
    """Build the user prompt listing every aligned segment with all source texts."""
    header = (
        f"Please correct the following {len(aligned)} transcript segment(s). "
        "Output one corrected line per segment in the specified format.\n\n"
    )
    parts = [header]
    for i, seg in enumerate(aligned, 1):
        ts = f"[{seg['start_str']} \u2192 {seg['end_str']}]"
        speaker_part = f" [{seg['speaker']}]:" if seg.get("speaker") else ""
        parts.append(f"Segment {i}: {ts}{speaker_part}")
        parts.append(f"  Azure:        {seg['azure_text']}")
        for src in other_source_names:
            val = seg.get(f"{src}_text")
            if val is not None:
                label = src.capitalize()
                parts.append(f"  {label}:{' ' * max(1, 8 - len(label))}{val}")
        parts.append("")  # blank line between segments for readability
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM correction (single streaming call)
# ---------------------------------------------------------------------------

def correct_all_segments(
    aligned: list[dict],
    other_source_names: list[str],
    client,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    """Send all aligned segments to GPT-5.4-nano in a single streaming call.

    Parameters
    ----------
    aligned:
        Output of :func:`align_to_azure`.
    other_source_names:
        Source keys present in *aligned* dicts, e.g. ``["typhoon"]``.
    client:
        ``AzureOpenAI`` client returned by :func:`get_openai_client`.
    progress_callback:
        Optional callable receiving a fraction (0.0 → 1.0) as tokens stream in.
        Used to drive the Gradio progress bar.

    Returns
    -------
    str
        Corrected transcript — one ``[HH:MM:SS → HH:MM:SS] ...`` line per segment.
    """
    system_prompt = _build_system_prompt()
    user_prompt   = _build_user_prompt(aligned, other_source_names)

    logger.info(
        "Sending %d aligned segments to %s (streaming)...",
        len(aligned), os.getenv("AZURE_DEPLOYMENT", "gpt-5.4-nano"),
    )

    estimated_total_tokens = max(1, len(aligned) * _TOKENS_PER_SEGMENT)

    stream = client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT", "gpt-5.4-nano"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_completion_tokens=128000,
        stream=True,
    )

    collected: list[str] = []
    token_count = 0

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            collected.append(delta.content)
            token_count += 1
            if progress_callback is not None:
                fraction = min(0.95, token_count / estimated_total_tokens)
                progress_callback(fraction)

    if progress_callback is not None:
        progress_callback(1.0)

    result = "".join(collected).strip()
    logger.info(
        "GPT-5.4-nano returned %d chars (~%d tokens streamed).",
        len(result), token_count,
    )
    return result


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def combine_and_correct(
    transcripts: dict[str, str | None],
    selected: list[str],
    progress=None,
) -> tuple[str, float]:
    """Combine and LLM-correct selected transcripts against the Azure time grid.

    Parameters
    ----------
    transcripts:
        ``{"azure": text, "typhoon": text|None, "thonburian": text|None}``
        Azure must always be present and non-empty.
    selected:
        Non-Azure source keys the user opted in, e.g. ``["typhoon", "thonburian"]``.
    progress:
        Gradio ``gr.Progress()`` instance for live progress-bar updates (optional).

    Returns
    -------
    tuple[str, float]
        ``(corrected_transcript_text, elapsed_seconds)``
    """
    t0 = time.perf_counter()

    def _progress(fraction: float, desc: str = "") -> None:
        if progress is not None:
            progress(fraction, desc=desc)

    # ---- 1. Validate -------------------------------------------------------
    azure_text = (transcripts.get("azure") or "").strip()
    if not azure_text:
        raise ValueError("Azure transcript is required but is empty.")

    # ---- 2. Parse ----------------------------------------------------------
    _progress(0.02, desc="Parsing transcripts…")

    azure_segs = parse_transcript(azure_text)
    if not azure_segs:
        raise ValueError(
            "Could not parse Azure transcript — no valid "
            "[HH:MM:SS → HH:MM:SS] timestamp lines found."
        )

    other_segs: dict[str, list[dict]] = {}
    for src in selected:
        src_text = (transcripts.get(src) or "").strip()
        if src_text:
            parsed = parse_transcript(src_text)
            if parsed:
                other_segs[src] = parsed
                logger.info("Parsed %d segments from '%s'.", len(parsed), src)
            else:
                logger.warning(
                    "Source '%s' selected but no parseable timestamp lines found — skipping.", src
                )
        else:
            logger.warning("Source '%s' selected but transcript is empty — skipping.", src)

    # ---- 3. Align ----------------------------------------------------------
    _progress(0.05, desc="Aligning segments to Azure time grid…")

    aligned = align_to_azure(azure_segs, other_segs)
    logger.info(
        "Aligned %d segments. Active sources: azure + %s",
        len(aligned), list(other_segs.keys()),
    )

    # ---- 4. Connect to Azure OpenAI ----------------------------------------
    _progress(0.08, desc="Connecting to Azure OpenAI…")
    client = get_openai_client()

    # ---- 5. Single streaming LLM call --------------------------------------
    def _streaming_progress(fraction: float) -> None:
        # Map [0 → 1] streaming fraction to overall progress [0.10 → 0.97]
        _progress(0.10 + fraction * 0.87, desc="GPT-5.4-nano correcting transcript…")

    corrected = correct_all_segments(
        aligned,
        list(other_segs.keys()),
        client,
        progress_callback=_streaming_progress,
    )

    _progress(1.0, desc="Done!")
    elapsed = time.perf_counter() - t0
    logger.info("combine_and_correct finished in %.2fs.", elapsed)
    return corrected, elapsed
