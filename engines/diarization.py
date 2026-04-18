"""Shared speaker diarization using pyannote/speaker-diarization-community-1."""

import logging
import os
import shutil

logger = logging.getLogger(__name__)

MODEL_ID = "pyannote/speaker-diarization-community-1"

_diarization_pipeline = None


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        logger.warning(
            "ffmpeg not found in PATH — torchcodec audio decoding may fail. "
            "Install ffmpeg: https://ffmpeg.org/download.html"
        )


def _get_diarization_pipeline():
    """Lazy-load the pyannote community diarization pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    import torch
    from pyannote.audio import Pipeline

    _check_ffmpeg()

    hf_token = os.getenv("HF_TOKEN")
    logger.info("Loading pyannote speaker diarization pipeline (%s)...", MODEL_ID)

    _diarization_pipeline = Pipeline.from_pretrained(
        MODEL_ID,
        use_auth_token=hf_token,
    )

    # Use CUDA if available, otherwise CPU (pyannote uses PyTorch, not OpenVINO)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    _diarization_pipeline = _diarization_pipeline.to(device)
    logger.info("Pyannote diarization pipeline ready on %s.", device)
    return _diarization_pipeline


def diarize(audio_path: str, num_speakers: int = 0, min_speakers: int = 0, max_speakers: int = 0) -> list[dict]:
    """Run speaker diarization on audio file.

    Returns list of {"start": float, "end": float, "speaker": str}
    sorted by start time.
    """
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    pipe = _get_diarization_pipeline()

    kwargs = {}
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    elif min_speakers > 0 or max_speakers > 0:
        if min_speakers > 0:
            kwargs["min_speakers"] = min_speakers
        if max_speakers > 0:
            kwargs["max_speakers"] = max_speakers

    logger.info("Running diarization on %s (num_speakers=%s)...", audio_path, num_speakers or "auto")

    with ProgressHook() as hook:
        diarization = pipe(audio_path, hook=hook, **kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start":   turn.start,
            "end":     turn.end,
            "speaker": speaker,
        })

    segments.sort(key=lambda s: s["start"])

    # Remap pyannote labels (SPEAKER_00, SPEAKER_01 …) to 1-based (SPEAKER_01, SPEAKER_02 …)
    unique_speakers = sorted({s["speaker"] for s in segments})
    speaker_map = {spk: f"SPEAKER_{i + 1:02d}" for i, spk in enumerate(unique_speakers)}
    for s in segments:
        s["speaker"] = speaker_map[s["speaker"]]

    logger.info("Diarization complete: %d segments, speakers: %s",
                len(segments), sorted({s["speaker"] for s in segments}))
    return segments


def _find_speaker(start: float | None, end: float | None, segments: list[dict]) -> str:
    """Return the speaker with greatest time overlap in [start, end]."""
    if start is None or end is None or not segments:
        return "SPEAKER"

    best_speaker = "SPEAKER"
    best_overlap = 0.0

    for seg in segments:
        overlap = max(0.0, min(end, seg["end"]) - max(start, seg["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker


def assign_speakers(result: dict, diarization_segments: list[dict]) -> str:
    """Align Whisper output chunks with speaker segments.

    Args:
        result: Whisper pipeline output dict with "text" and "chunks" keys.
        diarization_segments: Output of diarize().
    Returns:
        Formatted transcript string with [SPEAKER_XX]: labels.
    """
    chunks = result.get("chunks", [])

    # Fallback: no timestamps → return plain text
    if not chunks:
        return result.get("text", "").strip() or "(no speech detected)"

    lines: list[str] = []
    current_speaker: str | None = None
    current_words: list[str] = []
    group_start: float | None = None
    group_end: float | None = None

    for chunk in chunks:
        ts = chunk.get("timestamp", (None, None))
        c_start, c_end = (ts if ts else (None, None))
        text = chunk.get("text", "").strip()
        if not text:
            continue

        speaker = _find_speaker(c_start, c_end, diarization_segments)

        if speaker != current_speaker:
            if current_words and current_speaker is not None:
                ts_str = f"[{_fmt_ts(group_start)} → {_fmt_ts(group_end)}] " if group_start is not None else ""
                lines.append(f"{ts_str}[{current_speaker}]: {' '.join(current_words)}")
            current_speaker = speaker
            current_words = [text]
            group_start = c_start
            group_end = c_end
        else:
            current_words.append(text)
            if c_end is not None:
                group_end = c_end

    if current_words and current_speaker is not None:
        ts_str = f"[{_fmt_ts(group_start)} → {_fmt_ts(group_end)}] " if group_start is not None else ""
        lines.append(f"{ts_str}[{current_speaker}]: {' '.join(current_words)}")

    return "\n".join(lines) if lines else "(no speech detected)"

