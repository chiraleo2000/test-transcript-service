"""Typhoon Whisper Large v3 — Thai ASR via OpenVINO."""

import logging
import os

logger = logging.getLogger(__name__)

MODEL_ID = "typhoon-ai/typhoon-whisper-large-v3"

_pipeline = None


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _get_pipeline():
    """Lazy-load the Typhoon Whisper pipeline with OpenVINO backend."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from transformers import AutoProcessor, pipeline
    from optimum.intel import OVModelForSpeechSeq2Seq
    from engines.hardware import detect_hardware

    hw = detect_hardware()
    device = hw["selected_device"]
    cache_dir = os.getenv("OV_CACHE_DIR", "./ov_cache")
    export_dir = os.path.join(cache_dir, "typhoon")
    hf_token = os.getenv("HF_TOKEN")

    logger.info("Loading Typhoon Whisper (%s) on device=%s ...", MODEL_ID, device)

    # Try loading from local cache first; otherwise export from HuggingFace
    if os.path.isdir(export_dir) and os.path.isfile(os.path.join(export_dir, "openvino_encoder_model.xml")):
        logger.info("Loading Typhoon from cached OpenVINO IR: %s", export_dir)
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            export_dir,
            device=device,
            compile=True,
        )
        processor = AutoProcessor.from_pretrained(export_dir)
    else:
        logger.info("Exporting Typhoon to OpenVINO IR (first run, may take several minutes)...")
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            export=True,
            device=device,
            compile=True,
            token=hf_token,
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
        # Save for future fast loading
        model.save_pretrained(export_dir)
        processor.save_pretrained(export_dir)
        logger.info("Typhoon OpenVINO IR saved to %s", export_dir)

    _pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        return_timestamps=True,
        max_new_tokens=448,
    )

    logger.info("Typhoon Whisper pipeline ready on %s.", device)
    return _pipeline


def load_model():
    """Pre-load the Typhoon Whisper model. Safe to call multiple times."""
    _get_pipeline()
    logger.info("Typhoon Whisper model pre-loaded.")


def transcribe_typhoon(audio_path: str, language: str = "thai", num_speakers: int = 0, min_speakers: int = 0, max_speakers: int = 0) -> str:
    """Transcribe audio using Typhoon Whisper Large v3.

    When num_speakers > 0 (or min/max set), runs pyannote diarization and labels each speaker.
    """
    pipe = _get_pipeline()
    result = pipe(audio_path, generate_kwargs={"language": language})

    if num_speakers > 0 or min_speakers > 0 or max_speakers > 0:
        from engines.diarization import diarize, assign_speakers
        segments = diarize(audio_path, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        return assign_speakers(result, segments)

    # Format chunks with timestamps (no diarization)
    chunks = result.get("chunks", [])
    if chunks:
        lines = []
        for chunk in chunks:
            ts = chunk.get("timestamp", (None, None))
            c_start, c_end = (ts if ts else (None, None))
            text = chunk.get("text", "").strip()
            if not text:
                continue
            if c_start is not None:
                lines.append(f"[{_fmt_ts(c_start)} \u2192 {_fmt_ts(c_end)}] {text}")
            else:
                lines.append(text)
        return "\n".join(lines) if lines else "(no speech detected)"

    return result.get("text", "").strip() or "(no speech detected)"
