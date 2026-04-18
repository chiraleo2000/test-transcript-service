"""Azure Speech-to-Text engine using custom endpoint."""

import logging
import os
import threading
import time
import wave

logger = logging.getLogger(__name__)


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ensure_wav_16k(audio_path: str) -> str:
    """Convert audio to 16kHz mono PCM WAV. Always writes converted file to avoid
    sending stereo or wrong-rate audio to Azure (librosa.load bug: always returns
    sr=16000 so the sr!=16000 check was always False for WAV files)."""
    import librosa
    import soundfile as sf

    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    wav_path = os.path.splitext(audio_path)[0] + "_16k.wav"
    sf.write(wav_path, y, 16000, subtype="PCM_16")
    return wav_path


def _get_wav_duration(wav_path: str) -> float:
    """Get duration of a WAV file in seconds."""
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def _transcribe_standard(speech_config, audio_config) -> str:
    """Standard continuous recognition (no diarization)."""
    import azure.cognitiveservices.speech as speechsdk

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    all_results: list[str] = []
    done_event = threading.Event()

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            offset_s = evt.result.offset / 10_000_000  # ticks to seconds
            duration_s = evt.result.duration / 10_000_000
            end_s = offset_s + duration_s
            ts_str = f"[{_fmt_ts(offset_s)} \u2192 {_fmt_ts(end_s)}] "
            all_results.append(f"{ts_str}{evt.result.text}")

    def on_canceled(evt):
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            logger.error("Azure STT error: %s", evt.cancellation_details.error_details)
        done_event.set()

    def on_session_stopped(evt):
        done_event.set()

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    recognizer.start_continuous_recognition()
    done_event.wait(timeout=300)
    recognizer.stop_continuous_recognition()

    return "\n".join(all_results) if all_results else "(no speech detected)"


def _transcribe_with_diarization(speech_config, audio_config, min_speakers: int = 0, max_speakers: int = 0) -> str:
    """Conversation transcription with speaker diarization."""
    import azure.cognitiveservices.speech as speechsdk

    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    all_results: list[str] = []
    done_event = threading.Event()
    _speaker_map: dict[str, str] = {}

    def _normalize_speaker(raw_id: str) -> str:
        """Map any Azure speaker ID to SPEAKER_01, SPEAKER_02, …"""
        if not raw_id or raw_id.lower() in ("unknown", ""):
            raw_id = "__unknown__"
        if raw_id not in _speaker_map:
            _speaker_map[raw_id] = f"SPEAKER_{len(_speaker_map) + 1:02d}"
        return _speaker_map[raw_id]

    def on_transcribed(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            speaker = _normalize_speaker(evt.result.speaker_id or "")
            offset_s = evt.result.offset / 10_000_000
            duration_s = evt.result.duration / 10_000_000
            end_s = offset_s + duration_s
            ts_str = f"[{_fmt_ts(offset_s)} \u2192 {_fmt_ts(end_s)}] "
            all_results.append(f"{ts_str}[{speaker}]: {evt.result.text}")

    def on_canceled(evt):
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            logger.error("Azure diarization error: %s", evt.cancellation_details.error_details)
        done_event.set()

    def on_session_stopped(evt):
        done_event.set()

    transcriber.transcribed.connect(on_transcribed)
    transcriber.canceled.connect(on_canceled)
    transcriber.session_stopped.connect(on_session_stopped)

    transcriber.start_transcribing_async().get()
    done_event.wait(timeout=300)
    transcriber.stop_transcribing_async().get()

    return "\n".join(all_results) if all_results else "(no speech detected)"


def transcribe_azure(
    audio_path: str,
    language: str = "th-TH",
    num_speakers: int = 0,
    min_speakers: int = 0,
    max_speakers: int = 0,
) -> str:
    """Transcribe audio using Azure Speech-to-Text.

    When num_speakers > 0 (or min/max_speakers set), uses ConversationTranscriber
    for speaker diarization.
    Requires AZURE_SPEECH_KEY and AZURE_SPEECH_ENDPOINT env vars.
    """
    import azure.cognitiveservices.speech as speechsdk

    key = os.getenv("AZURE_SPEECH_KEY")
    endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")
    if not key or not endpoint:
        raise RuntimeError(
            "Set AZURE_SPEECH_KEY and AZURE_SPEECH_ENDPOINT in your .env file."
        )

    wav_path = _ensure_wav_16k(audio_path)

    speech_config = speechsdk.SpeechConfig(
        subscription=key,
        endpoint=endpoint,
    )
    speech_config.speech_recognition_language = language

    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)

    if num_speakers > 0:
        return _transcribe_with_diarization(speech_config, audio_config, min_speakers=0, max_speakers=0)
    elif min_speakers > 0 or max_speakers > 0:
        return _transcribe_with_diarization(speech_config, audio_config, min_speakers=min_speakers, max_speakers=max_speakers)
    else:
        return _transcribe_standard(speech_config, audio_config)
