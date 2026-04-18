"""Simple Transcription App — Gradio UI with tabbed ASR engine output."""

import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

import gradio as gr

from engines.hardware import detect_hardware, hardware_summary

# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

LANGUAGES = {
    "Thai": {"whisper": "thai", "azure": "th-TH"},
    "English": {"whisper": "english", "azure": "en-US"},
    "Chinese": {"whisper": "chinese", "azure": "zh-CN"},
    "Japanese": {"whisper": "japanese", "azure": "ja-JP"},
    "Korean": {"whisper": "korean", "azure": "ko-KR"},
}

# ---------------------------------------------------------------------------
# Model preloading state
# ---------------------------------------------------------------------------

_SKIP_LOCAL = os.getenv("SKIP_LOCAL_MODELS", "0").strip() == "1"

_models_ready = threading.Event()
_load_status = {
    "typhoon": "skipped" if _SKIP_LOCAL else "pending",
    "thonburian": "skipped" if _SKIP_LOCAL else "pending",
}

if _SKIP_LOCAL:
    _models_ready.set()  # unlock UI immediately
    logger.info("SKIP_LOCAL_MODELS=1 — Whisper models skipped, Azure-only mode.")


def _preload_models():
    """Pre-load both Whisper models in background threads."""
    if _SKIP_LOCAL:
        return

    def load_typhoon():
        try:
            _load_status["typhoon"] = "loading..."
            from engines.typhoon_asr import load_model
            load_model()
            _load_status["typhoon"] = "ready"
            logger.info("Typhoon model loaded.")
        except Exception as exc:
            _load_status["typhoon"] = f"FAILED: {exc}"
            logger.error("Typhoon load failed: %s", exc, exc_info=True)

    def load_thonburian():
        try:
            _load_status["thonburian"] = "loading..."
            from engines.thonburian_asr import load_model
            load_model()
            _load_status["thonburian"] = "ready"
            logger.info("Thonburian model loaded.")
        except Exception as exc:
            _load_status["thonburian"] = f"FAILED: {exc}"
            logger.error("Thonburian load failed: %s", exc, exc_info=True)

    t1 = threading.Thread(target=load_typhoon, daemon=True)
    t2 = threading.Thread(target=load_thonburian, daemon=True)
    t1.start()
    t2.start()

    def _wait_for_both():
        t1.join()
        t2.join()
        _models_ready.set()
        logger.info("All models ready.")

    threading.Thread(target=_wait_for_both, daemon=True).start()


def _get_load_status() -> str:
    """Return current model loading status as markdown."""
    if _SKIP_LOCAL:
        return "### Azure-Only Mode (SKIP_LOCAL_MODELS=1)\n- Typhoon Whisper: skipped\n- Thonburian Whisper: skipped\n- Azure Speech: ready"
    ty = _load_status["typhoon"]
    th = _load_status["thonburian"]
    ready = _models_ready.is_set()
    if ready and ty == "ready" and th == "ready":
        return "### All Models Ready\n- Typhoon Whisper: ready\n- Thonburian Whisper: ready\n- Azure Speech: ready"
    lines = ["### Loading Models..."]
    lines.append(f"- Typhoon Whisper: **{ty}**")
    lines.append(f"- Thonburian Whisper: **{th}**")
    if not ready:
        lines.append("\n*Audio upload will be enabled once both models are loaded.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine dispatch helpers
# ---------------------------------------------------------------------------

def _run_azure(audio_path: str, language: str, min_speakers: int, max_speakers: int) -> tuple[str, float]:
    from engines.azure_asr import transcribe_azure
    azure_lang = LANGUAGES.get(language, LANGUAGES["Thai"])["azure"]
    t0 = time.perf_counter()
    text = transcribe_azure(audio_path, language=azure_lang, min_speakers=min_speakers, max_speakers=max_speakers)
    elapsed = time.perf_counter() - t0
    return text, elapsed


def _run_typhoon(audio_path: str, language: str, min_speakers: int, max_speakers: int) -> tuple[str, float]:
    if _SKIP_LOCAL:
        return "(Typhoon skipped — set SKIP_LOCAL_MODELS=0 and restart after models are downloaded)", 0.0
    from engines.typhoon_asr import transcribe_typhoon
    whisper_lang = LANGUAGES.get(language, LANGUAGES["Thai"])["whisper"]
    t0 = time.perf_counter()
    text = transcribe_typhoon(audio_path, language=whisper_lang, min_speakers=min_speakers, max_speakers=max_speakers)
    elapsed = time.perf_counter() - t0
    return text, elapsed


def _run_thonburian(audio_path: str, language: str, min_speakers: int, max_speakers: int) -> tuple[str, float]:
    if _SKIP_LOCAL:
        return "(Thonburian skipped — set SKIP_LOCAL_MODELS=0 and restart after models are downloaded)", 0.0
    from engines.thonburian_asr import transcribe_thonburian
    whisper_lang = LANGUAGES.get(language, LANGUAGES["Thai"])["whisper"]
    t0 = time.perf_counter()
    text = transcribe_thonburian(audio_path, language=whisper_lang, min_speakers=min_speakers, max_speakers=max_speakers)
    elapsed = time.perf_counter() - t0
    return text, elapsed


# ---------------------------------------------------------------------------
# Main transcription callback
# ---------------------------------------------------------------------------

def transcribe(audio_path, selected_engines, language, diarization, min_speakers, max_speakers, enhance):
    """Run selected ASR engines in parallel threads and return results."""
    if not audio_path:
        empty = "(no audio provided)"
        return empty, "", empty, "", empty, ""

    if not _models_ready.is_set():
        msg = "Models are still loading, please wait..."
        return msg, "", msg, "", msg, ""

    # Use enhanced audio if enhancement is enabled
    process_path = audio_path
    if enhance:
        from engines.preprocess import preprocess_audio
        process_path = preprocess_audio(audio_path)

    # Pass min/max speakers to engines (0 = disabled / auto)
    n_min = int(min_speakers) if diarization else 0
    n_max = int(max_speakers) if diarization else 0

    results: dict[str, tuple[str, float]] = {}

    # Run all selected engines in parallel threads
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for name in selected_engines:
            if name == "Azure Speech":
                futures[pool.submit(_run_azure, process_path, language, n_min, n_max)] = name
            elif name == "Typhoon Whisper":
                futures[pool.submit(_run_typhoon, process_path, language, n_min, n_max)] = name
            elif name == "Thonburian Whisper":
                futures[pool.submit(_run_thonburian, process_path, language, n_min, n_max)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                text, elapsed = future.result()
                results[name] = (text, elapsed)
                logger.info("%s finished in %.2fs", name, elapsed)
            except Exception as exc:
                results[name] = (f"ERROR: {exc}", 0.0)
                logger.error("%s failed: %s", name, exc, exc_info=True)

    # Build outputs: (transcript, timing) for each tab
    outputs = []
    for engine_name in ["Azure Speech", "Typhoon Whisper", "Thonburian Whisper"]:
        if engine_name in results:
            text, elapsed = results[engine_name]
            outputs.append(text)
            outputs.append(f"{elapsed:.2f}s")
        elif engine_name in selected_engines:
            outputs.append("(failed)")
            outputs.append("")
        else:
            outputs.append("")
            outputs.append("(not selected)")

    return tuple(outputs)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    hw_md = hardware_summary()

    with gr.Blocks(title="Simple Transcription Service") as app:
        gr.Markdown("# Simple Transcription Service")
        gr.Markdown("Upload or record audio, then transcribe with one or more ASR engines in parallel.")

        # Model loading status — updated by Timer below
        load_status = gr.Markdown(_get_load_status())

        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio Input",
                    interactive=False,  # disabled until models load
                )

            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Thai",
                    label="Language",
                )
                engine_selector = gr.CheckboxGroup(
                    choices=["Azure Speech", "Typhoon Whisper", "Thonburian Whisper"],
                    value=["Azure Speech", "Typhoon Whisper", "Thonburian Whisper"],
                    label="Select Engines",
                )
                enhance = gr.Checkbox(
                    label="Audio Enhancement (Noise Reduction)",
                    value=False,
                )
                diarization = gr.Checkbox(
                    label="Speaker Diarization",
                    value=False,
                )
                with gr.Row(visible=False) as speakers_row:
                    min_speakers = gr.Slider(
                        minimum=1, maximum=10, step=1, value=2,
                        label="Min Speakers",
                    )
                    max_speakers = gr.Slider(
                        minimum=1, maximum=10, step=1, value=3,
                        label="Max Speakers",
                    )
                transcribe_btn = gr.Button(
                    "Transcribe", variant="primary", size="lg",
                    interactive=False,  # disabled until models load
                )

        # Show/hide speaker sliders based on diarization checkbox
        diarization.change(
            fn=lambda d: gr.update(visible=d),
            inputs=[diarization],
            outputs=[speakers_row],
        )

        # Audio playback — original vs enhanced
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Original Audio")
                audio_original = gr.Audio(
                    value=None, label="Original", interactive=False, type="filepath",
                )
            with gr.Column():
                gr.Markdown("#### Enhanced Audio")
                audio_enhanced = gr.Audio(
                    value=None, label="Enhanced", interactive=False, type="filepath",
                )

        # Mirror uploaded audio into the Original player immediately
        audio_input.change(
            fn=lambda a: a,
            inputs=[audio_input],
            outputs=[audio_original],
        )

        # Run preprocessing when enhancement is toggled ON (if audio exists)
        def _run_enhance(audio_path, do_enhance):
            if not audio_path or not do_enhance:
                return None
            from engines.preprocess import preprocess_audio
            return preprocess_audio(audio_path)

        enhance.change(
            fn=_run_enhance,
            inputs=[audio_input, enhance],
            outputs=[audio_enhanced],
        )

        # Also run preprocessing when new audio is uploaded (if enhancement is ON)
        audio_input.change(
            fn=_run_enhance,
            inputs=[audio_input, enhance],
            outputs=[audio_enhanced],
        )

        # Tabbed output — one tab per engine
        with gr.Tabs():
            with gr.TabItem("Azure Speech"):
                azure_text = gr.Textbox(label="Transcript", lines=10, interactive=False)
                azure_time = gr.Textbox(label="Elapsed Time", interactive=False, max_lines=1)

            with gr.TabItem("Typhoon Whisper"):
                typhoon_text = gr.Textbox(label="Transcript", lines=10, interactive=False)
                typhoon_time = gr.Textbox(label="Elapsed Time", interactive=False, max_lines=1)

            with gr.TabItem("Thonburian Whisper"):
                thonburian_text = gr.Textbox(label="Transcript", lines=10, interactive=False)
                thonburian_time = gr.Textbox(label="Elapsed Time", interactive=False, max_lines=1)

        gr.Markdown(hw_md)

        # Wire up transcribe button
        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, engine_selector, language, diarization, min_speakers, max_speakers, enhance],
            outputs=[
                azure_text, azure_time,
                typhoon_text, typhoon_time,
                thonburian_text, thonburian_time,
            ],
        )

        # Timer to poll model loading status and enable UI when ready
        timer = gr.Timer(value=2)

        def check_ready():
            ready = _models_ready.is_set()
            return (
                _get_load_status(),
                gr.update(interactive=ready),  # audio_input
                gr.update(interactive=ready),  # transcribe_btn
            )

        timer.tick(
            fn=check_ready,
            outputs=[load_status, audio_input, transcribe_btn],
        )

    return app


if __name__ == "__main__":
    # Start model preloading before UI launches
    _preload_models()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
