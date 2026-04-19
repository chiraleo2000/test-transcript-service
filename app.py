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
from gradio.themes import Soft as _SoftTheme

from engines.hardware import detect_hardware, hardware_summary

# ---------------------------------------------------------------------------
# Engine name / label constants  (avoid duplicating literals)
# ---------------------------------------------------------------------------

ENGINE_AZURE = "Azure Speech"
ENGINE_TYPHOON = "Typhoon Whisper"
ENGINE_THONBURIAN = "Thonburian Whisper"
LABEL_ELAPSED = "Elapsed Time"
LABEL_DOWNLOAD = "Download .txt"

ALL_ENGINES = [ENGINE_AZURE, ENGINE_TYPHOON, ENGINE_THONBURIAN]

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
    # Load sequentially — concurrent from_pretrained with sharded
    # checkpoints can leave meta tensors due to shared global state.
    t1.start()
    t1.join()
    t2.start()

    def _wait_for_both():
        t2.join()
        _models_ready.set()
        logger.info("All models ready.")

    threading.Thread(target=_wait_for_both, daemon=True).start()


def _get_load_status() -> str:
    """Return current model loading status as markdown."""
    if _SKIP_LOCAL:
        return (
            f"### Azure-Only Mode (SKIP_LOCAL_MODELS=1)\n"
            f"- {ENGINE_TYPHOON}: skipped\n"
            f"- {ENGINE_THONBURIAN}: skipped\n"
            f"- {ENGINE_AZURE}: ready"
        )
    ty = _load_status["typhoon"]
    th = _load_status["thonburian"]
    ready = _models_ready.is_set()
    if ready and ty == "ready" and th == "ready":
        return (
            "### All Models Ready\n"
            f"- {ENGINE_TYPHOON}: ready\n"
            f"- {ENGINE_THONBURIAN}: ready\n"
            f"- {ENGINE_AZURE}: ready"
        )
    lines = ["### Loading Models..."]
    lines.append(f"- {ENGINE_TYPHOON}: **{ty}**")
    lines.append(f"- {ENGINE_THONBURIAN}: **{th}**")
    if not ready:
        lines.append("\n*Audio upload will be enabled once both models are loaded.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine dispatch helpers
# ---------------------------------------------------------------------------

def _run_azure(
    audio_path: str, language: str,
    min_speakers: int, max_speakers: int,
    diar_segments: list | None = None,  # unused — Azure handles diarization natively
) -> tuple[str, float]:
    from engines.azure_asr import transcribe_azure
    azure_lang = LANGUAGES.get(language, LANGUAGES["Thai"])["azure"]
    t0 = time.perf_counter()
    text = transcribe_azure(
        audio_path, language=azure_lang,
        min_speakers=min_speakers, max_speakers=max_speakers,
    )
    elapsed = time.perf_counter() - t0
    return text, elapsed


def _run_typhoon(
    audio_path: str, language: str,
    min_speakers: int, max_speakers: int,
    diar_segments: list | None = None,
) -> tuple[str, float]:
    if _SKIP_LOCAL:
        return (
            "(Typhoon skipped — set SKIP_LOCAL_MODELS=0"
            " and restart after models are downloaded)", 0.0,
        )
    from engines.typhoon_asr import transcribe_typhoon
    whisper_lang = LANGUAGES.get(language, LANGUAGES["Thai"])["whisper"]
    t0 = time.perf_counter()
    text = transcribe_typhoon(
        audio_path, language=whisper_lang,
        diarization_segments=diar_segments,
    )
    elapsed = time.perf_counter() - t0
    return text, elapsed


def _run_thonburian(
    audio_path: str, language: str,
    min_speakers: int, max_speakers: int,
    diar_segments: list | None = None,
) -> tuple[str, float]:
    if _SKIP_LOCAL:
        return (
            "(Thonburian skipped — set SKIP_LOCAL_MODELS=0"
            " and restart after models are downloaded)", 0.0,
        )
    from engines.thonburian_asr import transcribe_thonburian
    whisper_lang = LANGUAGES.get(language, LANGUAGES["Thai"])["whisper"]
    t0 = time.perf_counter()
    text = transcribe_thonburian(
        audio_path, language=whisper_lang,
        diarization_segments=diar_segments,
    )
    elapsed = time.perf_counter() - t0
    return text, elapsed


# Map engine names → runner functions
_ENGINE_RUNNERS: dict = {
    ENGINE_AZURE: _run_azure,
    ENGINE_TYPHOON: _run_typhoon,
    ENGINE_THONBURIAN: _run_thonburian,
}

# ---------------------------------------------------------------------------
# Main transcription callback
# ---------------------------------------------------------------------------


def _save_transcript(engine_name: str, text: str) -> str | None:
    """Write transcript to a temp .txt file and return its path."""
    import tempfile
    if not text or text.startswith(("(", "ERROR")):
        return None
    safe_name = engine_name.replace(" ", "_")
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
            prefix=f"{safe_name}_transcript_", encoding="utf-8",
        )
        tmp.write(text)
        tmp.close()
        return tmp.name
    except OSError:
        return None


def _build_outputs(results: dict, selected_engines: list) -> tuple:
    """Build flat (text, elapsed, download_update, ...) tuple for all engine tabs."""
    outputs = []
    for engine_name in ALL_ENGINES:
        if engine_name in results:
            text, elapsed = results[engine_name]
            fpath = _save_transcript(engine_name, text)
            outputs.extend([
                text,
                f"{elapsed:.2f}s",
                gr.update(value=fpath, interactive=fpath is not None),
            ])
        elif engine_name in selected_engines:
            outputs.extend(["(failed)", "", gr.update(value=None, interactive=False)])
        else:
            outputs.extend(["", "(not selected)", gr.update(value=None, interactive=False)])
    return tuple(outputs)

def transcribe(
    audio_path, selected_engines, language,
    diarization, min_speakers, max_speakers, enhance,
):
    """Run selected ASR engines in parallel and return results."""
    if not audio_path:
        empty = "(no audio provided)"
        _no_dl = gr.update(value=None, interactive=False)
        return empty, "", _no_dl, empty, "", _no_dl, empty, "", _no_dl

    if not _models_ready.is_set():
        msg = "Models are still loading, please wait..."
        _no_dl = gr.update(value=None, interactive=False)
        return msg, "", _no_dl, msg, "", _no_dl, msg, "", _no_dl

    process_path = audio_path
    if enhance:
        from engines.preprocess import preprocess_audio
        process_path = preprocess_audio(audio_path)

    n_min = int(min_speakers) if diarization else 0
    n_max = int(max_speakers) if diarization else 0

    # Run pyannote diarization ONCE (before ASR threads) so both open-ASR
    # engines share the same result without a thread-safety race.
    # NOTE: n_min/n_max may both be 0 = auto mode — that is valid and must run.
    diar_segments: list | None = None
    open_asr_selected = any(
        e in selected_engines for e in [ENGINE_TYPHOON, ENGINE_THONBURIAN]
    )
    if diarization and open_asr_selected:
        try:
            from engines.diarization import diarize
            logger.info(
                "Running speaker diarization (min=%d, max=%d)...", n_min, n_max,
            )
            diar_segments = diarize(
                process_path, min_speakers=n_min, max_speakers=n_max,
            )
            logger.info(
                "Diarization complete: %d segments.", len(diar_segments),
            )
        except Exception as exc:
            logger.error("Diarization failed: %s", exc, exc_info=True)
            # Continue without diarization rather than aborting all engines

    results: dict[str, tuple[str, float]] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(runner, process_path, language, n_min, n_max, diar_segments): name
            for name in selected_engines
            if (runner := _ENGINE_RUNNERS.get(name)) is not None
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                text, elapsed = future.result()
                results[name] = (text, elapsed)
                logger.info("%s finished in %.2fs", name, elapsed)
            except Exception as exc:
                results[name] = (f"ERROR: {exc}", 0.0)
                logger.error("%s failed: %s", name, exc, exc_info=True)

    return _build_outputs(results, selected_engines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct the Gradio UI."""  # pylint: disable=too-many-locals
    hw_md = hardware_summary()

    with gr.Blocks(title="Simple Transcription Service") as demo:
        gr.Markdown("# Simple Transcription Service")
        gr.Markdown(
            "Upload or record audio, then transcribe "
            "with one or more ASR engines in parallel."
        )

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
                    choices=ALL_ENGINES,
                    value=ALL_ENGINES,
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
        diarization.change(  # pylint: disable=no-member
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
        audio_input.change(  # pylint: disable=no-member
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

        enhance.change(  # pylint: disable=no-member
            fn=_run_enhance,
            inputs=[audio_input, enhance],
            outputs=[audio_enhanced],
        )

        # Also run preprocessing when new audio is uploaded (if enhancement is ON)
        audio_input.change(  # pylint: disable=no-member
            fn=_run_enhance,
            inputs=[audio_input, enhance],
            outputs=[audio_enhanced],
        )

        # Tabbed output — one tab per engine
        with gr.Tabs():
            with gr.TabItem(ENGINE_AZURE):
                azure_text = gr.Textbox(
                    label="Transcript", lines=20, max_lines=200,
                    buttons=["copy"], interactive=False,
                )
                with gr.Row():
                    azure_time = gr.Textbox(
                        label=LABEL_ELAPSED, interactive=False,
                        max_lines=1, scale=3,
                    )
                    azure_dl = gr.DownloadButton(
                        label=LABEL_DOWNLOAD, value=None,
                        scale=1, visible=True, interactive=False,
                    )

            with gr.TabItem(ENGINE_TYPHOON):
                typhoon_text = gr.Textbox(
                    label="Transcript", lines=20, max_lines=200,
                    buttons=["copy"], interactive=False,
                )
                with gr.Row():
                    typhoon_time = gr.Textbox(
                        label=LABEL_ELAPSED, interactive=False,
                        max_lines=1, scale=3,
                    )
                    typhoon_dl = gr.DownloadButton(
                        label=LABEL_DOWNLOAD, value=None,
                        scale=1, visible=True, interactive=False,
                    )

            with gr.TabItem(ENGINE_THONBURIAN):
                thonburian_text = gr.Textbox(
                    label="Transcript", lines=20, max_lines=200,
                    buttons=["copy"], interactive=False,
                )
                with gr.Row():
                    thonburian_time = gr.Textbox(
                        label=LABEL_ELAPSED, interactive=False,
                        max_lines=1, scale=3,
                    )
                    thonburian_dl = gr.DownloadButton(
                        label=LABEL_DOWNLOAD, value=None,
                        scale=1, visible=True, interactive=False,
                    )

        gr.Markdown(hw_md)

        # Wire up transcribe button
        transcribe_btn.click(  # pylint: disable=no-member
            fn=transcribe,
            inputs=[
                audio_input, engine_selector, language,
                diarization, min_speakers, max_speakers, enhance,
            ],
            outputs=[
                azure_text, azure_time, azure_dl,
                typhoon_text, typhoon_time, typhoon_dl,
                thonburian_text, thonburian_time, thonburian_dl,
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

        timer.tick(  # pylint: disable=no-member
            fn=check_ready,
            outputs=[load_status, audio_input, transcribe_btn],
        )

    return demo


if __name__ == "__main__":
    # --- Startup resource check ---
    hw = detect_hardware()
    logger.info("=" * 60)
    logger.info("RESOURCE CHECK")
    logger.info("  Torch      : %s", hw["torch_version"] or "NOT INSTALLED")
    logger.info(
        "  CUDA       : %s",
        f"{hw['cuda_device_name']} ({hw['cuda_vram_mb']} MB)"
        if hw["cuda"] else "not available",
    )
    logger.info(
        "  OpenVINO   : %s",
        hw["openvino_version"] or "not installed",
    )
    logger.info(
        "  OV Devices : %s",
        ", ".join(hw["available_devices"]) or "none",
    )
    logger.info("  FFmpeg     : %s", hw["ffmpeg"] or "NOT FOUND")
    logger.info("  Backend    : %s", hw["backend"].upper())
    logger.info("  Device     : %s", hw["selected_device"])
    logger.info("=" * 60)

    # Start model preloading before UI launches
    _preload_models()
    application = build_ui()
    application.launch(
        server_name="0.0.0.0", server_port=7860,
        max_threads=40, theme=_SoftTheme(),
    )
