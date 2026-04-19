"""Pre-export Whisper models to OpenVINO IR format.

Run this once after setup to avoid slow first-run export when launching app.py.
Usage:  python scripts/export_models.py
"""

import logging
import os
import sys

# Add project root so engines package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Authenticate with HuggingFace Hub using token from .env
_hf_token = os.getenv("HF_TOKEN")
if _hf_token and _hf_token != "your-hf-token-here":
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)
    logger.info("Logged in to HuggingFace Hub.")
else:
    logger.warning("HF_TOKEN not set — private/gated models will fail to download.")


def export_model(model_id: str, subfolder: str):
    """Download or export a single model for the detected backend."""
    from transformers import AutoProcessor
    from engines.hardware import detect_hardware

    hw = detect_hardware()
    device = hw["selected_device"]
    hf_token = os.getenv("HF_TOKEN")

    if hw["backend"] == "cuda":
        # CUDA: pre-download weights to HuggingFace cache (no OV conversion needed)
        import torch
        from transformers import AutoModelForSpeechSeq2Seq

        logger.info("CUDA mode — pre-downloading %s to HuggingFace cache ...", model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=torch.float16,
            use_safetensors=True,
            token=hf_token,
        )
        AutoProcessor.from_pretrained(model_id, token=hf_token)
        del model  # free memory; weights are now in HF cache
        logger.info("Downloaded %s — ready for CUDA inference.", model_id)
        return

    # --- OpenVINO export path ---
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq

    cache_dir = os.getenv("OV_CACHE_DIR", "./ov_cache")
    export_dir = os.path.join(cache_dir, subfolder)

    if os.path.isdir(export_dir) and os.path.isfile(
        os.path.join(export_dir, "openvino_encoder_model.xml")
    ):
        logger.info("Model already exported at %s — skipping.", export_dir)
        return

    logger.info("Exporting %s to OpenVINO IR (device=%s) ...", model_id, device)
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        export=True,
        device=device,
        compile=True,
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    model.save_pretrained(export_dir)
    processor.save_pretrained(export_dir)
    logger.info("Saved to %s", export_dir)


def main():
    """Export all configured ASR models."""
    logger.info("=== Pre-exporting Typhoon Whisper Large v3 ===")
    export_model("typhoon-ai/typhoon-whisper-large-v3", "typhoon")

    logger.info("=== Pre-exporting Thonburian Whisper Large v3 ===")
    export_model("biodatlab/distill-whisper-th-large-v3", "thonburian")

    logger.info("=== All models exported. ===")


if __name__ == "__main__":
    main()
