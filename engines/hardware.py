"""Hardware detection — NVIDIA CUDA first, then OpenVINO (Intel GPU/CPU/NPU).

Automatically selects the best available backend:
  1. NVIDIA CUDA (if torch+CUDA available)
  2. OpenVINO GPU (Intel iGPU / dGPU)
  3. OpenVINO CPU (fallback)
"""

import logging
import os
import shutil

logger = logging.getLogger(__name__)

_hw_cache: list = []  # populated on first call; single-element list avoids global statement


def _check_torch() -> dict:
    """Probe PyTorch and CUDA availability."""
    result = {"torch_version": None, "cuda": False, "cuda_device_count": 0,
              "cuda_device_name": "", "cuda_vram_mb": 0}
    try:
        import torch
        result["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            result["cuda"] = True
            result["cuda_device_count"] = torch.cuda.device_count()
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                result["cuda_vram_mb"] = total // (1024 * 1024)
            except (RuntimeError, AttributeError):
                pass
    except (ImportError, RuntimeError, OSError) as exc:
        logger.debug("PyTorch/CUDA probe failed: %s", exc)
    return result


def _check_openvino() -> dict:
    """Probe OpenVINO runtime and available devices."""
    result = {"openvino_version": None, "available_devices": [],
              "cpu": False, "gpu": False, "npu": False}
    try:
        from openvino import Core, get_version
        result["openvino_version"] = get_version()
        devices = Core().available_devices
        result["available_devices"] = devices
        result["cpu"] = "CPU" in devices
        result["gpu"] = any(d.startswith("GPU") for d in devices)
        result["npu"] = "NPU" in devices
    except ImportError:
        logger.debug("OpenVINO not installed.")
    except (RuntimeError, AttributeError, OSError) as exc:
        logger.debug("OpenVINO probe failed: %s", exc)
    return result


def _check_ffmpeg() -> str | None:
    """Return ffmpeg path if found on PATH, else None."""
    return shutil.which("ffmpeg")


def detect_hardware() -> dict:
    """Detect available compute devices and select the preferred one.

    Priority order: NVIDIA CUDA > OpenVINO GPU (Intel) > OpenVINO CPU

    Returns a dict with keys:
        cuda, cuda_device_count, cuda_device_name, cuda_vram_mb,
        torch_version, available_devices, cpu, gpu, npu,
        openvino_version, ffmpeg, selected_device, backend
    """
    if _hw_cache:
        return _hw_cache[0]

    torch_info = _check_torch()
    ov_info = _check_openvino()
    ffmpeg_path = _check_ffmpeg()

    info = {
        # torch / CUDA
        "torch_version": torch_info["torch_version"],
        "cuda": torch_info["cuda"],
        "cuda_device_count": torch_info["cuda_device_count"],
        "cuda_device_name": torch_info["cuda_device_name"],
        "cuda_vram_mb": torch_info["cuda_vram_mb"],
        # OpenVINO
        "openvino_version": ov_info["openvino_version"],
        "available_devices": ov_info["available_devices"],
        "cpu": ov_info["cpu"],
        "gpu": ov_info["gpu"],
        "npu": ov_info["npu"],
        # FFmpeg
        "ffmpeg": ffmpeg_path,
        # selection (filled below)
        "selected_device": "CPU",
        "backend": "openvino",
    }

    # --- Select backend ---
    if info["cuda"]:
        info["selected_device"] = "cuda"
        info["backend"] = "cuda"
        logger.info("NVIDIA CUDA detected: %s (%d MB VRAM, x%d) — using CUDA backend.",
                    info["cuda_device_name"], info["cuda_vram_mb"], info["cuda_device_count"])
    elif ov_info["openvino_version"]:
        devices = ov_info["available_devices"]
        env_device = os.getenv("OV_DEVICE", "").upper()
        if env_device == "AUTO":
            info["selected_device"] = "AUTO"
        elif env_device and env_device in devices:
            info["selected_device"] = env_device
        elif env_device and any(d.startswith(env_device + ".") for d in devices):
            info["selected_device"] = env_device
        elif info["gpu"]:
            info["selected_device"] = next(d for d in devices if d.startswith("GPU"))
        else:
            info["selected_device"] = "CPU"
        logger.info("OpenVINO %s | Devices: %s | Selected: %s",
                    ov_info["openvino_version"], devices, info["selected_device"])
    else:
        logger.warning("No CUDA and no OpenVINO — falling back to CPU (PyTorch).")
        info["backend"] = "cpu"
        info["selected_device"] = "cpu"

    if ffmpeg_path:
        logger.info("FFmpeg found: %s", ffmpeg_path)
    else:
        logger.warning("FFmpeg not found on PATH — audio preprocessing may fail.")

    logger.info("Torch %s | Backend: %s | Device: %s",
                info["torch_version"], info["backend"].upper(), info["selected_device"])

    _hw_cache.append(info)
    return info


def hardware_summary() -> str:
    """Return a Markdown summary for the Gradio UI footer."""
    hw = detect_hardware()
    lines = [
        "### Hardware Status",
        f"- **Backend:** {hw['backend'].upper()}",
        f"- **PyTorch:** {hw['torch_version'] or 'not installed'}",
    ]
    if hw["cuda"]:
        lines.append(
            f"- **NVIDIA GPU:** {hw['cuda_device_name']}"
            f" ({hw['cuda_vram_mb']} MB VRAM,"
            f" x{hw['cuda_device_count']})"
        )
    lines += [
        f"- **OpenVINO:** {hw['openvino_version'] or 'not installed'}",
        f"- **OpenVINO devices:** {', '.join(hw['available_devices']) or 'none'}",
        f"- **Selected device:** {hw['selected_device']}",
        f"- **FFmpeg:** {'available' if hw['ffmpeg'] else 'NOT FOUND'}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(hardware_summary())
