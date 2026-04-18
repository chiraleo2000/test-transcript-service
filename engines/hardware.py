"""Hardware detection for OpenVINO devices (CPU, GPU, NPU)."""

import logging
import os

logger = logging.getLogger(__name__)

_hw_info: dict | None = None


def detect_hardware() -> dict:
    """Detect available OpenVINO devices and select the preferred one.

    Returns a dict with keys:
        available_devices: list[str]  e.g. ["CPU", "GPU", "NPU"]
        cpu: bool
        gpu: bool
        npu: bool
        selected_device: str  - the device to use for inference
    """
    global _hw_info
    if _hw_info is not None:
        return _hw_info

    info = {
        "available_devices": [],
        "cpu": False,
        "gpu": False,
        "npu": False,
        "openvino_version": None,
        "selected_device": "CPU",
    }

    try:
        from openvino import Core, get_version

        info["openvino_version"] = get_version()
        core = Core()
        devices = core.available_devices
        info["available_devices"] = devices
        info["cpu"] = "CPU" in devices
        info["gpu"] = "GPU" in devices
        info["npu"] = "NPU" in devices

        # User override via env
        env_device = os.getenv("OV_DEVICE", "").upper()
        if env_device and env_device in devices:
            info["selected_device"] = env_device
        elif env_device == "AUTO":
            info["selected_device"] = "AUTO"
        elif "GPU" in devices:
            info["selected_device"] = "GPU"
        else:
            info["selected_device"] = "CPU"

        logger.info("OpenVINO %s | Devices: %s | Selected: %s",
                     info["openvino_version"], devices, info["selected_device"])
    except ImportError:
        logger.warning("OpenVINO is not installed – falling back to CPU.")
    except Exception as exc:
        logger.warning("OpenVINO device detection failed: %s", exc)

    _hw_info = info
    return info


def hardware_summary() -> str:
    """Return a Markdown summary for the Gradio UI footer."""
    hw = detect_hardware()
    lines = [
        "### Hardware Status",
        f"- **OpenVINO version:** {hw['openvino_version'] or 'not installed'}",
        f"- **Available devices:** {', '.join(hw['available_devices']) or 'none'}",
        f"- **GPU detected:** {'yes' if hw['gpu'] else 'no'}",
        f"- **NPU detected:** {'yes' if hw['npu'] else 'no'}",
        f"- **Selected device:** {hw['selected_device']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(hardware_summary())
