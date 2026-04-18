# Simple Transcript Service Plan

This documentation pack describes a simple transcript service built with Gradio, two Thai open ASR models, and Azure Speech-to-Text, with PostgreSQL storage, Docker deployment, and hardware-aware inference planning.

## What this project is

This project is a practical v1 transcription service for internal use. It is designed to let a user upload or record audio in a Gradio interface, run multiple ASR engines, compare outputs, edit the final transcript, and store results in PostgreSQL.

The service is intentionally simpler than a full production transcription platform. It focuses on getting a working comparison workflow online quickly while keeping the architecture clean enough to grow later into review queues, diarization, fine-tuning, training pipelines, and MCP integration.

## Main stack

- **Frontend/UI:** Gradio, because it provides direct audio upload and microphone recording components and is easy to wire to Python inference flows.
- **Open ASR model 1:** `typhoon-ai/typhoon-whisper-large-v3` as the main Thai open ASR engine.
- **Open ASR model 2:** Thonburian Whisper as the second Thai open ASR engine for cross-checking and fallback.
- **Managed ASR:** Azure Speech-to-Text for cloud transcription, phrase boosting, and future diarization extension.
- **Database:** PostgreSQL for users, jobs, transcripts, app settings, vocabulary JSON, and edit history.
- **Deployment:** Docker-based local or server deployment.
- **Acceleration layer:** CUDA for NVIDIA GPU workloads and OpenVINO for Intel CPU, Intel GPU, and Intel NPU execution paths where supported.

## Why this hardware-aware design matters

This app should not assume that every host has the same accelerator. OpenVINO documents support for CPU, GPU, and NPU devices in its runtime, while Docker documents GPU access for NVIDIA devices through the Docker GPU runtime and NVIDIA Container Toolkit. That means the app should detect available acceleration at startup and choose the best execution path automatically.

In practical terms, your app should support these hardware modes:

1. **NVIDIA GPU + CUDA** for Whisper inference and training when CUDA is available.
2. **Intel CPU only** as the universal fallback.
3. **Intel Arc or Intel integrated GPU** through OpenVINO GPU execution where supported.
4. **Intel NPU** through OpenVINO NPU execution on supported systems.
5. **Azure-only mode** when local acceleration is unavailable or disabled.

## Recommended hardware execution strategy

### 1. NVIDIA path

Use this when the machine has an NVIDIA GPU and Docker GPU support is configured.

Best for:
- Faster Whisper inference.
- Fine-tuning and retraining jobs.
- Higher-throughput transcription workloads.

Checks to run:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If both pass, enable CUDA-backed Typhoon and Thonburian inference.

### 2. Intel CPU path

Use this as the universal fallback path. Every deployment should support CPU mode even if it is slower.

Best for:
- Local testing.
- Small workloads.
- Backup when no accelerator is available.

### 3. Intel GPU path with OpenVINO

OpenVINO supports GPU inference, including Intel GPU devices, and its GPU configuration documentation notes that Intel graphics devices require the proper runtime packages. For Intel Arc discrete GPUs on Linux, OpenVINO documents additional requirements such as supported kernels and driver-related packages.

Best for:
- Systems with Intel Arc GPUs.
- Intel-centric developer machines.
- Lower-cost inference environments where CUDA hardware is not available.

Checks to run:
- Confirm OpenVINO runtime is installed.
- Query available devices from OpenVINO Core.
- Verify that `GPU` appears in available devices.

### 4. Intel NPU path with OpenVINO

OpenVINO also supports NPU devices where the system and drivers expose them correctly. This is most relevant on supported Intel platforms with on-device NPUs.

Best for:
- Future low-power inference workflows.
- Experimental acceleration on supported Intel AI PCs.

Checks to run:
- Query OpenVINO available devices.
- Verify that `NPU` appears.
- Run a small smoke test model load.

### 5. Azure-only path

If no suitable local acceleration is available, or local model loading fails, the app should still function in Azure-only mode. That gives a safe fallback path for simple deployments.

## Hardware detection policy

At startup, the app should run a capability detector and record the results in the admin status panel.

The detector should report:
- CPU available: yes/no.
- NVIDIA GPU visible: yes/no.
- CUDA available in Python: yes/no.
- OpenVINO installed: yes/no.
- OpenVINO available devices list.
- Intel GPU visible to OpenVINO: yes/no.
- Intel NPU visible to OpenVINO: yes/no.
- Selected inference backend for each model.

Suggested selection logic:

- Prefer **CUDA** for Whisper if CUDA is available.
- Else prefer **OpenVINO GPU** if Intel GPU is available and the model path supports it.
- Else prefer **OpenVINO NPU** only for supported converted models and verified inference paths.
- Else fall back to **CPU**.
- Azure Speech remains available regardless of local hardware.

## Suggested startup checks

### NVIDIA / CUDA

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### OpenVINO device discovery

```python
from openvino.runtime import Core
core = Core()
print(core.available_devices)
```

Expected possible outputs include device names such as `CPU`, `GPU`, and `NPU` depending on the machine and runtime configuration.

## Recommended app behavior by hardware mode

### Mode A: NVIDIA server
- Run Typhoon and Thonburian with CUDA.
- Allow larger files and faster batch jobs.
- Enable future fine-tuning workflows.

### Mode B: Intel Arc workstation
- Run OpenVINO-enabled inference where available.
- Keep Azure as fallback if a model cannot run well through the selected Intel path.

### Mode C: Intel AI PC with NPU
- Use OpenVINO device discovery and route supported lightweight inference to NPU only after validation.
- Keep CPU or Azure fallback because not every model path will be NPU-ready.

### Mode D: CPU-only host
- Allow smaller jobs only.
- Warn user that open-model inference may be slower.
- Keep Azure available as the fastest practical option for some deployments.

## Files in this pack

- `01-scope.md`
- `02-model-selection.md`
- `03-gradio-ui-plan.md`
- `04-backend-flow.md`
- `05-database-plan.md`
- `06-docker-and-gpu.md`
- `07-azure-integration.md`
- `08-build-order.md`

## v1 goal

Build a small internal service where a user can upload or record audio, run transcription with Typhoon Whisper Large v3, Thonburian Whisper, Azure Speech, or all engines together, compare outputs side by side, and save the selected final transcript into PostgreSQL.

## Recommended next build order

1. Finish Docker and PostgreSQL base setup.
2. Add startup hardware detection for CUDA and OpenVINO devices.
3. Add Gradio upload and compare UI.
4. Add Azure transcription.
5. Add Typhoon transcription.
6. Add Thonburian transcription.
7. Save final reviewed transcript to PostgreSQL.
8. Add export and history views.
9. Add admin status page for GPU, CPU, Intel GPU, and NPU visibility.
