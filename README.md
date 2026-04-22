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

---

## วิธีปรับปรุงคุณภาพการถอดความ (Thai Guide — Improving Transcript Quality)

คู่มือนี้อธิบายแนวทางทั้งหมดที่โปรเจกต์นี้ใช้ในการปรับปรุงความแม่นยำของผลการถอดความ  
เรียงลำดับจากขั้นตอนแรก (เตรียมเสียง) ไปจนถึงขั้นตอนสุดท้าย (รวมผลลัพธ์ด้วย LLM)

---

### ขั้นที่ 1 — การลดเสียงรบกวนพื้นหลัง (Background Noise Reduction)

ก่อนที่ไฟล์เสียงจะถูกส่งไปยัง AI ทุกตัว ระบบจะผ่านกระบวนการปรับปรุงเสียง 3 ขั้นตอน  
(เปิดใช้งานได้โดยติ๊ก **"Audio Enhancement (Noise Reduction)"** ในหน้า UI)

#### ขั้นตอนที่ 1.1 — FFmpeg Bandpass Filter

ไฟล์เสียงต้นฉบับจะถูกแปลงผ่าน FFmpeg ก่อนเป็นอันดับแรก

| พารามิเตอร์ | ค่าปัจจุบัน | ความหมาย |
| --- | --- | --- |
| High-pass filter | 80 Hz | ตัดเสียงฮัมต่ำ เสียงแอร์ เสียงเครื่องยนต์ |
| Low-pass filter | 8,000 Hz | ตัดเสียงสูงผิดปกติ เสียงรบกวนความถี่สูง |
| Sample rate | 16,000 Hz (16 kHz) | ขนาดมาตรฐานสำหรับ Whisper และ Azure Speech |
| Format | 16-bit PCM mono WAV | รูปแบบที่ ASR ทุกตัวรองรับ |

ผลลัพธ์: เสียงสะอาดขึ้น ขนาดไฟล์เล็กลง ส่งเข้า AI ได้เร็วขึ้น

#### ขั้นตอนที่ 1.2 — Spectral Gating ด้วย noisereduce

หลังจาก FFmpeg ระบบจะใช้ไลบรารี `noisereduce` แบบ **Non-stationary spectral gating**

| พารามิเตอร์ | ค่าปัจจุบัน | ความหมาย |
| --- | --- | --- |
| `stationary` | `False` | ติดตามเสียงรบกวนที่เปลี่ยนแปลงตลอดเวลา (ไม่ใช่แค่ช่วงเงียบ) |
| `prop_decrease` | `0.75` | ลดเสียงรบกวนลง 75% |
| `n_fft` | `2048` | ขนาด window สำหรับวิเคราะห์ความถี่ |
| `freq_mask_smooth_hz` | `500 Hz` | ความนุ่มนวลของ masking ตามความถี่ |
| `time_mask_smooth_ms` | `50 ms` | ความนุ่มนวลของ masking ตามเวลา |

> **เหตุผลที่ใช้ non-stationary:** ในสภาพแวดล้อมจริง เช่น ห้องประชุม ร้านกาแฟ หรือสายโทรศัพท์  
> เสียงรบกวนจะเปลี่ยนแปลงตลอดเวลา ไม่ใช่เสียงคงที่ ดังนั้นโหมดนี้จะได้ผลดีกว่า

**การปรับแต่งเพิ่มเติมสำหรับกรณีพิเศษ:**

- ถ้าผู้พูดเสียงเบามาก → ลด `prop_decrease` เป็น `0.5` เพื่อไม่ให้ตัดเสียงพูดออกไปด้วย
- ถ้าสภาพแวดล้อมเสียงดังมาก → เพิ่ม `prop_decrease` เป็น `0.85–0.90`

#### ขั้นตอนที่ 1.3 — Pedalboard DSP Chain

ขั้นสุดท้ายของการปรับปรุงเสียง ใช้ไลบรารี `pedalboard` ของ Spotify  
ประกอบด้วยเครื่องมือ 3 ชิ้นต่อกันเป็น chain

```text
[เสียงเข้า] → NoiseGate → Compressor → Limiter → [ตรวจ peak] → [เสียงออก]
```

| เครื่องมือ | ค่าปัจจุบัน | หน้าที่ |
| --- | --- | --- |
| **NoiseGate** | threshold -50 dB, ratio 4:1, attack 5 ms, release 200 ms | ปิดเสียงช่วงที่เงียบจริงๆ (ใต้ -50 dB) เพื่อกำจัด hiss และ hum ระหว่างประโยค |
| **Compressor** | threshold -18 dB, ratio 2.5:1, attack 10 ms, release 200 ms | บีบ dynamic range — ลดเสียงที่ดังเกินไป ทำให้ระดับเสียงสม่ำเสมอมากขึ้น |
| **Limiter** | threshold -1 dB, release 50 ms | ป้องกัน clipping ไม่ให้เสียงเกิน -1 dBFS เด็ดขาด |

**ขั้นตอน Peak Normalization (conditional):**

หลัง limiter ระบบจะตรวจสอบ peak level ของสัญญาณ:

- ถ้า peak อยู่ระหว่าง `0.001 – 0.1` (เสียงเบามากต่ำกว่า -20 dBFS) → boost ขึ้นให้ peak = 0.1
- ถ้า peak สูงกว่า 0.1 → ไม่แตะ (ป้องกัน noise boost)

> **ข้อจำกัดปัจจุบัน:** Peak normalization มองแค่จุดสูงสุดของไฟล์ทั้งหมด  
> ถ้าไฟล์มีจุดเสียงดังแม้แต่จุดเดียว จะทำให้ช่วงที่เสียงเบากว่าไม่ได้รับการ boost  
> ดูแนวทางปรับปรุงในส่วน "การเพิ่มระดับเสียงผู้พูด" ด้านล่าง

---

### ขั้นที่ 2 — การเพิ่มระดับเสียงผู้พูดที่เบาเกินไป (Speaker Loudness Enhancement)

ปัญหาที่พบบ่อยในการบันทึกเสียงหลายคน: ผู้พูดบางคนอยู่ไกลไมค์หรือพูดเบาตามธรรมชาติ  
ทำให้หลังการถอดความ คำพูดของผู้พูดนั้นหายไปหรือถอดผิด

**สาเหตุที่เกิดจาก pipeline ปัจจุบัน:**

1. Compressor ทำหน้าที่บีบเสียงดัง **ลง** แต่ไม่ยก (lift) เสียงเบา **ขึ้น** เพราะไม่มี makeup gain
2. Peak normalization ไม่จัดการ intra-file loudness variation ได้ (ดูข้างต้น)
3. noisereduce ที่ 0.75 อาจตัดสระและพยัญชนะที่เสียงอ่อนออกด้วย

**แนวทางปรับปรุงใน `engines/preprocess.py`:**

#### วิธีที่ 1 — เปลี่ยนจาก Peak Normalization เป็น RMS Normalization

แทนที่จะดูแค่ peak, ให้คำนวณ RMS (Root Mean Square) ของทั้งไฟล์แล้ว normalize ให้ถึงเป้าหมาย:

```python
# เป้าหมาย RMS ที่ -20 dBFS สำหรับเสียงพูด
target_rms = 10 ** (-20 / 20)   # ≈ 0.1
rms = np.sqrt(np.mean(processed ** 2))
if 0.001 < rms < target_rms:
    processed = processed * (target_rms / rms)
    # clamp ป้องกัน clipping หลัง boost
    processed = np.clip(processed, -0.99, 0.99)
```

#### วิธีที่ 2 — เพิ่ม Makeup Gain ใน Compressor

`pedalboard.Compressor` ยังไม่มี makeup_gain parameter โดยตรง  
แต่สามารถเพิ่ม gain หลัง compressor ด้วย numpy ได้:

```python
# หลังจาก board(audio, sr) เพิ่ม makeup gain +4 dB
makeup_linear = 10 ** (4 / 20)  # ≈ 1.585
processed = board(audio, sr) * makeup_linear
```

#### วิธีที่ 3 — ปรับ Compressor เพื่อยกเสียงเบาขึ้น (Upward Compression)

เปลี่ยน threshold ให้ต่ำลงและ ratio น้อยลง เพื่อให้ compressor ทำงานกับเสียงปานกลางด้วย:

```python
# แบบ aggressive สำหรับเสียงหลายคนที่ดังต่างกันมาก
Compressor(threshold_db=-28, ratio=3.0, attack_ms=5.0, release_ms=150.0),
```

#### วิธีที่ 4 (ขั้นสูง) — Loudness Normalization มาตรฐาน EBU R128 / ITU-R BS.1770

ใช้ไลบรารี `pyloudnorm` เพื่อ normalize ให้ได้ค่า LUFS มาตรฐาน:

```bash
pip install pyloudnorm
```

```python
import pyloudnorm as pyln

meter = pyln.Meter(sr)  # สร้าง meter ที่ sample rate ของไฟล์
loudness = meter.integrated_loudness(audio_np)
# normalize ให้ได้ -18 LUFS (เหมาะสำหรับงาน ASR)
audio_normalized = pyln.normalize.loudness(audio_np, loudness, -18.0)
```

> **ค่าแนะนำ:** -18 LUFS ถึง -20 LUFS เหมาะที่สุดสำหรับ Whisper และ Azure Speech  
> เพราะ model เหล่านี้ถูก train มาด้วยเสียงพูดในช่วงนี้

---

### ขั้นที่ 3 — การแยกผู้พูด (Speaker Diarization)

Diarization คือการแยก **ว่าใครพูดตอนไหน** ในไฟล์เสียงที่มีหลายคน  
ระบบนี้ใช้โมเดล `pyannote/speaker-diarization-community-1` จาก HuggingFace

**วิธีเปิดใช้งาน:**

1. ติ๊ก **"Speaker Diarization"** ใน UI
2. ตั้งค่า **Min Speakers** และ **Max Speakers** (ถ้าไม่รู้ตั้ง min=1, max=5 ไว้ก่อน)
3. กด Transcribe

**ผลลัพธ์ที่ได้:**

```text
[00:00:05 → 00:00:12] SPEAKER_00: สวัสดีครับ วันนี้มาประชุมเรื่องอะไร
[00:00:13 → 00:00:20] SPEAKER_01: มาคุยเรื่องแผนงาน Q3 ค่ะ
[00:00:21 → 00:00:35] SPEAKER_00: โอเค งั้นเริ่มเลยนะครับ...
```

**การทำงานของ Diarization Pipeline:**

```text
[ไฟล์เสียง (หลัง enhance)] 
    → pyannote โมเดลวิเคราะห์ speaker embedding
    → แบ่ง segment ตาม speaker
    → จับคู่กับ Whisper chunks ตาม timestamp
    → รวมเป็น transcript พร้อม speaker label
```

**Fallback เมื่อ timestamp ไม่สมบูรณ์:**

- ถ้า Whisper ไม่ส่ง timestamp กลับมา (transformers บางเวอร์ชัน) ระบบจะประมาณ timestamp จากตำแหน่ง chunk เทียบกับความยาวไฟล์
- ถ้า diarization ล้มเหลวทั้งหมด ระบบจะ fallback เป็น timestamp เท่านั้น (ไม่มี speaker label)

**เคล็ดลับการใช้งาน:**

| สถานการณ์ | แนะนำ |
| --- | --- |
| ประชุม 2–3 คน | Min=2, Max=3 |
| สัมภาษณ์ผู้สมัคร | Min=2, Max=2 (ตายตัว) |
| สัมมนาหลายคน | Min=2, Max=8 หรือ auto |
| ไม่แน่ใจจำนวนผู้พูด | Min=1, Max=5 (ระบบจัดการเอง) |

---

### ขั้นที่ 4 — การถอดความด้วยสามเครื่องยนต์ (Triple-Engine Transcription)

หัวใจของระบบคือการรันเครื่องยนต์ ASR สามตัวพร้อมกัน และเปรียบเทียบผลลัพธ์

#### Azure Speech-to-Text

| คุณสมบัติ | รายละเอียด |
| --- | --- |
| ประเภท | Cloud API (Microsoft Azure Cognitive Services) |
| จุดแข็ง | Timestamp แม่นยำระดับ word, รองรับ phrase boosting, Speaker diarization ในตัว |
| จุดอ่อน | ต้องใช้อินเทอร์เน็ต, มีค่าใช้จ่ายตามปริมาณ |
| เหมาะกับ | เป็น baseline หลักสำหรับ time grid และการรวมผล |
| ภาษา | รองรับภาษาไทยพร้อม custom vocabulary |

Azure ถูกออกแบบให้เป็น **time-grid baseline** เพราะ timestamp ของมันแม่นยำที่สุด  
เมื่อรวมผลกับโมเดลอื่น Azure จะเป็นโครงโครงของ timeline

#### Typhoon Whisper Large v3

| คุณสมบัติ | รายละเอียด |
| --- | --- |
| ประเภท | Local open-source ASR (typhoon-ai/typhoon-whisper-large-v3) |
| จุดแข็ง | ถูก fine-tune เฉพาะภาษาไทย, รู้จำคำศัพท์เฉพาะทางไทยได้ดี |
| จุดอ่อน | ต้องการ GPU/NPU/CPU ที่มีกำลังพอ, โหลด model นานครั้งแรก |
| เหมาะกับ | เนื้อหาภาษาไทย โดยเฉพาะประโยคทั่วไปและภาษาพูด |
| Acceleration | CUDA (NVIDIA) หรือ OpenVINO (Intel) |

#### Thonburian Whisper

| คุณสมบัติ | รายละเอียด |
| --- | --- |
| ประเภท | Local open-source ASR (biodatlab/distill-whisper-th-large-v3) |
| จุดแข็ง | Distilled model — เร็วกว่า Typhoon, เหมาะกับ CPU/NPU |
| จุดอ่อน | Distillation อาจสูญเสีย accuracy บางส่วนเมื่อเทียบ full model |
| เหมาะกับ | การถอดความเร็ว, ใช้เป็น cross-check กับ Typhoon |
| Acceleration | CUDA หรือ OpenVINO |

**การทำงานแบบขนาน (Parallel Inference):**

```text
[ไฟล์เสียงที่ enhance แล้ว]
    ├──→ [Thread 1] Azure Speech API ────────────────→ azure_text
    ├──→ [Thread 2] Typhoon Whisper (local) ─────────→ typhoon_text
    └──→ [Thread 3] Thonburian Whisper (local) ──────→ thonburian_text
                                                            ↓
                                              แสดงผลแยก Tab ใน UI
```

ทั้งสามรันพร้อมกันใน `ThreadPoolExecutor(max_workers=3)` เพื่อประหยัดเวลา

---

### ขั้นที่ 5 — การรวมและแก้ไขผลลัพธ์ด้วย LLM (Combine & Correct with GPT)

นี่คือขั้นตอนสุดท้ายและทรงพลังที่สุด  
ระบบจะส่งผลลัพธ์จากหลายเครื่องยนต์ไปให้ GPT ช่วย **รวม แก้ไข และตรวจสอบ** ให้ได้ transcript ที่ดีที่สุด

**วิธีใช้งาน:**

1. รัน Transcribe ให้ครบทั้งสามเครื่องยนต์ก่อน
2. ติ๊กเลือก **"Include Typhoon in Combine & Correction"** และ/หรือ **"Include Thonburian in Combine & Correction"** ตามต้องการ
3. กดปุ่ม **"Combine & Correct with GPT-5.4-nano"**
4. ผลลัพธ์จะปรากฏใน **"Combined & Corrected Transcript"** ด้านล่าง

**หลักการทำงานของ llm_combine:**

```text
Input:
  azure_text   = "สวัสดี วันนี้ ประชุม เรื่อง..."  (baseline + timestamp)
  typhoon_text = "สวัสดีครับ วันนี้ประชุมเรื่อง..."
  thonburian_text = "สวัสดีคร้าบ วันนี้ประชุมเรื่อง..."

↓ ส่งเข้า Azure OpenAI (GPT-5.4-nano)
↓ Prompt บอก GPT ว่า: "Azure คือ baseline timeline,
                        Typhoon/Thonburian คือ cross-check สำหรับคำศัพท์ภาษาไทย"
↓ GPT ตรวจจับ: คำที่ตรงกัน 2–3 เครื่องยนต์ = น่าเชื่อถือ
                คำที่ตรงกันแค่เครื่องยนต์เดียว = ต้องพิจารณาจาก context
↓

Output:
  combined_text = "สวัสดีครับ วันนี้ประชุมเรื่อง..."  (ถูกต้องที่สุด)
```

**เหตุผลที่ Azure เป็น baseline:**

- Azure ให้ timestamp ระดับ word ที่แม่นยำ ทำให้รู้ว่าแต่ละคำพูดตอนไหน
- เมื่อ word ไม่ถูกต้อง Typhoon และ Thonburian ที่ถูก fine-tune ภาษาไทยโดยเฉพาะจะช่วยแก้

**เหตุผลที่ใช้ GPT แทนการ vote แบบ simple majority:**

- Simple voting ไม่เข้าใจ context ภาษาไทย (เช่น คำพ้องเสียง, ทับศัพท์)
- GPT สามารถเลือก version ที่ถูกต้องตาม grammar และ context ของประโยคทั้งหมด
- GPT แก้ไขคำที่ทุกเครื่องยนต์ถอดผิดหมด (เช่น ชื่อเฉพาะ, ศัพท์เทคนิค) โดยใช้ความรู้ทั่วไป

---

### สรุปกระบวนการทั้งหมด (End-to-End Flow)

```text
[ไฟล์เสียงต้นฉบับ (.mp3 / .wav / .m4a ฯลฯ)]
          ↓
┌─────────────────────────────────────────────┐
│  Stage 1: FFmpeg Bandpass                   │
│  80 Hz highpass + 8 kHz lowpass             │
│  → 16 kHz mono PCM WAV                      │
└─────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────┐
│  Stage 2: noisereduce Spectral Gating       │
│  Non-stationary, prop_decrease=0.75         │
│  → ลดเสียงรบกวนพื้นหลังที่เปลี่ยนแปลง      │
└─────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────┐
│  Stage 3: Pedalboard DSP Chain              │
│  NoiseGate(-50dB) → Compressor(-18dB,2.5:1) │
│  → Limiter(-1dB) → Peak Normalize           │
│  → ปรับระดับเสียงให้สม่ำเสมอ               │
└─────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────┐
│  [ถ้าเปิด Diarization]                      │
│  pyannote speaker-diarization-community-1   │
│  → แยก segment ว่าใครพูดตอนไหน             │
└─────────────────────────────────────────────┘
          ↓ (ไฟล์เดียวกัน ส่งแบบขนาน)
┌──────────────┬──────────────┬───────────────┐
│ Azure Speech │    Typhoon   │  Thonburian   │
│  (Cloud API) │  (Local GPU) │  (Local GPU)  │
│              │              │               │
│ timestamp    │ Thai vocab   │ Thai vocab    │
│ word-level   │ fine-tuned   │ distilled     │
└──────────────┴──────────────┴───────────────┘
          ↓
┌─────────────────────────────────────────────┐
│  [ถ้าเปิด Diarization]                      │
│  assign_speakers()                          │
│  → จับคู่ chunk Whisper กับ speaker segment │
│  → แสดง [SPEAKER_00], [SPEAKER_01] ฯลฯ    │
└─────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────┐
│  LLM Combine & Correct                      │
│  GPT-5.4-nano via Azure OpenAI              │
│  Azure = baseline timeline                  │
│  Typhoon + Thonburian = Thai language check │
│  → Cross-engine correction                  │
│  → ได้ transcript ที่ถูกต้องที่สุด          │
└─────────────────────────────────────────────┘
          ↓
    [Combined & Corrected Transcript]
    พร้อม Download เป็น .txt
```

---

### ตารางสรุปแนวทางปรับปรุงเพิ่มเติม (Quick Reference)

| ปัญหา | สาเหตุที่เป็นไปได้ | วิธีแก้ที่แนะนำ |
| --- | --- | --- |
| ผู้พูดบางคนเสียงเบาเกินไป | Peak normalization ไม่ครอบคลุม intra-file | เปลี่ยนเป็น RMS หรือ LUFS normalization |
| คำพูดหายในช่วง noisy | `prop_decrease=0.75` ตัดเสียงพูดด้วย | ลดเป็น `0.50–0.65` |
| ประโยคต้น/ท้ายถูกตัด | NoiseGate ตัด soft onset/tail | ลด threshold เป็น -55 dB, ratio 2:1 |
| ถอดคำศัพท์เฉพาะทางผิด | ASR model ไม่รู้จำคำนั้น | เพิ่ม phrase list ใน Azure, ใช้ LLM combine |
| ผลต่างกันมากระหว่าง engine | สภาพเสียงยาก, accent, เสียงทับซ้อน | เพิ่ม enhancement + เปิด diarization |
| Diarization ผิดพลาด | ผู้พูดเสียงคล้ายกัน หรือพูดทับกัน | ตั้ง Min/Max speakers ให้แน่นอน |
| LLM แก้ผิด | Context ไม่เพียงพอ | เปิด Typhoon + Thonburian ทั้งคู่ก่อน combine |
