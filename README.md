# visiOnlink (IGEN 330)

Real-time obstacle awareness for visually impaired navigation by fusing **RGB camera detection (YOLO)** with an **8×8 VL53L5CX time-of-flight grid** streamed from an **ESP32**. The app runs on **Windows**, speaks alerts via **Windows SAPI**, and optionally describes the scene with **Qwen2-VL**. A single **1920×1080** dashboard shows the camera, ToF heatmap, and alert log. visiOnlink is discretely embedded within a vest, allowing visually impaired users to feel safe and confident in public.

---

## Features

- **YOLO object detection** — Custom-trained weights (e.g. `best6.pt`) with classes such as stairs, doors, vehicles, furniture, signage, etc.
- **ToF fusion** — Bounding boxes are mapped onto an 8×8 distance grid; alerts require **close range** in the overlapping cells plus **two-frame confirmation** to suppress spikes (with defined exceptions below).
- **Spoken alerts** — Template: *“Alert: there is a … to your …, … centimeters away”*; **global audio cooldown** avoids TTS overload.
- **Three modes** — Normal (excludes low-priority classes), Everything (+ sensor-only “unidentified” hints), Emergency (all classes; **Exit Sign** can alert without ToF).
- **Scene description (optional)** — Headphone **play/pause** (single tap, after debounce) or **`S`** key runs **Qwen2-VL-2B** on a frozen frame (GPU recommended).
- **Mode switching** — Headphone **next track** or **`M`**; play/pause **double-tap** can also cycle mode.
- **Unified UI** — One OpenCV window: title bar, ~1000×1000 camera pane, ToF panel, rolling audio-alert list.

---

## Architecture (high level)

```
USB Camera ──► preprocess (rotate + pad to square) ──► YOLO ──► bboxes
 │
ESP32 + VL53L5CX ──► serial (8×8 mm) ──► background thread ──► fuse ──► TTS + dashboard
```

- **Main thread:** capture, inference, fusion, drawing, `cv2.imshow`.
- **Sensor thread:** `pyserial` `readline()` parsing; updates shared `8×8` `numpy` buffer under a lock.
- **TTS:** Daemon thread feeding one long-lived **PowerShell** `System.Speech` process (low latency vs spawning per phrase).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `camera_and_boundingboxes_F.py` | Main application: camera, YOLO, fusion, UI, TTS, modes, optional Qwen, `pynput` input. |
| `working_cam_sensor/vl53l5cx_sensor.py` | ESP32 serial client; optional I2C path for direct sensor (not default on Windows). |
| `esp32_sensorcode_F/esp32_sensorcode_V2_F/esp32_sensorcode_V2_F.ino` | Firmware: VL53L5CX8×8 @ 15 Hz, CSV rows + frame delimiter, **250000** baud. |
| `working_cam_sensor/vl53l5cx_sensor_rpi.py` | Alternate helper for Raspberry Pi–style setups. |
| `best_ncnn_model/` | NCNN export assets (optional deployment path). |

Place your **Ultralytics `.pt` checkpoint** next to the main script (see **Configuration**).

---

## Requirements

- **OS:** Windows 10/11 (tested with USB webcam + USB-serial ESP32).
- **Python:** 3.10+ recommended (3.13 used in development).
- **GPU:** Strongly recommended for YOLO; **NVIDIA GPU + CUDA** for Qwen2-VL scene mode.

### Python dependencies (core)

```bash
pip install ultralytics opencv-python numpy pyserial pynput
```

### Optional — scene description (Qwen2-VL)

```bash
pip install transformers torch qwen-vl-utils Pillow accelerate
```

On first scene request the model is downloaded from Hugging Face (`Qwen/Qwen2-VL-2B-Instruct`). Set `device_map` / dtype in code if you use CPU or different VRAM.

---

## Hardware

1. **Webcam** — USB; index is set in code (`cv2.VideoCapture(1)` — change to `0` if needed).
2. **ESP32** + **VL53L5CX** (e.g. SparkFun board) flashed with `esp32_sensorcode_V2_F.ino`.
3. **USB cable** to PC; note the **COM port** (auto-detect prefers common USB-serial chips; override in code if required).

---

## Firmware & serial protocol

- **Baud:** `250000` (fallbacks exist in Python init).
- **Frame format:** 8 lines of `d0,d1,…,d7` (mm or `0` invalid), then a line with space + newline as end-of-frame marker. See the `.ino` for details.

---

## Configuration (quick edits in `camera_and_boundingboxes_F.py`)

| Item | Typical location / notes |
|------|---------------------------|
| YOLO weights | `model = YOLO("best6.pt")` — path to your `.pt` file. |
| Class list | `OBSTACLE_CLASSES` must match `model.names` from your checkpoint. |
| Camera index | `cv2.VideoCapture(1)` |
| Serial port | `VL53L5CXSensor(port=None, …)` auto-detect; or `port='COM5'`. |
| ToF “close” threshold | `CLOSE_THRESHOLD_MM` |
| Alert confidence | `ALERT_CONFIDENCE_THRESHOLD` |
| Min time between spoken alerts | `AUDIO_GLOBAL_COOLDOWN` |
| Demo auto-exit | `MAX_RUNTIME` — set to `None` for indefinite run. |
| Scene model | `SCENE_MODEL_NAME` |

**Camera orientation:** `_prepare_camera_frame` rotates **90° CCW** and **pads** to a square (`max(w,h)`) so the full FOV is kept; adjust if your mount differs.

---

## Run

From the repo root (with venv activated):

```bash
python camera_and_boundingboxes_F.py
```

- **Esc** — quit (when the OpenCV window is focused).
- **M** — cycle mode (headphones optional).
- **S** — scene description (if optional deps installed).
- Headphones: **play/pause** (scene / double-tap mode), **next track** (mode).

---

## Design notes (for contributors)

- **Fusion:** Most alerts need **YOLO + ToF close +2-frame confirmation**. **Exit Sign in Emergency** is camera-only by design.
- **Stairs:** Sensor cells are **not** limited to upper rows so bottom-of-frame stairs still fuse with ToF.
- **Dashboard:** Fixed **1920×1080** canvas to reduce aspect-ratio stretch on Full HD displays; ToF thumbnail is scaled **uniformly** (square) to avoid a stretched grid.

---

## Troubleshooting

| Issue | Things to check |
|-------|------------------|
| No sensor data | COM port, baud, ESP32 firmware running, `verbose=True` on sensor init. |
| Wrong camera | Change `VideoCapture` index. |
| No speech | PowerShell available; default/female SAPI voice; not muted. |
| Media keys ignored | Use **M** / **S**; Bluetooth routing varies on Windows. |
| Scene mode fails | GPU/CUDA, `transformers`/`torch` versions, Hugging Face access. |

---

## License / course context

Developed for **UBC IGEN 330** (or equivalent) assistive-technology coursework.

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- [OpenCV](https://opencv.org/)  
- [Qwen2-VL](https://huggingface.co/Qwen) (optional)  
- SparkFun / ST VL53L5CX ecosystem
