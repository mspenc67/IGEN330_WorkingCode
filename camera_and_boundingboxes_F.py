from ultralytics import YOLO
import cv2
import time
import threading
import queue as _queue
from collections import deque

import numpy as np
import subprocess
import os
from working_cam_sensor.vl53l5cx_sensor import VL53L5CXSensor

# Optional scene-description support (Qwen2-VL + keyboard hotkey)
# Install: pip install transformers torch qwen-vl-utils keyboard Pillow accelerate
try:
    from PIL import Image as _PILImage
    import torch as _torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    _has_scene = True
except ImportError:
    _PILImage = None          # type: ignore
    _torch = None             # type: ignore
    Qwen2VLForConditionalGeneration = None  # type: ignore
    AutoProcessor = None      # type: ignore
    process_vision_info = None  # type: ignore
    _has_scene = False


from datetime import datetime


def _sanitize_speech_text(text: str) -> str:
    """Escape single-quotes for PowerShell string literals."""
    return text.replace("'", "''")


# ── Persistent TTS engine ─────────────────────────────────────────────────────
# One PowerShell process lives for the whole program lifetime.
# speak_text() writes a line to its stdin — speech starts in ~30 ms instead of
# the ~400 ms it takes to spawn a fresh powershell.exe each call.
# A female voice is selected once at startup; SpeakAsync lets the loop return
# immediately so the next alert can cancel-and-replace the current one.

_TTS_PS_PREAMBLE = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Rate = 4; "
    "foreach ($v in $s.GetInstalledVoices()) { "
    "  if ($v.VoiceInfo.Gender -eq "
    "[System.Speech.Synthesis.VoiceGender]::Female) { "
    "    $s.SelectVoice($v.VoiceInfo.Name); break } }; "
    "while ($true) { "
    "  $t = [Console]::ReadLine(); "
    "  if ($null -eq $t) { break }; "
    "  if ($t -eq '__CANCEL__') { $s.SpeakAsyncCancelAll(); continue }; "
    "  $s.SpeakAsyncCancelAll(); $s.SpeakAsync($t) "
    "}"
)

_tts_queue: _queue.Queue = _queue.Queue()
_tts_ps_proc = None


def _tts_worker() -> None:
    """Feed the persistent PowerShell TTS process from the queue."""
    global _tts_ps_proc
    try:
        _tts_ps_proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-Command", _TTS_PS_PREAMBLE],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
    except Exception as exc:
        print(f"ALERT: persistent TTS failed to start ({exc})")
        return

    while True:
        text = _tts_queue.get()
        if text is None:
            break
        try:
            _tts_ps_proc.stdin.write(text + "\n")
            _tts_ps_proc.stdin.flush()
        except Exception:
            break


_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak_text(text: str) -> None:
    """Speak *text* via the warm persistent TTS process (~30 ms latency).

    Drops any queued-but-not-yet-spoken item so the freshest alert always wins.
    """
    # Discard stale queued items — only the latest alert matters
    while True:
        try:
            _tts_queue.get_nowait()
        except _queue.Empty:
            break
    _tts_queue.put_nowait(text)


# ── Alert-log panel ───────────────────────────────────────────────────────────
# Rolling list of the last N spoken alerts; rendered as a small cv2 window.
ALERT_LOG_MAX    = 14        # rows to keep
ALERT_PANEL_W    = 620
ALERT_PANEL_H    = 320

_alert_log: deque = deque(maxlen=ALERT_LOG_MAX)   # (hh:mm:ss, text, bgr_colour)
_alert_log_lock  = threading.Lock()

# colour palette for the panel
_CLR_OBSTACLE  = (0,  200, 255)   # amber  — close obstacle
_CLR_UNKNOWN   = (0,  80,  220)   # red    — unidentified sensor hit
_CLR_MODE      = (180, 255, 100)  # green  — mode change
_CLR_SCENE     = (255, 200,  80)  # blue   — scene description event
_CLR_DIM       = (120, 120, 120)  # grey   — timestamp / divider


def _log_alert(text: str, color: tuple = _CLR_OBSTACLE, speak: bool = True) -> None:
    """Append *text* to the on-screen alert log and optionally speak it."""
    ts = datetime.now().strftime('%H:%M:%S')
    with _alert_log_lock:
        _alert_log.appendleft((ts, text, color))
    if speak:
        speak_text(text)


def _natural_direction(direction: str) -> str:
    """Convert a grid direction string to a natural-sounding TTS phrase."""
    d = direction.lower()
    if d == "center":
        return "ahead of you"
    if d == "upper":
        return "above you"
    if d == "bottom":
        return "below you"
    return f"to your {d}"


def _format_alert_text(label: str, direction: str, distance_mm: int | None) -> str:
    """Return the spoken + displayed alert string in the standard format."""
    dir_phrase = _natural_direction(direction)
    if distance_mm is not None:
        dist_cm = max(1, round(distance_mm / 10))
        return f"There is a {label} {dir_phrase}, {dist_cm} centimeters away"
    return f"There is a {label} {dir_phrase}"


def create_alert_panel() -> np.ndarray:
    """Render the rolling alert log as a small BGR image for cv2.imshow."""
    canvas = np.full((ALERT_PANEL_H, ALERT_PANEL_W, 3), 25, dtype=np.uint8)

    # ── header bar ──────────────────────────────────────────────────────────
    with _mode_lock:
        mode_name = MODES[current_mode]["name"]
    cv2.rectangle(canvas, (0, 0), (ALERT_PANEL_W, 34), (45, 45, 45), -1)
    cv2.putText(canvas, f"Audio Alerts",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    mode_label = f"Mode {current_mode}: {mode_name}"
    (tw, _), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(canvas, mode_label,
                (ALERT_PANEL_W - tw - 8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CLR_MODE, 1)

    # divider
    cv2.line(canvas, (0, 35), (ALERT_PANEL_W, 35), (60, 60, 60), 1)

    # ── log rows ────────────────────────────────────────────────────────────
    with _alert_log_lock:
        entries = list(_alert_log)

    y = 56
    row_h = 20
    for ts, text, color in entries:
        if y + row_h > ALERT_PANEL_H:
            break
        # timestamp
        cv2.putText(canvas, ts, (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _CLR_DIM, 1)
        # alert text — truncate if wider than panel (ASCII "..." avoids OpenCV Unicode issues)
        max_chars = 62
        display = text if len(text) <= max_chars else text[:max_chars - 3] + "..."
        cv2.putText(canvas, display, (76, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        y += row_h

    if not entries:
        cv2.putText(canvas, "No alerts yet.", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _CLR_DIM, 1)

    return canvas


# ── Unified dashboard ─────────────────────────────────────────────────────────
# 1920×1080 canvas: matches Full HD so fullscreen is not stretched; camera is 1000².
# Video is square canvas (pad-only after rotate): full FOV, then scaled to 1000² slot.
#
#   Total width 1920 = 1000 (camera) + 920 (ToF + alerts, stacked to same height)

_FONT          = cv2.FONT_HERSHEY_DUPLEX   # more professional than SIMPLEX
_DASH_TITLE_H  = 80
_DASH_CAM_W    = 1000                      # square camera panel — max on this layout
_DASH_CAM_H    = 1000
_DASH_PAD      = 10                        # inner padding for all panels
_DASH_SIDE_W   = 1920 - _DASH_CAM_W        # 920
_DASH_GRID_H   = 520
_DASH_ALERT_H  = _DASH_CAM_H - _DASH_GRID_H   # 480
_DASH_TOTAL_W  = 1920
_DASH_TOTAL_H  = _DASH_TITLE_H + _DASH_CAM_H  # 1080

_DASH_ACCENT   = (194, 120, 40)    # BGR — gold/amber accent
_DASH_BG       = (14, 14, 14)      # near-black background
_DASH_WIN_NAME = "Accessibility Navigation Assistant"


def _prepare_camera_frame(img: np.ndarray) -> np.ndarray:
    """Rotate 90° CCW, then pad to a square with **no cropping**.

    Square side = max(width, height) so **every** camera pixel is kept. That is
    a wider field of view than center-cropping to min(w, h), which looked zoomed.
    Expect letterbox/pillar bands (_DASH_BG) on one axis when the sensor is not square.
    """
    if img is None or img.size == 0:
        return img
    rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = rot.shape[:2]
    side = max(w, h)
    out = np.full((side, side, 3), _DASH_BG, dtype=np.uint8)
    x0 = (side - w) // 2
    y0 = (side - h) // 2
    out[y0 : y0 + h, x0 : x0 + w] = rot
    return out


def _draw_dashboard(cam_frame: np.ndarray,
                    sensor_grid: np.ndarray) -> np.ndarray:
    """Compose camera, sensor grid, and alert log into one 1920×1080 image."""
    canvas = np.full((_DASH_TOTAL_H, _DASH_TOTAL_W, 3), _DASH_BG, dtype=np.uint8)
    top = _DASH_TITLE_H
    sx  = _DASH_CAM_W
    F   = _FONT

    # ── Title bar ─────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (_DASH_TOTAL_W, _DASH_TITLE_H), (22, 22, 22), -1)
    # accent stripe on the left
    cv2.rectangle(canvas, (0, 0), (8, _DASH_TITLE_H), _DASH_ACCENT, -1)

    cv2.putText(canvas, "Accessibility Navigation Assistant",
                (24, 46), F, 1.0, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "IGEN 330  |  Real-Time Obstacle Detection",
                (24, 70), F, 0.44, (130, 130, 130), 1, cv2.LINE_AA)

    # mode badge — right-aligned
    with _mode_lock:
        _m    = current_mode
        _mname = MODES[_m]["name"]
    badge = f"Mode {_m}:  {_mname}"
    (bw, bh), _ = cv2.getTextSize(badge, F, 0.58, 1)
    bx = _DASH_TOTAL_W - bw - 20
    by = _DASH_TITLE_H // 2 + bh // 2 - 2
    cv2.rectangle(canvas, (bx - 12, by - bh - 8), (bx + bw + 8, by + 8),
                  (38, 38, 38), -1)
    cv2.rectangle(canvas, (bx - 12, by - bh - 8), (bx + bw + 8, by + 8),
                  _DASH_ACCENT, 1)
    cv2.putText(canvas, badge, (bx, by), F, 0.58, _CLR_MODE, 1, cv2.LINE_AA)

    # bottom border of title bar
    cv2.line(canvas, (0, _DASH_TITLE_H - 1),
             (_DASH_TOTAL_W, _DASH_TITLE_H - 1), _DASH_ACCENT, 2)

    # ── Camera panel (square canvas from _prepare_camera_frame; resize to slot) ─
    ch, cw = cam_frame.shape[:2]
    if cw != _DASH_CAM_W or ch != _DASH_CAM_H:
        interp = cv2.INTER_AREA if max(ch, cw) > _DASH_CAM_W else cv2.INTER_LINEAR
        cam_disp = cv2.resize(cam_frame, (_DASH_CAM_W, _DASH_CAM_H), interpolation=interp)
    else:
        cam_disp = cam_frame
    canvas[top:top + _DASH_CAM_H, 0:_DASH_CAM_W] = cam_disp

    # red alert flash border around camera
    since_alert = time.time() - _last_audio_time
    if since_alert < 0.7:
        intensity = int(220 * (1.0 - since_alert / 0.7))
        cv2.rectangle(canvas, (2, top + 2),
                      (_DASH_CAM_W - 3, top + _DASH_CAM_H - 3),
                      (0, 0, intensity), 5)

    # ── Panel label helper ────────────────────────────────────────────────────
    def _panel_header(img, label, y0=0, w=None):
        """Draw a section header bar with label text."""
        w = w or img.shape[1]
        cv2.rectangle(img, (0, y0), (w, y0 + 28), (32, 32, 32), -1)
        cv2.rectangle(img, (0, y0), (5, y0 + 28), _DASH_ACCENT, -1)
        cv2.putText(img, label, (12, y0 + 20),
                    F, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(img, (0, y0 + 28), (w, y0 + 28), (55, 55, 55), 1)

    # ── Sensor grid panel (8×8 stays square — no non-uniform stretch) ──────
    grid_panel = np.full((_DASH_GRID_H, _DASH_SIDE_W, 3), 18, dtype=np.uint8)
    _panel_header(grid_panel, "ToF Distance Sensor  (8 x 8)")
    canvas[top:top + _DASH_GRID_H, sx:sx + _DASH_SIDE_W] = grid_panel

    inner_w = _DASH_SIDE_W - 2 * _DASH_PAD
    inner_h = _DASH_GRID_H - 28 - 2 * _DASH_PAD
    sq = min(inner_w, inner_h)
    # NEAREST keeps cell edges crisp on the square grid
    ginner = cv2.resize(sensor_grid, (sq, sq), interpolation=cv2.INTER_NEAREST)
    gx_off = sx + _DASH_PAD + (inner_w - sq) // 2
    gy_off = top + 28 + _DASH_PAD + (inner_h - sq) // 2
    canvas[gy_off : gy_off + sq, gx_off : gx_off + sq] = ginner

    # divider between grid and alert panel
    div_y = top + _DASH_GRID_H
    cv2.line(canvas, (sx, div_y), (sx + _DASH_SIDE_W, div_y), (55, 55, 55), 1)

    # ── Alert log panel ───────────────────────────────────────────────────────
    ay = div_y
    alert_panel = np.full((_DASH_ALERT_H, _DASH_SIDE_W, 3), 18, dtype=np.uint8)
    _panel_header(alert_panel, "Audio Alerts")

    with _alert_log_lock:
        entries = list(_alert_log)

    log_y  = 28 + _DASH_PAD + 18   # start below header
    row_h  = 26                     # generous row spacing
    ts_x   = _DASH_PAD
    txt_x  = ts_x + 72
    max_w  = _DASH_SIDE_W - txt_x - _DASH_PAD

    for ts, text, color in entries:
        if log_y + row_h > _DASH_ALERT_H - 4:
            break
        cv2.putText(alert_panel, ts, (ts_x, log_y),
                    F, 0.38, _CLR_DIM, 1, cv2.LINE_AA)
        # fit text to available width
        scale = 0.50
        (tw, _), _ = cv2.getTextSize(text, F, scale, 1)
        while tw > max_w and len(text) > 10:
            text = text[:-4] + "..."
            (tw, _), _ = cv2.getTextSize(text, F, scale, 1)
        cv2.putText(alert_panel, text, (txt_x, log_y),
                    F, scale, color, 1, cv2.LINE_AA)
        log_y += row_h

    if not entries:
        cv2.putText(alert_panel, "No alerts yet.",
                    (_DASH_PAD, 28 + _DASH_PAD + 22),
                    F, 0.5, _CLR_DIM, 1, cv2.LINE_AA)

    canvas[ay:ay + _DASH_ALERT_H, sx:sx + _DASH_SIDE_W] = alert_panel

    # ── Vertical divider between camera and right column ─────────────────────
    cv2.line(canvas, (sx, top), (sx, _DASH_TOTAL_H), (55, 55, 55), 1)

    return canvas


# ── Scene-description mode ────────────────────────────────────────────────────
# Triggered by a single press of the headphone power/BT button.
# The button sends a "play/pause media" HID event on most BT headsets when
# already connected.  If your X15 sends a different key, change the constant.
SCENE_TRIGGER_KEY = "play/pause media"
MODE_CYCLE_KEY    = "next track"       # headphone forward/next button cycles detection mode

# Qwen2-VL model to use.  2B is fast; swap to "Qwen/Qwen2-VL-7B-Instruct"
# for higher quality if your hardware allows it.
SCENE_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Prompt: require complete sentences so the spoken result never cuts off mid-thought.
SCENE_PROMPT = (
    "You are assisting a visually impaired person. "
    "Describe the scene in front of them in 2-3 complete sentences. "
    "Every sentence must end with a period. Do not start a sentence you cannot finish. "
    "Focus on the most important objects, people, or hazards."
)

# Shared state
_scene_active = threading.Event()   # set while a scene description is running
_current_frame = None               # latest camera frame (updated each main-loop iteration)
_qwen_model = None                  # lazily loaded
_qwen_processor = None


def _load_qwen():
    """Load Qwen2-VL model and processor on first use (lazy to keep startup fast)."""
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return True
    if not _has_scene:
        print("ALERT: scene-description packages not installed. "
              "Run: pip install transformers torch qwen-vl-utils Pillow accelerate")
        return False
    try:
        print("Loading Qwen2-VL model (first use — may take a moment)…")
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            SCENE_MODEL_NAME,
            torch_dtype=_torch.bfloat16,  # explicit bfloat16 — fits in 6 GB VRAM
            device_map="cuda:0",           # force GPU, not CPU fallback
        )
        _qwen_processor = AutoProcessor.from_pretrained(SCENE_MODEL_NAME)
        print("Qwen2-VL model ready.")
        return True
    except Exception as e:
        print(f"ALERT: could not load Qwen2-VL model: {e}")
        return False


def _speak_blocking(text: str) -> None:
    """Cancel any async speech and speak *text* via the warm persistent process.

    Puts a __CANCEL__ through the queue first to stop anything in progress,
    then sends the new text and waits long enough for it to finish.
    No new PowerShell process is spawned, so latency is ~30 ms.
    """
    # Drain stale queued items
    while True:
        try:
            _tts_queue.get_nowait()
        except _queue.Empty:
            break
    # Cancel current speech then speak the new text
    _tts_queue.put_nowait("__CANCEL__")
    _tts_queue.put_nowait(text)
    # Estimate how long the phrase will take at Rate=4 (~110 wpm → ~1.8 char/s)
    wait_s = max(2.0, len(text) / 14.0)
    time.sleep(wait_s)


def _run_scene_description(frame_bgr):
    """Run Qwen2-VL on *frame_bgr* and speak the result.  Runs in a background thread."""
    _scene_active.set()
    print("[SCENE] _run_scene_description() started")
    try:
        _log_alert("Analyzing scene. Please wait.", color=_CLR_SCENE, speak=False)
        _speak_blocking("Analyzing scene. Please wait.")
        print(f"[SCENE] _has_scene={_has_scene}")

        if not _load_qwen():
            _speak_blocking("Scene description unavailable. Required packages are missing.")
            return

        # Convert BGR (OpenCV) → RGB PIL image
        print("[SCENE] Step 1: converting frame to PIL image...")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = _PILImage.fromarray(rgb)
        print("[SCENE] Step 1: done")

        print("[SCENE] Step 2: building messages...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text": SCENE_PROMPT},
                ],
            }
        ]
        print("[SCENE] Step 2: done")

        print("[SCENE] Step 3: applying chat template...")
        text_input = _qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("[SCENE] Step 3: done")

        print("[SCENE] Step 4: process_vision_info...")
        image_inputs, video_inputs = process_vision_info(messages)
        print("[SCENE] Step 4: done")

        print("[SCENE] Step 5: tokenizing inputs...")
        inputs = _qwen_processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(_qwen_model.device)
        print(f"[SCENE] Step 5: done  (device={_qwen_model.device})")

        print("[SCENE] Step 6: running model.generate()  ← may take 30-120s on CPU...")
        with _torch.no_grad():
            output_ids = _qwen_model.generate(**inputs, max_new_tokens=200)
        print("[SCENE] Step 6: done")

        print("[SCENE] Step 7: decoding output...")
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        description = _qwen_processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Trim to the last complete sentence so we never speak a half-sentence
        last_end = max(description.rfind('.'), description.rfind('!'), description.rfind('?'))
        if last_end > len(description) // 4:
            description = description[:last_end + 1]

        print("[SCENE] Step 7: done")

        print(f"[SCENE] Result: {description}")
        _log_alert(description, color=_CLR_SCENE, speak=False)
        _speak_blocking(description)

    except Exception as e:
        import traceback
        print(f"[SCENE] FAILED: {e}")
        traceback.print_exc()
        _speak_blocking("Scene description failed.")
    finally:
        print("[SCENE] Done — returning to detection mode")
        _scene_active.clear()


def _on_scene_button():
    """Called when the headphone button is pressed."""
    global _current_frame
    print("[SCENE] _on_scene_button() called")
    if _scene_active.is_set():
        print("[SCENE] Already active — ignoring press")
        return
    snapshot = _current_frame.copy() if _current_frame is not None else None
    if snapshot is None:
        print("[SCENE] _current_frame is None — no frame captured yet")
        return
    print("[SCENE] Snapshot taken, starting description thread")
    threading.Thread(target=_run_scene_description, args=(snapshot,), daemon=True).start()


def get_direction_descriptor(cx: int, cy: int, frame_w: int, frame_h: int) -> str:
    """Return a human readable direction (e.g., upper right, center) for a point in the frame."""
    # Divide the frame into a 3x3 grid
    third_w = frame_w / 3.0
    third_h = frame_h / 3.0

    if cx < third_w:
        horiz = "left"
    elif cx < 2 * third_w:
        horiz = "center"
    else:
        horiz = "right"

    if cy < third_h:
        vert = "upper"
    elif cy < 2 * third_h:
        vert = "center"
    else:
        vert = "bottom"

    if vert == "center" and horiz == "center":
        return "center"
    if vert == "center":
        return horiz
    if horiz == "center":
        return vert
    return f"{vert} {horiz}"




# Create YOLO model (suppress verbose output to avoid flooding the terminal)
model = YOLO("best6.pt", verbose=True)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ALERT: camera could not be opened")
    cap = None

if cap is not None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep only the latest frame in the OS buffer
    cap.set(cv2.CAP_PROP_FPS, 60)          # request higher FPS from the camera driver
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize VL53L5CX sensor
# For serial communication (ESP32), use: sensor = VL53L5CXSensor(port='COM3')  # Replace with your COM port
# For direct I2C, use: sensor = VL53L5CXSensor(use_serial=False)
# Auto-detect ESP32 on serial port (quiet mode):
try:
    sensor = VL53L5CXSensor(port=None, baudrate=250000, use_serial=True, verbose=True)
    if sensor.serial_conn is not None:
        print(f"[SENSOR] Connected on {sensor.serial_conn.port} at {sensor.serial_conn.baudrate} baud")
    else:
        print("[SENSOR] WARNING: serial_conn is None after init — no data will arrive")
        sensor = None
except Exception as e:
    print(f"[SENSOR] ALERT: Could not initialize sensor: {e}")
    print("[SENSOR]   Check: ESP32 is plugged in, correct COM port, firmware running.")
    sensor = None

# Class names from best4.pt (Roboflow-trained, 17 classes).
# Verified with: python -c "from ultralytics import YOLO; m=YOLO('best4.pt'); print(m.names)"
OBSTACLE_CLASSES = [
    "Bike", "Bottle", "Branch", "Chair", "Emergency Blue Phone",
    "Exit Sign", "Garbage Can", "Person", "Phone", "Pole",
    "Push to Open Button", "Sanitizer", "Stairs", "Tree",
    "Vehicle", "Washroom", "Water Fountain",
]

# ── Detection modes ────────────────────────────────────────────────────────────
# next-track button (or double-tap play/pause) cycles: 1 → 2 → 3 → 1 …
#
#   Mode 1 — Normal    : physical obstacles only; exclude low-hazard informational items
#   Mode 2 — Everything: all 17 classes + sensor-only "unidentified object" alerts
#   Mode 3 — Emergency : all 17 classes, known obstacles only (no unknowns)
MODES = {
    1: {
        "name": "Normal",
        "spoken": "Mode 1. Normal mode.",
        # Exclude items that are rarely physical hazards in everyday navigation
        "excluded": {"Person", "Emergency Blue Phone", "Exit Sign"},
        "catch_unknown": False,
    },
    2: {
        "name": "Everything",
        "spoken": "Mode 2. Everything mode. All objects including unidentified.",
        "excluded": set(),
        "catch_unknown": True,
    },
    3: {
        "name": "Emergency",
        "spoken": "Mode 3. Emergency mode. All obstacle classes active.",
        "excluded": set(),
        "catch_unknown": False,
    },
}
NUM_MODES = len(MODES)

# Double-tap window (seconds): second play/pause tap within this window → mode cycle
# Single tap (or first tap, with no second) → scene description
DOUBLE_TAP_WINDOW = 0.45

# Current active mode — protected by a lock so the keyboard thread can write safely
_mode_lock = threading.Lock()
current_mode = 1


def get_active_classes() -> set:
    """Return the set of class labels active in the current detection mode."""
    with _mode_lock:
        excluded = MODES[current_mode]["excluded"]
    return set(OBSTACLE_CLASSES) - excluded


def mode_catches_unknown() -> bool:
    """Return True if the current mode should alert on unrecognised YOLO detections."""
    with _mode_lock:
        return MODES[current_mode].get("catch_unknown", False)


def _cycle_mode() -> None:
    """Advance to the next mode and announce it."""
    global current_mode
    with _mode_lock:
        current_mode = (current_mode % NUM_MODES) + 1
        mode_info = MODES[current_mode]
    excl_str = ", ".join(mode_info["excluded"]) if mode_info["excluded"] else "none"
    print(f"[MODE] Switched to Mode {current_mode}: {mode_info['name']}  "
          f"(excluded: {excl_str}, catch_unknown: {mode_info['catch_unknown']})")
    _log_alert(mode_info["spoken"], color=_CLR_MODE, speak=False)
    _speak_blocking(mode_info["spoken"])


# ── Double-tap state (play/pause button) ─────────────────────────────────────
_tap_lock = threading.Lock()
_last_tap_time = 0.0
_pending_scene_timer: "threading.Timer | None" = None


def _on_play_pause_tap():
    """Route the headphone play/pause button to scene or mode based on tap count.

    Single tap  → scene description (fires after DOUBLE_TAP_WINDOW expires)
    Double tap  → cycle detection mode (second tap cancels the pending scene timer)
    """
    global _last_tap_time, _pending_scene_timer
    now = time.time()
    is_double_tap = False
    old_timer = None

    with _tap_lock:
        if now - _last_tap_time < DOUBLE_TAP_WINDOW:
            # Second tap within window → cycle mode
            _last_tap_time = 0.0           # reset so a 3rd tap starts fresh
            old_timer = _pending_scene_timer
            _pending_scene_timer = None
            is_double_tap = True
        else:
            # First tap → schedule scene description after window
            _last_tap_time = now
            old_timer = _pending_scene_timer  # cancel any lingering timer
            t = threading.Timer(DOUBLE_TAP_WINDOW, _on_scene_button)
            _pending_scene_timer = t
            t.start()

    if old_timer is not None:
        old_timer.cancel()
    if is_double_tap:
        threading.Thread(target=_cycle_mode, daemon=True).start()


# Distance threshold for considering something "close" (in mm)
CLOSE_THRESHOLD_MM = 1300

# Minimum YOLO confidence to trigger an audio alert.
# Detections below this are still drawn on screen and logged, but not spoken.
# Keep this at 0.5: objects at 600-1300 mm appear smaller in frame and YOLO
# assigns confidence ~0.3-0.6; 0.8 was silencing the whole useful alert range.
ALERT_CONFIDENCE_THRESHOLD = 0.5

# How often we remind the user about the same object
# Global audio cooldown: after any spoken alert, silence all audio for this many
# seconds so the current message can be heard in full before the next one fires.
# The spoken phrase takes ~2 s at Rate=4; 2.5 s gives a small buffer.
AUDIO_GLOBAL_COOLDOWN  = 3.5   # seconds
_last_audio_time       = 0.0   # timestamp of the most recently spoken alert

# If an object stays in roughly the same place, we consider it "the same object".

# Tracking objects over time so we can re-alert only after a reminder interval

# Sensor cells that were below CLOSE_THRESHOLD_MM in the most recent frame.
# A reading is only treated as real if it appears in TWO consecutive frames,
# which eliminates single-frame sensor spikes without altering displayed values.
prev_close_cells: set = set()

# Sensor grid visualization parameters
GRID_COLOR = (50, 50, 50)   # Dark gray for grid lines (used in sensor grid)
SENSOR_GRID_SIZE = 600  # Size of sensor grid window (square)
SENSOR_CELL_SIZE = SENSOR_GRID_SIZE // 8  # Each cell is 1/8 of the grid
SENSOR_MAX_DISTANCE = 3500  # Sensor reliable range ceiling in mm (used for colour scaling only)
SENSOR_VOID_VALUE = 0       # Sensor outputs 0 for invalid / out-of-range readings
SENSOR_STALE_TIMEOUT = 0.5  # seconds before we treat the data as stale

# Shared state between the main loop and the sensor polling thread
last_sensor_print_time = 0.0
sensor_warning_printed = False
sensor_parse_warning_printed = False
sensor_data_lock = threading.Lock()
last_sensor_data = np.zeros((8, 8), dtype=np.int32)  # 0 = no data yet (matches sensor invalid value)
last_sensor_update_time = time.time()

# Thread stop signal (clean shutdown)
sensor_stop_event = threading.Event()


def _normalize_sensor_data(raw_sensor_data):
    """Normalize sensor data into an 8x8 numpy int32 array.

    Supports:
    - numpy arrays (1D length 64 or 2D 8x8)
    - list/tuple data (flat or nested)
    - dicts with keys like 'distances', 'data', or a single-value mapping

    Returns None when the input cannot be interpreted.
    """
    if raw_sensor_data is None:
        return None

    # Unwrap common dict wrappers
    if isinstance(raw_sensor_data, dict):
        for key in ("distances", "distance_mm", "ranging_data", "data", "grid", "frame"):
            if key in raw_sensor_data:
                return _normalize_sensor_data(raw_sensor_data[key])
        # If dict contains a single entry, try that value
        if len(raw_sensor_data) == 1:
            return _normalize_sensor_data(next(iter(raw_sensor_data.values())))
        return None

    # Convert to numpy array where possible
    try:
        arr = np.asarray(raw_sensor_data)
    except Exception:
        return None

    # If the data is already 8x8, we're good.
    if arr.shape == (8, 8):
        return arr.astype(np.int32, copy=False)

    # If it's 1D with 64 values, reshape to 8x8
    if arr.ndim == 1 and arr.size == 64:
        try:
            return arr.reshape((8, 8)).astype(np.int32, copy=False)
        except Exception:
            pass

    # If total entries == 64, try reshaping regardless of dims
    if arr.size == 64:
        try:
            return arr.reshape((8, 8)).astype(np.int32, copy=False)
        except Exception:
            pass

    # As a fallback, flatten and pad/truncate to 64 values.
    flat = arr.flatten()
    if flat.size < 64:
        # Pad with 0 (sensor's own invalid marker) — do NOT substitute a synthetic distance
        pad = np.zeros(64 - flat.size, dtype=np.int32)
        flat = np.concatenate([flat, pad])
    elif flat.size > 64:
        flat = flat[:64]

    # Ensure integer values; replace any non-finite floats with 0 (invalid marker)
    try:
        flat = flat.astype(np.int32, copy=False)
    except Exception:
        flat = np.array(flat, dtype=np.int32)

    flat = np.where(np.isfinite(flat.astype(float)), flat, 0).astype(np.int32)
    return flat.reshape((8, 8))


def _sensor_polling_thread():
    """Continuously poll the VL53L5CX sensor in a background thread.

    This decouples sensor I/O from the camera/YOLO loop so the camera stays smooth
    even if serial reads are slow or intermittent.
    Raw sensor values are written directly — no smoothing or value substitution.
    """
    global last_sensor_data, last_sensor_update_time, sensor_warning_printed, sensor_parse_warning_printed

    while not sensor_stop_event.is_set():
        if not sensor:
            # If sensor isn't initialized, avoid busy looping
            time.sleep(0.1)
            continue

        new_frame = sensor.get_ranging_data()
        if new_frame is None:
            # No new data: sleep briefly to avoid burning CPU
            time.sleep(0.005)
            continue

        last_sensor_update_time = time.time()

        normalized = _normalize_sensor_data(new_frame)
        if normalized is None:
            # Avoid spamming the console if we cannot parse sensor data
            if not sensor_parse_warning_printed:
                print("ALERT: sensor returned unexpected data format. Unable to normalize to 8x8 array.")
                sensor_parse_warning_printed = True
            continue

        # Write raw frame directly — no smoothing applied
        with sensor_data_lock:
            last_sensor_data = normalized

        # If the sensor is returning all-zero values, warn once (sensor may not be ready)
        if not sensor_warning_printed:
            if np.all(normalized == 0):
                print("ALERT: sensor returning all-zero values. Check wiring/position.")
                sensor_warning_printed = True


# Start background thread to keep sensor readings as fresh as possible
sensor_thread = threading.Thread(target=_sensor_polling_thread, daemon=True)
sensor_thread.start()

# Use pynput for headphone button detection — it correctly handles Bluetooth
# HID Consumer Control events that the 'keyboard' library cannot see.
try:
    from pynput import keyboard as _pynput_kb

    def _on_pynput_press(key):
        # Filter out held modifier spam before printing
        if key not in (_pynput_kb.Key.shift, _pynput_kb.Key.shift_l,
                       _pynput_kb.Key.shift_r, _pynput_kb.Key.ctrl,
                       _pynput_kb.Key.ctrl_l, _pynput_kb.Key.ctrl_r,
                       _pynput_kb.Key.alt, _pynput_kb.Key.alt_l,
                       _pynput_kb.Key.alt_r):
            print(f"[KEY] {key!r}")

        # ── Headphone media buttons ───────────────────────────────────────
        if key == _pynput_kb.Key.media_play_pause:
            _on_play_pause_tap()
        elif key == _pynput_kb.Key.media_next:
            threading.Thread(target=_cycle_mode, daemon=True).start()

        # ── Keyboard fallback (no headphones) ────────────────────────────
        # M → cycle detection mode    S → scene description
        elif hasattr(key, 'char'):
            if key.char == 'm':
                threading.Thread(target=_cycle_mode, daemon=True).start()
            elif key.char == 's':
                _on_scene_button()

    _pynput_listener = _pynput_kb.Listener(on_press=_on_pynput_press)
    _pynput_listener.start()
    print("Headphone controls active (pynput):")
    print("  play/pause      → scene description (single tap) / mode cycle (double tap)")
    print("  next track      → cycle detection mode")
    print("Keyboard fallback (no headphones):")
    print("  M               → cycle detection mode")
    print("  S               → scene description")

except Exception as _ke:
    print(f"[KEY] pynput listener failed: {_ke}")


def create_sensor_grid(sensor_data):
    """
    Create a visualization grid for the 8x8 TOF sensor data
    Color-codes cells based on distance (closer = red/orange, farther = blue/green)
    """
    canvas = np.zeros((SENSOR_GRID_SIZE, SENSOR_GRID_SIZE, 3), dtype=np.uint8)
    
    # Draw grid lines
    for i in range(9):
        x = i * SENSOR_CELL_SIZE
        cv2.line(canvas, (x, 0), (x, SENSOR_GRID_SIZE), GRID_COLOR, 2)
        cv2.line(canvas, (0, x), (SENSOR_GRID_SIZE, x), GRID_COLOR, 2)
    
    # Fill cells with color based on distance
    for row in range(8):
        for col in range(8):
            distance = sensor_data[row, col]
            
            # Calculate cell position
            x1 = col * SENSOR_CELL_SIZE
            y1 = row * SENSOR_CELL_SIZE
            x2 = (col + 1) * SENSOR_CELL_SIZE
            y2 = (row + 1) * SENSOR_CELL_SIZE
            
            # Color mapping: closer objects = red/orange, farther = blue/green
            # Use raw distance in mm (0-3300), clamp to max distance
            clamped_dist = min(distance, SENSOR_MAX_DISTANCE)
            
            if distance == 0:
                # Invalid - gray
                color = (50, 50, 50)
            else:
                # Color gradient: red (close) -> yellow -> green -> blue (far)
                # Map 0-3300 to 0-1
                normalized_dist = clamped_dist / SENSOR_MAX_DISTANCE
                
                if normalized_dist < 0.25:
                    # Close: Red to Orange
                    r = 255
                    g = int(255 * (normalized_dist / 0.25))
                    b = 0
                elif normalized_dist < 0.5:
                    # Medium-close: Orange to Yellow
                    r = 255
                    g = 255
                    b = int(255 * ((normalized_dist - 0.25) / 0.25))
                elif normalized_dist < 0.75:
                    # Medium-far: Yellow to Green
                    r = int(255 * (1 - (normalized_dist - 0.5) / 0.25))
                    g = 255
                    b = 0
                else:
                    # Far: Green to Blue
                    r = 0
                    g = int(255 * (1 - (normalized_dist - 0.75) / 0.25))
                    b = int(255 * ((normalized_dist - 0.75) / 0.25))
                
                color = (int(b), int(g), int(r))  # BGR format for OpenCV
            
            # Fill cell with color
            cv2.rectangle(canvas, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), color, -1)
            
            # Add distance text (in mm, or '---' if sensor reports 0 / invalid)
            if distance == 0:
                text = "---"
            else:
                text = f"{int(distance)}"
            
            # Calculate text position (centered in cell)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x1 + (SENSOR_CELL_SIZE - text_size[0]) // 2
            text_y = y1 + (SENSOR_CELL_SIZE + text_size[1]) // 2
            
            # Use white or black text based on cell brightness
            brightness = sum(color) / 3
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            
            cv2.putText(canvas, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    # Add title
    cv2.putText(canvas, "TOF Sensor Grid (mm)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return canvas

def map_camera_to_sensor_grid(x1, y1, x2, y2, frame_height, frame_width):
    """
    Map camera frame bounding box coordinates to sensor grid coordinates (8x8)
    Assumes sensor is centered and aligned with camera view
    Returns list of (row, col) tuples for sensor cells that overlap with bbox
    """
    # Normalize bbox to 0-1 range
    norm_x1 = x1 / frame_width
    norm_y1 = y1 / frame_height
    norm_x2 = x2 / frame_width
    norm_y2 = y2 / frame_height
    
    # Map to 8x8 sensor grid (0-7 for rows and cols)
    sensor_col1 = int(norm_x1 * 8)
    sensor_row1 = int(norm_y1 * 8)
    sensor_col2 = int(norm_x2 * 8)
    sensor_row2 = int(norm_y2 * 8)
    
    # Clamp to valid range
    sensor_col1 = max(0, min(sensor_col1, 7))
    sensor_row1 = max(0, min(sensor_row1, 7))
    sensor_col2 = max(0, min(sensor_col2, 7))
    sensor_row2 = max(0, min(sensor_row2, 7))
    
    # Generate list of sensor cells covering the bbox
    sensor_cells = []
    for row in range(sensor_row1, sensor_row2 + 1):
        for col in range(sensor_col1, sensor_col2 + 1):
            sensor_cells.append((row, col))
    
    return sensor_cells

def check_sensor_close_in_region(sensor_data, sensor_cells, distance_threshold=1300):
    """
    Check if sensor detected an object within distance_threshold (mm) in the specified region
    Returns tuple: (detected_close: bool, min_distance: int or None)
    """
    if sensor_data is None or len(sensor_cells) == 0:
        return False, None
    
    distances = []
    for row, col in sensor_cells:
        distance = sensor_data[row, col]
        # Valid distance reading
        if distance > 0:
            distances.append(distance)
    
    if len(distances) == 0:
        return False, None
    
    min_dist = min(distances)
    detected_close = min_dist < distance_threshold
    return detected_close, min_dist


# last_sensor_data is initialized and updated by the background sensor polling thread.

# (Debug printing disabled: only alerts and audio should be emitted)


# ── Threaded camera capture ───────────────────────────────────────────────────
# Captures and prepares frames in a background thread so cap.read() never
# stalls the display/YOLO loop.  read() always returns the freshest frame.

class _FrameBuffer:
    """Background thread that continuously grabs and prepares camera frames."""
    def __init__(self, cap):
        self._cap   = cap
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            ret, raw = self._cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            prepared = _prepare_camera_frame(raw)
            with self._lock:
                self._frame = prepared

    def read(self):
        """Return a copy of the latest prepared frame, or None if not ready yet."""
        with self._lock:
            f = self._frame
        return None if f is None else f.copy()

    def stop(self):
        self._stop.set()


frame_buffer = _FrameBuffer(cap) if cap is not None else None


# ── Threaded YOLO inference ───────────────────────────────────────────────────
# YOLO runs in a dedicated thread so the display loop is never blocked.
# Detections are stored as plain tuples; the display loop reads the latest batch.

YOLO_IMGSZ = 320   # inference input size: 320 = ~4× faster than 640, slightly less accurate

_yolo_detections: list = []            # latest: [(x1,y1,x2,y2,cls_int,conf_float), ...]
_yolo_det_lock   = threading.Lock()
_yolo_input      = None                # frame submitted for next inference pass
_yolo_input_lock = threading.Lock()
_yolo_trigger    = threading.Event()   # set by main loop, cleared by worker
_yolo_stop_ev    = threading.Event()


def _yolo_worker():
    global _yolo_detections
    while not _yolo_stop_ev.is_set():
        if not _yolo_trigger.wait(timeout=0.1):
            continue
        _yolo_trigger.clear()
        with _yolo_input_lock:
            frame = _yolo_input
        if frame is None:
            continue
        try:
            dets = []
            for r in model(frame, stream=True, verbose=False, imgsz=YOLO_IMGSZ, conf=0.20):
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    dets.append((x1, y1, x2, y2, int(b.cls[0]), float(b.conf[0])))
            with _yolo_det_lock:
                _yolo_detections = dets
        except Exception as _yolo_exc:
            print(f"[YOLO] inference error: {_yolo_exc}")


_yolo_inf_thread = threading.Thread(target=_yolo_worker, daemon=True)
_yolo_inf_thread.start()


# Dashboard window: explicit pixel size (1920×1080) — do not use AUTOSIZE or the
# aspect ratio can drift and the layout looks stretched / non-square.
cv2.namedWindow(_DASH_WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(_DASH_WIN_NAME, _DASH_TOTAL_W, _DASH_TOTAL_H)

# optional auto-termination to prevent unresponsive pop-ups during testing
MAX_RUNTIME = 30  # seconds; set to None to run indefinitely
start_time = time.time()

while True:
    if frame_buffer is None:
        # No camera available — use a black placeholder frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        frame = frame_buffer.read()
        if frame is None:
            # Buffer not ready yet — yield and retry
            if cv2.waitKey(1) == 27:
                break
            continue

    # Keep a reference to the latest frame so the scene-description thread can grab it
    _current_frame = frame

    # ── Scene-description mode: pause all detection while active ─────────────
    if _scene_active.is_set():
        cv2.imshow(_DASH_WIN_NAME, _draw_dashboard(frame, create_sensor_grid(last_sensor_data)))
        if cv2.waitKey(1) == 27:
            break
        continue

    # Copy latest sensor data (thread-safe) for visualization/detection
    with sensor_data_lock:
        sensor_data = last_sensor_data.copy()
        last_update = last_sensor_update_time

    # If the sensor has not updated in a while, treat it as stale (show all-zero / no-data)
    _now = time.time()
    if _now - last_update > SENSOR_STALE_TIMEOUT:
        sensor_data = np.zeros((8, 8), dtype=np.int32)
        if sensor is not None and _now - last_update > 3.0 and not getattr(_sensor_polling_thread, '_stale_warned', False):
            print("[SENSOR] WARNING: no data received for >3 s — check ESP32 connection")
            _sensor_polling_thread._stale_warned = True
    
    # Submit this frame to the async YOLO inference thread (non-blocking)
    with _yolo_input_lock:
        _yolo_input = frame   # frame_buffer.read() already returns a fresh copy
    _yolo_trigger.set()

    # Fetch the most recent detections (may be ~1 frame old — acceptable for real-time use)
    with _yolo_det_lock:
        current_detections = list(_yolo_detections)

    # Track whether we've already spoken an alert this frame (one audio at a time)
    audio_alert_sent = False
    frame_now = time.time()

    # Track sensor cells covered by a YOLO detection (for unidentifiable-object check)
    covered_sensor_cells: set = set()
    # Collect every cell that is close this frame; used to update prev_close_cells
    current_close_cells: set = set()

    # Resolve mode state once per frame (avoids repeated lock acquisition)
    active_classes = get_active_classes()
    catch_unknown  = mode_catches_unknown()

    for (x1, y1, x2, y2, cls, conf) in current_detections:
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        label = model.names[cls]

        # Determine whether this detection should be processed and under what name
        if label in active_classes:
            display_label = label                   # known, active class
        elif catch_unknown and label not in OBSTACLE_CLASSES:
            display_label = "unidentified object"   # mode 2: YOLO sees something off-list
        else:
            continue                                 # excluded or not catching unknowns

        location_str = f"({center_x}, {center_y})"
        direction = get_direction_descriptor(center_x, center_y, frame.shape[1], frame.shape[0])

        cv2.putText(frame, f"{display_label} ({conf:.1f})", (int(x1), int(y1)-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)

        # ── Special case: Exit Sign in Emergency mode ─────────────────────────
        # Alert purely on camera detection — no distance check needed.
        with _mode_lock:
            _em_mode = current_mode == 3
        if display_label == "Exit Sign" and _em_mode:
            print(f"EXIT SIGN detected {direction} (conf={conf:.2f})")
            if (not audio_alert_sent
                    and conf >= ALERT_CONFIDENCE_THRESHOLD
                    and (frame_now - _last_audio_time) >= AUDIO_GLOBAL_COOLDOWN):
                _last_audio_time = frame_now
                msg = f"Exit Sign {_natural_direction(direction)}"
                _log_alert(msg, color=_CLR_OBSTACLE)
                audio_alert_sent = True
            # Skip sensor-distance logic for Exit Sign
            continue

        # ── Sensor-distance gating ────────────────────────────────────────────
        # Stairs: check ALL sensor rows (they appear at the bottom of frame).
        # Everything else: restrict to upper rows to reduce false positives.
        if display_label == "Stairs":
            sensor_cells = map_camera_to_sensor_grid(x1, y1, x2, y2, frame.shape[0], frame.shape[1])
        else:
            sensor_cells = map_camera_to_sensor_grid(x1, y1, x2, y2, frame.shape[0], frame.shape[1])
            sensor_cells = [(sr, sc) for sr, sc in sensor_cells if sr < 6]

        sensor_close, sensor_distance = check_sensor_close_in_region(
            sensor_data, sensor_cells, distance_threshold=CLOSE_THRESHOLD_MM)

        # Track which cells are close this frame (for next-frame confirmation)
        for _r, _c in sensor_cells:
            if 0 < sensor_data[_r, _c] < CLOSE_THRESHOLD_MM:
                current_close_cells.add((_r, _c))

        # Require the close reading to have been present last frame too
        sensor_close_confirmed = sensor_close and any(
            cell in prev_close_cells for cell in sensor_cells
            if 0 < sensor_data[cell[0], cell[1]] < CLOSE_THRESHOLD_MM
        )

        # Mark these cells as covered by a known YOLO object
        covered_sensor_cells.update(sensor_cells)

        # Alert every confirmed-close detection; AUDIO_GLOBAL_COOLDOWN
        # prevents back-to-back audio — no per-object tracking needed.
        if sensor_close_confirmed:
            print(f"{display_label} at {sensor_distance}mm {direction} (conf={conf:.2f})")
            if (not audio_alert_sent
                    and conf >= ALERT_CONFIDENCE_THRESHOLD
                    and (frame_now - _last_audio_time) >= AUDIO_GLOBAL_COOLDOWN):
                _last_audio_time = frame_now
                msg = _format_alert_text(display_label, direction, sensor_distance)
                _log_alert(msg, color=_CLR_OBSTACLE)
                audio_alert_sent = True

    # ── Unidentifiable-object check ──────────────────────────────────────────────
    # If the ToF sensor sees something close in the upper half (rows 0-3) but YOLO
    # did not identify any object there, alert as "unidentifiable object".
    # Confirmation required: cell must also have been close in the previous frame.
    unidentifiable_close = False
    min_unidentified_dist = None
    for row in range(4):          # upper half of the 8×8 sensor grid
        for col in range(8):
            if (row, col) in covered_sensor_cells:
                continue          # already accounted for by a known detection
            dist = int(sensor_data[row, col])
            if 0 < dist < (CLOSE_THRESHOLD_MM-200):
                current_close_cells.add((row, col))
                if (row, col) in prev_close_cells:   # confirmed across two frames
                    unidentifiable_close = True
                    if min_unidentified_dist is None or dist < min_unidentified_dist:
                        min_unidentified_dist = dist
    # Sensor-only unidentified alert — active only in Mode 2 (catch_unknown)
    if catch_unknown and unidentifiable_close:
        print(f"unidentified object at ~{min_unidentified_dist}mm (sensor only, no YOLO match)")
        if not audio_alert_sent and (frame_now - _last_audio_time) >= AUDIO_GLOBAL_COOLDOWN+3:
            _last_audio_time = frame_now
            _log_alert(_format_alert_text("unidentified object", "center", min_unidentified_dist), color=_CLR_UNKNOWN)
            audio_alert_sent = True

    # Advance the close-cell history for the next frame's confirmation check
    prev_close_cells = current_close_cells

    
    # Create sensor visualization grid (use last data, or empty grid if no data yet)
    if sensor_data is not None:
        sensor_grid = create_sensor_grid(sensor_data)
    else:
        # Create empty grid until first data arrives
        sensor_grid = np.zeros((SENSOR_GRID_SIZE, SENSOR_GRID_SIZE, 3), dtype=np.uint8)
        cv2.putText(sensor_grid, "Waiting for sensor data...", (SENSOR_GRID_SIZE//2 - 150, SENSOR_GRID_SIZE//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    dashboard = _draw_dashboard(frame, sensor_grid)
    cv2.imshow(_DASH_WIN_NAME, dashboard)

    # window event handling and exit conditions
    if cv2.waitKey(1) == 27:
        break

# Cleanup
sensor_stop_event.set()
_yolo_stop_ev.set()

if sensor_thread.is_alive():
    sensor_thread.join(timeout=1.0)
if _yolo_inf_thread.is_alive():
    _yolo_inf_thread.join(timeout=1.0)
if frame_buffer is not None:
    frame_buffer.stop()

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
if sensor:
    sensor.close()





#if label in OBSTACLE_CLASSES and conf > 0.5:
    