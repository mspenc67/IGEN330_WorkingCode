"""
code_for_rpi1.py  —  Raspberry Pi runtime
==========================================
Stripped-down obstacle-detection loop for headless Pi deployment:
  • YOLO object detection   (best.pt — Roboflow custom model)
  • VL53L5CX ToF distance sensing
  • Audio alerts via espeak-ng  (sudo apt install espeak-ng)
  • No display windows, no Excel logging, no scene-description model

Hardware assumptions
--------------------
  Camera  : USB webcam on /dev/video0  (or Pi Camera via V4L2)
  Sensor  : VL53L5CX connected through ESP32 over serial (same as main file)
  Audio   : 3.5 mm jack or Bluetooth headset — espeak-ng writes to ALSA

Install dependencies on Pi
--------------------------
  sudo apt update && sudo apt install -y espeak-ng
  pip install ultralytics opencv-python-headless numpy
  # sensor library already present in working_cam_sensor/
"""

from ultralytics import YOLO
import cv2
import time
import threading
import numpy as np
import subprocess

from working_cam_sensor.vl53l5cx_sensor import VL53L5CXSensor


# ─────────────────────────────────────────────────────────────────────────────
# Text-to-Speech  (espeak-ng — lightweight, works offline on Pi)
# ─────────────────────────────────────────────────────────────────────────────
TTS_PROCESS = None


def speak_text(text: str) -> None:
    """Speak *text* using espeak-ng (non-blocking, one utterance at a time)."""
    global TTS_PROCESS
    if TTS_PROCESS is not None and TTS_PROCESS.poll() is None:
        return  # Don't interrupt speech already in progress
    try:
        # -s 160 = words-per-minute  (default 175; lower = clearer on bone-conduction)
        # -v en  = English voice
        TTS_PROCESS = subprocess.Popen(
            ["espeak-ng", "-s", "160", "-v", "en", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        if not hasattr(speak_text, "_warned"):
            print("ALERT: espeak-ng not found. Run: sudo apt install espeak-ng")
            speak_text._warned = True
    except Exception as e:
        if not hasattr(speak_text, "_warned"):
            print(f"ALERT: TTS failed ({e})")
            speak_text._warned = True


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Class names must match best.pt exactly.
# Verify with: python -c "from ultralytics import YOLO; m=YOLO('best.pt'); print(m.names)"
OBSTACLE_CLASSES = [
    "Bike", "Car", "Chair", "Emergency Blue Phone",
    "Exit sign", "Person", "Pole", "Stairs", "Tree", "Washroom",
]

# Distance threshold — object is "close" when sensor reads below this (mm)
CLOSE_THRESHOLD_MM = 1000

# Minimum seconds before re-alerting about the same object
ALERT_REMINDER_SECONDS = 1.0

# Object tracking thresholds
TRACKED_OBJECT_MOVE_THRESHOLD_PX = 80   # px before we treat it as a new object
TRACKED_OBJECT_MAX_AGE = 10.0           # seconds before forgetting a still object

# Sensor settings
SENSOR_MAX_DISTANCE = 3500   # reliable ceiling in mm (used for nothing here — kept for reference)
SENSOR_STALE_TIMEOUT = 0.5   # seconds before treating sensor data as stale


# ─────────────────────────────────────────────────────────────────────────────
# Object tracking helpers
# ─────────────────────────────────────────────────────────────────────────────
tracked_objects = []

# Cells that were close last frame — used to confirm real detections vs. spikes
prev_close_cells: set = set()


def get_direction_descriptor(cx: int, cy: int, frame_w: int, frame_h: int) -> str:
    """Return a spoken direction string, e.g. 'upper left', 'center'."""
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


def _euclidean_dist(p0, p1):
    return ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5


def _find_tracked_object(label: str, center: tuple) -> dict | None:
    for obj in tracked_objects:
        if obj["label"] == label and _euclidean_dist(center, obj["center"]) < TRACKED_OBJECT_MOVE_THRESHOLD_PX:
            return obj
    return None


def _cleanup_tracked_objects(now: float) -> None:
    global tracked_objects
    tracked_objects = [o for o in tracked_objects if now - o.get("last_seen", 0) < TRACKED_OBJECT_MAX_AGE]


# ─────────────────────────────────────────────────────────────────────────────
# Sensor (VL53L5CX — polled in a background thread)
# ─────────────────────────────────────────────────────────────────────────────
sensor_data_lock = threading.Lock()
last_sensor_data = np.zeros((8, 8), dtype=np.int32)
last_sensor_update_time = time.time()
sensor_stop_event = threading.Event()
_sensor_warning_printed = False
_sensor_parse_warning_printed = False


def _normalize_sensor_data(raw):
    """Coerce raw sensor output into an (8, 8) int32 numpy array, or return None."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in ("distances", "distance_mm", "ranging_data", "data", "grid", "frame"):
            if key in raw:
                return _normalize_sensor_data(raw[key])
        if len(raw) == 1:
            return _normalize_sensor_data(next(iter(raw.values())))
        return None
    try:
        arr = np.asarray(raw)
    except Exception:
        return None
    if arr.shape == (8, 8):
        return arr.astype(np.int32, copy=False)
    if arr.size == 64:
        try:
            return arr.reshape((8, 8)).astype(np.int32, copy=False)
        except Exception:
            pass
    flat = arr.flatten()
    if flat.size < 64:
        flat = np.concatenate([flat, np.zeros(64 - flat.size, dtype=np.int32)])
    else:
        flat = flat[:64]
    flat = np.where(np.isfinite(flat.astype(float)), flat, 0).astype(np.int32)
    return flat.reshape((8, 8))


def _sensor_polling_thread():
    global last_sensor_data, last_sensor_update_time
    global _sensor_warning_printed, _sensor_parse_warning_printed

    while not sensor_stop_event.is_set():
        if not sensor:
            time.sleep(0.1)
            continue
        new_frame = sensor.get_ranging_data()
        if new_frame is None:
            time.sleep(0.005)
            continue
        last_sensor_update_time = time.time()
        normalized = _normalize_sensor_data(new_frame)
        if normalized is None:
            if not _sensor_parse_warning_printed:
                print("ALERT: sensor returned unexpected data format.")
                _sensor_parse_warning_printed = True
            continue
        with sensor_data_lock:
            last_sensor_data = normalized
        if not _sensor_warning_printed and np.all(normalized == 0):
            print("ALERT: sensor returning all-zero values — check wiring/position.")
            _sensor_warning_printed = True


def map_camera_to_sensor_grid(x1, y1, x2, y2, frame_h: int, frame_w: int) -> list:
    """Return list of (row, col) sensor cells that overlap with the bounding box."""
    c1 = max(0, min(int((x1 / frame_w) * 8), 7))
    r1 = max(0, min(int((y1 / frame_h) * 8), 7))
    c2 = max(0, min(int((x2 / frame_w) * 8), 7))
    r2 = max(0, min(int((y2 / frame_h) * 8), 7))
    return [(row, col) for row in range(r1, r2 + 1) for col in range(c1, c2 + 1)]


def check_sensor_close(sensor_data, sensor_cells, threshold: int):
    """Return (is_close: bool, min_distance: int|None) for the given cells."""
    if sensor_data is None or not sensor_cells:
        return False, None
    distances = [int(sensor_data[r, c]) for r, c in sensor_cells if sensor_data[r, c] > 0]
    if not distances:
        return False, None
    min_d = min(distances)
    return min_d < threshold, min_d


# ─────────────────────────────────────────────────────────────────────────────
# Hardware initialisation
# ─────────────────────────────────────────────────────────────────────────────
print("Loading YOLO model (best.pt)…")
model = YOLO("best.pt", verbose=False)
print(f"Model ready. Classes: {list(model.names.values())}")

# Camera — index 0 for USB webcam; use cv2.CAP_V4L2 flag if needed on Pi
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ALERT: camera could not be opened on index 0.")
    cap = None
else:
    # Lower resolution for faster YOLO inference on Pi CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ToF sensor (ESP32 serial bridge — same setup as laptop version)
try:
    sensor = VL53L5CXSensor(port=None, baudrate=250000, use_serial=True, verbose=False)
except Exception as e:
    print(f"ALERT: Could not initialise sensor: {e}")
    sensor = None

# Start background sensor polling
sensor_thread = threading.Thread(target=_sensor_polling_thread, daemon=True)
sensor_thread.start()

print("System ready. Press Ctrl+C to stop.")
speak_text("System ready")


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────
try:
    while True:
        # ── Grab frame ────────────────────────────────────────────────────────
        if cap is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret:
                print("ALERT: camera frame grab failed.")
                time.sleep(0.1)
                continue

        # ── Grab sensor snapshot (thread-safe) ────────────────────────────────
        with sensor_data_lock:
            sensor_data = last_sensor_data.copy()
            last_update = last_sensor_update_time

        if time.time() - last_update > SENSOR_STALE_TIMEOUT:
            sensor_data = np.zeros((8, 8), dtype=np.int32)

        # ── YOLO inference ────────────────────────────────────────────────────
        results = model(frame, stream=True, verbose=False)

        audio_alert_sent = False
        frame_now = time.time()
        _cleanup_tracked_objects(frame_now)

        current_close_cells: set = set()

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                cls   = int(b.cls[0])
                conf  = float(b.conf[0])
                label = model.names[cls]

                if label not in OBSTACLE_CLASSES:
                    continue  # ignore classes not in our list

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Map bounding box → ToF sensor cells (top half of sensor only)
                sensor_cells = map_camera_to_sensor_grid(
                    x1, y1, x2, y2, frame.shape[0], frame.shape[1]
                )
                sensor_cells = [(row, col) for row, col in sensor_cells if row < 4]

                sensor_close, sensor_distance = check_sensor_close(
                    sensor_data, sensor_cells, CLOSE_THRESHOLD_MM
                )

                # Collect cells that are close this frame
                for _r, _c in sensor_cells:
                    if 0 < sensor_data[_r, _c] < CLOSE_THRESHOLD_MM:
                        current_close_cells.add((_r, _c))

                # Require the close reading to be confirmed across two consecutive frames
                # (eliminates single-frame sensor spikes)
                sensor_close_confirmed = sensor_close and any(
                    cell in prev_close_cells for cell in sensor_cells
                    if 0 < sensor_data[cell[0], cell[1]] < CLOSE_THRESHOLD_MM
                )

                if sensor_close_confirmed:
                    obj = _find_tracked_object(label, (center_x, center_y))
                    should_alert = False

                    if obj is None:
                        obj = {
                            "label": label,
                            "center": (center_x, center_y),
                            "last_alert": frame_now,
                            "last_seen": frame_now,
                        }
                        tracked_objects.append(obj)
                        should_alert = True
                    else:
                        moved = _euclidean_dist((center_x, center_y), obj["center"])
                        obj["center"] = (center_x, center_y)
                        obj["last_seen"] = frame_now
                        if (frame_now - obj.get("last_alert", 0)) > ALERT_REMINDER_SECONDS \
                                and moved < TRACKED_OBJECT_MOVE_THRESHOLD_PX:
                            should_alert = True
                            obj["last_alert"] = frame_now

                    if should_alert and not audio_alert_sent:
                        direction = get_direction_descriptor(
                            center_x, center_y, frame.shape[1], frame.shape[0]
                        )
                        print(f"ALERT: {label} — {sensor_distance} mm — {direction}")
                        speak_text(f"{label} {direction}")
                        audio_alert_sent = True

        # Advance frame history for next-frame confirmation check
        prev_close_cells = current_close_cells

except KeyboardInterrupt:
    print("\nCtrl+C — shutting down…")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
sensor_stop_event.set()
if sensor_thread.is_alive():
    sensor_thread.join(timeout=1.0)

if cap is not None:
    cap.release()

if sensor:
    sensor.close()

print("Done.")
