"""
Raspberry Pi runtime with feature parity to camera_and_boundingboxes_F.py:
  - YOLO + VL53L5CX fusion alerts
  - Mode cycling on volume+ button
  - Scene-description trigger on play/pause button
  - Optional tiny Qwen model for spoken phrasing/summaries
"""

from collections import deque
import os
import subprocess
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from qwen_pi_assistant import QwenPiAssistant
from working_cam_sensor.vl53l5cx_sensor_rpi import VL53L5CXSensorRPI

try:
    import keyboard as _keyboard  # type: ignore
    _has_keyboard = True
except Exception:
    _keyboard = None
    _has_keyboard = False

try:
    from evdev import InputDevice, ecodes, list_devices  # type: ignore
    _has_evdev = True
except Exception:
    InputDevice = None
    ecodes = None
    list_devices = None
    _has_evdev = False


TTS_PROCESS = None


def speak_text(text: str) -> None:
    global TTS_PROCESS
    if TTS_PROCESS is not None and TTS_PROCESS.poll() is None:
        return
    try:
        TTS_PROCESS = subprocess.Popen(
            ["espeak-ng", "-s", "160", "-v", "en", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        if not hasattr(speak_text, "_warned"):
            print(f"ALERT: TTS failed ({exc})")
            speak_text._warned = True


def _speak_blocking(text: str) -> None:
    global TTS_PROCESS
    if TTS_PROCESS is not None and TTS_PROCESS.poll() is None:
        TTS_PROCESS.terminate()
        TTS_PROCESS.wait(timeout=2.0)
        TTS_PROCESS = None
    speak_text(text)
    if TTS_PROCESS is not None:
        TTS_PROCESS.wait(timeout=10.0)


OBSTACLE_CLASSES = [
    "Bike", "Car", "Chair", "Emergency Blue Phone",
    "Exit sign", "Person", "Pole", "Stairs", "Tree", "Washroom",
]

MODES = {
    1: {
        "name": "Normal",
        "spoken": "Mode 1. Normal mode.",
        "excluded": {"Person", "Exit sign"},
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
MODE_CYCLE_KEYS = {"volume up", "volume+", "vol up"}
SCENE_TRIGGER_KEYS = {"play/pause media", "play/pause", "pause/play"}

CLOSE_THRESHOLD_MM = 1000
ALERT_REMINDER_SECONDS = 1.0
TRACKED_OBJECT_MOVE_THRESHOLD_PX = 80
TRACKED_OBJECT_MAX_AGE = 10.0
SENSOR_MAX_DISTANCE = 3500
SENSOR_MIN_VALID_MM = 60
SENSOR_STALE_TIMEOUT = 0.5
SENSOR_HISTORY_FRAMES = 3

YOLO_IMGSZ = 416
YOLO_CONF_THRESHOLD = 0.35
INFER_EVERY_N_FRAMES = 2

ENABLE_QWEN_ASSISTANT = os.getenv("ENABLE_QWEN_ASSISTANT", "0") == "1"
QWEN_GGUF_PATH = os.getenv("QWEN_GGUF_PATH", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")

tracked_objects = []
prev_close_cells: set = set()
_mode_lock = threading.Lock()
current_mode = 1
_scene_active = threading.Event()
_frame_lock = threading.Lock()
_current_frame = None
_latest_scene_snapshot = []
_latest_scene_mode = "Normal"
_scene_lock = threading.Lock()

sensor_data_lock = threading.Lock()
last_sensor_data = np.zeros((8, 8), dtype=np.int32)
last_sensor_update_time = time.time()
sensor_stop_event = threading.Event()
sensor_frame_history = deque(maxlen=SENSOR_HISTORY_FRAMES)


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


def get_direction_descriptor(cx: int, cy: int, frame_w: int, frame_h: int) -> str:
    third_w = frame_w / 3.0
    third_h = frame_h / 3.0
    horiz = "left" if cx < third_w else "center" if cx < 2 * third_w else "right"
    vert = "upper" if cy < third_h else "center" if cy < 2 * third_h else "bottom"
    if vert == "center" and horiz == "center":
        return "center"
    if vert == "center":
        return horiz
    if horiz == "center":
        return vert
    return f"{vert} {horiz}"


def get_active_classes() -> set:
    with _mode_lock:
        excluded = MODES[current_mode]["excluded"]
    return set(OBSTACLE_CLASSES) - excluded


def mode_catches_unknown() -> bool:
    with _mode_lock:
        return MODES[current_mode]["catch_unknown"]


def _cycle_mode() -> None:
    global current_mode
    with _mode_lock:
        current_mode = (current_mode % NUM_MODES) + 1
        mode_info = MODES[current_mode]
    print(f"[MODE] Switched to Mode {current_mode}: {mode_info['name']}")
    _speak_blocking(mode_info["spoken"])


def _normalize_sensor_data(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in ("distances", "distance_mm", "ranging_data", "data", "grid", "frame"):
            if key in raw:
                return _normalize_sensor_data(raw[key])
        if len(raw) == 1:
            return _normalize_sensor_data(next(iter(raw.values())))
        return None
    arr = np.asarray(raw)
    if arr.shape == (8, 8):
        return arr.astype(np.int32, copy=False)
    if arr.size == 64:
        return arr.reshape((8, 8)).astype(np.int32, copy=False)
    flat = arr.flatten()
    if flat.size < 64:
        flat = np.concatenate([flat, np.zeros(64 - flat.size, dtype=np.int32)])
    else:
        flat = flat[:64]
    return np.where(np.isfinite(flat.astype(float)), flat, 0).astype(np.int32).reshape((8, 8))


def _sensor_polling_thread():
    global last_sensor_data, last_sensor_update_time
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
            continue
        normalized = np.where(
            (normalized >= SENSOR_MIN_VALID_MM) & (normalized <= SENSOR_MAX_DISTANCE),
            normalized,
            0,
        ).astype(np.int32, copy=False)
        with sensor_data_lock:
            sensor_frame_history.append(normalized)
            if len(sensor_frame_history) >= 2:
                stack = np.stack(sensor_frame_history, axis=0)
                last_sensor_data = np.median(stack, axis=0).astype(np.int32)
            else:
                last_sensor_data = normalized


def map_camera_to_sensor_grid(x1, y1, x2, y2, frame_h: int, frame_w: int) -> list:
    c1 = max(0, min(int((x1 / frame_w) * 8), 7))
    r1 = max(0, min(int((y1 / frame_h) * 8), 7))
    c2 = max(0, min(int((x2 / frame_w) * 8), 7))
    r2 = max(0, min(int((y2 / frame_h) * 8), 7))
    return [(row, col) for row in range(r1, r2 + 1) for col in range(c1, c2 + 1)]


def check_sensor_close(sensor_data, sensor_cells, threshold: int):
    if sensor_data is None or not sensor_cells:
        return False, None
    distances = [
        int(sensor_data[r, c])
        for r, c in sensor_cells
        if SENSOR_MIN_VALID_MM <= int(sensor_data[r, c]) <= SENSOR_MAX_DISTANCE
    ]
    if not distances:
        return False, None
    robust_d = int(np.percentile(distances, 25))
    return robust_d < threshold, robust_d


def _scene_description_worker():
    _scene_active.set()
    try:
        _speak_blocking("Analyzing scene. Please wait.")
        with _scene_lock:
            obs = list(_latest_scene_snapshot)
            mode_name = _latest_scene_mode
        summary = qwen_assistant.generate_scene_summary(obs, mode_name=mode_name)
        print(f"[SCENE] {summary}")
        _speak_blocking(summary)
    except Exception as exc:
        print(f"[SCENE] failed: {exc}")
        _speak_blocking("Scene description failed.")
    finally:
        _scene_active.clear()


def _on_scene_button() -> None:
    if _scene_active.is_set():
        return
    threading.Thread(target=_scene_description_worker, daemon=True).start()


def _keyboard_hook(event):
    if event.event_type != "down":
        return
    key_name = (event.name or "").lower().strip()
    if key_name in SCENE_TRIGGER_KEYS:
        _on_scene_button()
    elif key_name in MODE_CYCLE_KEYS:
        _cycle_mode()


def _start_button_listeners() -> None:
    if _has_keyboard:
        try:
            _keyboard.hook(_keyboard_hook, suppress=False)
            print("Buttons active via keyboard hook: play/pause=scene, volume+=mode.")
            return
        except Exception as exc:
            print(f"ALERT: keyboard hook unavailable ({exc}); trying evdev fallback.")

    if _has_evdev:
        def _evdev_loop():
            try:
                devices = [InputDevice(path) for path in list_devices()]
            except Exception as exc:
                print(f"ALERT: evdev discovery failed ({exc})")
                return
            for dev in devices:
                dev.grab_context = None
            while True:
                for dev in devices:
                    try:
                        for ev in dev.read():
                            if ev.type != ecodes.EV_KEY or ev.value != 1:
                                continue
                            if ev.code == ecodes.KEY_PLAYPAUSE:
                                _on_scene_button()
                            elif ev.code == ecodes.KEY_VOLUMEUP:
                                _cycle_mode()
                    except Exception:
                        continue
                time.sleep(0.01)

        threading.Thread(target=_evdev_loop, daemon=True).start()
        print("Buttons active via evdev fallback: play/pause=scene, volume+=mode.")
        return

    print("ALERT: no media-key listener available. Install keyboard or python-evdev.")


print("Loading YOLO model (best.pt)...")
model = YOLO("best.pt", verbose=False)
print(f"Model ready. Classes: {list(model.names.values())}")

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
except Exception:
    pass

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ALERT: camera could not be opened on index 0.")
    cap = None
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    sensor = VL53L5CXSensorRPI(port=None, baudrate=250000, use_serial=True, verbose=False)
except Exception as exc:
    print(f"ALERT: Could not initialize sensor: {exc}")
    sensor = None

sensor_thread = threading.Thread(target=_sensor_polling_thread, daemon=True)
sensor_thread.start()

qwen_assistant = QwenPiAssistant(model_path=QWEN_GGUF_PATH, enabled=ENABLE_QWEN_ASSISTANT)
_start_button_listeners()

print("System ready. Press Ctrl+C to stop.")
speak_text("System ready")

try:
    frame_idx = 0
    last_results = []
    while True:
        if cap is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

        with _frame_lock:
            _current_frame = frame.copy()

        if _scene_active.is_set():
            time.sleep(0.02)
            continue

        with sensor_data_lock:
            sensor_data = last_sensor_data.copy()
            last_update = last_sensor_update_time
        if time.time() - last_update > SENSOR_STALE_TIMEOUT:
            sensor_data = np.zeros((8, 8), dtype=np.int32)

        frame_idx += 1
        if frame_idx % INFER_EVERY_N_FRAMES == 0:
            last_results = list(
                model.predict(
                    source=frame,
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF_THRESHOLD,
                    verbose=False,
                    device="cpu",
                )
            )
        results = last_results

        audio_alert_sent = False
        frame_now = time.time()
        _cleanup_tracked_objects(frame_now)

        active_classes = get_active_classes()
        catch_unknown = mode_catches_unknown()
        current_close_cells: set = set()
        covered_sensor_cells: set = set()
        scene_observations = []

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                raw_label = model.names[cls]

                if raw_label in active_classes:
                    label = raw_label
                elif catch_unknown and raw_label not in OBSTACLE_CLASSES:
                    label = "unidentified object"
                else:
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                direction = get_direction_descriptor(center_x, center_y, frame.shape[1], frame.shape[0])

                sensor_cells = map_camera_to_sensor_grid(x1, y1, x2, y2, frame.shape[0], frame.shape[1])
                sensor_cells = [(row, col) for row, col in sensor_cells if row < 4]
                covered_sensor_cells.update(sensor_cells)

                sensor_close, sensor_distance = check_sensor_close(sensor_data, sensor_cells, CLOSE_THRESHOLD_MM)

                for _r, _c in sensor_cells:
                    if SENSOR_MIN_VALID_MM <= int(sensor_data[_r, _c]) < CLOSE_THRESHOLD_MM:
                        current_close_cells.add((_r, _c))

                sensor_close_confirmed = sensor_close and any(
                    cell in prev_close_cells for cell in sensor_cells
                    if SENSOR_MIN_VALID_MM <= int(sensor_data[cell[0], cell[1]]) < CLOSE_THRESHOLD_MM
                )

                scene_observations.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "direction": direction,
                        "distance_mm": sensor_distance,
                        "is_close": sensor_close_confirmed,
                    }
                )

                if not sensor_close_confirmed:
                    continue

                obj = _find_tracked_object(label, (center_x, center_y))
                should_alert = False
                if obj is None:
                    tracked_objects.append(
                        {
                            "label": label,
                            "center": (center_x, center_y),
                            "last_alert": frame_now,
                            "last_seen": frame_now,
                        }
                    )
                    should_alert = True
                else:
                    moved = _euclidean_dist((center_x, center_y), obj["center"])
                    obj["center"] = (center_x, center_y)
                    obj["last_seen"] = frame_now
                    if (frame_now - obj.get("last_alert", 0)) > ALERT_REMINDER_SECONDS and moved < TRACKED_OBJECT_MOVE_THRESHOLD_PX:
                        should_alert = True
                        obj["last_alert"] = frame_now

                if should_alert and not audio_alert_sent:
                    print(f"ALERT: {label} — {sensor_distance} mm — {direction}")
                    speak_text(
                        qwen_assistant.generate_alert_phrase(
                            label=label,
                            direction=direction,
                            distance_mm=sensor_distance,
                        )
                    )
                    audio_alert_sent = True

        unidentifiable_close = False
        min_unidentified_dist = None
        if catch_unknown:
            for row in range(4):
                for col in range(8):
                    if (row, col) in covered_sensor_cells:
                        continue
                    dist = int(sensor_data[row, col])
                    if SENSOR_MIN_VALID_MM <= dist < (CLOSE_THRESHOLD_MM - 200):
                        current_close_cells.add((row, col))
                        if (row, col) in prev_close_cells:
                            unidentifiable_close = True
                            if min_unidentified_dist is None or dist < min_unidentified_dist:
                                min_unidentified_dist = dist
        if catch_unknown and unidentifiable_close:
            print(f"ALERT: unidentified object ~{min_unidentified_dist} mm (sensor-only)")
            scene_observations.append(
                {
                    "label": "unidentified object",
                    "confidence": 0.0,
                    "direction": "front",
                    "distance_mm": min_unidentified_dist,
                    "is_close": True,
                }
            )
            if not audio_alert_sent:
                speak_text("Alert: unidentified object detected")
                audio_alert_sent = True

        with _mode_lock:
            mode_name = MODES[current_mode]["name"]
        with _scene_lock:
            _latest_scene_snapshot = sorted(scene_observations, key=lambda o: o["confidence"], reverse=True)[:8]
            _latest_scene_mode = mode_name

        prev_close_cells = current_close_cells

except KeyboardInterrupt:
    print("\nCtrl+C — shutting down...")

sensor_stop_event.set()
if sensor_thread.is_alive():
    sensor_thread.join(timeout=1.0)
if cap is not None:
    cap.release()
if sensor:
    sensor.close()
print("Done.")
