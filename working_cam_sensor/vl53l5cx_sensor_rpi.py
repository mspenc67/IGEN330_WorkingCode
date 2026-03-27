"""Raspberry Pi specific VL53L5CX serial parser.

Keeps the desktop helper untouched while matching ESP32 semantics used on Pi:
- 0 means invalid / no return
- optional frame marker lines are accepted
"""

from __future__ import annotations

import numpy as np

from working_cam_sensor.vl53l5cx_sensor import VL53L5CXSensor


class VL53L5CXSensorRPI(VL53L5CXSensor):
    """Pi-focused subclass with ESP32-frame-aware serial parsing."""

    def _read_serial_data(self):
        if not self.serial_conn.in_waiting:
            return None

        data = self.serial_conn.read(self.serial_conn.in_waiting).decode("utf-8", errors="ignore")
        self._serial_buffer += data

        lines = self._serial_buffer.split("\n")
        distances = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.lower() in {"f", "frame_end"}:
                if len(distances) == 8:
                    self._serial_buffer = "\n".join(lines[i + 1 :])
                    return np.array(distances, dtype=np.int32)
                continue

            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            row = []
            for tok in parts:
                if not tok:
                    continue
                if tok == "---":
                    row.append(0)
                    continue
                try:
                    row.append(int(tok))
                except ValueError:
                    row.append(0)

            if len(row) < 8:
                row.extend([0] * (8 - len(row)))
            elif len(row) > 8:
                row = row[:8]

            distances.append(row)
            if len(distances) == 8:
                self._serial_buffer = "\n".join(lines[i + 1 :])
                return np.array(distances, dtype=np.int32)

        if len(self._serial_buffer) > 10000:
            self._serial_buffer = "\n".join(lines[-20:])
        return None
