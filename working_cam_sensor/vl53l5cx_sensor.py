"""VL53L5CX sensor support.

This module is a convenient Python interface to the VL53L5CX ToF sensor.
It supports both ESP32-based serial output and direct I2C via a compatible
Python library.

This is a thin port of the ``vl53l5cx_sensor_F.py`` helper in the project root.
"""

import serial
import serial.tools.list_ports
import time
import numpy as np


class VL53L5CXSensor:
    """Python interface for VL53L5CX sensor - equivalent to Arduino code"""
    
    def __init__(self, port=None, baudrate=250000, use_serial=True, verbose=True):
        """Initialize the sensor.

        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux).
                 If None, will auto-detect ESP32.
            baudrate: Serial baud rate.
            use_serial: If True, use serial communication. If False, use direct I2C.
            verbose: If False, suppress informational prints.
        """
        self.use_serial = use_serial
        self.verbose = verbose
        self.image_resolution = 0
        self.image_width = 0
        self.max_distance_mm = 4000  # Used for mapping 'no return' values
        self.serial_conn = None
        self.i2c_sensor = None
        self._serial_buffer = ""  # buffer to accumulate partial serial blocks
        self._last_no_data_warn = 0.0

        if use_serial:
            self._init_serial(port, baudrate)
        else:
            self._init_i2c()
    
    def _init_serial(self, port, baudrate):
        """Initialize serial communication with ESP32"""
        # Determine candidate ports to try
        if port is None:
            ports = serial.tools.list_ports.comports()
            if self.verbose:
                print("Available serial ports:")
                for p in ports:
                    print(f"  {p.device}: {p.description}")

            # Prefer common ESP32 / CH340 USB serial adapters (avoid generic Bluetooth COM ports)
            preferred_keywords = ["ch340", "cp21", "usb serial", "silicon labs", "usb-serial"]
            preferred = [p.device for p in ports if any(k in (p.description or "").lower() for k in preferred_keywords)]
            if preferred:
                candidate_ports = preferred + [p.device for p in ports if p.device not in preferred]
            else:
                candidate_ports = [p.device for p in ports]
        else:
            candidate_ports = [port]

        # Try each port with best-known baud rates
        baud_candidates = [baudrate, 250000, 115200, 921600]
        last_exception = None
        for p in candidate_ports:
            for b in baud_candidates:
                try:
                    self.serial_conn = serial.Serial(p, b, timeout=1)
                    time.sleep(2)
                    if self.verbose:
                        print(f"Connected to ESP32 on {p} at {b} baud")
                    return
                except Exception as e:
                    last_exception = e
                    # Continue trying other baud rates/ports
                    continue

        # If we reach here, we failed to open any port at any baud
        if self.verbose:
            print("Error connecting to serial port:", last_exception)
            print("Make sure ESP32 is connected, the correct port is used, and the firmware is running.")
        raise Exception("Failed to connect at any baud rate")
    
    def _init_i2c(self):
        """Initialize direct I2C connection (requires sparkfun-qwiic-vl53l5cx)"""
        try:
            import qwiic_vl53l5cx  # type: ignore
            self.i2c_sensor = qwiic_vl53l5cx.QwiicVL53L5CX()
            
            if not self.i2c_sensor.begin():
                raise Exception("Sensor not found - check your wiring")
            
            self.i2c_sensor.set_resolution(8*8)  # Enable all 64 pads
            self.image_resolution = self.i2c_sensor.get_resolution()
            self.image_width = int(np.sqrt(self.image_resolution))
            self.i2c_sensor.start_ranging()
            if self.verbose:
                print("VL53L5CX sensor initialized via I2C")
        except ImportError:
            if self.verbose:
                print("I2C mode requires: pip install sparkfun-qwiic-vl53l5cx")
            raise
        except Exception as e:
            if self.verbose:
                print(f"Error initializing I2C sensor: {e}")
            raise
    
    def is_data_ready(self):
        """Check if sensor data is ready"""
        if self.use_serial:
            # For serial mode, we'll read when data is available
            return self.serial_conn.in_waiting > 0
        else:
            return self.i2c_sensor.is_data_ready()
    
    def flush_serial_buffer(self):
        """Clear stale data from the serial input buffer"""
        if self.use_serial and self.serial_conn:
            self.serial_conn.reset_input_buffer()
    
    def get_ranging_data(self):
        """
        Read distance data into array (equivalent to Arduino getRangingData)
        Returns: 8x8 numpy array of distances in mm, or None if no data
        """
        if self.use_serial:
            return self._read_serial_data()
        else:
            return self._read_i2c_data()

    # Compatibility helper for callers expecting a "read_frame" API
    def read_frame(self):
        """Alias for get_ranging_data (returns 8x8 numpy array or None)."""
        return self.get_ranging_data()
    
    def _read_serial_data(self):
        """Read sensor data from ESP32 via serial.

        The ESP32 outputs an 8x8 grid of values (and "---" for no return).
        The serial stream may contain partial frames, so we buffer until we have a full 8x8 frame.
        """
        # Read any available bytes
        if not self.serial_conn.in_waiting:
            now = time.time()
            if self.verbose and now - self._last_no_data_warn > 5.0:
                print("WARNING: No serial data arriving from ESP32 (in_waiting=0).")
                self._last_no_data_warn = now
            return None

        data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
        self._serial_buffer += data

        # Debug: print raw data once to help confirm format (avoid flooding terminal)
        if self.verbose and not hasattr(self, '_raw_dump_shown'):
            print(f"Raw data sample (first 200 chars): {repr(data[:200])}")
            self._raw_dump_shown = True

        lines = self._serial_buffer.split("\n")
        distances = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Treat empty lines as frame delimiters
            if not stripped:
                if len(distances) == 8:
                    self._serial_buffer = "\n".join(lines[i+1:])
                    return np.array(distances, dtype=np.int32)
                continue

            # Split by comma first (common), otherwise by whitespace
            parts = [p.strip() for p in line.split(',')] if ',' in line else line.split()
            row = []
            for tok in parts:
                if not tok:
                    continue
                if tok == "---":
                    row.append(self.image_resolution or self.max_distance_mm)
                    continue
                try:
                    val = int(tok)
                    # Treat 0 as max distance (no return)
                    row.append(self.image_resolution or self.max_distance_mm if val == 0 else val)
                except ValueError:
                    # Non-numeric tokens treated as max distance
                    row.append(self.image_resolution or self.max_distance_mm)

            # Pad/truncate to 8 values in case of formatting issues
            if len(row) < 8:
                row.extend([self.image_resolution or self.max_distance_mm] * (8 - len(row)))
            elif len(row) > 8:
                row = row[:8]

            distances.append(row)

            if len(distances) == 8:
                self._serial_buffer = "\n".join(lines[i+1:])
                return np.array(distances, dtype=np.int32)

        # Keep buffer trimmed to avoid unbounded growth
        if len(self._serial_buffer) > 10000:
            self._serial_buffer = "\n".join(lines[-20:])

        return None
    
    def _read_i2c_data(self):
        """Read sensor data directly via I2C"""
        if not self.i2c_sensor.is_data_ready():
            return None
        
        measurement_data = self.i2c_sensor.get_ranging_data()
        
        # Convert to 8x8 numpy array
        # The ST library returns data transposed, so we need to reshape
        distances = np.array(measurement_data.distance_mm)
        distances_2d = distances.reshape((8, 8))
        
        # Transpose to match Arduino output format (increasing y, decreasing x)
        return distances_2d
    
    def print_distance_array(self, distance_array):
        """
        Pretty-print distance array (equivalent to Arduino print format)
        The ST library returns the data transposed from zone mapping shown in datasheet
        Pretty-print data with increasing y, decreasing x to reflect reality
        """
        if distance_array is None:
            return
        
        image_width = distance_array.shape[1]
        
        # Print with increasing y, decreasing x (like Arduino code)
        for y in range(image_width):
            row = []
            for x in range(image_width - 1, -1, -1):
                row.append(f"{distance_array[y, x]}")
            if self.verbose:
                print("\t".join(row))
        if self.verbose:
            print()
            print()
    
    def close(self):
        """Close connections"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        if self.verbose:
            print("Sensor connection closed")
