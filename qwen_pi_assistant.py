"""Tiny Qwen helper for Raspberry Pi obstacle-alert phrasing.

This module is optional. If dependencies/model are missing, it gracefully
falls back to deterministic template text.
"""

from __future__ import annotations

import os
import time


class QwenPiAssistant:
    """Generate short spoken alert phrases using a tiny local Qwen GGUF model."""

    def __init__(self, model_path: str, enabled: bool = False) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self._llm = None
        self._last_err_print = 0.0
        self._load_model()

    def _load_model(self) -> None:
        if not self.enabled:
            return
        if not os.path.exists(self.model_path):
            print(
                "Qwen disabled: GGUF model not found at "
                f"{self.model_path}. Falling back to template alerts."
            )
            self.enabled = False
            return
        try:
            from llama_cpp import Llama  # type: ignore

            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=256,
                n_threads=max(1, (os.cpu_count() or 2) - 1),
                n_batch=64,
                verbose=False,
            )
            print(f"Qwen assistant ready: {self.model_path}")
        except Exception as exc:
            print(f"Qwen disabled: could not load llama-cpp model ({exc})")
            self.enabled = False

    @staticmethod
    def _fallback(label: str, direction: str, distance_mm: int | None) -> str:
        if distance_mm is None:
            return f"{label} {direction}"
        return f"{label} {direction}, {distance_mm} millimeters"

    def generate_alert_phrase(self, label: str, direction: str, distance_mm: int | None) -> str:
        if not self.enabled or self._llm is None:
            return self._fallback(label, direction, distance_mm)

        distance_text = "unknown" if distance_mm is None else f"{distance_mm} millimeters"
        prompt = (
            "You are a mobility assistant. Return one short phrase (max 8 words),\n"
            "plain English, no punctuation except commas.\n"
            f"Object: {label}\nDirection: {direction}\nDistance: {distance_text}\n"
            "Output:"
        )
        try:
            out = self._llm(
                prompt,
                max_tokens=16,
                temperature=0.2,
                top_p=0.9,
                stop=["\n"],
            )
            text = out["choices"][0]["text"].strip()
            if text:
                return text
        except Exception as exc:
            now = time.time()
            if now - self._last_err_print > 5.0:
                print(f"Qwen generation failed; using fallback ({exc})")
                self._last_err_print = now
        return self._fallback(label, direction, distance_mm)

    def generate_scene_summary(self, observations: list[dict], mode_name: str = "Normal") -> str:
        if not observations:
            return f"Mode {mode_name}. Path looks clear right now."

        obs_lines = []
        for obs in observations[:8]:
            label = obs.get("label", "object")
            direction = obs.get("direction", "center")
            distance_mm = obs.get("distance_mm")
            is_close = bool(obs.get("is_close", False))
            dist_text = "unknown" if distance_mm is None else f"{int(distance_mm)} millimeters"
            close_text = "close" if is_close else "not close"
            obs_lines.append(f"{label} at {direction}, {dist_text}, {close_text}")

        if not self.enabled or self._llm is None:
            top = observations[0]
            label = top.get("label", "object")
            direction = top.get("direction", "center")
            distance_mm = top.get("distance_mm")
            if distance_mm is None:
                return f"Mode {mode_name}. Main object is {label} at {direction}."
            return f"Mode {mode_name}. Main object is {label} at {direction}, {int(distance_mm)} millimeters."

        prompt = (
            "You are a mobility assistant. Summarize this scene in 1-2 short spoken sentences.\n"
            "Keep it simple and practical for navigation.\n"
            f"Mode: {mode_name}\n"
            "Observations:\n"
            + "\n".join(obs_lines)
            + "\nSummary:"
        )
        try:
            out = self._llm(
                prompt,
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                stop=["\n\n"],
            )
            text = out["choices"][0]["text"].strip()
            if text:
                return text
        except Exception as exc:
            now = time.time()
            if now - self._last_err_print > 5.0:
                print(f"Qwen scene summary failed; using fallback ({exc})")
                self._last_err_print = now
        return f"Mode {mode_name}. {obs_lines[0]}."
