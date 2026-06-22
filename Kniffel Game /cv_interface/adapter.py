"""
CV Integration Adapter
======================
This is the handshake point between the game and your teammate's OpenCV module.

Your teammate's detector should call `adapter.push(detections)` whenever it
has a new set of die readings. The game reads the latest frame on each roll.

Detection dict format:
    {
        "value":      int,            # face value 1–6
        "bbox":       (x, y, w, h),  # bounding box in camera pixel coords
        "confidence": float,          # 0.0–1.0 (optional, defaults to 1.0)
    }

Example integration (your teammate's side):
    from cv_interface.adapter import cv_adapter

    # After running their detection pipeline:
    detections = [
        {"value": 3, "bbox": (120, 80, 60, 60), "confidence": 0.97},
        {"value": 5, "bbox": (210, 80, 60, 60), "confidence": 0.95},
        ...
    ]
    cv_adapter.push(detections)
"""

import threading
from typing import Optional


class CVAdapter:
    def __init__(self):
        self._lock = threading.Lock()
        self._detections: list[dict] = []
        self._new_data = False

    def push(self, detections: list[dict]) -> None:
        """Called by the CV module to deliver new dice readings."""
        sanitized = []
        for d in detections:
            if "value" not in d or "bbox" not in d:
                continue
            sanitized.append(
                {
                    "value": int(d["value"]),
                    "bbox": tuple(d["bbox"]),
                    "confidence": float(d.get("confidence", 1.0)),
                }
            )
        with self._lock:
            self._detections = sanitized
            self._new_data = bool(sanitized)

    def consume(self) -> Optional[list[dict]]:
        """Returns latest detections and clears the 'new data' flag."""
        with self._lock:
            if not self._new_data:
                return None
            self._new_data = False
            return list(self._detections)

    def peek(self) -> list[dict]:
        """Returns latest detections without consuming."""
        with self._lock:
            return list(self._detections)

    def has_feed(self) -> bool:
        with self._lock:
            return bool(self._detections)

    def clear(self) -> None:
        with self._lock:
            self._detections = []
            self._new_data = False


# Singleton — import this in both the game and the CV module
cv_adapter = CVAdapter()
