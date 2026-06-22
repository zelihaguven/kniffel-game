"""
Background camera thread.
Continuously captures frames, runs dice detection, and draws bounding-box overlays.
The renderer reads the latest annotated frame and debug mask via get_latest().
"""

import threading
import cv2
import numpy as np
from .detector import detect_dice_from_frame, get_debug_mask, DetectorMode


class CameraFeed:
    def __init__(self, source: int = 0):
        self._cap        = cv2.VideoCapture(source)
        self._lock       = threading.Lock()
        self._annotated: np.ndarray | None  = None
        self._debug_mask: np.ndarray | None = None
        self._detections: list[dict]        = []
        self._running    = False
        self._thread: threading.Thread | None = None
        self._mode       = DetectorMode.CONTOURS

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not self._cap.isOpened():
            return False
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        self._cap.release()

    def is_open(self) -> bool:
        return self._cap.isOpened()

    # ── detector mode ─────────────────────────────────────────────────────────

    def set_mode(self, mode: DetectorMode) -> None:
        with self._lock:
            self._mode = mode

    def get_mode(self) -> DetectorMode:
        with self._lock:
            return self._mode

    def toggle_mode(self) -> DetectorMode:
        with self._lock:
            self._mode = (DetectorMode.WATERSHED
                          if self._mode == DetectorMode.CONTOURS
                          else DetectorMode.CONTOURS)
            return self._mode

    # ── read ──────────────────────────────────────────────────────────────────

    def get_latest(self) -> tuple[np.ndarray | None, np.ndarray | None, list[dict]]:
        """Returns (annotated_frame, debug_mask, detections). Thread-safe."""
        with self._lock:
            return self._annotated, self._debug_mask, list(self._detections)

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    continue
                with self._lock:
                    mode = self._mode
                detections = detect_dice_from_frame(frame, mode=mode)
                annotated  = self._draw_overlay(frame.copy(), detections)
                mask       = get_debug_mask(frame, mode=mode)
                with self._lock:
                    self._annotated   = annotated
                    self._debug_mask  = mask
                    self._detections  = detections
        except Exception as e:
            print(f"[CameraFeed] Error in thread: {e}")
            import traceback
            traceback.print_exc()

    def _draw_overlay(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        for d in detections:
            x, y, w, h = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 215, 80), 2)
            cv2.putText(frame, str(d["value"]), (x + 4, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 80), 2)
        count = len(detections)
        label = f"Detected: {count}/5"
        color = (0, 215, 80) if count == 5 else (0, 180, 255)
        cv2.putText(frame, label, (12, frame.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return frame
