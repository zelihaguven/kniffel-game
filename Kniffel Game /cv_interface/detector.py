"""
Dice detector — two interchangeable pipelines:

  CONTOURS  (BinContours.py)   adaptive threshold + morphology
  WATERSHED (BinWatershed.py)  HSV white-mask + watershed + K-means split

Both expose the same output format:
    list of {'value': int, 'bbox': (x,y,w,h), 'confidence': float}

Standalone usage:
    python -m cv_interface.detector images/roll1.jpg --debug [--mode watershed]
"""

import cv2
import numpy as np
from enum import Enum
from pathlib import Path
import argparse


class DetectorMode(Enum):
    CONTOURS  = "contours"   # BinContours.py  — any dice colour / lighting
    WATERSHED = "watershed"  # BinWatershed.py — white dice, handles touching


# ── Shared: pip counting via blob detector ────────────────────────────────────

def _make_blob_detector() -> cv2.SimpleBlobDetector:
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold        = 5
    p.maxThreshold        = 255
    p.thresholdStep       = 5
    p.filterByArea        = True;  p.minArea        = 30;   p.maxArea    = 50000
    p.filterByCircularity = False; p.minCircularity = 0.7
    p.filterByConvexity   = False; p.minConvexity   = 0.5
    p.filterByInertia     = True;  p.minInertiaRatio = 0.60
    p.minDistBetweenBlobs = 1
    return cv2.SimpleBlobDetector_create(p)


_BLOB_DETECTOR = _make_blob_detector()


def _count_pips_in_box(frame: np.ndarray, box: np.ndarray) -> int:
    x, y, w, h = cv2.boundingRect(box)
    y1, y2 = max(0, y), min(frame.shape[0], y + h)
    x1, x2 = max(0, x), min(frame.shape[1], x + w)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh  = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 4)
    return len(_BLOB_DETECTOR.detect(thresh))


def _boxes_to_detections(frame: np.ndarray, boxes: list[np.ndarray]) -> list[dict]:
    detections: list[dict] = []
    for box in boxes:
        value = _count_pips_in_box(frame, box)
        if 1 <= value <= 6:
            x, y, w, h = cv2.boundingRect(box)
            detections.append({"value": value, "bbox": (x, y, w, h), "confidence": 0.90})
    return sorted(detections, key=lambda d: d["bbox"][0])


# ── CONTOURS pipeline (BinContours.py) ───────────────────────────────────────

def _contours_preprocess(frame: np.ndarray) -> np.ndarray:
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (7, 7), 0)
    binary = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            cv2.drawContours(filled, [cnt], -1, 255, cv2.FILLED)

    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel, iterations=1)


def _contours_find_boxes(mask: np.ndarray, max_dice: int = 5) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Estimate single-die size from median of square-ish blobs (robust to noise)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        if max(w, h) / min(w, h) < 1.3:
            candidates.append(area)

    if not candidates:
        return []

    base_area = np.median(candidates)
    base_side = np.sqrt(base_area)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    boxes: list[np.ndarray] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < base_area * 0.5:
            continue

        rect     = cv2.minAreaRect(cnt)
        center   = rect[0]
        angle    = rect[2]

        # Snap near-axis angles for cleaner boxes
        if abs(angle) < 8:
            angle = 0.0
        elif abs(abs(angle) - 90) < 8:
            angle = 90.0

        num_dice = int(round(area / base_area))
        num_dice = max(1, min(max_dice, num_dice))

        if num_dice == 1:
            boxes.append(np.int64(cv2.boxPoints(
                (center, (base_side * 0.92, base_side * 0.92), angle)
            )))
            continue

        # K-means to split merged blobs — works for any touching geometry
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        pts = np.column_stack(np.where(blob_mask > 0))[:, ::-1].astype(np.float32)

        try:
            _, _, centers = cv2.kmeans(pts, num_dice, None, criteria, 10,
                                       cv2.KMEANS_PP_CENTERS)
        except cv2.error:
            continue

        for cx, cy in centers:
            boxes.append(np.int64(cv2.boxPoints(
                ((float(cx), float(cy)), (base_side * 0.90, base_side * 0.90), angle)
            )))

    return boxes[:max_dice]


def _detect_contours(frame: np.ndarray) -> list[dict]:
    mask  = _contours_preprocess(frame)
    boxes = _contours_find_boxes(mask)
    return _boxes_to_detections(frame, boxes)


# ── WATERSHED pipeline (BinWatershed.py) ─────────────────────────────────────

def _watershed_preprocess(frame: np.ndarray) -> np.ndarray:
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))

    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    dist   = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    _, fg  = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    fg     = np.uint8(fg)
    bg     = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=3)
    unk    = cv2.subtract(bg, fg)

    _, markers = cv2.connectedComponents(fg)
    markers    = markers + 1
    markers[unk == 255] = 0
    markers    = cv2.watershed(frame, markers)

    filled = np.zeros(frame.shape[:2], dtype=np.uint8)
    filled[markers > 1] = 255
    return filled


def _watershed_find_boxes(mask: np.ndarray, frame: np.ndarray,
                          max_dice: int = 5) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Estimate unit area from smallest square-ish blob
    unit_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        if max(w, h) / min(w, h) < 1.3:
            unit_candidates.append(area)

    if not unit_candidates:
        return []

    min_unit_area = min(unit_candidates)
    side          = np.sqrt(min_unit_area)

    boxes: list[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_unit_area * 0.7:
            continue

        num_dice = int(round(area / min_unit_area))

        if num_dice <= 1:
            boxes.append(np.int64(cv2.boxPoints(cv2.minAreaRect(cnt))))
            continue

        # K-means to find centers of merged dice blobs
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        pts = np.column_stack(np.where(blob_mask > 0))[:, ::-1].astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pts, num_dice, None, criteria, 10,
                                   cv2.KMEANS_PP_CENTERS)
        _, _, angle = cv2.minAreaRect(cnt)
        dim = side * 0.95
        for cx, cy in centers:
            boxes.append(np.int64(cv2.boxPoints(((cx, cy), (dim, dim), angle))))

    return boxes[:max_dice]


def _detect_watershed(frame: np.ndarray) -> list[dict]:
    mask  = _watershed_preprocess(frame)
    boxes = _watershed_find_boxes(mask, frame)
    return _boxes_to_detections(frame, boxes)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_dice_from_frame(
    frame: np.ndarray,
    mode: DetectorMode = DetectorMode.CONTOURS,
) -> list[dict]:
    """
    Detect dice in a BGR frame. Returns list of:
        {'value': int, 'bbox': (x,y,w,h), 'confidence': float}
    sorted left-to-right.
    """
    if mode == DetectorMode.WATERSHED:
        return _detect_watershed(frame)
    return _detect_contours(frame)


def get_debug_mask(
    frame: np.ndarray,
    mode: DetectorMode = DetectorMode.CONTOURS,
) -> np.ndarray:
    """Return the binary detection mask as a BGR frame (for live preview)."""
    gray = (_watershed_preprocess if mode == DetectorMode.WATERSHED
            else _contours_preprocess)(frame)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def detect_dice(image_path: str, debug: bool = False,
                mode: DetectorMode = DetectorMode.CONTOURS) -> list[dict]:
    """Same as detect_dice_from_frame but reads from a file path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    detections = detect_dice_from_frame(img, mode=mode)
    if debug:
        _save_debug_image(img, detections, image_path)
    return detections


# ── Debug output ──────────────────────────────────────────────────────────────

def _save_debug_image(img: np.ndarray, detections: list[dict],
                      source_path: str) -> None:
    out = img.copy()
    for d in detections:
        x, y, w, h = d["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 80), 2)
        cv2.putText(out, str(d["value"]), (x + 4, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 80), 2)
    total = sum(d["value"] for d in detections)
    cv2.putText(out, f"Dice: {len(detections)}  Sum: {total}",
                (16, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 80), 2)
    out_path = Path(source_path).with_suffix(".debug.jpg")
    cv2.imwrite(str(out_path), out)
    print(f"[debug] saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Detect dice values in an image")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--mode",   choices=["contours", "watershed"],
                        default="contours")
    args  = parser.parse_args()
    mode  = DetectorMode(args.mode)

    for path in args.images:
        print(f"\n── {path} ──")
        try:
            dets = detect_dice(path, debug=args.debug, mode=mode)
        except ValueError as e:
            print(f"  ERROR: {e}"); continue
        if not dets:
            print("  No dice detected"); continue
        for d in dets:
            x, y, w, h = d["bbox"]
            print(f"  [{d['value']}]  bbox=({x},{y},{w},{h})")
        print(f"  → values: {[d['value'] for d in dets]}  sum: {sum(d['value'] for d in dets)}")


if __name__ == "__main__":
    _cli()
