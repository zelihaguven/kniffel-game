# Kniffel Vision AI

A computer-vision powered Kniffel (Yahtzee) game for two players, built as a school project at Hochschule Heilbronn (HHN).

The game reads real physical dice from a webcam using OpenCV and renders a full Kniffel scorecard in a pygame UI at 60 FPS.

---

## Features

- **Live dice detection** via webcam (OpenCV) — no manual input needed
- **Two interchangeable CV pipelines** — Contours and Watershed, togglable mid-game with `D`
- **Full Kniffel rules** — all 13 scoring categories, upper-section bonus (+35), 3 rolls per turn
- **Two-player support** — custom player names, per-player scorecards with potential-score preview
- **Hold / unlock dice** between rolls — held dice are excluded from the next capture
- **Fallback random roll** when no camera is connected (`SPACE`)
- **Casino-themed UI** — 1280 × 720 @ 60 FPS, felt-green left panel + warm paper scorecard

---

## Project Structure

```
Kniffel_ImageProcessing/
├── main.py                   # Entry point and pygame event loop
├── run.sh                    # Quick-launch script (macOS / Linux)
├── requirements.txt
│
├── cv_interface/
│   ├── detector.py           # Two CV pipelines: CONTOURS and WATERSHED
│   ├── camera.py             # Background capture thread
│   ├── adapter.py            # Thread-safe handshake for external CV modules
│   └── mock.py               # Fake detections for integration testing
│
├── engine/
│   ├── dice.py               # Die / DiceSet data model
│   ├── scoring.py            # All 13 Kniffel categories + Scorecard class
│   └── game.py               # Turn state machine (ROLL → HOLD → SCORE)
│
├── ui/
│   ├── renderer.py           # pygame renderer — all drawing logic
│   └── constants.py          # Colors, layout constants, pip dot positions
│
├── BinContours.py            # Original standalone Contours research script
└── BinWatershed.py           # Original standalone Watershed research script
```

---

## Requirements

- Python 3.10 or newer
- pygame ≥ 2.5
- numpy ≥ 1.24
- opencv-python ≥ 4.8

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Game

```bash
python main.py
```

Select a different camera (default is `0` — built-in webcam):

```bash
python main.py --camera 1
```

macOS / Linux convenience script:

```bash
./run.sh
```

for the testing or more insights of the detection : 
```bash
python BinCountours.py
```

---

## Controls

| Key / Action | Effect |
|---|---|
| `ROLL DICE` button | Capture current dice from camera |
| `SPACE` | Random roll (camera-free fallback) |
| Click die face or `LOCK` button | Hold / unlock that individual die |
| `RELEASE ALL` button | Release all held dice |
| `HOLD ALL` button | Hold all dice at once |
| Click a scorecard row | Record score for that category |
| `D` | Toggle detector mode (Contours ↔ Watershed) |
| `C` | Switch camera source (0 ↔ 1) |
| `R` | Return to name entry and reset game |
| `NEW GAME` button | Full reset to name entry screen |

---

## CV Detection Pipelines

### CONTOURS (default)

Works on any dice color and lighting condition.

1. Grayscale → Gaussian blur (7×7) → adaptive threshold (inverted binary mask)
2. Morphological close (5×5 kernel) to fill gaps inside die faces
3. Fill all external contours larger than 200 px²
4. Estimate single-die size from the **median area** of square-ish blobs (robust to noise)
5. For each blob: estimate how many dice it contains (`round(blob_area / median_die_area)`)
6. If multiple dice merged: **K-means clustering** splits the blob into individual die centers
7. For each die ROI: `cv2.SimpleBlobDetector` counts pips → face value 1–6

### WATERSHED

Optimized for white or cream-colored dice.

1. Convert to HSV → mask near-white pixels (H 0–180, S 0–60, V 150–255)
2. Morphological close to fill holes
3. Distance transform → threshold at 40 % of max → sure foreground markers
4. Dilate mask → sure background; subtract → unknown region
5. **cv2.watershed** with connected-component markers to separate touching dice
6. Extract contours from the watershed result → same K-means + blob-detector pip step

Toggle between modes at runtime with the `D` key. The active mode is shown in the HUD overlay.

---

## Standalone Detector CLI

Run the detector on still images without launching the full game:

```bash
python -m cv_interface.detector roll.jpg --debug
python -m cv_interface.detector roll.jpg --mode watershed --debug
```

`--debug` saves an annotated copy as `roll.debug.jpg` in the same folder.

Multiple images can be passed at once:

```bash
python -m cv_interface.detector img1.jpg img2.jpg --debug
```

---

## Scoring Categories

| Category | Points |
|---|---|
| Ones | Sum of all 1s |
| Twos | Sum of all 2s |
| Threes | Sum of all 3s |
| Fours | Sum of all 4s |
| Fives | Sum of all 5s |
| Sixes | Sum of all 6s |
| **Upper Bonus** | **+35 if upper section total ≥ 63** |
| Three of a Kind | Sum of all dice (if ≥ 3 of one face) |
| Four of a Kind | Sum of all dice (if ≥ 4 of one face) |
| Full House | 25 pts |
| Small Straight | 30 pts (any 4-value sequence) |
| Large Straight | 40 pts (5-value sequence) |
| Kniffel! | 50 pts (all 5 dice equal) |
| Chance | Sum of all dice (always valid) |

Potential scores for all unfilled categories are shown in grey on the scorecard after each roll.

---

## Software Used

- Python 3.13
- OpenCV 4.x
- pygame 2.x
- NumPy
- VS Code 
