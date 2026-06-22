
import cv2
import os
import numpy as np

mode = input("Choose mode: Picture (1), Camera (0), Video (2): ")

cap = None
frame = None
printing = False
# -----------------------------
# INIT SECTION
# -----------------------------
if mode == "1":
    folder_path = r"C:\PlaatjesVerwerken\Data\Active"
    valid_images = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    
    # Get all matching file paths
    all_image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                       if f.lower().endswith(valid_images)]

    if not all_image_paths:
        raise FileNotFoundError("No images found in the specified folder.")
elif mode == "2":
    folder_path = r"C:\PlaatjesVerwerken\Data\Active"
    valid_videos = (".mp4", ".avi", ".mov", ".mkv")

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_videos)]

    if not files:
        raise FileNotFoundError("No video files found.")

    video_path = os.path.join(folder_path, files[0])
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

elif mode == "0":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

else:
    raise ValueError("Invalid choice")

# -----------------------------
# PRE-PROCESS FUNCTIONS
# -----------------------------
def process_all(frame):
    import cv2
    import numpy as np

    original = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Lower saturation (0-60) and high value (150-255) targets white calibrated for purple background
    lower_white = np.array([0, 0, 150]) 
    upper_white = np.array([180, 60, 255])
    binary = cv2.inRange(blur, lower_white, upper_white)

    kernel_pips = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_pips, iterations=2)

    #  Distance Transform to find centers of touching dice
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Watershed Marker Labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 7. Apply Watershed
    markers = cv2.watershed(frame, markers)

    filled = np.zeros_like(gray)
    filled[markers > 1] = 255

    return original, gray, blur, binary, filled
# -----------------------------
# DICE DETECTION
# -----------------------------
def find_dice(cleaned, frame, max_dice=5):
    import cv2
    import numpy as np

    contour_view = frame.copy()
    dice_boxes = []

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], contour_view

    
    unit_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150: # Minimum noise floor
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if h == 0 or w == 0: continue
            if max(w, h) / min(w, h) < 1.3: # Must be square-ish
                unit_candidates.append(area)

    if not unit_candidates:
        return [], contour_view

    #Smallest valid die area and its side length
    min_unit_area = min(unit_candidates)
    side = np.sqrt(min_unit_area)

    # 2. Process Blobs with Strict Area-Based Subdivision
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_unit_area * 0.7: continue # Skip noise

        # Coverage Rule: Force division based on how many units fit in the area
        # e.g., if area is 4x min_unit_area, num_dice = 4
        num_dice = int(round(area / min_unit_area))

        if num_dice <= 1:
            dice_boxes.append(np.int64(cv2.boxPoints(cv2.minAreaRect(cnt))))
            continue

        # 3. Handle Merged Dice (Point-to-Point or Edge-to-Edge)
        # Use K-Means to find centers. This works for diagonal (point-to-point) 
        # because the pixel density will cluster at the centers of the dice.
        mask = np.zeros(cleaned.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        pixel_points = np.column_stack(np.where(mask > 0))
        pixel_points = pixel_points[:, ::-1].astype(np.float32)

        # Force K-Means to find exactly the number of dice the area says should exist
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixel_points, num_dice, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # Apply a uniform box to each discovered center
        _, _, clump_angle = cv2.minAreaRect(cnt)
        for center in centers:
            cx, cy = center
            # Use 0.95 side to ensure >90% coverage without boxes overlapping messy edges
            box_dim = side * 0.95
            box = cv2.boxPoints(((cx, cy), (box_dim, box_dim), clump_angle))
            dice_boxes.append(np.int64(box))

    # 4. Draw results
    dice_boxes = dice_boxes[:max_dice]
    for box in dice_boxes:
        cv2.drawContours(contour_view, [box], -1, (0, 255, 0), 2)

    return dice_boxes, contour_view



# -----------------------------
# PIP DETECTION
# -----------------------------
def count_pips(frame, dice_boxes):
    global printing
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 5
    params.maxThreshold = 255
    params.thresholdStep = 5

    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 50000

    params.filterByCircularity = False
    params.minCircularity = 0.7

    params.filterByConvexity = False
    params.minConvexity = 0.5

    params.filterByInertia = True
    params.minInertiaRatio = 0.60

    params.minDistBetweenBlobs = 1

    detector = cv2.SimpleBlobDetector_create(params)

    total_sum = 0
    pipCountArray = []

    for box in dice_boxes:

        x, y, w, h = cv2.boundingRect(box)

        y1, y2 = max(0, y), min(frame.shape[0], y + h)
        x1, x2 = max(0, x), min(frame.shape[1], x + w)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Pre-process
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Detect pips
        keypoints = detector.detect(thresh)

        count = len(keypoints)

        pipCountArray.append(count)

        total_sum += count

        # Draw green dots
        for kp in keypoints:

            kx, ky = kp.pt

            center_on_frame = (
                int(x1 + kx),
                int(y1 + ky)
            )

            cv2.circle(frame, center_on_frame, 3, (0, 255, 0), -1)

        # Draw die contour
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        cx, cy = x + w // 2, y + h // 2

        label = f"V: {count}"

        (t_w, t_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )

        cv2.rectangle(
            frame,
            (cx - t_w // 2 - 5, cy - 55 - t_h),
            (cx + t_w // 2 + 5, cy - 50),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (cx - t_w // 2, cy - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    # Total banner
    cv2.rectangle(frame, (10, 10), (220, 60), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"TOTAL: {total_sum}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    #  # PRINT VALUES LIVE
    # if len(pipCountArray) == 5:
    #     if printing == 0:
    #      printing = 1
    #      print("Dice values:", pipCountArray)
    # else:
    #     printing = 0




    # Picture album
    
    count_pips.ym_vorige_worp = None

    # Picture album / Print logica
    if len(pipCountArray) == 5:
        ym_huidige_worp = sorted(pipCountArray) 
        
        
        # ym_huidige_worp = pipCountArray               # gebruik als volgorde boeit

        if ym_huidige_worp != count_pips.ym_vorige_worp:
            print("Dice values:", pipCountArray)
            count_pips.ym_vorige_worp = ym_huidige_worp
            
    return frame
    return frame

# -----------------------------
# DICE CHECK
# -----------------------------
def check_throw(frame, dice_boxes):
    import cv2
    
    
    num_dice = len(dice_boxes)
    
    if num_dice != 5:
        
        text = f"INVALID THROW: {num_dice}/5 DICE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        color = (0, 0, 255) 
        
        
        (t_w, t_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        
        text_x = (frame.shape[1] - t_w) // 2
        text_y = frame.shape[0] - 50
        
        
        cv2.rectangle(frame, (text_x - 10, text_y - t_h - 10), 
                      (text_x + t_w + 10, text_y + 10), (0, 0, 0), -1)
        
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
    return frame


# -----------------------------
# DISPLAY SECTION
# -----------------------------


if mode == "1":
    # PROCESS ALL IMAGES IN FOLDER
    for path in all_image_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        # Pipeline
        output0, output1, output2, output3, output4 = process_all(frame)
        DiceContours, DiceVieuw = find_dice(output4, frame)
        
        # We work on a copy to keep the 'original' clean for display
        results_frame = frame.copy()
        results = count_pips(results_frame, DiceContours)
        checked_results = check_throw(results, DiceContours)
        cv2.imshow("Final Result", checked_results)      # Dice detection + pip count
        cv2.imshow("Original Image", output0)           # Original input image
        cv2.imshow("Binary Threshold", output3)         # Adaptive threshold result
        cv2.imshow("Cleaned Mask", output4)             # Morphologically cleaned mask
        print(f"Processed: {os.path.basename(path)}")

        # Wait for key: press 'q' to quit folder, any other key for next image
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
else:
    # PROCESS CAMERA OR VIDEO STREAM
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output0, output1, output2, output3, output4 = process_all(frame)
        DiceContours, DiceVieuw = find_dice(output4, frame)
        
        # Overlay
        results_frame = frame.copy()
        results = count_pips(results_frame, DiceContours)
        checked_results = check_throw(results, DiceContours)

        cv2.imshow("original", output0)
        cv2.imshow("BIN", output3)
        cv2.imshow("DICEPIPS", checked_results)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# -----------------------------
# CLEANUP
# -----------------------------
if cap is not None:
    cap.release()

cv2.destroyAllWindows()