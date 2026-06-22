#!/usr/bin/env python3
"""
Test camera and detector without pygame
"""
import cv2
import time
from cv_interface.camera import CameraFeed

print("Starting camera test...")
camera = CameraFeed(source=0)
ok = camera.start()

if not ok:
    print("❌ Camera failed to start")
    exit(1)

print("✓ Camera started successfully")

# Let it warm up
time.sleep(2)

# Read 10 frames
for i in range(10):
    frame, detections = camera.get_latest()
    
    if frame is not None:
        print(f"Frame {i}: {frame.shape} | Detections: {len(detections)}")
        if detections:
            for d in detections:
                print(f"  - Die value: {d['value']}, confidence: {d['confidence']:.2f}")
    else:
        print(f"Frame {i}: No frame yet")
    
    time.sleep(0.5)

camera.stop()
print("✓ Camera test complete")
