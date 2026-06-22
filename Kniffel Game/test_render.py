#!/usr/bin/env python3
"""Test if frame rendering works"""
import pygame
import cv2
import numpy as np
from cv_interface.camera import CameraFeed
from ui.renderer import _frame_to_surface

camera = CameraFeed(source=0)
camera.start()

import time
time.sleep(2)

frame, _ = camera.get_latest()
camera.stop()

if frame is not None:
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
    print(f"Frame min/max: {frame.min()}/{frame.max()}")
    
    # Try converting
    try:
        surf = _frame_to_surface(frame, 650, 390)
        print(f"✓ Surface created: {surf.get_size()}")
        
        # Save to file to check
        pygame.image.save(surf, "/tmp/test_surface.png")
        print("✓ Saved to /tmp/test_surface.png")
    except Exception as e:
        print(f"✗ Error converting: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No frame captured")
