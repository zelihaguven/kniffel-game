"""
Mock detector for testing without a camera.
Simulates what your teammate's CV module will produce.

Usage:
    from cv_interface.mock import inject_mock_detections
    inject_mock_detections()   # pushes 5 random dice into the adapter
"""

import random
from .adapter import cv_adapter


def inject_mock_detections(count: int = 5) -> None:
    """Push fake CV detections — useful for integration testing."""
    detections = []
    x = 100
    for _ in range(count):
        detections.append(
            {
                "value": random.randint(1, 6),
                "bbox": (x, 80, 60, 60),
                "confidence": round(random.uniform(0.88, 0.99), 2),
            }
        )
        x += 80
    cv_adapter.push(detections)
