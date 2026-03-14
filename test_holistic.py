import cv2
import numpy as np
from holistic_detector import HolisticDetector
import time

def test_holistic():
    print("Testing HolisticDetector initialization...")
    try:
        detector = HolisticDetector()
        print("Initialization successful.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("Testing process on blank image...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    timestamp_ms = int(time.time() * 1000)
    try:
        face_results, hand_results = detector.process(img, timestamp_ms)
        print("Process successful.")
        
        print("Testing draw_ultimate_art...")
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.draw_ultimate_art(canvas, face_results, hand_results, (255, 255, 255))
        print("Drawing successful.")
        
    except Exception as e:
        print(f"Process/Draw failed: {e}")

if __name__ == "__main__":
    test_holistic()
