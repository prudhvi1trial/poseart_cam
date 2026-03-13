import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import numpy as np

class HolisticDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # 1. Paths
        self.face_model_path = 'face_landmarker.task'
        self.hand_model_path = 'hand_landmarker.task'
        
        self._ensure_models_exist()

        # 2. Face Landmarker
        base_options_face = python.BaseOptions(model_asset_path=self.face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

        # 3. Hand Landmarker
        base_options_hand = python.BaseOptions(model_asset_path=self.hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

        self.last_face_results = None
        self.last_hand_results = None

    def _ensure_models_exist(self):
        models = {
            self.face_model_path: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            self.hand_model_path: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        }
        for path, url in models.items():
            if not os.path.exists(path):
                print(f"Downloading {path}...")
                urllib.request.urlretrieve(url, path)
                print("Download complete.")

    def process(self, img, timestamp_ms, skip_inference=False):
        if skip_inference and self.last_face_results is not None:
            return self.last_face_results, self.last_hand_results

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Face detection
        self.last_face_results = self.face_landmarker.detect_for_video(mp_image, int(timestamp_ms))
        # Hand detection
        self.last_hand_results = self.hand_landmarker.detect_for_video(mp_image, int(timestamp_ms))
        
        return self.last_face_results, self.last_hand_results

    def draw_ultimate_art(self, canvas, face_results, hand_results, theme_color):
        h, w, _ = canvas.shape

        # 1. Face Mesh (Connected)
        if face_results and face_results.face_landmarks:
            for face_landmarks in face_results.face_landmarks:
                pts_map = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks)}
                
                # A. Structural Contours (Robust definition)
                contours = {
                    "face": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 
                             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
                    "left_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263],
                    "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
                    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185],
                    "left_ebrow": [276, 283, 282, 295, 285],
                    "right_ebrow": [46, 53, 52, 65, 55]
                }
                
                for name, indices in contours.items():
                    poly = np.array([pts_map[idx] for idx in indices if idx in pts_map], np.int32)
                    if len(poly) > 1:
                        cv2.polylines(canvas, [poly], name != "lips_outer", theme_color, 1 if name == "face" else 2)

                # B. Generative 'Mesh' Look (Cross-connections)
                # Vertical Ribs
                ribs_v = [
                    [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 13, 14, 15, 16, 17], # Midline
                    [297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400], # Left Cheek
                    [67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176], # Right Cheek
                    [109, 67, 103, 54, 21, 162, 127], # Far Right
                    [338, 297, 332, 284, 251, 389, 356] # Far Left
                ]
                # Horizontal Rings (Decimated for performance)
                ribs_h = [
                    [103, 67, 109, 10, 338, 297, 332], # Forehead
                    [21, 54, 103, 67, 10, 338, 297, 332, 284, 251], # Brow level
                    [234, 127, 162, 21, 5, 251, 389, 356, 454], # Mid face
                    [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288] # Jawline
                ]
                
                for rib in ribs_v + ribs_h:
                    poly = np.array([pts_map[idx] for idx in rib if idx in pts_map], np.int32)
                    if len(poly) > 1:
                        cv2.polylines(canvas, [poly], False, theme_color, 1)

                # C. Points for detail
                for i in range(0, len(face_landmarks), 10):
                    cx, cy = pts_map[i]
                    cv2.circle(canvas, (cx, cy), 1, theme_color, -1)

        # 2. Hands (Simplified)
        if hand_results and hand_results.hand_landmarks:
            for hand_landmarks in hand_results.hand_landmarks:
                # Connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
                    (0, 5), (5, 6), (6, 7), (8, 7),      # Index
                    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
                    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
                    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
                    (0, 17) # Palm base
                ]
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                
                for conn in connections:
                    cv2.line(canvas, points[conn[0]], points[conn[1]], theme_color, 2)
                
                # Finger joints as small highlights
                for p in [4, 8, 12, 16, 20]:
                    cv2.circle(canvas, points[p], 3, (255, 255, 255), -1)
        
        return canvas
