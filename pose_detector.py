import cv2  # type: ignore
import mediapipe as mp  # type: ignore
from mediapipe.tasks import python  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore
import math
import os
import urllib.request
import numpy as np  # type: ignore

class PoseDetector:
    def __init__(self, model_asset_path='pose_landmarker_lite.task', num_poses=1,
                 min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, 
                 min_tracking_confidence=0.5):
        
        self.model_path = model_asset_path
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.lm_list = []
        self.results = None
        
    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading model to {self.model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def find_pose(self, img, timestamp_ms, draw=True):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Process the image for the given timestamp
        self.results = self.detector.detect_for_video(mp_image, int(timestamp_ms))
        
        # Draw the original skeleton if requested
        if draw and self.results.pose_landmarks:
            for pose_landmarks in self.results.pose_landmarks:
                # We need to manually draw the points and connections since the tasks api 
                # drawing utils differs a bit and expects normalized coords
                self.draw_landmarks(img, pose_landmarks)
        return img
    
    def draw_landmarks(self, img, pose_landmarks):
        h, w, c = img.shape
        # Just simple connections for standard 33 landmarks
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
            (24, 26), (26, 28)
        ]
        
        points = {}
        for idx, lm in enumerate(pose_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[idx] = (cx, cy)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            
        for connection in connections:
            if connection[0] in points and connection[1] in points:
                cv2.line(img, points[connection[0]], points[connection[1]], (0, 255, 0), 2)
    
    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results and self.results.pose_landmarks:
            # We assume single person for now
            pose_landmarks = self.results.pose_landmarks[0]
            h, w, c = img.shape
            for id, lm in enumerate(pose_landmarks):
                # normalized x and y to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        if len(self.lm_list) == 0:
            return 0
            
        try:
            x1, y1 = self.lm_list[p1][1:]
            x2, y2 = self.lm_list[p2][1:]
            x3, y3 = self.lm_list[p3][1:]
        except IndexError:
            return 0

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        
        if angle < 0:
            angle += 360
            
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
        
def main():
    import time
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        
        timestamp_ms = int(time.time() * 1000)
        img = detector.find_pose(img, timestamp_ms)
        lm_list = detector.find_position(img, draw=False)
        if len(lm_list) != 0:
            angle = detector.find_angle(img, 12, 14, 16)
            
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
