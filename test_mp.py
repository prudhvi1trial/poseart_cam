import urllib.request
print("Downloading model...")
urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task", "pose_landmarker_lite.task")
print("Downloaded model.")

import mediapipe as mp  # type: ignore
from mediapipe.tasks import python  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore

print("Initializing PoseLandmarker...")
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE)
detector = vision.PoseLandmarker.create_from_options(options)
print("Success!")
