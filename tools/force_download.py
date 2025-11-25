import torch
from ultralytics import YOLO

# Force download yolo10n.pt (smallest YOLOv10 variant)
model = YOLO("yolov10n.pt")

# Verify it's loaded
print(f"Model loaded: {model.model}")
