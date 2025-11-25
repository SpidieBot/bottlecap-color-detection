from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

# Export once
model.export(format="ncnn", imgsz=320)  # creates ncnn_model/
model.export(format="openvino", imgsz=320)  # creates openvino_model/
