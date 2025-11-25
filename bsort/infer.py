import os
from typing import Any, Dict, List

import cv2
import yaml
from ultralytics import YOLO


def run_inference(config_path: str, image_path: str) -> List[Dict[str, Any]]:
    """
    Run inference on a single image using a trained YOLO model from config.

    This function loads the model from the specified path in config,
    performs detection on the provided image, and returns results.
    It also saves an annotated version of the image for visualization.

    Args:
        config_path (str): Path to the YAML configuration file.
        image_path (str): Path to the input image for inference.

    Returns:
        List[Dict[str, Any]]: List of detection results, each with class, confidence, and bbox.
    """
    # Load configuration
    with open(config_path, "r") as f:
        params: Dict[str, Any] = yaml.safe_load(f)

    # Load trained model (assume 'model_path' in config points to best.pt or exported model)
    model_path: str = params.get("model_path", "runs/detect/train/weights/best.pt")
    model: YOLO = YOLO(model_path)

    # Run inference
    results = model(
        image_path, conf=params.get("conf", 0.25), iou=params.get("iou", 0.45)
    )

    # Process and visualize results
    detections: List[Dict[str, Any]] = []
    img = cv2.imread(image_path)
    class_names = params.get("class_names", ["light_blue", "dark_blue", "others"])

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detection = {
                "class": class_names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
            detections.append(detection)

            # Draw bbox and label
            color = (
                (0, 255, 0)
                if cls_id == 0
                else (0, 0, 255) if cls_id == 1 else (255, 0, 0)
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{detection['class']} {conf:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    # Save annotated image
    output_dir = params.get("output_dir", "inference_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to: {output_path}")

    return detections


# For CLI integration (e.g., in cli.py: infer(config, image))
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise ValueError("Usage: python infer.py <config_path> <image_path>")
    results = run_inference(sys.argv[1], sys.argv[2])
    print("Detections:", results)
