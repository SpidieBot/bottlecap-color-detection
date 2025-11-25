import json
import os
from typing import Any, Dict, List

import cv2
import yaml
from ultralytics import YOLO


def run_inference(
    config_path: str, input_path: str, output_json: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run inference on a single image or all images in a directory using a trained YOLO model from config.

    This function loads the model from the specified path in config,
    performs detection on the provided input (file or dir), and returns results.
    It also saves annotated versions of the images for visualization,
    and prints detailed inference timings per image and averages.

    Args:
        config_path (str): Path to the YAML configuration file.
        input_path (str): Path to the input image or directory of images.
        output_json (bool): Whether to save detections as JSON (default: False).

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with image paths as keys and list of detections as values.
    """
    # Load configuration
    with open(config_path, "r") as f:
        params: Dict[str, Any] = yaml.safe_load(f)

    # Load trained model
    model_path: str = params.get("model_path", "runs/detect/train/weights/best.pt")
    model: YOLO = YOLO(model_path)

    # Prepare class names and output dir
    class_names = params.get("class_names", ["light_blue", "dark_blue", "others"])
    output_dir = params.get("output_dir", "inference_results")
    os.makedirs(output_dir, exist_ok=True)

    # Collect image paths (single file or all .jpg in dir)
    image_paths: List[str] = []
    if os.path.isdir(input_path):
        image_paths = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_paths:
            raise ValueError(f"No image files found in directory: {input_path}")
    elif os.path.isfile(input_path) and input_path.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        image_paths = [input_path]
    else:
        raise ValueError(
            f"Invalid input: {input_path} must be an image file or directory containing images."
        )

    # Run inference on each image and collect timings
    all_detections: Dict[str, List[Dict[str, Any]]] = {}
    inference_times: List[float] = []  # To calculate average

    for img_path in image_paths:
        results = model(
            img_path, conf=params.get("conf", 0.25), iou=params.get("iou", 0.45)
        )

        detections: List[Dict[str, Any]] = []
        img = cv2.imread(img_path)

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
        output_path = os.path.join(
            output_dir, f"annotated_{os.path.basename(img_path)}"
        )
        cv2.imwrite(output_path, img)
        print(f"Annotated image saved to: {output_path}")

        # Print per-image timings
        speed = results[0].speed
        print(f"Image: {os.path.basename(img_path)}")
        print(f"  Preprocess: {speed['preprocess']:.1f}ms")
        print(f"  Inference: {speed['inference']:.1f}ms")
        print(f"  Postprocess: {speed['postprocess']:.1f}ms")
        inference_times.append(speed["inference"])

        all_detections[img_path] = detections

    # Print average inference time if batch
    if len(image_paths) > 1:
        avg_inference = sum(inference_times) / len(inference_times)
        print(f"\nBatch Summary:")
        print(f"  Total Images: {len(image_paths)}")
        print(f"  Average Inference: {avg_inference:.1f}ms")

    # Optional JSON export
    if output_json:
        json_path = os.path.join(output_dir, "detections.json")
        with open(json_path, "w") as jf:
            json.dump(all_detections, jf, indent=4)
        print(f"Detections saved to JSON: {json_path}")

    return all_detections


# For CLI integration (e.g., in cli.py: infer(config, input_path, output_json))
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise ValueError("Usage: python infer.py <config_path> <input_path> [--json]")
    output_json = len(sys.argv) > 3 and sys.argv[3] == "--json"
    results = run_inference(sys.argv[1], sys.argv[2], output_json)
    print("Detections:", results)
