import os

import cv2
import numpy as np


def get_dominant_color(crop: np.ndarray) -> int:
    """
    Determine color class based on mean HSV.

    Args:
        crop (np.ndarray): Cropped image region.

    Returns:
        int: Class (0: light blue, 1: dark blue, 2: others).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])
    if 100 <= mean_h <= 130 and mean_s > 50:
        if mean_v > 128:
            return 0  # light blue
        else:
            return 1  # dark blue
    return 2  # others


def adjust_labels(image_dir: str, label_dir: str, output_label_dir: str) -> None:
    """
    Adjust YOLO labels by color.

    Args:
        image_dir (str): Path to images.
        label_dir (str): Path to original labels.
        output_label_dir (str): Path for new labels.
    """
    os.makedirs(output_label_dir, exist_ok=True)
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            image_file = label_file.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)
            output_path = os.path.join(output_label_dir, label_file)

            if not os.path.exists(image_path):
                continue

            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            with open(label_path, "r") as f, open(output_path, "w") as out:
                for line in f:
                    parts = line.strip().split()
                    cls, x, y, bw, bh = parts[0], *map(float, parts[1:])
                    xmin = int((x - bw / 2) * w)
                    xmax = int((x + bw / 2) * w)
                    ymin = int((y - bh / 2) * h)
                    ymax = int((y + bh / 2) * h)
                    crop = img[ymin:ymax, xmin:xmax]

                    new_cls = get_dominant_color(crop)
                    out.write(f"{new_cls} {x} {y} {bw} {bh}\n")


adjust_labels("dataset/raw/images", "dataset/raw/labels", "dataset/raw/adjusted_labels")
