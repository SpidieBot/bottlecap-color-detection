import os
import random
import shutil
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np


def get_classes_from_label(label_path: str) -> set:
    """Extract unique classes from a YOLO label file."""
    classes = set()
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls = int(line.strip().split()[0])
                classes.add(cls)
    return classes


def pre_crop_images(
    image_dir: str, label_dir: str, output_image_dir: str, output_label_dir: str
):
    """Crop each bounding box to a new image and create corresponding label (centered at 0.5 0.5 1 1)."""
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    crop_count = 0

    for img_file in os.listdir(image_dir):
        if not img_file.endswith(".jpg"):
            continue
        img_path = os.path.join(image_dir, img_file)
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = parts[0]
                x, y, bw, bh = map(float, parts[1:])
                xmin = max(0, int((x - bw / 2) * w))
                xmax = min(w, int((x + bw / 2) * w))
                ymin = max(0, int((y - bh / 2) * h))
                ymax = min(h, int((y + bh / 2) * h))

                # Crop and save new image
                crop_img = img[ymin:ymax, xmin:xmax]
                if crop_img.size == 0:
                    continue
                crop_filename = f"crop_{crop_count}.jpg"
                crop_path = os.path.join(output_image_dir, crop_filename)
                cv2.imwrite(crop_path, crop_img)

                # Create new label: class 0.5 0.5 1 1
                crop_label_path = os.path.join(
                    output_label_dir, crop_filename.replace(".jpg", ".txt")
                )
                with open(crop_label_path, "w") as out:
                    out.write(f"{cls} 0.5 0.5 1 1\n")

                crop_count += 1

    print(f"Created {crop_count} cropped images.")


def stratified_split_dataset(image_dir: str, label_dir: str, train_ratio: float = 0.8):
    """Split dataset with stratification to ensure all classes in train."""
    files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # Group files by their class sets for stratification
    class_groups = defaultdict(list)
    for f in files:
        label_f = f.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_f)
        classes = frozenset(get_classes_from_label(label_path))
        class_groups[classes].append(f)

    # Shuffle within groups and allocate to prioritize diversity in train
    train_files = []
    test_files = []
    for group_files in class_groups.values():
        random.shuffle(group_files)
        train_end = max(
            1, int(len(group_files) * train_ratio)
        )  # Ensure at least one per group in train if possible
        train_files.extend(group_files[:train_end])
        test_files.extend(group_files[train_end:])

    # Create directories and copy
    for split, split_files in [("train", train_files), ("test", test_files)]:
        os.makedirs(f"dataset/{split}/images", exist_ok=True)
        os.makedirs(f"dataset/{split}/labels", exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(image_dir, f), f"dataset/{split}/images/{f}")
            label_f = f.replace(".jpg", ".txt")
            shutil.copy(
                os.path.join(label_dir, label_f), f"dataset/{split}/labels/{label_f}"
            )


# Usage: First pre-crop, then split the cropped dataset
crop_image_dir = "dataset/cropped/images"
crop_label_dir = "dataset/cropped/labels"
pre_crop_images(
    "dataset/raw/images", "dataset/raw/adjusted_labels", crop_image_dir, crop_label_dir
)
stratified_split_dataset(crop_image_dir, crop_label_dir, train_ratio=0.8)
