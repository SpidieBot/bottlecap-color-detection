import os
import random
import shutil
from collections import defaultdict
from typing import List, Tuple


def get_classes_from_label(label_path: str) -> set:
    """
    Extract unique classes from a YOLO label file.

    Args:
        label_path (str): Path to the label file.

    Returns:
        set: Set of unique class IDs in the file.
    """
    classes = set()
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls = int(line.strip().split()[0])
                classes.add(cls)
    return classes


def stratified_split_dataset(
    image_dir: str, label_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.0
):
    """
    Split dataset into train/val/test with stratification to ensure class diversity, especially for small datasets.

    Args:
        image_dir (str): Directory of images.
        label_dir (str): Directory of labels.
        train_ratio (float): Ratio for training set (default 0.7).
        val_ratio (float): Ratio for validation set (default 0.15; remainder for test).
    """
    files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # Group files by their unique class sets for stratification
    class_groups = defaultdict(list)
    for f in files:
        label_f = f.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_f)
        classes = frozenset(get_classes_from_label(label_path))
        class_groups[classes].append(f)

    # Shuffle within groups and allocate proportionally, prioritizing train diversity
    train_files = []
    val_files = []
    test_files = []
    for group_files in class_groups.values():
        random.shuffle(group_files)
        group_size = len(group_files)
        train_end = max(
            0, int(group_size * train_ratio)
        )  # Ensure some in train if possible
        val_end = train_end + max(0, int(group_size * val_ratio))

        train_files.extend(group_files[:train_end])
        val_files.extend(group_files[train_end:val_end])
        test_files.extend(group_files[val_end:])

    # Create directories and copy files
    for split, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        os.makedirs(f"dataset/{split}/images", exist_ok=True)
        os.makedirs(f"dataset/{split}/labels", exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(image_dir, f), f"dataset/{split}/images/{f}")
            label_f = f.replace(".jpg", ".txt")
            shutil.copy(
                os.path.join(label_dir, label_f), f"dataset/{split}/labels/{label_f}"
            )

    print(
        f"Split complete: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}"
    )


# Usage example (adjust paths as needed)
stratified_split_dataset("dataset/raw/images", "dataset/raw/adjusted_labels")
