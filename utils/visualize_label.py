import os

import cv2


def visualize(image_paths, label_paths, output_dir):
    """
    Visualize YOLO bounding boxes on images and save the results.

    This function can accept either:
    - Lists of image paths and corresponding label paths (must be same length).
    - Directory paths as strings for images and labels (automatically matches files).

    Args:
        image_paths (list[str] or str): List of image file paths or image directory.
        label_paths (list[str] or str): List of label file paths or label directory.
        output_dir (str): Directory to save visualized images.
    """
    if isinstance(image_paths, str) and isinstance(label_paths, str):
        # Assume directories; collect matching files
        image_dir = image_paths
        label_dir = label_paths
        image_paths = []
        label_paths = []
        for img_file in sorted(os.listdir(image_dir)):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                base_name = os.path.splitext(img_file)[0]
                lab_file = f"{base_name}.txt"
                lab_path = os.path.join(label_dir, lab_file)
                if os.path.exists(lab_path):
                    image_paths.append(os.path.join(image_dir, img_file))
                    label_paths.append(lab_path)

    if len(image_paths) != len(label_paths):
        raise ValueError("Number of image paths must match number of label paths.")

    os.makedirs(output_dir, exist_ok=True)

    for img_path, lab_path in zip(image_paths, label_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        h, w = img.shape[:2]

        try:
            with open(lab_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = float(parts[0])
                    x, y, bw, bh = map(float, parts[1:])
                    xmin = int((x - bw / 2) * w)
                    xmax = int((x + bw / 2) * w)
                    ymin = int((y - bh / 2) * h)
                    ymax = int((y + bh / 2) * h)
                    color = (
                        (0, 255, 0)
                        if cls == 0
                        else (0, 0, 255) if cls == 1 else (255, 0, 0)
                    )
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(
                        img,
                        f"Class {int(cls)}",
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
        except Exception as e:
            print(f"Error processing label {lab_path}: {e}")
            continue

        output_path = os.path.join(output_dir, f"vis_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, img)
        print(f"Saved visualized image to {output_path}")


# Example usage with directories (adjust paths as needed)
visualize("dataset/raw/images", "dataset/raw/adjusted_labels", "dataset/visualized")

# Alternative usage with lists
# image_list = ["path/to/image1.jpg", "path/to/image2.jpg"]
# label_list = ["path/to/label1.txt", "path/to/label2.txt"]
# visualize(image_list, label_list, "dataset/visualized")
