from typing import Any, Dict

import wandb
import yaml
from ultralytics import YOLO

# List of valid YOLO training arguments (from Ultralytics docs)
VALID_TRAIN_ARGS = [
    "data",
    "epochs",
    "batch",
    "imgsz",
    "lr0",
    "optimizer",
    "time",
    "patience",
    "save",
    "save_period",
    "cache",
    "device",
    "workers",
    "pretrained",
    "seed",
    "deterministic",
    "single_cls",
    "classes",
    "rect",
    "multi_scale",
    "cos_lr",
    "close_mosaic",
    "resume",
    "amp",
    "fraction",
    "profile",
    "freeze",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "dfl",
    "pose",
    "kobj",
    "nbs",
    "overlap_mask",
    "mask_ratio",
    "dropout",
    "val",
    "plots",
    "compile",
    # Augmentation params
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
]


def train_model(config_path: str) -> Dict[str, Any]:
    """
    Train a YOLO model using parameters from a YAML config file and track with WandB.

    This function loads the configuration, filters to valid YOLO train args,
    initializes WandB, trains the model, and returns results. Ensure WandB project
    is public on wandb.ai.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Training results (e.g., best mAP, model path).
    """
    # Load configuration
    with open(config_path, "r") as f:
        params: Dict[str, Any] = yaml.safe_load(f)

    # Initialize WandB
    wandb.init(
        project=params.get("project", "bottle_caps"),
        config=params,
        name=params.get("name", "exp1"),
        mode="online",
    )

    # Load pretrained model
    model: YOLO = YOLO(
        params.get("model", "yolo11n.pt")
    )  # Default to yolo11n if not specified

    # Filter params to only valid train args (exclude infer-specific like model_path, class_names)
    train_params = {k: v for k, v in params.items() if k in VALID_TRAIN_ARGS}

    # Train with filtered parameters
    results = model.train(**train_params)

    # Export for edge devices
    export_path: str = model.export(format="ncnn")
    print(f"Model exported to: {export_path}")

    # Log export and finish WandB
    wandb.log({"export_path": export_path})
    wandb.finish()

    return results.__dict__


# CLI entry
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise ValueError("Usage: python train.py <config_path>")
    train_model(sys.argv[1])
