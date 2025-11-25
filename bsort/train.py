import os
from typing import Any, Dict

import wandb
import yaml
from ultralytics import YOLO


def train_model(config_path: str) -> Dict[str, Any]:
    """
    Train a YOLO model using parameters from a YAML config file and track with WandB.

    This function loads the configuration, initializes WandB for public tracking,
    trains the model, and returns training results. Ensure WandB project is set to
    public on wandb.ai for accessibility.

    Args:
        config_path (str): Path to the YAML configuration file containing training params.

    Returns:
        Dict[str, Any]: Dictionary of training results (e.g., best mAP, model path).
    """
    # Load configuration
    with open(config_path, "r") as f:
        params: Dict[str, Any] = yaml.safe_load(f)

    # Initialize WandB (logs metrics, checkpoints, and visualizations automatically)
    wandb.init(
        project=params.get("project", "bottle_caps"),
        config=params,
        name=params.get("name", "exp1"),
        mode="online",  # Ensures syncing; set to 'offline' if needed
    )

    # Load pretrained model
    model: YOLO = YOLO(params["model"])

    # Train with extracted parameters (Ultralytics handles the rest)
    results = model.train(
        **{k: v for k, v in params.items() if k not in ["model", "project", "name"]}
    )

    # Export best model for inference (e.g., for Raspberry Pi)
    export_path: str = model.export(format="ncnn")  # Or 'onnx'/'tflite'
    print(f"Model exported to: {export_path}")

    # Finish WandB run and log export path
    wandb.log({"export_path": export_path})
    wandb.finish()

    return results.__dict__  # Return results for further analysis


# CLI entry (integrate with bsort cli.py)
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise ValueError("Usage: python train.py <config_path>")
    train_model(sys.argv[1])
