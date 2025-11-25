from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Bottle Cap Color Sorter - bsort CLI")


@app.command()
def train(config: Path = "configs/settings.yaml"):
    """Train YOLO model using config file."""
    from bsort.train import train_model

    train_model(config)


@app.command()
def infer(
    config: Path = "configs/settings.yaml",
    image: Optional[Path] = None,
    dir: Optional[Path] = None,
    json: bool = False,
):
    """Run inference on image or directory."""
    from bsort.infer import run_inference

    if image:
        run_inference(config, image.parent, json)
    elif dir:
        run_inference(config, dir, json)
    else:
        typer.echo("Error: Provide --image or --dir")
        raise typer.Exit(1)


@app.command()
def export(weights: Path, format: str = "ncnn"):
    """Export trained model to edge format."""
    from ultralytics import YOLO

    model = YOLO(weights)
    model.export(format=format, imgsz=320)
    typer.echo(f"Exported to {format.upper()}")


if __name__ == "__main__":
    app()
