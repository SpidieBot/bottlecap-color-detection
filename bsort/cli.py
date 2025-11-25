import typer
import yaml

from bsort.infer import run_inference
from bsort.train import train_model

app = typer.Typer()


@app.command()
def train(config: str):
    """Train the model using config.

    Args:
        config (str): Path to config.yaml.
    """
    with open(config, "r") as f:
        params = yaml.safe_load(f)
    train_model(params)


@app.command()
def infer(config: str, image: str):
    """Run inference on an image using config.

    Args:
        config (str): Path to settings.yaml.
        image (str): Path to sample.jpg.
    """
    results = run_inference(config, image)

    # Optionally format/print results


if __name__ == "__main__":
    app()
