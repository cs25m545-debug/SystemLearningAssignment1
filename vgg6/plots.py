import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES = {
    "training_loss": ("train/loss", "Training cross-entropy"),
    "training_accuracy": ("train/accuracy", "Training top-1 accuracy (%)"),
    "validation_loss": ("validation/loss", "Validation cross-entropy"),
    "validation_accuracy": ("validation/accuracy", "Validation top-1 accuracy (%)"),
}


def read_history(directory: Path):
    with (directory / "history.csv").open(newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def plot_run(directory: Path):
    history = read_history(directory)
    if not history:
        raise ValueError("History is empty")
    output = directory / "plots"
    output.mkdir(exist_ok=True)
    paths = {}
    for name, (metric, label) in CURVES.items():
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot([row["epoch"] for row in history], [row[metric] for row in history])
        axis.set(xlabel="Epoch", ylabel=label, title=label)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        paths[name] = output / f"{name}.png"
        figure.savefig(paths[name], dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(
        [row["step"] for row in history], [row["validation/accuracy"] for row in history], s=20
    )
    axis.set(
        xlabel="Optimizer update step",
        ylabel="Validation top-1 accuracy (%)",
        title="Validation accuracy vs. step",
    )
    axis.grid(alpha=0.3)
    figure.tight_layout()
    paths["validation_scatter"] = output / "validation_scatter.png"
    figure.savefig(paths["validation_scatter"], dpi=160)
    plt.close(figure)
    return paths
