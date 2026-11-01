import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import wandb
from torch import nn

from vgg6.config import Config
from vgg6.data import training_loaders
from vgg6.evaluation import evaluate
from vgg6.model import VGG6


def seed_everything(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(request: str):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(request)


def make_optimizer(model, config: Config):
    shared = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer in ("sgd", "nesterov"):
        return torch.optim.SGD(
            model.parameters(),
            momentum=config.momentum,
            nesterov=config.optimizer == "nesterov",
            **shared,
        )
    factory = {
        "adam": torch.optim.Adam,
        "adagrad": torch.optim.Adagrad,
        "rmsprop": torch.optim.RMSprop,
        "nadam": torch.optim.NAdam,
    }[config.optimizer]
    return factory(model.parameters(), **shared)


def environment(device):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train(
    config: Config,
    output: Path,
    mode="disabled",
    project="vgg6-cifar10",
    entity=None,
    tracking_run=None,
):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config.to_dict())
    if tracking_run is not None:
        return _train(config, output, tracking_run)
    with wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        config=config.to_dict(),
        name=output.name,
        group=config.group,
        dir=str(output),
    ) as run:
        return _train(config, output, run)


def _train(config: Config, output: Path, run):
    seed_everything(config.seed)
    device = resolve_device(config.device)
    write_json(output / "environment.json", environment(device))
    run.define_metric("step")
    for metric in ("train/loss", "train/accuracy", "validation/loss", "validation/accuracy"):
        run.define_metric(metric, step_metric="step")
    run.define_metric("validation/accuracy", summary="max")
    model = VGG6(config.activation).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler == "cosine"
        else None
    )
    training, validation = training_loaders(config)
    criterion = nn.CrossEntropyLoss()
    best_accuracy, best_epoch, step = -1.0, 0, 0
    started = time.perf_counter()
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        loss_sum, correct, total = 0.0, 0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, targets in training:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, step {step}")
            loss.backward()
            optimizer.step()
            count = targets.numel()
            loss_sum += loss.item() * count
            correct += (logits.argmax(1) == targets).sum().item()
            total += count
            step += 1
        if not total:
            raise ValueError("Cannot train with an empty loader")
        metrics = evaluate(model, validation, device)
        row = {
            "epoch": epoch,
            "step": step,
            "train/loss": loss_sum / total,
            "train/accuracy": 100.0 * correct / total,
            "validation/loss": metrics["loss"],
            "validation/accuracy": metrics["accuracy"],
            "learning_rate": learning_rate,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        with (output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        run.log(row)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = metrics["accuracy"], epoch
            checkpoint = {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": config.to_dict(),
                "epoch": epoch,
                "step": step,
                "validation_accuracy": best_accuracy,
            }
            torch.save(checkpoint, output / "best.pt")
        if scheduler is not None:
            scheduler.step()
        print(
            f"{output.name} epoch={epoch}/{config.epochs} "
            f"train={row['train/accuracy']:.2f}% validation={metrics['accuracy']:.2f}%",
            flush=True,
        )
    from vgg6.plots import plot_run

    plots = plot_run(output)
    for name, path in plots.items():
        run.log({f"plots/{name}": wandb.Image(str(path))})
    summary = {
        "status": "completed",
        "config": config.to_dict(),
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "final_validation_accuracy": history[-1]["validation/accuracy"],
        "elapsed_seconds": time.perf_counter() - started,
        "steps": step,
        "wandb_url": run.url if run.settings.mode == "online" else None,
        "wandb_mode": run.settings.mode,
        "wandb_run_id": run.id,
        "synthetic": config.synthetic,
    }
    if not math.isfinite(best_accuracy):
        raise FloatingPointError("Nonfinite best accuracy")
    run.summary["best_validation_accuracy"] = best_accuracy
    run.summary["best_epoch"] = best_epoch
    run.summary["synthetic"] = config.synthetic
    write_json(output / "summary.json", summary)
    return summary
