import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import wandb
import yaml

from vgg6.config import Config, controlled_configs
from vgg6.data import evaluation_loader
from vgg6.evaluation import evaluate
from vgg6.model import VGG6
from vgg6.training import resolve_device, seed_everything, train, write_json


def run_suite(base: Config, output: Path, mode: str, project: str, entity=None):
    output.mkdir(parents=True, exist_ok=False)
    configs = controlled_configs(base)
    write_json(output / "manifest.json", [config.to_dict() for config in configs])
    for index, config in enumerate(configs):
        train(config, output / f"{index:02d}-{config.group}", mode, project, entity)


def create_sweeps(base: Config, output: Path, project: str, entity=None):
    """Register one grid per controlled axis and an optional joint random search."""
    output.mkdir(parents=True, exist_ok=False)
    axes = {
        "activation": ["relu", "sigmoid", "tanh", "silu", "gelu"],
        "optimizer": ["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"],
        "batch_size": [64, 128, 256],
        "epochs": [25, 50, 100],
        "learning_rate": [0.0001, 0.001, 0.01],
    }
    ids = {}
    for axis, values in [*axes.items(), ("joint", None)]:
        fixed = replace(base, group=axis).to_dict()
        parameters = {key: {"value": value} for key, value in fixed.items()}
        if axis == "joint":
            for key, choices in axes.items():
                parameters[key] = {"values": choices}
        else:
            parameters[axis] = {"values": values}
        specification = {
            "name": f"vgg6-{axis}",
            "method": "random" if axis == "joint" else "grid",
            "metric": {"name": "best_validation_accuracy", "goal": "maximize"},
            "parameters": parameters,
        }
        (output / f"{axis}.yaml").write_text(yaml.safe_dump(specification, sort_keys=False))
        ids[axis] = wandb.sweep(specification, project=project, entity=entity)
        write_json(output / "ids.json", ids)
    return ids


def sweep_agent(sweep_id: str, output: Path, count: int, project: str, entity=None):
    output.mkdir(parents=True, exist_ok=True)

    def execute():
        with wandb.init(project=project, entity=entity) as run:
            config = Config(**dict(run.config))
            train(config, output / run.id, tracking_run=run)

    wandb.agent(sweep_id, function=execute, count=count, project=project, entity=entity)


def completed_runs(root: Path):
    results = []
    for summary_path in sorted(root.rglob("summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "completed":
            results.append((summary_path.parent, summary))
    if not results:
        raise ValueError(f"No completed runs in {root}")
    datasets = {summary["synthetic"] for _, summary in results}
    protocols = {
        (
            summary["config"]["split_seed"],
            summary["config"]["validation_size"],
            summary["config"]["data_dir"],
        )
        for _, summary in results
    }
    if len(datasets) != 1 or len(protocols) != 1:
        raise ValueError("Cannot compare runs with different datasets or validation splits")
    return results


def sync_runs(root: Path, project: str, entity: str):
    """Upload offline logs, then verify remote measurements before recording their URLs."""
    if not wandb.api.api_key:
        raise RuntimeError("Authenticate first with: uv run --locked wandb login")
    api = wandb.Api()
    synced = []
    for directory, summary in completed_runs(root):
        if summary["wandb_mode"] in ("online", "synced"):
            continue
        if summary["wandb_mode"] != "offline":
            raise ValueError(f"{directory} has no offline W&B logs to sync")
        run_id = summary["wandb_run_id"]
        log_dirs = list((directory / "wandb").glob(f"offline-run-*-{run_id}"))
        if len(log_dirs) != 1:
            raise ValueError(f"Expected exactly one offline log directory for {directory}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wandb",
                "sync",
                "--entity",
                entity,
                "--project",
                project,
                str(log_dirs[0]),
            ],
            check=True,
        )
        remote = api.run(f"{entity}/{project}/{run_id}")
        if remote.summary.get("best_validation_accuracy") != summary["best_validation_accuracy"]:
            raise ValueError(f"Synced accuracy does not match local measurements: {directory}")
        if any(remote.config.get(key) != value for key, value in summary["config"].items()):
            raise ValueError(
                f"Synced configuration does not match local configuration: {directory}"
            )
        summary["wandb_url"] = remote.url
        summary["wandb_mode"] = "synced"
        write_json(directory / "summary.json", summary)
        synced.append(remote.url)
    return synced


def select_best(root: Path):
    return max(
        completed_runs(root),
        key=lambda item: (item[1]["best_validation_accuracy"], -item[1]["best_epoch"]),
    )


def evaluate_checkpoint(checkpoint_path: Path, split: str, device="auto", data_dir=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    config = replace(config, device=device, **({"data_dir": data_dir} if data_dir else {}))
    seed_everything(config.seed)
    resolved = resolve_device(config.device)
    model = VGG6(config.activation).to(resolved)
    model.load_state_dict(checkpoint["model"], strict=True)
    return evaluate(model, evaluation_loader(config, split), resolved)


def finalize(root: Path, output: Path, device: str, data_dir=None):
    directory, summary = select_best(root)
    checkpoint = directory / "best.pt"
    validation = evaluate_checkpoint(checkpoint, "validation", device, data_dir)
    if abs(validation["accuracy"] - summary["best_validation_accuracy"]) > 1e-8:
        raise ValueError("Checkpoint validation accuracy does not match its summary")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint, output / "best.pt")
    shutil.copy2(directory / "config.json", output / "config.json")
    for name in ("history.csv", "environment.json", "summary.json"):
        shutil.copy2(directory / name, output / name)
    test = evaluate_checkpoint(output / "best.pt", "test", device, data_dir)
    result = {
        "selected_run": str(directory.resolve()),
        "config": summary["config"],
        "best_epoch": summary["best_epoch"],
        "best_validation_accuracy": validation["accuracy"],
        "test_accuracy": test["accuracy"],
        "test_loss": test["loss"],
        "test_samples": test["samples"],
        "synthetic": summary["synthetic"],
    }
    write_json(output / "final.json", result)
    return result
