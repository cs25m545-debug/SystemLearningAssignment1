import argparse
import json
from dataclasses import replace
from pathlib import Path

from vgg6.config import ACTIVATIONS, OPTIMIZERS, Config
from vgg6.experiments import (
    create_sweeps,
    evaluate_checkpoint,
    finalize,
    run_suite,
    sweep_agent,
    sync_runs,
)
from vgg6.report import build_report
from vgg6.training import train


def configuration_arguments(parser):
    parser.add_argument("--config", type=Path)
    fields = {
        "activation": (str, ACTIVATIONS),
        "optimizer": (str, OPTIMIZERS),
        "batch-size": (int, None),
        "epochs": (int, None),
        "learning-rate": (float, None),
        "weight-decay": (float, None),
        "momentum": (float, None),
        "seed": (int, None),
        "split-seed": (int, None),
        "validation-size": (int, None),
        "workers": (int, None),
        "device": (str, ("auto", "cpu", "cuda", "mps")),
        "data-dir": (str, None),
        "scheduler": (str, ("cosine", "none")),
        "cutout": (int, None),
    }
    for name, (kind, choices) in fields.items():
        parser.add_argument(f"--{name}", type=kind, choices=choices)
    parser.add_argument("--synthetic", action="store_true", default=None)


def tracking_arguments(parser):
    parser.add_argument("--project", default="vgg6-cifar10")
    parser.add_argument("--entity")


def get_config(args):
    config = Config.load(args.config) if args.config else Config()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in config.to_dict() and value is not None
    }
    return replace(config, **overrides)


def main():
    parser = argparse.ArgumentParser(description="CS6886W VGG6/CIFAR-10 experiment system")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "suite", "sweeps"):
        command = commands.add_parser(name)
        configuration_arguments(command)
        tracking_arguments(command)
        command.add_argument("--output", type=Path, required=True)
        if name != "sweeps":
            command.add_argument(
                "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
            )
    agent = commands.add_parser("agent")
    tracking_arguments(agent)
    agent.add_argument("--sweep-id", required=True)
    agent.add_argument("--output", type=Path, required=True)
    agent.add_argument("--count", type=int, default=25)
    sync = commands.add_parser("sync")
    sync.add_argument("--runs", type=Path, required=True)
    sync.add_argument("--project", default="vgg6-cifar10")
    sync.add_argument("--entity", required=True)
    sync.add_argument(
        "--replay",
        action="store_true",
        help="Upload saved measured CSV history through online logging",
    )
    for name in ("evaluate", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
        command.add_argument("--data-dir")
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--split", choices=("validation", "test"), default="validation")
        else:
            command.add_argument("--runs", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--runs", type=Path, required=True)
    report.add_argument("--final", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--github-url", default="")
    report.add_argument("--wandb-parallel", type=Path)
    report.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        result = train(get_config(args), args.output, args.wandb_mode, args.project, args.entity)
    elif args.command == "suite":
        result = run_suite(
            get_config(args), args.output, args.wandb_mode, args.project, args.entity
        )
    elif args.command == "sweeps":
        if get_config(args).synthetic:
            parser.error("Synthetic sweeps are not supported; use train for smoke tests")
        result = create_sweeps(get_config(args), args.output, args.project, args.entity)
    elif args.command == "agent":
        if args.count <= 0:
            parser.error("--count must be positive")
        result = sweep_agent(args.sweep_id, args.output, args.count, args.project, args.entity)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.checkpoint, args.split, args.device, args.data_dir)
    elif args.command == "finalize":
        result = finalize(args.runs, args.output, args.device, args.data_dir)
    elif args.command == "sync":
        result = sync_runs(args.runs, args.project, args.entity, args.replay)
    else:
        result = str(
            build_report(
                args.runs, args.final, args.output, args.github_url, args.wandb_parallel, args.draft
            )
        )
    if result is not None:
        print(json.dumps(result, indent=2))
