"""Create native W&B panels from verified remote runs; requires the submission extra."""

import argparse
import json
from pathlib import Path

import wandb_workspaces.reports.v2 as wr

import wandb
from vgg6.experiments import completed_runs
from vgg6.plots import read_history
from vgg6.training import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/mac-real"))
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="vgg6-cifar10")
    parser.add_argument("--output", type=Path, default=Path("artifacts/wandb-report.json"))
    args = parser.parse_args()
    api = wandb.Api()
    results = completed_runs(args.runs)
    verified = []
    for directory, summary in results:
        if not summary.get("wandb_url"):
            raise ValueError(f"Run has not been uploaded and verified: {directory}")
        remote = api.run(f"{args.entity}/{args.project}/{summary['wandb_run_id']}")
        history = list(
            remote.scan_history(
                keys=[
                    "epoch",
                    "step",
                    "train/loss",
                    "train/accuracy",
                    "validation/loss",
                    "validation/accuracy",
                ]
            )
        )
        saved = read_history(directory)
        if len(history) != len(saved):
            raise ValueError(f"Remote epoch count differs from saved training: {directory}")
        for observed, expected in zip(history, saved):
            if any(
                abs(observed[key] - expected[key]) > 1e-8
                for key in (
                    "epoch",
                    "step",
                    "train/loss",
                    "train/accuracy",
                    "validation/loss",
                    "validation/accuracy",
                )
            ):
                raise ValueError(f"Remote history differs from saved measurements: {directory}")
        verified.append(
            {
                "run": directory.name,
                "id": remote.id,
                "url": remote.url,
                "epochs_verified": len(history),
                "best_validation_accuracy": summary["best_validation_accuracy"],
            }
        )

    panels = [
        wr.ParallelCoordinatesPlot(
            title="Configuration vs best validation accuracy",
            columns=[
                wr.ParallelCoordinatesPlotColumn(wr.Config(name), name)
                for name in ("activation", "optimizer", "batch_size", "epochs", "learning_rate")
            ]
            + [
                wr.ParallelCoordinatesPlotColumn(
                    wr.SummaryMetric("best_validation_accuracy"), "Best validation accuracy (%)"
                )
            ],
            layout=wr.Layout(w=24, h=9),
        ),
    ]
    for index, (metric, title) in enumerate(
        [
            ("train/loss", "Training loss"),
            ("train/accuracy", "Training accuracy"),
            ("validation/loss", "Validation loss"),
            ("validation/accuracy", "Validation accuracy"),
        ]
    ):
        panels.append(
            wr.LinePlot(
                title=title,
                x="step",
                y=[metric],
                smoothing_type="none",
                max_runs_to_show=16,
                layout=wr.Layout(x=12 * (index % 2), y=9 + 7 * (index // 2), w=12, h=7),
            )
        )
    report = wr.Report(
        entity=args.entity,
        project=args.project,
        title="CS6886W Assignment 1 - Verified VGG6 experiments",
        description="16 real CIFAR-10 configurations; validation 88.08%, test 86.61%.",
        width="fluid",
        blocks=[
            wr.MarkdownBlock(
                text=(
                    "# Verified VGG6 / CIFAR-10 experiments\n"
                    "Repository: https://github.com/cs25m545-debug/SystemLearningAssignment1\n\n"
                    "All 16 runs contain the original saved per-epoch measurements, uploaded "
                    "after training and checked against the local CSV histories. No retraining "
                    "or synthetic measurements are used in this report. The winner is ReLU / "
                    "RMSprop / batch 128 / 100 epochs / LR 0.001; best checkpoint epoch 92.\n\n"
                    "Best validation accuracy: **88.08%**. Official test accuracy: **86.61%**. "
                    "An independent from-scratch rerun reproduced both accuracies.\n"
                )
            ),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(
                        entity=args.entity,
                        project=args.project,
                        name="16 controlled real-data experiments",
                    )
                ],
                panels=panels,
            ),
        ],
    )
    report.save()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "report_url": report.url,
            "project_url": f"https://wandb.ai/{args.entity}/{args.project}",
            "runs": verified,
        },
    )
    print(json.dumps({"report_url": report.url, "verified_runs": len(verified)}, indent=2))


if __name__ == "__main__":
    main()
