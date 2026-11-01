import json
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from vgg6.config import ACTIVATIONS, OPTIMIZERS
from vgg6.experiments import completed_runs, select_best
from vgg6.plots import plot_run, read_history

MECHANISMS = {
    "activation": (
        "ReLU has a nonsaturating positive branch but can have inactive units. Sigmoid saturates "
        "and is not zero-centered; tanh is zero-centered but also saturates. SiLU and GELU "
        "are smooth gates that preserve small negative outputs. Batch normalization and the "
        "shared template initialization are held fixed, so this comparison does not tune "
        "initialization independently for each activation."
    ),
    "optimizer": (
        "SGD uses momentum 0.9; Nesterov-SGD uses its look-ahead correction. Adam combines "
        "first and second moment estimates; Nadam adds Nesterov-style momentum to Adam. "
        "Adagrad accumulates squared gradients, which can shrink its effective step size "
        "over long runs. RMSprop uses an exponential squared-gradient average rather than "
        "an ever-growing accumulator. Defaults are PyTorch defaults except learning rate, "
        "weight decay, and SGD momentum. A shared learning rate isolates the optimizer "
        "choice but is not each optimizer's individually tuned optimum."
    ),
    "batch_size": (
        "Smaller batches give more optimizer updates per epoch and noisier gradients. Larger "
        "batches can improve hardware throughput but give fewer updates per epoch. Compare "
        "both elapsed time and step counts, not epochs alone; no linear learning-rate "
        "scaling is applied in this controlled comparison."
    ),
    "epochs": (
        "Longer runs give more optimization opportunities but may overfit. Best and final "
        "validation accuracies expose late degradation. With cosine scheduling, changing "
        "the epoch budget also changes the schedule horizon; these are budget comparisons, "
        "not identical learning-rate trajectories truncated at different epochs."
    ),
    "learning_rate": (
        "Very small learning rates can converge slowly; large rates can oscillate or diverge. "
        "The initial rate is varied while cosine scheduling, optimizer, seed, and split "
        "remain fixed. Observed differences below are single-seed measurements, not "
        "evidence of statistical significance."
    ),
}


def controlled_subset(results, axis):
    subset = [
        (directory, summary)
        for directory, summary in results
        if summary["config"]["group"] in ("baseline", axis)
    ]
    fixed = {
        json.dumps(
            {key: value for key, value in summary["config"].items() if key not in ("group", axis)},
            sort_keys=True,
        )
        for _, summary in subset
    }
    if len(fixed) > 1:
        raise ValueError(f"{axis} comparison changes more than one factor")
    return subset


def _coverage(results):
    required = {
        "activation": set(ACTIVATIONS),
        "optimizer": set(OPTIMIZERS),
        "batch_size": {64, 128, 256},
        "epochs": {25, 50, 100},
        "learning_rate": {0.0001, 0.001, 0.01},
    }
    missing = []
    for axis, expected in required.items():
        observed = {s["config"][axis] for _, s in controlled_subset(results, axis)}
        if not expected <= observed:
            missing.append(axis)
    return missing


def build_report(
    root: Path,
    final_dir: Path,
    output: Path,
    github_url: str,
    parallel_image: Path | None = None,
    draft=False,
):
    results = completed_runs(root)
    selected_dir, selected = select_best(root)
    final = json.loads((final_dir / "final.json").read_text())
    if final["config"] != selected["config"] or (
        final["best_validation_accuracy"] != selected["best_validation_accuracy"]
    ):
        raise ValueError("Final model does not match the best run in this experiment collection")
    missing = _coverage(results)
    parsed = urlparse(github_url)
    valid_repo = (
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and len(parsed.path.strip("/").split("/")) == 2
    )
    incomplete = [
        path.parent
        for path in root.rglob("config.json")
        if not (path.parent / "summary.json").exists()
    ]
    if parallel_image is not None and not parallel_image.is_file():
        raise FileNotFoundError(parallel_image)
    if not draft:
        if final["synthetic"]:
            raise ValueError("Synthetic measurements cannot be used in a submission report")
        if missing or incomplete:
            raise ValueError(f"Incomplete experiments: missing axes={missing}, runs={incomplete}")
        if not valid_repo:
            raise ValueError("A real https://github.com/OWNER/REPO URL is required")
        if parallel_image is None:
            raise ValueError("Export the W&B parallel-coordinate panel and pass --wandb-parallel")
        if any(not s.get("wandb_url") for _, s in results):
            raise ValueError(
                "Submission requires online or verified synced W&B runs; other runs are drafts"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []

    def paragraph(text, style="BodyText"):
        story.extend([Paragraph(escape(text), styles[style]), Spacer(1, 8)])

    def table(rows):
        rendered = [
            [Paragraph(escape(str(value)), styles["BodyText"]) for value in row] for row in rows
        ]
        item = Table(rendered, repeatRows=1, hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.extend([item, Spacer(1, 12)])

    def image(path):
        width, height = ImageReader(str(path)).getSize()
        scale = min(470 / width, 340 / height)
        story.extend([Image(str(path), width=width * scale, height=height * scale), Spacer(1, 10)])

    paragraph("CS6886W Assignment 1: VGG6 on CIFAR-10", "Title")
    if draft:
        paragraph(
            "DRAFT - NOT FOR SUBMISSION. "
            + (
                "Synthetic smoke-test data; these are NOT CIFAR-10 results."
                if final["synthetic"]
                else "Submission prerequisites have not been enforced."
            ),
            "Heading2",
        )
        paragraph(f"Missing controlled comparisons: {missing}. Incomplete runs: {incomplete}.")
        if final["synthetic"]:
            paragraph(
                "The smoke run actually uses 24 random training tensors, 12 validation "
                "tensors, and 12 test tensors without augmentation. The CIFAR-10 protocol "
                "described below applies only to future real-data runs."
            )
    paragraph(f"GitHub repository: {github_url or 'Not supplied'}")
    paragraph("1. Baseline and data preparation", "Heading1")
    paragraph(
        "The linked template topology is preserved: 3x3 convolutions with channels "
        "64, 64, 128, 128; batch normalization and the chosen activation after every "
        "convolution; 2x2 max pooling after each pair; adaptive global average pooling; "
        "a 128-to-10 linear classifier. Cross-entropy consumes logits."
    )
    paragraph(
        f"CIFAR-10 training data is split into "
        f"{50000 - final['config']['validation_size']} training and "
        f"{final['config']['validation_size']} validation examples "
        "by a fixed seeded permutation. "
        "The official 10,000-example test set is never used during sweeps. Training transforms: "
        "random 32x32 crop with padding 4, random horizontal flip (p=0.5), tensor conversion, "
        "normalization with mean (0.4914, 0.4822, 0.4465) and standard deviation "
        f"(0.2023, 0.1994, 0.2010), then one clipped cutout square of side "
        f"{final['config']['cutout']} (zero disables cutout). Validation "
        "and test use tensor conversion and the same normalization only. Training accuracy "
        "is measured on augmented minibatches before each update; validation accuracy uses "
        "eval mode on unaugmented held-out data."
    )
    paragraph(
        f"Selected checkpoint test top-1 accuracy: {final['test_accuracy']:.4f}%. "
        f"Test loss: {final['test_loss']:.6f}; evaluated samples: {final['test_samples']}."
    )
    paragraph("2. Configuration comparisons", "Heading1")
    paragraph(
        "The baseline is shared across one-factor comparisons. Joint-search runs are "
        "excluded from causal one-factor conclusions but remain eligible for final selection. "
        "Accuracies are percentages. t60 is the first wall-clock time reaching 60% "
        "validation accuracy; '-' means the threshold was never reached."
    )
    for axis, explanation in MECHANISMS.items():
        paragraph(axis.replace("_", " ").title(), "Heading2")
        paragraph(explanation)
        subset = controlled_subset(results, axis)
        if not subset:
            paragraph("No controlled measurements available for this axis.")
            continue
        rows = [["Value", "Best val %", "Final val %", "Best epoch", "Steps", "Seconds", "t60"]]
        for directory, summary in subset:
            history = read_history(directory)
            threshold = next(
                (r["elapsed_seconds"] for r in history if r["validation/accuracy"] >= 60), None
            )
            rows.append(
                [
                    summary["config"][axis],
                    f"{summary['best_validation_accuracy']:.3f}",
                    f"{summary['final_validation_accuracy']:.3f}",
                    summary["best_epoch"],
                    summary["steps"],
                    f"{summary['elapsed_seconds']:.1f}",
                    "-" if threshold is None else f"{threshold:.1f}",
                ]
            )
        table(rows)
        winner = max(subset, key=lambda item: item[1]["best_validation_accuracy"])[1]
        baseline = next((s for _, s in subset if s["config"]["group"] == "baseline"), None)
        text = (
            f"The strongest observed {axis} value in this controlled comparison is "
            f"{winner['config'][axis]}, at {winner['best_validation_accuracy']:.3f}% "
            "best validation accuracy."
        )
        if baseline:
            delta = winner["best_validation_accuracy"] - baseline["best_validation_accuracy"]
            text += f" Its difference from the baseline is {delta:+.3f} percentage points."
        paragraph(text + " This conclusion applies only to the measured configuration budget.")
    story.append(PageBreak())
    paragraph("3. Plots", "Heading1")
    report_metadata = final_dir.parent / "wandb-report.json"
    if report_metadata.exists():
        metadata = json.loads(report_metadata.read_text())
        paragraph(f"W&B report: {metadata['report_url']}")
    if parallel_image:
        paragraph("W&B parallel-coordinate panel exported from the experiment workspace.")
        image(parallel_image)
    else:
        paragraph("W&B parallel-coordinate export is missing; add it after online sweeps.")
    paragraph(
        "The following selected-run plots are generated automatically and logged as W&B "
        "images by the training program. Native W&B scalar panels also use the same logged "
        "history with optimizer update step as their x-axis."
    )
    for name, path in plot_run(selected_dir).items():
        paragraph(name.replace("_", " ").title(), "Heading2")
        image(path)
    paragraph("4. One final configuration", "Heading1")
    paragraph(
        f"Exactly one winner was selected by highest validation accuracy; ties prefer the "
        f"earlier best epoch, then sorted run path. Best validation accuracy: "
        f"{final['best_validation_accuracy']:.4f}%; checkpoint epoch: {final['best_epoch']}. "
        "The best checkpoint, not the last epoch, was evaluated on the test set."
    )
    table([["Parameter", "Value"], *[[key, value] for key, value in final["config"].items()]])
    paragraph(f"Selected W&B run: {selected.get('wandb_url') or 'Not an online run'}")
    paragraph("5. Reproducibility and repository", "Heading1")
    paragraph(
        "See README.md for exact installation, sweep, selection, evaluation, and PDF commands. "
        "Python, NumPy, PyTorch, shuffle generators, and worker RNGs are seeded. Deterministic "
        "algorithms are enabled and cuDNN benchmarking is disabled. Identical software and "
        "hardware are required for bitwise reproduction; cross-device exact accuracy is "
        "not guaranteed. uv.lock pins transitive dependencies."
    )
    reproduction_path = final_dir / "reproduction.json"
    if reproduction_path.exists():
        reproduction = json.loads(reproduction_path.read_text())
        if reproduction["status"] != "passed":
            raise ValueError("Saved from-scratch reproduction did not pass")
        paragraph(
            "An independent from-scratch rerun on the same Mac reproduced validation "
            f"accuracy {reproduction['reproduced_validation_accuracy']:.4f}% and test "
            f"accuracy {reproduction['reproduced_test_accuracy']:.4f}%, selecting epoch "
            f"{reproduction['reproduced_best_epoch']}. The verification is saved in "
            f"{reproduction_path.as_posix()}."
        )
    env = json.loads((selected_dir / "environment.json").read_text())
    table([["Environment", "Value"], *[[key, value] for key, value in env.items()]])
    paragraph(
        f"The trained checkpoint is {final_dir.as_posix()}/best.pt. Its configuration, results, "
        "history, and environment are saved alongside it. The saved model is a state dictionary "
        "with its configuration; evaluation reconstructs the architecture without pickled "
        "model objects. Confirm the repository and checkpoint are accessible to evaluators."
    )
    SimpleDocTemplate(
        str(output), title="CS6886W Assignment 1", topMargin=36, bottomMargin=36
    ).build(story)
    return output
