# CS6886W Assignment 1: VGG6 / CIFAR-10

Repository: https://github.com/cs25m545-debug/SystemLearningAssignment1

This is an original, modular Python implementation of the system required by
`A1.pdf`. It implements all five activations, all six optimizers, controlled
batch-size/epoch/learning-rate comparisons, W&B sweeps and automatic plots,
best-validation checkpoint selection, separate test evaluation, and a PDF report.

**Status:** all 16 real CIFAR-10 controlled experiments completed on Apple MPS
(1,475 epochs). The selected checkpoint achieved **88.08% validation accuracy**
and **86.61% test accuracy** on the official 10,000-example test set.
The real checkpoint is in `artifacts/mac-final/best.pt`, and the detailed
final submission PDF with the repository URL is
[artifacts/assignment1.pdf](artifacts/assignment1.pdf).
All 16 experiments and their automatic plots have been uploaded to W&B; every
remote epoch was checked against its saved local measurements. The native W&B
parallel-coordinate panel is [artifacts/wandb-parallel.png](artifacts/wandb-parallel.png).
See [the W&B report](https://wandb.ai/cs25m545-indian-institute-of-technology-madras/vgg6-cifar10/reports/CS6886W-Assignment-1---Verified-VGG6-experiments--VmlldzoxODIzNzMyMw==)
and `artifacts/wandb-report.json` for verified run URLs.
The W&B team project currently requires team access; all submission results and
plots are also included in the public GitHub repository and PDF.

**From-scratch reproduction passed:** retraining the selected configuration on
the same Mac reproduced **88.08% validation accuracy**, **86.61% test accuracy**,
and the same selected epoch **92**. The verification record is
`artifacts/mac-final/reproduction.json`; the independent rerun is stored outside
the selection collection in `runs/mac-reproduction/`.

## Measured results and exactly one selected configuration

The winner uses **ReLU, RMSprop, batch size 128, 100 epochs, initial LR 0.001,
weight decay 0.0005, cosine scheduling, seed 42, split seed 42, cutout 16,
workers 0, and Apple MPS**. The selected checkpoint is epoch **92**, not epoch 100.
Its test cross-entropy is **0.4080603**. The complete configuration is saved
with the model in `artifacts/mac-final/config.json`; test results are in
`artifacts/mac-final/final.json`. Only the validation-selected winner was tested.

All other factors are held at the Adam baseline for each comparison below.
Times are measured seconds including training, validation, and end-of-run plotting,
but excluding dataset download.

| Factor | Value | Best validation accuracy | Seconds |
|---|---|---:|---:|
| Baseline | ReLU / Adam / 128 / 100 / 0.001 | 87.74% | 794.3 |
| Activation | Sigmoid | 71.68% | 775.9 |
| Activation | Tanh | 79.90% | 778.8 |
| Activation | SiLU | 86.96% | 781.2 |
| Activation | GELU | 87.50% | 782.9 |
| Optimizer | SGD | 74.66% | 763.2 |
| Optimizer | Nesterov-SGD | 74.98% | 765.7 |
| Optimizer | Adagrad | 62.48% | 769.0 |
| Optimizer | RMSprop | **88.08%** | 770.8 |
| Optimizer | Nadam | 87.98% | 778.4 |
| Batch size | 64 | 87.82% | 942.9 |
| Batch size | 256 | 86.92% | 749.0 |
| Epochs | 25 | 82.52% | 194.6 |
| Epochs | 50 | 86.08% | 388.5 |
| Initial LR | 0.0001 | 77.98% | 777.8 |
| Initial LR | 0.01 | 79.24% | 776.2 |

ReLU performed best among activations; the saturating sigmoid/tanh alternatives
were substantially worse. RMSprop narrowly exceeded Nadam and Adam, whereas SGD,
Nesterov-SGD, and Adagrad were worse **at the shared LR 0.001**; these measurements
do not establish each optimizer's individually tuned optimum. Batch size 64
slightly improved accuracy over 128 but required more time; batch size 256 was
faster but less accurate. Increasing the budget from 25 to 50 to 100 epochs
improved validation accuracy, while also changing the cosine schedule horizon.
LR 0.001 exceeded both 0.0001 and 0.01. The RMSprop run's final validation
accuracy was 87.76%, below its best 88.08%, supporting best-checkpoint selection.
These are single-seed observations, not statistical-significance claims.

Exact commands for the already-selected model:

```bash
uv run --locked vgg6 evaluate --checkpoint artifacts/mac-final/best.pt \
  --split validation --device mps
uv run --locked vgg6 evaluate --checkpoint artifacts/mac-final/best.pt \
  --split test --device mps
OMP_NUM_THREADS=1 uv run --locked vgg6 train \
  --config artifacts/mac-final/config.json --wandb-mode offline \
  --output runs/mac-reproduction
```

The final command is a from-scratch reproduction and requires a fresh output path.
It is deliberately outside `runs/mac-real`, so it does not alter model selection.

## Environment and installation

Use Python **3.11** (3.12 is also supported) and `uv`. Direct dependencies are
pinned in `pyproject.toml`; all transitive versions are pinned in `uv.lock`.
The reference packages are PyTorch 2.6.0, torchvision 0.21.0, NumPy 2.2.3,
W&B 0.22.3, matplotlib 3.10.0, PyYAML 6.0.2, and ReportLab 4.3.1.
The original measured runs used W&B 0.19.8; the tracking client was subsequently
upgraded to accept the newer `wandb_v1_` API-key format. Model-training dependencies
and saved results are unchanged; existing offline logs do not require retraining.

```bash
uv sync --locked --extra dev --python 3.11
uv run --locked --extra dev pytest -q
uv run --locked --extra dev ruff check .
uv run --locked --extra dev ruff format --check .
uv run --locked vgg6 --help
```

On Linux use a CUDA-compatible NVIDIA driver; `--device cuda` fails explicitly if
CUDA is unavailable. `auto` chooses CUDA when available, otherwise CPU. MPS is
opt-in (`--device mps`), and unsupported deterministic operations fail rather than
silently weakening reproducibility. Mac CPU is suitable for smoke tests, not the
full 100-epoch experiment collection. Training does not use mixed precision.

## Architecture and protocol

The architecture matches the linked assignment template: four 3x3 convolutions
(64, 64, 128, 128 channels), batch normalization and the selected activation after
each convolution, two max pools, global average pooling, and a 128-to-10 linear
head. The name VGG6 follows the assignment; no larger stock torchvision VGG is
substituted. We use the same template-style Kaiming initialization for every
activation, so initialization is not another swept factor.

The official CIFAR-10 training set is partitioned into **45,000 training / 5,000
validation** examples with a seeded random permutation. Training uses random
crop (32, padding 4), horizontal flip (p=0.5), tensor conversion, normalization
(mean 0.4914/0.4822/0.4465; std 0.2023/0.1994/0.2010), and a clipped 16-pixel
cutout after normalization. Validation and test use only tensor conversion and
normalization. No augmentation leaks into validation. The separate official
10,000-example test set is loaded only by evaluation/finalization, never training.

Defaults: ReLU, Adam, batch size 128, 100 epochs, initial LR 0.001, weight decay
0.0005, SGD momentum 0.9, cosine scheduler over the full epoch budget, seed 42,
split seed 42, workers 0. Other optimizer settings are PyTorch defaults.
CLI overrides and a saved `--config` JSON are supported. Model and data RNGs,
worker RNGs, shuffle generators, cuDNN, and deterministic algorithms are set.
Identical hardware/software is needed for exact reproduction across runs;
cross-device bitwise reproducibility is not promised.

Training accuracy is sample-weighted on augmented minibatches before each
optimizer update. Validation metrics use eval mode after each epoch. `step`
counts actual optimizer updates, not epochs. Best validation accuracy selects the
checkpoint; ties retain the earliest epoch. Across runs, ties prefer the earlier
best epoch, then sorted directory path. Test accuracy is never a selection metric.

## Local smoke check (no download or W&B account)

Choose fresh output paths; existing run directories are never overwritten.

```bash
uv run --locked vgg6 train --synthetic --device cpu --epochs 2 --batch-size 8 \
  --wandb-mode disabled --output runs/smoke/baseline
uv run --locked vgg6 finalize --runs runs/smoke --device cpu \
  --output artifacts/smoke
uv run --locked vgg6 report --runs runs/smoke --final artifacts/smoke \
  --draft --output reports/smoke.pdf
```

`reports/smoke.pdf` is visibly labeled synthetic and not for submission.
Do not include smoke runs in the real experiment directory.

## Real GPU experiments and W&B plots

W&B stands for **Weights & Biases**, the experiment-tracking service required by
the assignment's plot question. Create or sign into your account at
https://wandb.ai, then authenticate the local CLI. Existing offline logs can be
uploaded without repeating training; see the `vgg6 sync` command below.

Log in interactively; never commit an API key. Replace `YOUR_ENTITY` with your
W&B user or team. W&B uploads metrics, configuration, generated plots, and machine
metadata; use your own authorized account.

```bash
uv run --locked wandb login
uv run --locked vgg6 sweeps --device cuda --workers 4 \
  --project vgg6-cifar10 --entity YOUR_ENTITY --output artifacts/sweeps
```

`artifacts/sweeps/ids.json` holds six registered sweep IDs, with their YAML
definitions beside it. Run the five controlled sweeps first:

```bash
uv run --locked vgg6 agent --sweep-id ACTIVATION_ID --count 5 \
  --project vgg6-cifar10 --entity YOUR_ENTITY --output runs/real
uv run --locked vgg6 agent --sweep-id OPTIMIZER_ID --count 6 \
  --project vgg6-cifar10 --entity YOUR_ENTITY --output runs/real
uv run --locked vgg6 agent --sweep-id BATCH_SIZE_ID --count 3 \
  --project vgg6-cifar10 --entity YOUR_ENTITY --output runs/real
uv run --locked vgg6 agent --sweep-id EPOCHS_ID --count 3 \
  --project vgg6-cifar10 --entity YOUR_ENTITY --output runs/real
uv run --locked vgg6 agent --sweep-id LEARNING_RATE_ID --count 3 \
  --project vgg6-cifar10 --entity YOUR_ENTITY --output runs/real
```

These are grid sweeps varying one factor while all others remain fixed:
ReLU/Sigmoid/Tanh/SiLU/GELU; SGD/Nesterov-SGD/Adam/Adagrad/RMSprop/Nadam;
batch size 64/128/256; epochs 25/50/100; initial LR 0.0001/0.001/0.01.
Grid sweeps repeat the common baseline, giving **20 runs**. For optional joint
optimization after controlled comparisons:

```bash
uv run --locked vgg6 agent --sweep-id JOINT_ID --count 25 \
  --project vgg6-cifar10 --entity YOUR_ENTITY --output runs/real
```

The joint sweep randomly searches combinations; `--count` is its explicit compute
budget, not a guarantee of global optimality. W&B agents can run on multiple GPUs
using separate processes and `CUDA_VISIBLE_DEVICES`. Use shared experiment storage,
or collect all completed run directories onto one machine before selection.
Do not run independent processes with the same run output directory.

Each run automatically logs four scalar curves, a validation-accuracy-vs-update
scatter image, and four curve images to W&B. In the W&B project workspace create
a **Parallel coordinates** panel containing activation, optimizer, batch_size,
epochs, learning_rate, and `best_validation_accuracy`, restricted to this
experiment collection. Sweep pages also provide the parallel-coordinate view.
Export the project panel as `artifacts/wandb-parallel.png`; the project-wide view
must include the eventual winner, including any joint sweep. Native curve panels
use `step`, and the logged `plots/validation_scatter` is the requested scatter plot.
The PDF embeds the exported W&B parallel panel and the exact automatically logged
curve/scatter images from the selected run.

Without W&B, the same controlled matrix can run locally (16 unique runs):

```bash
uv run --locked vgg6 suite --device cuda --workers 4 --wandb-mode disabled \
  --output runs/local
```

Disabled logging is useful for debugging but cannot satisfy the assignment's W&B
requirement. `--wandb-mode offline` records complete local W&B logs. After logging
in, upload them and verify their remote measurements without retraining:

```bash
uv run --locked wandb login
uv run --locked vgg6 sync --runs runs/mac-real \
  --project vgg6-cifar10 --entity YOUR_ENTITY
```

`sync` checks the uploaded configuration and best validation accuracy before
recording a verified W&B URL. Unsynced offline summaries permit draft reports only.
Use online mode or verified synchronization for the final collection.

### Full collection on this Mac

`scripts/run_mac.sh` resumes the official dataset download, verifies its checksum,
runs all 16 unique controlled configurations on MPS (1,475 total epochs), selects
and tests one checkpoint, and produces a real-results draft PDF. It uses offline
W&B because no account is authenticated. The process is long-running; keep this
Mac powered and awake. The current launch uses `caffeinate -i` to prevent idle
sleep, but closing the lid or powering off still interrupts it.

```bash
caffeinate -i /bin/bash scripts/run_mac.sh > artifacts/mac-training.log 2>&1
```

Output: `runs/mac-real/`, `artifacts/mac-final/`, and
`artifacts/mac-assignment-draft.pdf`. Follow progress in `artifacts/mac-training.log`
and each run's `history.csv`. Do not start a second copy while the first is running.
An interrupted completed-run collection is not automatically resumed; existing
run output directories are intentionally protected from overwrite.

After synchronization, export the W&B project parallel panel and generate the
submission PDF with `--runs runs/mac-real --final artifacts/mac-final` and your
actual GitHub URL. The draft is not a substitute for those final submission steps.

## Select, verify, and reproduce exactly one final model

Wait for all intended experiments to finish. Incomplete runs have no
`summary.json` and are excluded from selection; a submission report rejects them.
Validation split and dataset identity must agree across all compared runs.

```bash
uv run --locked vgg6 finalize --runs runs/real --device cuda \
  --output artifacts/final
uv run --locked vgg6 evaluate --checkpoint artifacts/final/best.pt \
  --split validation --device cuda
uv run --locked vgg6 evaluate --checkpoint artifacts/final/best.pt \
  --split test --device cuda
uv run --locked vgg6 train --config artifacts/final/config.json \
  --wandb-mode online --project vgg6-cifar10 --entity YOUR_ENTITY \
  --output runs/reproduction
```

The last command retrains the one selected configuration from scratch, outside
the experiment collection. Compare its best validation accuracy to
`artifacts/final/final.json`. Reuse the recorded device, versions, seed, and
workers for strict reproduction. `evaluate` and `finalize` accept `--data-dir`
when moving the CIFAR-10 cache. `train --config ... --data-dir ...` also overrides
the recorded path.

`artifacts/final` contains the model, exact configuration, selected history,
environment, summary, and test result. Checkpoint format uses tensor state
dictionaries loaded with `weights_only=True`. It is not a resumable optimizer
checkpoint: interrupted runs must be rerun in a fresh directory.

## GitHub and the single submission PDF

The code, real model, results, and final PDF are published in the public repository
linked above. Regenerate the submitted PDF from the committed evidence without
retraining or authenticating to W&B:

```bash
uv run --locked vgg6 report --runs artifacts/experiments --final artifacts/mac-final \
  --github-url https://github.com/cs25m545-debug/SystemLearningAssignment1 \
  --wandb-parallel artifacts/wandb-parallel.png \
  --output artifacts/assignment1.pdf
```

`artifacts/experiments/` contains all 16 configurations, verified summaries,
per-epoch histories, and recorded environments. W&B's legacy offline sync command
reported completion without remotely creating runs in this environment. The
preserved measurements were therefore uploaded with the explicit replay option:

```bash
uv run --locked vgg6 sync --runs runs/mac-real --replay \
  --project vgg6-cifar10 --entity cs25m545-indian-institute-of-technology-madras
```

This logs the original measured history and automatically generated plot images,
marks their source as replayed history, and verifies remote configurations and
best accuracies. It does not rerun optimization or invent measurements. The
programmatic report creation additionally verified every remote epoch and metric
against the original local CSV histories.

To recreate the native W&B report and its screenshot after authenticated upload:

```bash
uv sync --locked --extra submission --python 3.11
uv run --locked --extra submission playwright install chromium
uv run --locked --extra submission python scripts/create_wandb_report.py \
  --runs artifacts/experiments \
  --entity cs25m545-indian-institute-of-technology-madras
uv run --locked --extra submission python scripts/export_wandb_panel.py
```

The exporter renders the actual native W&B report panel, not a local substitute.
Browser credentials are kept in memory and restricted to the W&B API; no browser
profile or credential file is committed. `wandb-workspaces==0.2.0` and
`playwright==1.51.0` are pinned in the optional `submission` extra.

The generator refuses a non-draft PDF without real data, all five controlled
comparison axes, completed runs, online W&B URLs, a GitHub-shaped repository URL,
and a W&B parallel-panel export. It does **not** verify remote repository access
or inspect whether an image really came from W&B; check these yourself. It records
measured comparisons, convergence time/steps, theoretical explanations, one best
configuration, test accuracy, plots, and environment. Review the generated
analysis against W&B and add any further interpretation required by your course.

Commit code, `uv.lock`, `README.md`, `artifacts/final/`, the sweep definitions, the
W&B panel export, and `artifacts/assignment1.pdf`. Keep datasets, virtualenvs,
W&B local logs, and raw run directories out of Git. The model is small enough
for normal GitHub Git storage; confirm with `ls -lh artifacts/final/best.pt`.
Make the repository accessible to evaluators; its actual URL is included in the PDF.
The README's generic instructions plus the committed final `config.json` reproduce
the selected configuration without choosing among multiple "best" options.

The course handout `A1.pdf` remains local and is not redistributed in this repository.

## Layout and result limitations

`artifacts/results.csv` contains measured results for all 16 configurations.
`artifacts/figures/` contains the selected run's four curves and validation
scatter. `artifacts/assignment1.pdf` is the final PDF containing the genuine W&B
parallel-coordinate export, measured comparisons, all requested curves/scatter,
one selected configuration, test accuracy, environment and reproduction proof.
The older `artifacts/assignment1-draft.pdf` is retained only as historical output;
submit the final PDF, not the draft.

`config.py` validates parameters; `model.py` defines the network; `data.py`
prepares datasets; `training.py` trains/logs/checkpoints; `evaluation.py` computes
metrics; `experiments.py` runs sweeps and selects/evaluates the winner;
`plots.py` generates figures; `report.py` builds the PDF; `cli.py` exposes commands.

The generated comparison narrative distinguishes theory from measured findings.
Shared learning rates do not establish individually tuned optimizer superiority.
Changing epoch budgets also changes the cosine schedule horizon. Single-seed
results do not establish statistical significance. Longer schedules and joint
search may improve accuracy, but no unmeasured improvement is claimed.
