#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p data artifacts
echo "Downloading CIFAR-10; resuming the partial download when supported."
curl --fail --location --silent --show-error --retry 3 --continue-at - \
  --output data/cifar-10-python.tar.gz \
  'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
.venv/bin/python -c \
  "import sys; from torchvision.datasets.utils import check_integrity; sys.exit(0 if check_integrity('data/cifar-10-python.tar.gz', 'c58f30108f718f92721af3b95e74349a') else 'CIFAR-10 archive checksum mismatch')"

echo "Starting 16 controlled real-data experiments on Apple MPS."
.venv/bin/python -m vgg6 suite --device mps --workers 0 \
  --wandb-mode offline --output runs/mac-real
.venv/bin/python -m vgg6 finalize --runs runs/mac-real --device mps \
  --output artifacts/mac-final
.venv/bin/python -m vgg6 report --runs runs/mac-real --final artifacts/mac-final \
  --draft --output artifacts/mac-assignment-draft.pdf
echo "Training and test evaluation complete; W&B synchronization and the GitHub URL remain pending."
