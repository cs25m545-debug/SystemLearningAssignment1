import math

import torch
from torch import nn


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets).item()
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        loss_sum += loss
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.numel()
    if total == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return {"loss": loss_sum / total, "accuracy": 100.0 * correct / total, "samples": total}
