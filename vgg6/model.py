import torch
from torch import nn

from vgg6.config import ACTIVATIONS


class VGG6(nn.Module):
    """Template topology: four convolutions, two pools, global pooling, linear head."""

    def __init__(self, activation: str = "relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        factory = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }[activation]
        layers = []
        channels = 3
        for index, output in enumerate((64, 64, 128, 128)):
            layers.extend(
                [nn.Conv2d(channels, output, 3, padding=1), nn.BatchNorm2d(output), factory()]
            )
            if index in (1, 3):
                layers.append(nn.MaxPool2d(2))
            channels = output
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)).flatten(1))
