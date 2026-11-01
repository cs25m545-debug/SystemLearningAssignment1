import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ACTIVATIONS = ("relu", "sigmoid", "tanh", "silu", "gelu")
OPTIMIZERS = ("sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam")


@dataclass(frozen=True)
class Config:
    activation: str = "relu"
    optimizer: str = "adam"
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.9
    seed: int = 42
    split_seed: int = 42
    validation_size: int = 5000
    workers: int = 0
    device: str = "auto"
    data_dir: str = "data"
    scheduler: str = "cosine"
    cutout: int = 16
    synthetic: bool = False
    group: str = "baseline"

    def __post_init__(self):
        if self.activation not in ACTIVATIONS or self.optimizer not in OPTIMIZERS:
            raise ValueError("Unsupported activation or optimizer")
        for name in ("batch_size", "epochs", "validation_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("workers", "seed", "split_seed", "cutout"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.cutout > 32 or self.validation_size >= 50000:
            raise ValueError("cutout must be <=32 and validation_size must be <50000")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.momentum) or not 0 < self.momentum < 1:
            raise ValueError("momentum must be between zero and one")
        if self.scheduler not in ("cosine", "none"):
            raise ValueError("scheduler must be cosine or none")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a boolean")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text()))


def controlled_configs(base: Config):
    """One-factor comparisons share the baseline and a fixed validation split."""
    configs = [base]
    axes = {
        "activation": ACTIVATIONS,
        "optimizer": OPTIMIZERS,
        "batch_size": (64, 128, 256),
        "epochs": (25, 50, 100),
        "learning_rate": (0.0001, 0.001, 0.01),
    }
    for field, values in axes.items():
        for value in values:
            if value != getattr(base, field):
                configs.append(replace(base, **{field: value, "group": field}))
    return configs
