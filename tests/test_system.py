import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vgg6.config import ACTIVATIONS, OPTIMIZERS, Config, controlled_configs
from vgg6.data import MEAN, STD, Cutout, split_indices, training_loaders
from vgg6.evaluation import evaluate
from vgg6.experiments import (
    completed_runs,
    create_sweeps,
    evaluate_checkpoint,
    finalize,
    select_best,
    sync_runs,
)
from vgg6.model import VGG6
from vgg6.report import build_report, controlled_subset
from vgg6.training import make_optimizer, train

torch.set_num_threads(1)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_template_topology_and_activation_gradient(activation):
    model = VGG6(activation)
    model.train()
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert sum(isinstance(m, nn.Conv2d) for m in model.modules()) == 4
    assert sum(isinstance(m, nn.MaxPool2d) for m in model.modules()) == 2
    assert model.features(torch.randn(2, 3, 32, 32)).shape == (2, 128, 8, 8)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_optimizer_performs_update(optimizer):
    model = nn.Linear(2, 2)
    before = model.weight.detach().clone()
    instance = make_optimizer(model, Config(optimizer=optimizer))
    model(torch.ones(2, 2)).square().sum().backward()
    instance.step()
    assert not torch.equal(before, model.weight)
    if optimizer in ("sgd", "nesterov"):
        assert instance.defaults["nesterov"] == (optimizer == "nesterov")


def test_validation_split_is_fixed_and_disjoint():
    training, validation = split_indices(50, 5, 42)
    assert (training, validation) == split_indices(50, 5, 42)
    assert set(training).isdisjoint(validation)
    assert sorted(training + validation) == list(range(50))
    assert validation != split_indices(50, 5, 43)[1]


def test_cutout_preserves_input_and_handles_odd_length():
    image = torch.ones(3, 32, 32)
    torch.manual_seed(1)
    result = Cutout(5)(image)
    assert torch.all(image == 1)
    assert (result == 0).any()
    assert torch.equal(result[0], result[1])
    assert torch.equal(Cutout(0)(image), image)


def test_evaluation_weights_uneven_batches_and_restores_eval_mode():
    model = nn.Identity()
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert metrics["accuracy"] == pytest.approx(200 / 3)
    assert metrics["loss"] == pytest.approx(nn.functional.cross_entropy(logits, targets).item())
    assert not model.training
    with pytest.raises(ValueError, match="empty"):
        evaluate(model, [], torch.device("cpu"))


def test_controlled_matrix_changes_only_one_axis():
    baseline = Config()
    configs = controlled_configs(baseline)
    assert len(configs) == 16
    for config in configs[1:]:
        changed = {
            key for key, value in config.to_dict().items() if value != baseline.to_dict()[key]
        }
        assert changed == {"group", config.group}
    assert {c.optimizer for c in configs} == set(OPTIMIZERS)
    assert {c.activation for c in configs} == set(ACTIVATIONS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": -1},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"validation_size": 50000},
        {"momentum": 0},
        {"cutout": 33},
        {"seed": -1},
        {"synthetic": "false"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_training_reproduction_selection_evaluation_and_pdf(tmp_path):
    config = Config(epochs=2, batch_size=8, synthetic=True, device="cpu")
    root = tmp_path / "runs"
    first = train(config, root / "first")
    second = train(config, root / "second")
    assert first["wandb_mode"] == "disabled" and first["wandb_url"] is None
    assert first["best_validation_accuracy"] == second["best_validation_accuracy"]
    saved_first = torch.load(root / "first" / "best.pt", weights_only=True)
    saved_second = torch.load(root / "second" / "best.pt", weights_only=True)
    assert all(
        torch.equal(value, saved_second["model"][key])
        for key, value in saved_first["model"].items()
    )
    validation = evaluate_checkpoint(root / "first" / "best.pt", "validation", "cpu")
    assert validation["accuracy"] == first["best_validation_accuracy"]
    assert select_best(root)[0].name == "first"
    final_dir = tmp_path / "final"
    result = finalize(root, final_dir, "cpu")
    assert result["test_samples"] == 12
    draft = build_report(root, final_dir, tmp_path / "draft.pdf", "", draft=True)
    assert draft.read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="Synthetic"):
        build_report(root, final_dir, tmp_path / "submission.pdf", "https://github.com/a/b")
    with pytest.raises(FileExistsError):
        train(config, root / "first")


def test_incompatible_validation_protocol_is_rejected(tmp_path):
    for index in (0, 1):
        directory = tmp_path / str(index)
        directory.mkdir()
        config = replace(Config(), split_seed=index).to_dict()
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "config": config,
                    "synthetic": False,
                }
            )
        )
    with pytest.raises(ValueError, match="different datasets or validation splits"):
        completed_runs(tmp_path)


def test_evaluation_rejects_nonfinite_loss():
    loader = DataLoader(
        TensorDataset(torch.full((2, 2), float("nan")), torch.tensor([0, 1])), batch_size=2
    )
    with pytest.raises(FloatingPointError):
        evaluate(nn.Identity(), loader, torch.device("cpu"))


def test_cifar_training_does_not_load_test_set_or_augment_validation(monkeypatch):
    created = []

    class SmallCifar:
        def __init__(self, root, train, download, transform):
            assert train and download
            self.transform = transform
            created.append(self)

        def __len__(self):
            return 20

        def __getitem__(self, index):
            return self.transform(Image.new("RGB", (32, 32), (255, 255, 255))), index % 10

    monkeypatch.setattr("vgg6.data.datasets.CIFAR10", SmallCifar)
    training, validation = training_loaders(Config(validation_size=5, batch_size=4))
    assert len(created) == 2
    assert set(training.dataset.indices).isdisjoint(validation.dataset.indices)
    image, _ = validation.dataset[0]
    expected = torch.tensor([(1 - mean) / std for mean, std in zip(MEAN, STD)])
    assert torch.allclose(image, expected[:, None, None].expand_as(image))
    assert torch.equal(image, validation.dataset[0][0])
    assert not any(isinstance(transform, Cutout) for transform in created[1].transform.transforms)


def test_sweep_definitions_fix_all_nonvaried_parameters(tmp_path, monkeypatch):
    registered = []

    def register(specification, project, entity):
        registered.append(specification)
        assert project == "test-project" and entity == "test-entity"
        return f"id-{len(registered)}"

    monkeypatch.setattr("vgg6.experiments.wandb.sweep", register)
    ids = create_sweeps(Config(), tmp_path / "sweeps", "test-project", "test-entity")
    assert len(ids) == 6
    assert json.loads((tmp_path / "sweeps" / "ids.json").read_text()) == ids
    for axis, specification in zip(ids, registered):
        varied = {key for key, value in specification["parameters"].items() if "values" in value}
        assert varied == (
            {axis}
            if axis != "joint"
            else {"activation", "optimizer", "batch_size", "epochs", "learning_rate"}
        )
        assert specification["metric"]["name"] == "best_validation_accuracy"


def test_report_rejects_confounded_controlled_comparisons():
    results = [
        (None, {"config": Config().to_dict()}),
        (None, {"config": Config(activation="gelu", epochs=25, group="activation").to_dict()}),
    ]
    with pytest.raises(ValueError, match="more than one factor"):
        controlled_subset(results, "activation")


def test_online_training_persists_wandb_url_using_settings_mode(tmp_path, monkeypatch):
    run = SimpleNamespace(
        id="test",
        mode="run",
        settings=SimpleNamespace(mode="online"),
        url="https://wandb.ai/test/project/runs/test",
        summary={},
        define_metric=lambda *args, **kwargs: None,
        log=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("vgg6.training.wandb.init", lambda **kwargs: nullcontext(run))
    summary = train(
        Config(synthetic=True, epochs=1, batch_size=8), tmp_path / "online", mode="online"
    )
    assert summary["wandb_mode"] == "online"
    assert summary["wandb_url"] == run.url


@pytest.mark.parametrize("matches", [True, False])
def test_offline_sync_requires_verified_remote_measurements(tmp_path, monkeypatch, matches):
    config = Config().to_dict()
    run_dir = tmp_path / "run"
    logs = run_dir / "wandb" / "offline-run-20261101-test"
    logs.mkdir(parents=True)
    summary = {
        "status": "completed",
        "config": config,
        "synthetic": False,
        "wandb_mode": "offline",
        "wandb_run_id": "test",
        "wandb_url": None,
        "best_validation_accuracy": 70.0,
    }
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary))
    remote = SimpleNamespace(
        config=config,
        summary={"best_validation_accuracy": 70.0 if matches else 69.0},
        url="https://wandb.ai/entity/project/runs/test",
    )
    monkeypatch.setattr("vgg6.experiments.wandb.api", SimpleNamespace(api_key="test-only"))
    monkeypatch.setattr(
        "vgg6.experiments.wandb.Api", lambda: SimpleNamespace(run=lambda name: remote)
    )
    uploads = []
    monkeypatch.setattr(
        "vgg6.experiments.subprocess.run", lambda args, check: uploads.append((args, check))
    )
    if matches:
        assert sync_runs(tmp_path, "project", "entity") == [remote.url]
        updated = json.loads(path.read_text())
        assert updated["wandb_mode"] == "synced" and updated["wandb_url"] == remote.url
    else:
        with pytest.raises(ValueError, match="accuracy does not match"):
            sync_runs(tmp_path, "project", "entity")
        assert json.loads(path.read_text()) == summary
    assert uploads[0][0][-1] == str(logs)
    assert uploads[0][1] is True
