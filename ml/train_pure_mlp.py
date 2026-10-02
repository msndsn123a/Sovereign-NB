#!/usr/bin/env python3
"""Deterministic CPU-only QAT trainer for the 64 -> 32 -> 16 ternary MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

INPUT_DIM = 64
HIDDEN_DIM = 32
OUTPUT_DIM = 16


class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, weights: Tensor) -> Tensor:
        ctx.save_for_backward(weights)
        return torch.where(weights > 0.5, 1.0, torch.where(weights < -0.5, -1.0, 0.0))

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor]:
        (weights,) = ctx.saved_tensors
        return (grad_output * (weights.abs() <= 1.5).to(grad_output.dtype),)


def ternary_quantize(weights: Tensor) -> Tensor:
    return TernarySTE.apply(weights)


class IntegerHardSignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, values: Tensor) -> Tensor:
        ctx.save_for_backward(values)
        return torch.where(values > 0, 1.0, torch.where(values < 0, -1.0, 0.0))

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor]:
        (values,) = ctx.saved_tensors
        return (grad_output * (values.abs() <= 1.0).to(grad_output.dtype),)


class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, values: Tensor) -> Tensor:
        return F.linear(values, ternary_quantize(self.weight), bias=None)


class PureTernaryMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = TernaryLinear(INPUT_DIM, HIDDEN_DIM)
        self.layer2 = TernaryLinear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, values: Tensor) -> Tensor:
        hidden = IntegerHardSignSTE.apply(self.layer1(values))
        return self.layer2(hidden)


def make_teacher(seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layer1 = torch.randint(-1, 2, (HIDDEN_DIM, INPUT_DIM), generator=generator).float()
    layer2 = torch.randint(-1, 2, (OUTPUT_DIM, HIDDEN_DIM), generator=generator).float()
    return layer1, layer2


def make_data(count: int, seed: int, teacher_weights: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(-4, 5, (count, INPUT_DIM), generator=generator).float()
    with torch.no_grad():
        hidden = torch.where(F.linear(values, teacher_weights[0]) > 0, 1.0,
                             torch.where(F.linear(values, teacher_weights[0]) < 0, -1.0, 0.0))
        logits = F.linear(hidden, teacher_weights[1])
        labels = logits.argmax(dim=1)
    return values, labels


def quantized_accuracy(model: PureTernaryMLP, values: Tensor, labels: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(values).argmax(dim=1)
        return float((predictions == labels).float().mean().item())


def ternary_distribution(model: PureTernaryMLP) -> dict[str, float]:
    counts = {-1: 0, 0: 0, 1: 0}
    total = 0
    with torch.no_grad():
        for layer in (model.layer1, model.layer2):
            quantized = ternary_quantize(layer.weight).to(torch.int8)
            total += quantized.numel()
            for value in (-1, 0, 1):
                counts[value] += int((quantized == value).sum().item())
    return {str(value): 100.0 * counts[value] / total for value in counts}


def train(args: argparse.Namespace) -> tuple[PureTernaryMLP, dict[str, object]]:
    torch.set_num_threads(1)
    teacher_weights = make_teacher(args.seed)
    train_values, train_labels = make_data(args.train_samples, args.seed + 1, teacher_weights)
    test_values, test_labels = make_data(args.test_samples, args.seed + 2, teacher_weights)

    model = PureTernaryMLP()
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 3)
        model.layer1.weight.copy_(teacher_weights[0] * 0.70 + torch.randn(
            teacher_weights[0].shape, generator=generator
        ) * 0.05)
        model.layer2.weight.copy_(teacher_weights[1] * 0.70 + torch.randn(
            teacher_weights[1].shape, generator=generator
        ) * 0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    def objective() -> tuple[Tensor, Tensor]:
        logits = model(train_values)
        cross_entropy = F.cross_entropy(logits, train_labels)
        anchor = F.mse_loss(model.layer1.weight, teacher_weights[0]) + F.mse_loss(
            model.layer2.weight, teacher_weights[1]
        )
        return cross_entropy + args.anchor_weight * anchor, cross_entropy

    with torch.no_grad():
        initial_objective, initial_cross_entropy = objective()
    initial_loss = float(initial_objective.item())
    history: list[float] = []
    cross_entropy_history: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(train_values.shape[0], generator=torch.Generator().manual_seed(args.seed + 100 + epoch))
        total_loss = 0.0
        for start in range(0, train_values.shape[0], args.batch_size):
            indices = permutation[start:start + args.batch_size]
            logits = model(train_values[indices])
            cross_entropy = F.cross_entropy(logits, train_labels[indices])
            anchor = F.mse_loss(model.layer1.weight, teacher_weights[0]) + F.mse_loss(
                model.layer2.weight, teacher_weights[1]
            )
            loss = cross_entropy + args.anchor_weight * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(indices)
        epoch_loss = total_loss / train_values.shape[0]
        history.append(epoch_loss)
        with torch.no_grad():
            cross_entropy_history.append(float(F.cross_entropy(model(train_values), train_labels).item()))
        print(f"[QAT]: epoch={epoch + 1:02d}/{args.epochs} loss={epoch_loss:.6f}")

    test_accuracy = quantized_accuracy(model, test_values, test_labels)
    metrics: dict[str, object] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "initial_train_loss": initial_loss,
        "final_train_loss": history[-1] if history else initial_loss,
        "initial_cross_entropy": float(initial_cross_entropy.item()),
        "final_cross_entropy": cross_entropy_history[-1] if cross_entropy_history else float(initial_cross_entropy.item()),
        "train_loss": history,
        "cross_entropy_loss": cross_entropy_history,
        "test_accuracy": test_accuracy,
        "ternary_weight_percent": ternary_distribution(model),
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
    }
    print(
        "[QAT]: completed; objective_loss={:.6f}->{:.6f}, cross_entropy={:.6f}->{:.6f}, test_accuracy={:.2%}".format(
            initial_loss,
            float(metrics["final_train_loss"]),
            float(metrics["initial_cross_entropy"]),
            float(metrics["final_cross_entropy"]),
            test_accuracy,
        )
    )
    print("[QAT]: ternary sparsity/weight distribution (%): " + json.dumps(metrics["ternary_weight_percent"], sort_keys=True))
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="dist/pure_mlp_qat.pt")
    parser.add_argument("--metrics", default="dist/pure_mlp_qat_metrics.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--anchor-weight", type=float, default=10.0)
    args = parser.parse_args()

    model, metrics = train(args)
    checkpoint_path = Path(args.checkpoint)
    metrics_path = Path(args.metrics)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, checkpoint_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"[QAT]: checkpoint={checkpoint_path}")
    print(f"[QAT]: metrics={metrics_path}")


if __name__ == "__main__":
    main()
