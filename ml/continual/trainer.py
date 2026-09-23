"""Continual training and regression gating for forensic detection models."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .replay_buffer import ReplaySample, StratifiedReplayBuffer

LOGGER = logging.getLogger(__name__)

ReplayBatchLoader = Callable[
    [Sequence[ReplaySample], torch.device],
    tuple[Tensor, Tensor],
]


class ContinualForensicTrainer:
    """Train a forensic detector on successive Avalanche-style experiences.

    Each experience dataset must yield ``(image_tensor, binary_label)`` pairs,
    where labels are 0 for Real and 1 for AI. Replay examples are retrieved
    from ``StratifiedReplayBuffer`` and converted to tensors by
    ``replay_batch_loader``.

    Regularization options:
    - ``ewc``: diagonal Fisher-weighted penalty from the previous experience.
    - ``lwf``: distillation from a frozen copy of the pre-update model.

    Args:
        model: Model returning a mapping with ``binary_logits`` and optionally
            ``generator_logits``.
        optimizer: Optimizer whose parameters belong to ``model``.
        replay_buffer: Buffer holding prior examples.
        replay_batch_loader: Loads the images and labels for selected replay
            samples, returning ``(images, labels)`` tensors.
        regularizer: Either ``"ewc"`` or ``"lwf"``.
        replay_batch_size: Number of replay items per training batch.
        regularization_strength: EWC penalty multiplier or LwF loss weight.
        distillation_temperature: Softmax temperature for LwF.
        device: Model device. Defaults to CUDA when available.
        checkpoint_dir: Directory where per-experience checkpoints are saved.
        max_memory_samples: Maximum examples used to estimate Fisher.

    Notes:
        This runner uses the Avalanche experience object as the source of each
        new dataset, but performs the optimization loop directly. This is
        useful when replay selection is delegated to the custom stratified
        buffer rather than Avalanche's built-in memory plugin.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: StratifiedReplayBuffer,
        replay_batch_loader: ReplayBatchLoader,
        *,
        regularizer: str = "ewc",
        replay_batch_size: int = 16,
        regularization_strength: float = 10.0,
        distillation_temperature: float = 2.0,
        device: torch.device | str | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        max_memory_samples: int = 512,
    ) -> None:
        if regularizer not in {"ewc", "lwf"}:
            raise ValueError("regularizer must be 'ewc' or 'lwf'")
        if replay_batch_size < 0:
            raise ValueError("replay_batch_size cannot be negative")
        if regularization_strength < 0:
            raise ValueError("regularization_strength cannot be negative")
        if distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive")

        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.replay_batch_loader = replay_batch_loader
        self.regularizer = regularizer
        self.replay_batch_size = replay_batch_size
        self.regularization_strength = regularization_strength
        self.distillation_temperature = distillation_temperature
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_memory_samples = max_memory_samples

        self.model.to(self.device)
        self._teacher: nn.Module | None = None
        self._ewc_anchor: dict[str, Tensor] | None = None
        self._fisher: dict[str, Tensor] | None = None

    @staticmethod
    def _binary_logits(output: Any) -> Tensor:
        if isinstance(output, Mapping):
            return output["binary_logits"]
        if isinstance(output, (tuple, list)):
            return output[0]
        if isinstance(output, Tensor):
            return output
        raise TypeError("Model output must contain binary logits")

    def _regularization_loss(self, images: Tensor) -> Tensor:
        zero = torch.zeros((), device=self.device)

        if self.regularizer == "ewc":
            if self._ewc_anchor is None or self._fisher is None:
                return zero
            penalty = zero
            for name, parameter in self.model.named_parameters():
                if name in self._ewc_anchor:
                    penalty = (
                        penalty
                        + (
                            self._fisher[name]
                            * (parameter - self._ewc_anchor[name]).square()
                        ).sum()
                    )
            return 0.5 * self.regularization_strength * penalty

        if self._teacher is None:
            return zero

        self._teacher.eval()
        with torch.no_grad():
            teacher_logits = self._binary_logits(self._teacher(images))

        student_logits = self._binary_logits(self.model(images))
        temperature = self.distillation_temperature
        return (
            nn.functional.kl_div(
                nn.functional.log_softmax(student_logits / temperature, dim=1),
                nn.functional.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            )
            * (temperature**2)
            * self.regularization_strength
        )

    def _estimate_fisher(self, experience: Any) -> None:
        """Estimate diagonal Fisher information on the current experience."""
        self.model.eval()
        fisher = {
            name: torch.zeros_like(parameter, device=self.device)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

        dataset = experience.dataset
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

        seen = 0
        for batch in loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            self.model.zero_grad(set_to_none=True)
            logits = self._binary_logits(self.model(images))
            loss = nn.functional.cross_entropy(logits, labels.long())
            loss.backward()

            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None and name in fisher:
                    fisher[name] += parameter.grad.detach().square()

            seen += 1
            if seen >= self.max_memory_samples:
                break

        divisor = max(seen, 1)
        self._fisher = {name: value / divisor for name, value in fisher.items()}
        self._ewc_anchor = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self._fisher
        }
        LOGGER.info("Estimated EWC Fisher diagonal from %d samples", seen)

    def _snapshot_teacher(self) -> None:
        self._teacher = copy.deepcopy(self.model).to(self.device)
        self._teacher.eval()
        for parameter in self._teacher.parameters():
            parameter.requires_grad_(False)

    def train_experience(
        self,
        experience: Any,
        *,
        experience_id: str | int,
        epochs: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Path:
        """Train on one new experience and save the resulting checkpoint.

        Args:
            experience: Avalanche ``Experience`` or compatible object exposing
                a map-style ``dataset`` yielding image and label pairs.
            experience_id: Identifier included in the checkpoint filename.
            epochs: Number of passes over the new experience.
            batch_size: New-experience minibatch size.
            num_workers: DataLoader worker count.

        Returns:
            Path to the saved checkpoint.
        """
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not hasattr(experience, "dataset"):
            raise TypeError("experience must expose a dataset attribute")

        if self.regularizer == "lwf":
            self._snapshot_teacher()

        loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0

            for batch in loader:
                new_images, new_labels = batch[0].to(self.device), batch[1].to(
                    self.device
                )
                images, labels = new_images, new_labels.long()

                replay_count = min(
                    self.replay_batch_size,
                    len(self.replay_buffer),
                )
                if replay_count:
                    replay_samples = self.replay_buffer.select_replay_batch(
                        replay_count
                    )
                    if replay_samples:
                        replay_images, replay_labels = self.replay_batch_loader(
                            replay_samples,
                            self.device,
                        )
                        images = torch.cat(
                            (images, replay_images.to(self.device)),
                            dim=0,
                        )
                        labels = torch.cat(
                            (labels, replay_labels.to(self.device).long()),
                            dim=0,
                        )

                self.optimizer.zero_grad(set_to_none=True)
                logits = self._binary_logits(self.model(images))
                task_loss = nn.functional.cross_entropy(logits, labels)
                total_loss = task_loss + self._regularization_loss(images)
                total_loss.backward()
                self.optimizer.step()

                running_loss += float(total_loss.detach())
                steps += 1

            mean_loss = running_loss / max(steps, 1)
            LOGGER.info(
                "Experience %s epoch %d/%d loss=%.4f",
                experience_id,
                epoch + 1,
                epochs,
                mean_loss,
            )

        if self.regularizer == "ewc":
            self._estimate_fisher(experience)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"forensic_{experience_id}.pt"
        torch.save(
            {
                "experience_id": str(experience_id),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "regularizer": self.regularizer,
                "fisher": self._fisher,
                "ewc_anchor": self._ewc_anchor,
            },
            checkpoint_path,
        )
        LOGGER.info("Saved updated model checkpoint to %s", checkpoint_path)
        return checkpoint_path


class EvaluationRegressionGate:
    """Compare pre-update and post-update accuracy across evaluation groups.

    Expected evaluation mapping keys:
    - ``historical_real``
    - ``historical_sdxl``
    - ``historical_flux``
    - ``new_generator``
    - ``unseen_firefly``
    - ``unseen_midjourney``

    Each value is an iterable of batches yielding ``(images, binary_labels)``.
    """

    REQUIRED_GROUPS = (
        "historical_real",
        "historical_sdxl",
        "historical_flux",
        "new_generator",
        "unseen_firefly",
        "unseen_midjourney",
    )

    def __init__(
        self,
        *,
        min_stability: float = 0.90,
        max_forgetting_rate: float = 0.05,
        device: torch.device | str | None = None,
    ) -> None:
        self.min_stability = min_stability
        self.max_forgetting_rate = max_forgetting_rate
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @staticmethod
    def _logits(output: Any) -> Tensor:
        if isinstance(output, Mapping):
            return output["binary_logits"]
        if isinstance(output, (tuple, list)):
            return output[0]
        if isinstance(output, Tensor):
            return output
        raise TypeError("Model output must contain binary logits")

    @torch.inference_mode()
    def _accuracy(self, model: nn.Module, loader: Iterable[Any]) -> float:
        model.eval()
        correct = 0
        total = 0

        for batch in loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            predictions = self._logits(model(images)).argmax(dim=1)
            correct += int((predictions == labels.long()).sum())
            total += int(labels.numel())

        return correct / total if total else float("nan")

    def evaluate(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        evaluation_sets: Mapping[str, Iterable[Any]],
    ) -> dict[str, float | bool]:
        """Compute regression metrics and return the gate decision."""
        missing = set(self.REQUIRED_GROUPS) - set(evaluation_sets)
        if missing:
            raise ValueError(f"Missing evaluation groups: {sorted(missing)}")

        model_before = model_before.to(self.device)
        model_after = model_after.to(self.device)

        group_accuracies: dict[str, tuple[float, float]] = {}
        for group in self.REQUIRED_GROUPS:
            loader = evaluation_sets[group]
            before_accuracy = self._accuracy(model_before, loader)
            after_accuracy = self._accuracy(model_after, loader)
            group_accuracies[group] = (before_accuracy, after_accuracy)

        historical_groups = (
            "historical_real",
            "historical_sdxl",
            "historical_flux",
        )
        old_before = sum(group_accuracies[g][0] for g in historical_groups) / len(
            historical_groups
        )
        old_after = sum(group_accuracies[g][1] for g in historical_groups) / len(
            historical_groups
        )
        stability = old_after / old_before if old_before > 0 else 0.0
        forgetting = old_before - old_after
        plasticity = group_accuracies["new_generator"][1]
        generalization = (
            sum(group_accuracies[g][1] for g in ("unseen_firefly", "unseen_midjourney"))
            / 2.0
        )

        pass_gate = (
            stability >= self.min_stability and forgetting <= self.max_forgetting_rate
        )

        LOGGER.info(
            "Regression metrics: stability=%.4f, plasticity=%.4f, "
            "generalization=%.4f, forgetting_rate=%.4f",
            stability,
            plasticity,
            generalization,
            forgetting,
        )
        for group, (before, after) in group_accuracies.items():
            LOGGER.info(
                "  %-20s before=%.4f after=%.4f",
                group,
                before,
                after,
            )
        LOGGER.info(
            "Regression gate: %s (stability >= %.2f; forgetting <= %.2f)",
            "PASS" if pass_gate else "FAIL",
            self.min_stability,
            self.max_forgetting_rate,
        )

        return {
            "stability": stability,
            "plasticity": plasticity,
            "generalization": generalization,
            "forgetting_rate": forgetting,
            "pass_gate": pass_gate,
        }
