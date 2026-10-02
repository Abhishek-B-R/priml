"""Training step for the ETTh1 DLinear baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, cast, override

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor

import torch

from priml.baselines.etth1.model import DLinear
from priml.loss.custom_types import LossOutput
from priml.loss.simple_loss import SimpleLoss, mse
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.train.custom_types import TrainStepOutput


class Etth1TrainStep(TrainStep):
    """DLinear model plus the reference ETTh1 optimization recipe."""

    class Config(
        Makes["Etth1TrainStep"],
        TrainStep.Config[DLinear.Config],
        kw_only=True,
    ):
        """Model and optimizer for the canonical DLinear experiment."""

        model: DLinear.Config = field(default_factory=DLinear.Config)
        """DLinear forecasting model."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(
                torch.optim.Adam,
                lr=1e-4,
            ),
        )
        """Adam optimizer matching the reference DLinear recipe."""

        loss: Makeable[Callable[..., LossOutput]] = field(
            default_factory=lambda: SimpleLoss.Config(
                loss_fn=mse,
                kwargs={"reduction": "mean"},
            ),
        )
        """Mean squared error, reduced in the same order as the reference."""

        seed: int | None = None
        """Unsalted reference Torch seed; inherited from the loop when specified."""

        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """The canonical reference uses eager forward and backward."""

    def __init__(self, config: Config) -> None:
        """Initialize with the reference DLinear Torch RNG stream."""
        if config.seed is not None:
            # The reference seeds Torch directly.
            torch.manual_seed(config.seed)
        super().__init__(config)

    @property
    def net(self) -> DLinear:
        """Return the forecasting model under its concrete type."""
        return cast(DLinear, self.model)

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one DLinear optimization step."""
        media = batch["media"]
        assert isinstance(media, Tensor)

        label = batch["label"]
        assert isinstance(label, Tensor)

        self.model.train()
        self.optimizer.zero_grad()

        prediction = self.net(media)
        loss = self.loss(prediction, **batch)["loss"]

        loss.backward()

        with self.timer_step:
            self.apply_learning_rate()
            self.optimizer.step()

        return {
            "loss": loss.detach().reshape(1),
            "model": prediction.detach(),
        }

    @override
    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Compute training loss without updating parameters."""
        media = batch["media"]
        assert isinstance(media, Tensor)

        label = batch["label"]
        assert isinstance(label, Tensor)

        self.model.train()

        with torch.no_grad():
            prediction = self.net(media)
            loss = self.loss(prediction, **batch)["loss"]

        return {
            "loss": loss.detach().reshape(1),
            "model": prediction.detach(),
        }

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Compute forecasting MSE without updating parameters."""
        media = batch["media"]
        assert isinstance(media, Tensor)

        label = batch["label"]
        assert isinstance(label, Tensor)

        self.model.eval()

        with torch.no_grad():
            prediction = self.net(media)
            loss = self.loss(prediction, **batch)["loss"]

        return {
            "loss": loss.detach().reshape(1),
            "model": prediction.detach(),
        }
