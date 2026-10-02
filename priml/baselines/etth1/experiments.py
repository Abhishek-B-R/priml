"""ETTh1 long-horizon forecasting experiments."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Self, cast, override

from configgle import Makes, PartialConfig

from priml.baselines.etth1.checkpointer import Etth1Checkpointer
from priml.baselines.etth1.data import Etth1Data
from priml.baselines.etth1.metrics import ForecastMSE
from priml.baselines.etth1.train_step import Etth1TrainStep
from priml.runtime import SingleProcess
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from collections.abc import Mapping


def dlinear_type1(progress: float, *, epochs: float = 10) -> float:
    """Match the reference DLinear type1 learning-rate schedule."""
    epoch = int(progress * epochs + 1e-9)
    drops = max(0, epoch - 1)
    return 0.5**drops


class Etth1TrainLoop(TrainLoop):
    """Training loop reproducing the reference DLinear stopping rule."""

    class Config(
        Makes["Etth1TrainLoop"],
        TrainLoop.Config[Etth1TrainStep.Config, Etth1Data.Config],
    ):
        """Training loop with ETTh1 and DLinear installed."""

        step: Etth1TrainStep.Config = field(
            default_factory=Etth1TrainStep.Config,
        )
        """DLinear model and optimization recipe."""

        dataset: Etth1Data.Config = field(
            default_factory=Etth1Data.Config,
        )
        """ETTh1 forecasting dataset."""

        patience: int = 3
        """Reference early-stopping patience in validation epochs."""

        @override
        def finalize(self) -> Self:
            self.step.model.seq_len = self.dataset.seq_len
            self.step.model.pred_len = self.dataset.pred_len
            self.step.model.channels = self.dataset.channels
            if self.step.seed is None:
                self.step.seed = self.seed
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.patience <= 0:
            raise ValueError("Early-stopping patience must be positive.")
        self.patience = config.patience
        self.best_validation_loss = float("inf")
        self.bad_validation_epochs = 0
        self.stop_early = False
        self.validation_losses: list[float] = []
        super().__init__(config)

    @override
    def _publish_eval_metrics(
        self,
        eval_metrics: dict[str, object],
        *,
        eval_time: float,
        step: int,
        is_final: bool,
    ) -> dict[str, float]:
        scalar_metrics = super()._publish_eval_metrics(
            eval_metrics,
            eval_time=eval_time,
            step=step,
            is_final=is_final,
        )

        validation_loss = scalar_metrics["total_loss"]
        self.validation_losses.append(validation_loss)

        if validation_loss <= self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.bad_validation_epochs = 0
        else:
            self.bad_validation_epochs += 1

        if self.bad_validation_epochs >= self.patience:
            self.stop_early = True

        return scalar_metrics

    @override
    def _should_stop_early(self) -> bool:
        return self.stop_early

    @override
    def _on_epoch_boundary(self) -> None:
        super()._on_epoch_boundary()

        # Save after validation so a resume starts after the full epoch.
        if self.checkpointer is not None and self.local_step > 0:
            self.checkpointer.save(self, self.step.global_step)

        if self.stop_early:
            self._terminal_epoch_evaluated = True
            raise StopIteration

    class StateDict(TrainLoop.StateDict):
        """Training state plus the reference early-stopping decision."""

        best_validation_loss: float
        bad_validation_epochs: int
        stop_early: bool
        validation_losses: list[float]

    @override
    def state_dict(self) -> StateDict:
        """Save stopping history together with model, loader, optimizer, and RNG."""
        return {
            **super().state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "bad_validation_epochs": self.bad_validation_epochs,
            "stop_early": self.stop_early,
            "validation_losses": list(self.validation_losses),
        }

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore stopping history before continuing a saved run."""
        if "best_validation_loss" not in state_dict:
            raise ValueError(
                "This legacy ETTh1 checkpoint lacks resume state. "
                "Evaluate it with scripts.evaluate or start in a new run directory.",
            )
        super().load_state_dict(state_dict)
        state = cast(Etth1TrainLoop.StateDict, state_dict)
        self.best_validation_loss = state["best_validation_loss"]
        self.bad_validation_epochs = state["bad_validation_epochs"]
        self.stop_early = state["stop_early"]
        self.validation_losses = list(state["validation_losses"])


def exp000() -> Etth1TrainLoop.Config:
    """DLinear on ETTh1 with a 336-step history and 96-step forecast.

    This is the canonical forecasting baseline.

    Hypothesis:
      A decomposition-linear model provides a compact canonical baseline for
      long-horizon multivariate forecasting and a fast target for automated
      experiment hillclimbing.

    Returns:
      cfg: Canonical ETTh1 DLinear training configuration.

    References:
      https://arxiv.org/abs/2205.13504
      Zeng et al. 2023. Are Transformers Effective for Time Series Forecasting?

    Results:
      Reference-hardware benchmark: TBD. See README.md for the separately
      recorded local CPU reproduction and source-parity evidence.

    """
    cfg = Etth1TrainLoop.Config()

    cfg.study_name = "etth1"
    cfg.experiment_name = "exp000"
    cfg.seed = 2021

    steps_per_epoch = (
        cfg.dataset.train_rows - cfg.dataset.seq_len - cfg.dataset.pred_len + 1
    ) // cfg.dataset.batch_size

    cfg.max_epochs = 10
    cfg.max_steps = 10 * steps_per_epoch

    cfg.step.train_budget_epochs = cfg.max_epochs
    cfg.step.lr_schedule = PartialConfig(dlinear_type1, epochs=cfg.max_epochs)

    cfg.num_steps_eval = float("inf")
    cfg.eval_every_epoch = True

    cfg.num_steps_log = 100
    cfg.early_train_log_steps = 0

    cfg.metrics_eval[""] = ForecastMSE.Config()

    cfg.checkpointer = Etth1Checkpointer.Config()
    # The epoch hook saves after validation.
    cfg.checkpointer.save_every = cfg.max_steps
    cfg.checkpointer.keep_last_n = 3
    cfg.checkpointer.best_metric = "total_loss"
    cfg.checkpointer.best_mode = "min"

    cfg.runtime = SingleProcess.Config(device="cpu")

    return cfg


def exp_smoke() -> Etth1TrainLoop.Config:
    """Tiny ETTh1 DLinear run for end-to-end integration checking.

    Returns:
      cfg: Minimal forecasting run that exercises model, data, loss,
      optimization, and evaluation.

    Hypothesis:
      A short, narrow run exercises the canonical data and training wiring.

    References:
      exp000.

    Results:
      Integration only; not a forecasting quality benchmark.

    """
    cfg = exp000()

    cfg.experiment_name = "exp_smoke"

    cfg.max_epochs = 1
    cfg.max_steps = 3
    cfg.dataset.seq_len = 5
    cfg.dataset.pred_len = 3
    cfg.dataset.batch_size = 2
    cfg.dataset.eval_batch_size = 2

    cfg.num_steps_eval = cfg.max_steps
    cfg.eval_every_epoch = False

    cfg.checkpointer = None

    return cfg
