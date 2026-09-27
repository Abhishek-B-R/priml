r"""Reproduce the ARC-AGI-1 blog-post experiment with public Priml.

From the Priml repository root, set up the environment and exp004 data:
  uv sync --all-groups
  priml/baselines/arcagi1/scripts/prepare_data.py --experiment exp004

Train from scratch on eight visible H200 GPUs and score at step 280,000:
  uv run torchrun --standalone --nproc_per_node=8 \
    priml/baselines/arcagi1/scripts/reproduce_blog_post.py
On successful completion, Priml scores first, then writes a distributed
checkpoint directory at
  /opt/scratch/runs/arcagi1/exp004_blog_8gpu/checkpoints/step_00280000.pt/

To evaluate the recovered internal 8xH100 checkpoint instead, run with
  --checkpoint /path/to/historical-hps-step280000.pt
This option accepts only the SHA-pinned historical archive, not a new Priml
checkpoint produced by the training command above.

To train on one GX10/GB10 (128 GB shared memory):
  uv run python priml/baselines/arcagi1/scripts/reproduce_blog_post.py
Its final checkpoint is the native file
  /opt/scratch/runs/arcagi1/exp004_blog_gx10/checkpoints/step_00280000.pt
The GX10 path was smoke-tested for two optimizer steps. It uses one GPU and
batch 8 instead of 8x96; at 280,000 steps it sees 96x fewer examples, so its
score is not like-for-like. A full GX10 run and this launcher's eight-H200 run
have not been completed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import argparse
import hashlib
import logging
import os

import torch

from priml.baselines.arcagi1.experiments import HPSArcTrainLoop, exp004
from priml.runtime import SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import NoParallel
from priml.train.tracker import FileTracker


if TYPE_CHECKING:
    from priml.baselines.sudoku.prefix import PrefixStack, SparsePuzzleEmbedding

STEP = 280_000
SHA256 = "64bd795f8dc984d7e647be155afc098bb41c7a8c22daeac36f57c4ad76fc1cff"
RENAMES = {
    "embed_tokens.weight": "embedding.embed_tokens.weight",
    "embed_feedback": "embedding.channels.0.embed_feedback",
    "q_head.weight": "halt_head.weight",
    "q_head.bias": "halt_head.bias",
    "puzzle_emb.weights": "prefix.parts.0.weights",
}


def _overlay(archive: dict[str, object], into: dict[str, object]) -> dict[str, object]:
    """Map historical model and EMA onto a fresh eval-only Priml state."""
    old = cast("dict[str, object]", archive["step"])
    fresh = cast("dict[str, object]", into["step"])
    source = cast("dict[str, torch.Tensor]", old["model"])
    old_ema = cast("dict[str, torch.Tensor]", old["ema"])
    model = {RENAMES.get(name, name): value for name, value in source.items()}
    ema = {RENAMES.get(name, name): value for name, value in old_ema.items()}
    fresh["model"] = model
    fresh["ema"] = {"shadow_params": ema, "global_step": STEP}
    fresh["timer_step"] = {"global_count": STEP, "global_sec": 0.0}
    return into


def recipe(*, single_gpu: bool, checkpoint: bool) -> HPSArcTrainLoop:
    """Set the same model recipe for distributed or single-GPU execution."""
    cfg = exp004()
    cfg.experiment_name = "exp004_blog_gx10" if single_gpu else "exp004_blog_8gpu"
    cfg.max_steps = STEP
    cfg.max_time = float("inf")
    cfg.num_steps_eval = -1  # Final scoring uses the trained EMA in memory.
    cfg.eval_warmup_batches = 0
    cfg.tracker = FileTracker.Config()
    assert isinstance(cfg.checkpointer, Checkpointer.Config)
    cfg.checkpointer.resume = False
    # Keep exp004's model and optimizer; only adapt runtime and batch to GB10.
    if single_gpu:
        cfg.runtime = SingleProcess.Config(device="cuda")
        cfg.step.parallelism = NoParallel.Config(device="cuda")
        cfg.step.batch_size = cfg.dataset.batch_size = 8
        cfg.dataset.eval_batch_size = 32
        prefix = cast("PrefixStack.Config", cfg.step.model.prefix)
        cast("SparsePuzzleEmbedding.Config", prefix.parts[0]).batch_size = 8
    if checkpoint:
        cfg.experiment_name += "_historical_eval"
        cfg.eval_only = True
        cfg.checkpointer = None
    return cfg


def main() -> None:
    """Train and score a new model, or score the historical archive."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, help="SHA-pinned historical archive")
    path = cast("Path | None", parser.parse_args().checkpoint)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (world_size, torch.cuda.device_count()) not in {(1, 1), (8, 8)}:
        raise SystemExit("Use one CUDA GPU or torchrun with eight CUDA GPUs")
    cfg = recipe(single_gpu=world_size == 1, checkpoint=path is not None)
    logging.basicConfig(level=logging.INFO)
    if path is not None:
        with path.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != SHA256:
                raise SystemExit(f"Historical checkpoint SHA256 must be {SHA256}")
    loop = cfg.make()
    if path is not None:
        archive = cast("dict[str, object]", torch.load(path, "cpu", weights_only=True))
        loop.load_state_dict(_overlay(archive, dict(loop.state_dict())))
    loop.run()


if __name__ == "__main__":
    main()
