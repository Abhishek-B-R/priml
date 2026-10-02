# ETTh1 forecasting

`exp000` reproduces the multivariate DLinear ETTh1 recipe: 336 hours of
history, a 96-hour forecast, seven channels, moving average width 25,
Adam at `1e-4`, batch size 8, seed 2021, and the source `type1` learning-rate
schedule. The budget is at most 10 epochs / 10,260 updates; validation
patience is 3 epochs. The source's unused decoder is retained because its
initialization consumes RNG draws.

The target is held-out MSE in training-standardized units. Training uses the
first 8,640 rows; validation and test each cover the next 2,880 rows with
history overlapping the preceding split. Normalization fits training rows
only. Training and validation shuffle; test is ordered. All loaders drop the
last incomplete batch, matching the reference, so test scoring covers 2,784
windows. Validation averages batch losses in float32. Test MSE/MAE use the
source's float32 NumPy reductions over concatenated forecasts.

## Run locally

Run from the PRIML repository with its frozen environment. `base_dir` selects
a portable resource root; the example uses the git-ignored `.scratch/`.
Choose a fresh base directory if you have an older checkpoint there.

```sh
uv sync --frozen
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.prepare_data \
  --directory .scratch/datasets/etth1
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  uv --quiet run --frozen python -m priml \
  priml.baselines.etth1.experiments.exp000 --override base_dir=.scratch
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.evaluate \
  --directory .scratch/datasets/etth1 \
  --checkpoint .scratch/runs/etth1/exp000/checkpoints
```

Preparation downloads a pinned ETDataset revision and verifies SHA-256 before
installing the CSV atomically. To prepare from an existing copy, add
`--source /path/to/ETTh1.csv`. A corrupt existing destination fails explicitly.
On macOS installations without a configured Python certificate bundle,
`SSL_CERT_FILE=/etc/ssl/cert.pem` selects the system trust store.

Checkpoints preserve sampler position, optimizer, learning-rate progress,
early-stopping history, and RNG state. Epoch checkpoints are saved after
validation. Equal validation scores select the later checkpoint, as in the
source. Old checkpoints that lack this resume state can still be scored with
`scripts.evaluate`; resume from them fails with an explanatory message.

## Source verification and goldens

The reference is the user's [ql-denoising/DLinear checkout](https://github.com/ql-denoising/DLinear)
at `da9e67442b95af76488b8e4e1806cc3185723dd9`. Source files and their checksums
are recorded in `testdata/source.json`. The only accepted compatibility edit
is `np.Inf` to `np.inf` in `utils/tools.py`, for NumPy 2. The reference checkout
is otherwise read-only during verification.

```sh
# Run with the reference's pandas, scikit-learn and matplotlib dependencies
# available in the invoking environment; normal baseline use needs none of them.
uv --quiet run --frozen python -m priml.baselines.etth1.scripts.verify_reference \
  --reference /path/to/DLinear --directory .scratch/datasets/etth1
```

The verifier compares native and portable tiny traces, then three canonical
real-data updates, with exact equality for initialization, batches, predictions,
losses, gradients, parameters, Adam state and RNG position. The tiny probe uses
distinct dimensions (batch 2, history 5, horizon 3, channels 4) and three updates
across the first learning-rate decay. Add `--mint` only when intentionally
recording verified **source** tensors. It never mints from the port. Model and
training goldens are each below 32 KiB and replay through PRIML's portable
numerics harness without a source checkout or downloaded dataset.

```sh
uv --quiet run --frozen pytest priml/baselines/etth1 -o addopts= -q
uv --quiet run --frozen pytest priml/baselines/etth1 -o addopts= \
  --cov=priml/baselines/etth1 --cov-report=term-missing
```

Tests also check that a one-ULP weight change and a changed learning rate fail
the goldens, split boundaries and normalization, source loader RNG behavior,
resume equivalence, latest-tie checkpoint selection, pure config propagation,
and model compute-cost accounting. Tiny and canonical model/train-step probes
execute identical lines and branches.

## Local reproduction

The local CPU run on 2026-10-02 (Apple Silicon, Python 3.12.3, PyTorch 2.11.0,
NumPy 2.5.3, one Torch thread) stopped after 5 epochs / 5,130 updates. The best
checkpoint was step 2,052, with validation MSE **0.6461290121**. Its test MSE
was **0.3748246133** and MAE **0.3994735181**.

A complete run of the original training loop produced identical validation
losses, all retained epoch parameters, best-model parameters, final Torch RNG,
and all **1,870,848** test prediction elements. `results/local_cpu.json`
records the comparison. These are local reproduction results; the
reference-hardware benchmark remains TBD.

For hillclimbing, add `exp001()` by deriving from `exp000()` and making one
explicit change. Keep the split, normalization, MSE definition, seed, validation
selection and maximum compute budget fixed; report actual updates and runtime.
Do not tune against the held-out test split. Follow the repository skill's
hypothesis / references / results convention for each experiment.
