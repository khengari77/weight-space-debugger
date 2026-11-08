# Training-path debugging: make the simplest thing learn

Before attempting exotic weight analysis, determine whether the model is failing because the *training process* is incorrect. This file is for access to the training loop, data pipeline, losses and optimizer. It does not recommend writing arbitrary inference weights as a substitute for repairing broken training.

## Preserve the investigation baseline

Training checks are conditional on the requested investigation and allowed compute.
Access to training code alone does not authorize optimizer steps. In read-only
forensics, inspect code, saved logs and existing gradients without running training.
Skip these checks when training, backward passes or gradients are prohibited by
the experiment (including weight-only and strictly gradient-free investigations).

When training diagnostics are in scope, use a disposable model instance loaded
from the same trusted checkpoint, or a deep copy when supported. Create a separate
optimizer bound only to that instance; do not reuse the baseline optimizer or its
state. Run train-mode forward passes, backward passes and updates only on this
instance. A shallow state dictionary or `functional_call` with aliased buffers is
not sufficient isolation: BatchNorm and custom forwards can update buffers.

Snapshot and compare the baseline's complete state (parameters and buffers),
existing `.grad` values, per-module training flags and any existing optimizer state.
Isolate RNG effects with `torch.random.fork_rng` (including relevant CUDA devices)
and save/restore any Python or NumPy RNGs used by the harness. Reject the run if the
baseline changed. If a model cannot be safely copied within budget, report that
training diagnostics were skipped and continue with the available evidence.

## Minimal executable sanity ladder

1. **One example forward pass.** Print exact input/target shapes, ranges, dtype, device, and a few human-interpretable samples. Verify label/action semantics by hand. Check for leakage and accidental transforms (channel order, unnormalized values, token masking).
2. **One batch, one train step.** Inspect loss terms, logits, gradients, optimizer parameter coverage and weight change. Ensure every intended trainable parameter participates; inspect `requires_grad`, detached tensors, gradient accumulation/zeroing, no-grad scopes and optimizer groups.
3. **Overfit one tiny batch.** If a moderately expressive model cannot memorize 8–32 examples, suspect implementation/data/loss before optimization tricks. This test is not a generalization test; it is a pipeline sanity test.
4. **Increase complexity one dimension at a time.** Fix the seed and compare learning curves when adding augmentation, dropout, regularization, multiple workers, mixed precision, gradient clipping, distribution, quantization and schedules.
5. **Only then** investigate generalization: independent holdout, leakage, label noise, data shift, under/overfit, calibration, distribution robustness, imbalance and loss-metric mismatch.

## Numerical checks

- Use anomaly detection or diagnostic hooks only where necessary; they have runtime cost.
- Log loss, gradient global/per-module norms and finite fraction, update-to-weight ratio, activation distributions, learning rate, optimizer state. Include zero gradients and missing gradients distinctly.
- Track scaled vs unscaled gradients under AMP; avoid claiming a gradient is zero when the scaler skipped the update.
- BatchNorm/Dropout train/eval modes change outputs. Compare frozen inference with training mode deliberately.
- Beware hidden state or KV cache carried across independent sequences. RL recurrent policy state resets are essential.
- If divergence first appears under `torch.compile` or mixed precision, compare a minimal eager FP32 run with identical inputs, then isolate the first mismatching intermediate.

## Gradient health and update statistics

For an update step, a practical statistic is `||ΔW||_2 / (||W||_2 + ε)` per parameter group. Record whether weight decay, momentum, frozen parameters or normalization mean the interpretation is ambiguous. Large/small values are not automatically bad; compare against training-phase baselines and stability.

If the loss doesn't improve on the tiny batch, test: target alignment, loss reduction (`sum` vs `mean`), sign for gradient ascent/descent, initialization/symmetry, final layer and activation compatibility, optimizer step placement, accidental `detach()`, incorrect broadcasting, masking all loss positions, numerical underflow, bad clip thresholds.

## RL training-specific sanity

Plot mean episode return vs success rate; a rising shaped reward can coexist with a failing task. Inspect action mask, environment reset, advantage normalization, entropy, clipping, KL and value function fit. Run a deterministic 'tiny MDP' with known optimal behavior, and test with a scripted policy before scaling simulator throughput.

## A hard stop

If a one-batch overfit check fails due to a confirmed training defect, repair it when that is within the requested scope; otherwise report it. Establish a new baseline before attempting circuit surgery. Do not assert a meaningful 'bad neuron' discovered inside a model trained with corrupted labels or invalid actions. Keep training issues distinct from learned-but-undesirable policy strategies.

PyTorch numerical debugging references:
https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection
https://docs.pytorch.org/tutorials/recipes/debug_mode_tutorial.html
