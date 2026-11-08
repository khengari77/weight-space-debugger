# Weight edit engineering, acceptance gates, and audit trails

## An edit is an artifact with a proof obligation

Never edit the only copy of a checkpoint. Prefer `torch.func.functional_call` with a parameter dictionary containing only cloned tensors that change, or load a second in-memory model when manageable. Before modification, compute full model/file identifiers and exactly which parameter objects are shared. Keep a machine-readable delta manifest. Re-apply to a fresh copy and rerun smoke tests.

A suggested edit MUST specify:

- Base checkpoint ID/hash and architecture/revision/dtype/quantization.
- Exact parameter names, indices, mathematical transformation, shape, dtype, orientation, and any ties to other tensors.
- Changed-element count, % of model, `||Δ||_2 / ||W||_2`, nonfinite check and maximum coordinate change.
- Behavioral and mechanistic prediction, negative controls, preserved capability targets and test budget.
- How to revert and how to verify no other tensors changed.

`patching.py` implements a small set of safe `set`, `add`, and `scale` operations to an isolated cloned tensor; it is a **primitive**, not an autonomous semantic editor. Only apply it to trusted model tensors after documenting what the coordinates mean. For complex model sharding/tied weights, write and independently test an architecture-specific adapter instead of using generic sparse assignment.

The helper counts changes in the original dtype and accumulates norms in bounded
float64 chunks. It clones edited tensors, but unchanged parameters and buffers
still alias the baseline. `functional_call` does not make those aliases read-only.
Use a disposable instance for stateful/training forwards; otherwise verify the
entire baseline state and existing gradients remain unchanged after the experiment.

## Acceptance metrics

Let `S_base` be baseline target performance and `S_edit` edited performance. Let `P_base` and `P_edit` be performance on preservation tasks. Agree on thresholds `δ_gain>0`, `ε_preserve>=0` *before* testing. Accept only when:

1. `S_edit - S_base >= δ_gain` with uncertainty bound suitable for the sample size;
2. `P_edit >= P_base - ε_preserve` across predefined preserved tasks;
3. no critical safety/robustness/behavioral regressions, especially outside the local edit task;
4. repeated independent runs and a fresh reload reproduce the improvement;
5. model, weights, evaluation and data provenance are recorded.

For LLM factual edits also measure paraphrase generalization, specificity/locality, portability into dependent reasoning, other-task retention, and repeated-edit accumulation. For RL measure unseen seeds, domain randomization and episode-level safety constraints. For classifiers measure balanced subsets and calibration, not just headline accuracy.

## Negative controls and ablations

- **Sham edit**: identical copying without changing weights or outputs.
- **Equal-footprint random edit**: same parameter count and norm in comparable eligible tensors.
- **Random direction edit**: same norm but random orthogonal/different direction.
- **Non-weight baseline**: stop/tokenizer fix, decoding change, input preprocessing or small transparent policy.
- **Dose curve**: `α ∈ {0, 0.25, 0.5, 1.0, 1.5}` if safe; a monotonic useful trend can help.
- **Ablation**: what happens if only a subset of the patch is included?
- **Red-team preservation tests**: if behavior is changed, what unseen tasks degrade?

Ensure comparisons are properly powered and report confidence intervals. Multiple candidate edits tried on the same development samples must be counted; otherwise improvements may be overfit. Do not publish claims based on sealed test-driven iteration.

## Stop / rollback triggers

Revert upon any prohibited capability loss, invalid outputs, nonfinite tensors, cross-device/tied-weight inconsistency, inability to reproduce original baseline, unexplained scaling of effects with quantization, or evaluation contamination. Keep *rejected edits* in the experiment log, not in the released model.

Edits to LLMs can cause gradual or catastrophic forgetting when applied repeatedly; test cumulative effects rather than assuming sparse changes are independent.

Research: https://aclanthology.org/2024.findings-acl.902/
PyTorch functional_call: https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html
