# Executable investigation protocol

Use this workflow when inference is available and a behavioral defect is the task.
For checkpoint-only requests, use [static forensics](09-static-forensics.md) instead.
Read-only investigations stop before interventions; causal debugging permits only
the authorized ephemeral interventions; persistent repairs require weight-surgery scope.
Training diagnostics are a separate conditional activity, never a read-only step.


### 0. Scope, permissions, and budget

Obtain or infer: model path/ID and revision; architecture and framework; hardware/memory; allowed inference/data; objective and failure example; whether gradients are accessible; edit permission; desired preservation constraints. If missing, proceed with read-only work that does not require them, listing assumptions. Never download untrusted model code or load pickle checkpoints as a side effect. Prefer `safetensors`, `trust_remote_code=False`, offline checks where possible.

Write a one-paragraph investigation contract: *observed defect*, *measurable target*, *protected behaviors*, *resource budget*, *allowed intervention*, *stop criteria*. Use an explicit experiment budget; scale from cheapest probe upward. Do not export private inputs or model weights to a third-party model without permission. Snapshot version and environment; save immutable input hashes.

### 1. Reproduce behavior before analyzing weights

- Freeze checkpoint/revision, preprocessing, prompts, chat template, generation parameters, environment, precision/device, seed, and scoring implementation. `model.eval()` for inference, except when investigating training.
- Build three disjoint sets: **diagnostic** (hypothesis discovery), **development** (edit selection), **sealed test** (final acceptance). For RL: independent environment seeds and randomized initial states. Never tune to the sealed test.
- Separate a reproducible *failure* from an anecdote. Record failure rate, error distribution, representative full traces, and counterexamples. For LLMs record token budget, reasoning-loop rate, repetition, stop reasons, prompt/tokenization differences, and accuracy. For RL record rollouts and termination causes.
- Check trivial causes first: evaluation bug, leakage, invalid action mapping, state normalization, reward shaping, sampling, stop tokens, cache inconsistencies, mixed precision, train/eval mode, bad labels, distribution shift, quantization, data transforms. Inspect training code and logs when relevant. Run a one-batch training check only when training diagnostics are in scope, on a disposable model with a separate optimizer; preserve the baseline state, gradients, modes and RNG. Skip it if gradients or training are prohibited; see [training isolation](07-training-debugging.md).
- Compare a no-op rerun and at least one non-weight alternative (decoding/config/preprocessing/environment change) before considering weight surgery.

**Gate A:** If no reliably measured failure exists, do not assert a mechanistic root cause. Report how to reproduce it.

### 2. Map the model and run cheap integrity checks

Create a compact **model map**: input/output semantics, token/channel/layout conventions, module graph, parameter and buffer names/shapes/dtypes, tied/shared tensors, weight quantization, residual branches, normalization, heads and MLP structure, checkpoint ancestry.

Run `scripts/checkpoint_audit.py` on a trusted `.safetensors` checkpoint (see `--help`). It hashes the file and summarizes tensor anomalies. Use `scripts/probe.py` for bounded activation/gradient summaries during executable runs. Inspect in order:

1. Load/shape/config mismatches, NaNs/Infs, saturated or dead channels, accidental zeroed layers, explosions, normalization and clipping errors.
2. Layer-wise norms and distribution **relative to comparators**, not arbitrary thresholds; across versions, report deltas only with matching parameter semantics.
3. Active computational paths: representative hidden activations, attention head outputs, MLP writes, logit contributions, gradients only where accessible and relevant.
4. Domain-specific analyses via `03-transformers.md` or `04-rl-and-control.md`.

**Never send all large-model weights to the reasoning model.** Summarize, prioritize interesting components, retrieve small slices and structured statistics on demand. Count I/O, VRAM, and runtime costs.

### 3. Generate a falsifiable hypothesis ledger

For each candidate failure mechanism record:

- observed evidence and exact cases;
- mechanism with precise component, tensor orientation, token position/state and computation;
- prediction under a **counterfactual intervention**;
- predicted preservation of unrelated behavior;
- alternative explanations and a negative control;
- cost, confidence, and falsifying result.

Maintain at least two competing hypotheses and a null hypothesis. Prefer the experiment with maximum discrimination per unit cost. Distinguish **architecture**, **weight**, **activation**, **training process**, and **inference-time** causes. Review `01-evidence-and-method.md` before using causal terms.

### 4. Execute causal experiments, not just visualizations

Prefer matched clean/corrupted examples, carefully aligned positions and a scalar task metric. Escalation ladder:

1. Behavioral counterfactuals, controlled input changes, and baseline/no-op.
2. Activation inspection and calibrated comparisons (correlational).
3. Activation patching: replace candidate component at appropriate token/time with a clean counterpart, measuring intervention vs sham. Test **both directions** when meaningful.
4. Component ablation, resampling, alternative-value injection, and path-specific mediation when supported; negative/random-target controls.
5. Gradient/attribution patching as a **screening approximation**, verified by actual intervention; compare rank agreement and nonlinearity.
6. Weight-level counterfactuals on an isolated copy, only after predicting effects.

Test necessity, sufficiency, and specificity separately; a patch that repairs output in one example is not a complete circuit explanation. Control for off-manifold activation replacements, sequence-position mismatches, broken layer norms, and perturbation size. An attention map is **not** a causal explanation. A high-probe-accuracy direction is **not** necessarily used by the model.

**Gate B:** A claim may graduate to *causal support* only when a preregistered prediction survives a controlled intervention and a meaningful control. Otherwise label it correlational or speculative.

### 5. Weight-space analysis and repair

Read `02-weight-space.md` before naming a neuron, direction, or weight matrix as a mechanism. Respect parameter symmetries: hidden-unit permutations and rescalings can change weights without changing the computed function. Do not interpret raw differences between independently trained networks as evidence of behavioral change without alignment and functional tests.

If authorized, design *one minimal edit* at a time:

- Specify exact tensor names, axis definitions, indices, algebra (e.g. rank-one removal), scope, expected effect, and maximum changed entries/Frobenius norm.
- Prefer reversible in-memory `functional_call` overrides or `scripts/patching.py` utilities; preserve pristine base. No optimization/gradient step if the experiment specifically claims **gradient-free** editing. Do not disguise search/finetuning as a hand-written patch.
- Verify shape/dtype/finite values, shared/tied weights, shard ownership, quantization and normalization interactions, parameter hashing, and patch footprint.
- Compare against equal-footprint random edits, ablations, scale changes, and non-weight baselines. Do not keep searching the test set.
- If the objective is debugging rather than patching, an **unrepaired but well-localized failure** is a valid outcome.

Read `05-editing-and-validation.md` for locality, reversibility, collateral regressions and sequential-edit risks.

**Gate C:** No persistent edit may be called a fix unless it clears explicit held-out success and preservation thresholds. Otherwise mark experimental/rejected and retain the base.

### 6. Final verification and report

Use the incident template `assets/incident-report.md`; archive a compact machine-readable experiment registry (`assets/experiment-registry.json` provides fields). Report:

- before/after with denominators, independent trials, effect size and uncertainty (use paired comparison/bootstraps as suitable);
- time/memory and total forward/backward passes; exact seeds and revisions;
- side effects, OOD robustness, adversarial counterexamples, locality, and non-weight baselines;
- actual *mechanistic evidence* vs interpretations, including failed tests;
- patch location/shape, hash, changed-entry fraction, total norm, reproducible code path, rollback method;
- residual uncertainty, alternative explanations and stop/next-experiment decisions.

Use these evidence labels: **Observed**, **Correlational**, **Causally supported (scoped)**, **Replicated**, **Unresolved**, **Falsified**. Do not report a global mechanism from one prompt, one seed, or one local intervention. Stop if evidence is too weak, preservation fails, eval is contaminated, or budgets are exhausted. **Never silently replace or deploy a model.**

