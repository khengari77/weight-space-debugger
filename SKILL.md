---
name: weight-space-debugger
description: Inspect neural-network checkpoints, investigate reproducible model failures, and test authorized reversible weight edits. Use for PyTorch, transformer, or RL checkpoint comparisons, numerical anomalies, causal circuit investigations, and targeted repairs; supports checkpoint-only access without inference.
license: MIT
---

# Weight-Space Debugger

Build discriminating evidence about neural-network behavior. Treat weight patterns,
attention maps, probes and an LLM's explanations as hypotheses until tested.

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab. The concept is to reason
directly about network weights and their computation, including deliberate edits
rather than optimizer-driven training. This skill applies that idea to debugging;
it does not reproduce the authors' experiments or promise general model repair.

## Route by access and requested work

Determine the available access independently of intervention permission:

| Access | Start here | Valid output |
|---|---|---|
| Checkpoint only, or inference disallowed | [Static forensics](references/09-static-forensics.md) | Integrity, structure and aligned deltas; untested behavioral hypotheses |
| Executable model | [Executable investigation](references/08-executable-investigation.md) | Reproduction, behavioral measurements and authorized causal tests |
| API only | Behavioral reproduction and config/input comparisons | Behavioral evidence; no weight-level claims |

Choose the least invasive mode that satisfies the request:

- **Forensics** (default): inspect weights, existing traces, or ordinary inference. No parameter edits, activation interventions or training steps.
- **Causal debugging**: ephemeral activation or parameter interventions on an isolated run. No persistent checkpoint edits.
- **Weight surgery**: when requested or already authorized, test minimal reversible parameter edits on a copy; accept only after held-out success and preservation checks.

Training diagnostics are conditional on the task and experiment restrictions, not
implied by access to training code. Follow [training isolation](references/07-training-debugging.md):
use a disposable model and separate optimizer, preserve baseline parameters,
buffers, gradients, module modes and RNG, and skip backward/optimizer steps when
prohibited. Never train the investigation baseline.

## Shared constraints and evidence gates

- Infer model identity/revision, tensor semantics, allowed data/inference, resource budget and preservation targets from available artifacts. Continue with useful inspection when inference or data is missing.
- Use trusted model code and safe serialization. The audit supports `.safetensors`; do not auto-load pickle or enable remote model code. Do not export private weights or inputs without authorization.
- For behavioral failures, freeze preprocessing, decoding/environment, precision, seeds and scoring. Rule out input, evaluation and inference defects before blaming weights. Static forensics needs no behavioral failure or datasets.
- For causal claims, register a prediction, run the actual intervention, and include a meaningful control. Follow the [evidence method](references/01-evidence-and-method.md); distinguish necessity, sufficiency and specificity.
- Before interpreting neurons, matrices or checkpoint deltas, check orientation, ties and parameter symmetries using [weight-space reasoning](references/02-weight-space.md). Unaligned coordinate differences do not establish changed behavior.
- For repairs, use diagnostic/development/sealed-test splits and protected behaviors. Compare no-op, non-weight and norm/sparsity-matched controls. Follow [editing and validation](references/05-editing-and-validation.md). Static audits do not require these repair gates.
- Preserve pristine checkpoints. `functional_call` can mutate aliased parameters/buffers when the forward updates state: use a disposable instance for stateful or training runs. Never silently replace or deploy a model.
- Respect runtime and memory budgets. Retrieve small slices and aggregate statistics rather than dumping large weights or activation caches into context.

A reproducible failure is required for a behavioral root-cause claim; a controlled
intervention is required for scoped causal support; held-out success and preservation
are required to call a persistent edit a fix. Report unsupported conclusions as
**Unresolved**, with the next useful experiment or missing access.

## Tools and conditional references

- Checkpoint inventory: `python scripts/checkpoint_audit.py FILE.safetensors --output audit.json`.
- Aligned checkpoint deltas: `python scripts/compare_checkpoints.py BASE.safetensors OTHER.safetensors --output delta.json`.
- In-process helpers: `scripts/probe.py` for sampled activation/gradient summaries; `scripts/causal.py` for single-tensor capture/patching; `scripts/patching.py` for reversible overrides and footprints; `scripts/metrics.py` for exploratory paired bootstrap intervals.
- Transformer work: [tokenizer, decoding and circuit guidance](references/03-transformers.md). RL work: [observations, actions and control](references/04-rl-and-control.md).
- Tool selection and primary sources: [toolbox](references/06-toolbox-and-sources.md). Read installed-version documentation before architecture adapters or model conversion; verify logit parity after conversion.
- Loop investigation: [worked blueprint](examples/transformer-loop-investigation.md). Helper demonstration: `python examples/toy_case.py` (synthetic known fault).

Paths above are relative to this skill's directory; resolve them there even when
working in another project. Load only the references relevant to the selected task.

## Provenance and contribution claims

Disclose sources of hypotheses, methods and code before presenting results. If a
fix was found in prior work, say so explicitly; do not describe it as independently
discovered. Separate reproduction, adaptation, engineering refinements and new
findings. Compare claimed additions with the source: inherited evaluation controls
are not original contributions. Use **novelty not established** when significance
has not been demonstrated. Keep failed interventions and evidence limits visible;
a successful reproduction does not establish autonomous discovery ability.

## Deliverables and stopping

Use [the incident template](assets/incident-report.md) and
[experiment registry](assets/experiment-registry.json) for substantive investigations.
Mark unavailable behavioral, split and repair fields as not applicable for static
forensics; record zero model forward/backward passes. Report artifacts/revisions,
sampling coverage, observed facts, hypotheses, experiments, limitations, resource
use and optional patch/rollback details.

Use **Observed**, **Correlational**, **Causally supported (scoped)**, **Replicated**,
**Unresolved** and **Falsified** accurately. Stop when the task is answered, evidence
or preservation fails, or the budget is exhausted. A localized unrepaired failure
or a useful static audit is a valid result.

Validate helper changes with `python -m pytest -q` and `python scripts/package_skill.py`.
These checks do not establish agent reasoning quality: use the separate
[hidden-fault evaluation cards](examples/skill-evaluation.md) for that assessment.
