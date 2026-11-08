# Evidence, experimental design, and calibrated claims

## What is a root cause?

A useful mechanism makes predictions beyond the data used to invent it. A high activation on failed examples is correlation; successful rescue with a specific controlled intervention is scoped causal evidence. A model may contain multiple redundant circuits: ablation of one alone can show no effect even when it contributes. Conversely, a highly disruptive intervention may change performance for uninteresting reasons. Treat these cases separately.

Use an **evidence ladder**:

| Label | Minimum evidence | What it does NOT establish |
|---|---|---|
| Observed | Frozen setup and reproducible measurements | Causation |
| Correlational | Feature/activation/weight statistic predicts behavior on held-out examples | The model uses that feature |
| Causally supported (scoped) | Controlled activation/path/parameter intervention yields the registered effect and survives control(s) | Unique or complete mechanism |
| Replicated | Consistent across reruns, cases, seeds, and/or nearby models | Universal generality |
| Falsified | Predicted effect absent under a valid powered test | Every broader theory impossible |
| Unresolved | Insufficient intervention or ambiguous controls | Success |

## Hypothesis ledger example

| ID | Mechanism | Prediction | Falsifying experiment | Control | Status |
|---|---|---|---|---|---|
| H1 | Decoder stop tokens wrong | Correct stop ID ends looping without accuracy loss | Hold weights fixed; change stop handling | Equivalent non-looping prompts | Untested |
| H2 | MLP writes repetition-driving vector | Remove predicted write: loop rate falls, nonloops preserved | Activation patch at specific layer/position | Random equal-norm direction | Untested |
| H3 | Training data cannot support task | No sparse patch improves genuine reasoning errors | Blind independent tasks | Decode and prompt baseline | Untested |

Store *prediction before result*, expected direction, metric, experimental identifier, seeds, and sample sizes. Avoid post-hoc explaining the same result by whatever story fits.

## Dataset and evaluation discipline

- Diagnostic/development/sealed-test splits must be disjoint by **underlying task** (not just paraphrase), particularly for repeated benchmark questions.
- Define task score, failure incidence, and **preservation** score before attempting a patch. Record exact prompts and output parsers. If private, hash and store locally instead of exporting.
- Prefer matched clean/corrupt paired examples differing in one causally relevant factor. For token sequences, log tokenizer IDs and align positions rather than assuming equal rendered text length.
- Include a sham patch, random target, and, for weight edits, equal-norm/equal-sparsity random patch. Count attempted candidates to avoid selection bias.
- For stochastic evaluation, use shared random seeds/common random numbers where feasible, and verify with independent seeds afterward. A paired bootstrap across *items* is useful; for RL bootstrap at episode/seed level, not timestep level.
- If tuning across many hypotheses, do not repeatedly peek at sealed-test performance; one final evaluation for accepted edit and a separate audit of any necessary second test.

## Causal intervention rules

**Activation restoration:** corrupted forward, with a selected activation from clean run substituted. Measure whether clean behavior is restored. This measures a counterfactual at an intervention site, not automatically a neuron-level semantic program.

**Activation ablation:** replace with zero/mean/another-sample activation. Off-manifold artifacts are common; use at least two baselines or matched resampling. **Patching both directions** (clean→corrupt and corrupt→clean) helps distinguish necessity/sufficiency but is not conclusive alone.

**Path patching:** isolate effects of one writer to one reader; plain activation replacement can also alter other readers and downstream routes. Check the precise intervention semantics of your library.

**Gradient attribution:** cheap first-order screening estimate, not a measured intervention; saturating nonlinearities, residual routing, correlated sites and large deltas can invalidate the approximation. Verify strongest predictions with exact patching.

**Finite differencing:** if possible, perturb candidate components incrementally (dose-response curve) and monitor secondary metrics, not just a binary switch.

## Stopping rules

Stop and report *Unresolved* when: the failure is not reproducible; token/state alignment is unclear; intervention silently fails; model architecture is unsupported; suspected circuit effects vanish on holdout; a different simple configuration fix solves the issue; changes damage protected capabilities; budget or permissions are exceeded.

## Example questions that distinguish hypotheses

- Does the defect occur with greedy decoding? Does it appear before reaching EOS? Is it a tokenizer/chat-template mismatch?
- Is the activation merely a readout of failure already decided upstream? Does patching it *before* the decisive token alter behavior?
- Does ablating a head eliminate the *specific* behavior but preserve controls, or does it just degrade all outputs?
- Could a rank-one weight direction be a byproduct of optimization rather than a semantically meaningful circuit?
- Is an edited model better at reasoning or merely more likely to complete a response before the token limit?

Relevant references: https://www.jmlr.org/papers/v26/23-0058.html ; https://transformerlensorg.github.io/TransformerLens/content/analysis_tools.html
