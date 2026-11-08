# Worked research plan: diagnosing a language-model repetition loop

This is a **protocol**, not an assertion that any particular model has the same mechanism as the November 2026 weight-space-programming experiment. Do not copy weight edit coordinates from that experiment into unrelated checkpoints.

## Stage 0: verify the boring stuff

Pin checkpoint, tokenizer, chat template, model revision, precision, stopping IDs, and sampling. Collect matched examples that loop and don't loop. Log token IDs, full outputs, max-token hit fraction, exact answer parsing and per-prompt score. Perform no-weight controls: greedy decoding, alternate documented decoding presets, correct EOS handling, reduced context, corrected chat template. If these solve the problem, report that finding and skip weight surgery.

## Stage 1: define discriminating metrics

- `loop_rate`: proportion of prompts with predeclared n-gram/cycle-based repetition threshold, not just long answers.
- `length_limit_rate`: proportion reaching generation cap without a valid final answer.
- `conditional_correctness`: accuracy conditional on producing a final answer, plus unconditional accuracy to avoid hiding length effects.
- `locality`: nonlooping tasks, instruction following, fact questions, concise response tasks, negative prompts.
- `per_step` signals: probability and rank of repeated continuation tokens vs task-relevant next-token alternatives; stop-token probability.

Use disjoint diagnostic and development prompts; save benchmark evaluation for a sealed acceptance run. Do not optimize to benchmark labels.

## Stage 2: weight and activation reconnaissance

Start with architecture + numerical audit. Collect a few short failing trajectories and matched successful trajectories, then selected MLP outputs/residual streams per layer at the **onset** of repetition. Check whether a reproducible directional signal differs, and whether candidate MLP contributions align with it. Avoid a dense activation dump. When inspecting Qwen-like architecture, verify the exact actual parameter orientation, MLP gate/up/down semantics and normalized residual path.

Create ledger: H1 decoding/stop bug; H2 one MLP write feeds repetitive loop; H3 loop is distributed dynamics over many layers; H4 task beyond capability.

## Stage 3: causal localization

Screen component/layer positions with attribution patching if supported; confirm with direct hook-based interventions on actual matched cases. For hypothesis H2, subtract a proposed direction from a candidate layer output only during certain positions; test onset, mid-loop, and unrelated prompts. Rescue should occur with **smaller perturbations** and should not collapse to truncated, incoherent or formulaic endings. Test inverse-direction injection in nonlooping examples and random equal-norm directions. Record sign, dose and metrics.

An edit that only causes the model to stop earlier, without preserving answer correctness, does NOT validate a math-capability mechanism.

## Stage 4: reversible edit

Only if causal evidence survives:

1. Choose selected input or output subspace and precisely derive delta matrix dimensions in the installed architecture.
2. Apply a **reversible** sparse/rank-limited override to a copied checkpoint or through `functional_call` if feasible.
3. Measure changed-entry count, delta norm and targeted/residual contribution shifts.
4. Evaluate candidates on development data; compare equal-footprint random edits, a no-op, and decoding-only baselines.
5. Freeze chosen candidate before the sealed test.

## Stage 5: acceptance

Report MATH/GSM-style reasoning, instruction-following and non-math retention separately, paired when possible. Record loop-rate reduction, accuracy conditional on completion, length distribution, seed variation, gradient-use status, forward-pass budget and all unsuccessful attempts. If repeated edit cycles cause further degradation, reject. Only claim a mechanism scoped to the tested cases, not universal reasoning improvement.

For context, in the article's Qwen3.5-0.8B experiment a 0.016% parameter edit coincided with reduced length-limit hits and large accuracy gains. Its separate Gemma experiment failed to add missing math ability. The takeaway is a **hypothesis worth investigating**, not a reusable recipe that already works across LMs.

Reference: https://rikiyat.github.io/blog/weight-space-programming/
