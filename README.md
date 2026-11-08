# Weight-Space Debugger

An [Agent Skill](https://agentskills.io/specification) for neural-network checkpoint
forensics, causal debugging, and authorized reversible weight edits.

Inspired by **[Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)**
by **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab** (November 2, 2026).
Weight-Space Programming uses an LLM's reasoning to inspect and deliberately write
network parameters rather than relying solely on optimizer-driven training. This
skill applies the concept to debugging with explicit evidence and preservation checks.
It is an independent project, not a reproduction of the authors' experiments.

## Install

Clone the complete skill into a discovery directory supported by your agent.
For agents supporting project-local `.agents/skills/` discovery, run from your project:

```bash
git clone https://github.com/khengari77/weight-space-debugger.git .agents/skills/weight-space-debugger
```

Other agents may use a different discovery directory. Keep the root `SKILL.md`,
`agents/`, `scripts/`, `references/`, and `assets/` together. Installing the skill
makes instructions available; running the helpers also requires Python dependencies.

The helpers and CI are validated on **Python 3.12 with CPU PyTorch**. No GPU,
model download, cloud service, or vendor-specific runtime is required for the tests.

```bash
cd .agents/skills/weight-space-debugger
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To install CPU-only PyTorch, run `python -m pip install 'torch>=2.4' --index-url https://download.pytorch.org/whl/cpu`
before installing the requirements. Use a separate environment to avoid changing
an existing model runtime.

## Use

Invoke `$weight-space-debugger` in an agent that supports named skills, or ask it
to use [SKILL.md](SKILL.md). Example requests:

- “Compare these two safetensors checkpoints without loading or running the model.”
- “Investigate why my policy freezes in this state. Reproduce and explain it without editing weights.”
- “Find out why this LM loops on math prompts; test decoding before circuit interventions.”
- “Inspect my training code read-only; do not run optimizer steps.”
- “I authorize a reversible patch on a copied model if causal tests support it; protect these evaluations.”

| Available access | Work performed | Evidence limits |
|---|---|---|
| Checkpoint only | Hashes, inventory, sampled numerical checks, aligned differences | No measured behavioral or causal conclusions |
| Executable model | Reproduction, probes, and authorized controlled interventions | Claims scoped to tested cases |
| API only | Input/configuration and behavioral debugging | No weight-level inference |

Read-only forensics is the default. Training diagnostics require a disposable
model and separate optimizer when in scope. Persistent repairs require authorization,
held-out success, and preserved behavior. The baseline checkpoint is retained.

## Helpers

```bash
python scripts/checkpoint_audit.py model.safetensors --output audit.json
python scripts/compare_checkpoints.py base.safetensors candidate.safetensors --output delta.json
python examples/toy_case.py
```

The audit hashes the whole file but samples tensor values in bounded axis-0 chunks.
Rows exceeding the chunk budget are skipped and reported. A clean sample is not
a full finite-value guarantee. Inspect checkpoint semantics and ancestry before
interpreting coordinate differences.

In-process utilities provide sampled activation/gradient telemetry, single-tensor
activation capture/patching, reversible parameter overrides, and exploratory paired
bootstrap intervals. Overrides clone edited tensors; unchanged parameters and buffers
are aliased. Use an isolated instance when forwards update state. These are research
primitives, not an architecture-general automatic model editor.

## Package and validation

```text
weight-space-debugger/
  SKILL.md                  Agent instructions and routing
  agents/openai.yaml        UI metadata and default invocation
  scripts/                  Audits, probes, interventions, package validation
  references/               Conditional methodological guidance
  assets/                   Incident report and experiment registry templates
  examples/                 Synthetic demonstration and evaluation blueprints
  tests/                    Helper and package regression tests
```

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python examples/toy_case.py
python scripts/package_skill.py
```

GitHub Actions runs these checks on Python 3.12 with CPU PyTorch. After intentional
package changes, regenerate the demonstration with
`python examples/toy_case.py --output examples/toy-result.json` and refresh the
file hashes with `python scripts/package_skill.py --update` before validation.
`MANIFEST.json` lists the distributed source files; it excludes itself, Git metadata,
environments, caches, model checkpoints, and generated investigation reports.

The demonstration has a known, hand-constructed defect. Its binary inputs repeat
across independent dataset draws; it does not establish generalization to unseen
tasks. Dense random controls match norm only. The four one-entry controls match
both norm and footprint and include the repair itself, so the example claims no
superiority over all controls. See [the loop investigation blueprint](examples/transformer-loop-investigation.md)
and [the separate agent reasoning evaluation cards](examples/skill-evaluation.md).
Script tests do not validate an agent's mechanistic reasoning, and no blind real-model
repair benchmark has been performed.

## Reproduction of an existing Whisper repair method

**Codex found the fix in [Kevin Bondzio’s published weight-patching work](https://github.com/kevinpl07/weight-patching/tree/d7df0d9bd15afb952b1e0cb64a067806563aa8e3)
and adapted it. It did not independently discover the fix.** The decoder helpers
are substantially copied from that project, and the repair recipe and evaluation
safeguards were already present upstream.

[Our narrower Whisper-small reproduction](examples/whisper-case-study.md) measured
a default-rule failure rate of **38% before the patch and 1.5% afterward**, with
**200/200 speech transcripts unchanged**, after editing **1,537 parameters**.
These are measurements from our run, not evidence of a novel repair algorithm,
a better method than upstream, or autonomous discovery by the skill.

The useful additions are a smaller evaluation setup, implementation refinements,
new locally measured results, and Agent Skill packaging. **No significant new
repair method or methodological advance has been established.** The observed
GPU account-credit decrease was **$0.0280**; we have not demonstrated a cost
advantage over the upstream study. See the case study for the contribution audit,
controls, negative interventions, and limitations.

## Real-model case study

[RobustMNIST: a 455,991-parameter published classifier](examples/robustmnist-case-study.md)
was tested on real handwritten digits. The tested distortions did not meet the
predeclared threshold for surgery; the report includes measured errors and a
negative replication rather than claiming a successful repair.

## License and attribution

Original code and documentation are available under the [MIT license](LICENSE),
copyright 2026 Abdulrahman Khengari. The linked research and its authors receive
credit for the inspiration and Weight-Space Programming concept; the package's
license does not relicense their article or imply their endorsement.
