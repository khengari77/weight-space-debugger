# Model Investigation: <ID>

**Access:** Checkpoint only / Executable / API only  
**Mode:** Forensics / Causal debugging / Weight surgery  
**Status:** Observed / Correlational / Causally supported (scoped) / Replicated / Unresolved / Falsified  
**Investigator / date:**  
**Base checkpoint:** ID/revision/hash, license, architecture  
**Code and runtime:** Git SHA, dependency versions, device/precision  
**Permissions and constraints:**  

## Executive finding

Two or three sentences that separate the observed failure from its best-supported explanation; explicitly state what you do NOT know.

For checkpoint-only forensics, mark unavailable behavioral metrics, dataset splits,
causal tests and repair fields as not applicable; record zero model passes.

## Provenance and contribution assessment

- Prior work and pinned source revisions (hypotheses, method, code):
- What was known from those sources before investigation:
- Borrowed or copied components, licences and attribution:
- Reproduction / adaptation / engineering refinements / independently developed findings:
- Comparison of claimed additions with upstream evidence:
- Significance demonstrated, or **novelty not established**:
- Negative interventions and limits on causal, generality and discovery claims:

## Reproduction and baseline

- Failure specification:
- Preprocessing/tokenizer/action mapping:
- Exact evaluation parameters and seed policy:
- Sample counts (diagnostic/development/sealed-test):
- Target metric baseline including uncertainty:
- Preservation metric baselines:
- Non-weight baseline results:

## Model and weight inspection

- Model map and relevant tensor names/shapes/dtypes:
- Findings (nonfinite, saturation, drift, low-rank indications):
- Evidence IDs and data provenance:

## Hypotheses (including rejected)

| ID | Precise mechanism | Prediction registered BEFORE experiment | Result | Control | Verdict |
|---|---|---|---|---|---|
| H1 | | | | | |
| H2 | | | | | |
| Null | | | | | |

## Causal experiments

Specify exact intervention site/time/token, input matching, control, metric, repetitions, effect size, uncertainty, limitations. Document necessity and sufficiency separately if claimed.

## Optional patch and rollback

- Base SHA256/checkpoint revision:
- Patch method (`set`, `add`, rank-one, projection, etc.):
- Tensor names / indices / orientation:
- Changed entries, % of model, Frobenius norms:
- Artifact SHA256 / patch manifest:
- Reapply and rollback commands:

## Held-out evaluation and guardrails

| Metric (denominator) | Base | Candidate | Difference / CI | Pass? |
|---|---:|---:|---:|---|
| Target | | | | |
| Preservation | | | | |
| OOD/robustness | | | | |
| Safety constraints | | | | |

- Stochastic run policy / seeds:
- Random-control edit results:
- Negative / counterfactual tests:
- Test leakage or multiple-testing concerns:
- Time, VRAM and number of forward/backward passes:

## Verdict and remaining uncertainty

Accepted / Rejected / Unresolved. The strongest falsifying evidence; next high-information experiment; whether a non-weight fix is superior. No deployment performed without explicit approval.
