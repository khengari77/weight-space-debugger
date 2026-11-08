# Checkpoint-only forensics

Use this route for inventory, corruption checks and checkpoint comparisons when
inference is unavailable, prohibited, or unnecessary for the request. A working
model, behavioral failure, diagnostic/development/test datasets and a training
loop are not prerequisites. Do not invent those artifacts or block a useful audit
because they are absent.

## Inspect

1. Record checkpoint path/revision, file hash, format, available config and the
   inspection budget. Check whether a multi-file checkpoint is sharded; inventory
   all provided shards and disclose missing ones rather than treating one as the model.
2. Run `scripts/checkpoint_audit.py` on each provided `.safetensors` file. It hashes
   the whole file but samples tensor values. Report `scanned_elements`,
   `complete_scan`, skipped tensors and rows too wide for the chunk budget.
   A sample without NaNs is not a guarantee that the tensor is finite.
3. Map names, shapes, dtypes, metadata and known tensor semantics. Read trusted
   architecture/config sources when available; do not execute unknown model code
   merely to inspect a checkpoint. File-only data may not identify runtime ties.
4. Compare names/shapes and numerical deltas with `scripts/compare_checkpoints.py`
   when requested. Establish matching semantics/shared ancestry before interpreting
   coordinate changes. For independent training runs, report alignment uncertainty
   and prefer structural comparisons over behavioral interpretations of raw deltas.
5. Use [weight-space reasoning](02-weight-space.md) for geometry and invariances.
   Record candidate anomalies and falsifiable hypotheses only when justified;
   do not manufacture multiple explanations for a routine inventory task.

## Report and stop

Separate exact file/metadata observations from sampled numerical findings.
A changed tensor, large norm or sparse delta alone does not establish what the
model does, why behavior changed, or whether a repair would help.

Use [the incident template](../assets/incident-report.md), marking behavioral
reproduction, dataset splits, causal tests and repair acceptance as not applicable
unless actually performed. In the [registry](../assets/experiment-registry.json),
record `access: checkpoint_only`, retain the `forensics` mode, leave unavailable
metrics/splits null, and record zero model forward/backward passes. Describe the
minimum trusted inference harness/data needed to test a behavioral hypothesis.
Finish when the requested audit or comparison is complete; inference is a proposed
follow-up, not a prerequisite for completing static work.

## Routing walkthrough

Request: compare two authorized safetensors files without running the network.
Expected response: hashes, tensor inventory, key/shape differences, aligned sampled
deltas, scan limitations and unresolved behavioral implications. No model loading,
training steps, fabricated failure example or demanded test split.

Request: inspect training code without changing the supplied model.
Expected response: static code/log inspection; describe any unperformed disposable
training check as a follow-up. Do not execute it just because training code exists.
