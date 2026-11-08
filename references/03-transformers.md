# Language models and transformer circuit debugging

## Rule out outer-loop failures first

Before circuit inspection verify tokenizer/model revision; padding and EOS IDs; chat template; system/developer/user prompt construction; role delimiters; sampling temperature/top-p/top-k/repetition and presence penalties; inference cache/kv-cache; positional encodings; generation stopping; context truncation; offload and quantization; task output extraction.

Instrument **full trajectories**: generated token IDs, decoded text, stop reason, token-limit hit, local repetition spans, response length, log-prob of correct answer, entropy and competing token ranks. Record when behavior departs from a successful run. A model can be poor at a benchmark because it never finishes rather than because it cannot solve the tasks.

## Three levels of questions

1. **Behavior:** Does a prompt or decode change fix the issue? A paired set measures effects in end-to-end outputs.
2. **Where:** Which layer/head/MLP/position conveys useful information? Use residual-stream patching, head output patching or neuron/path experiments.
3. **How:** What computation does that component implement? Analyze QK patterns, OV write, upstream signals, downstream readers, interventions and circuit faithfulness.

An attention head's high probability on a token describes attention allocation, not whether information from it caused an answer. Similarly a linear probe decoding a fact from a residual stream does not demonstrate the network actually uses that fact.

## Patching discipline

Define clean/corrupt pair, token positions, expected answer, logit-difference or likelihood metric. Verify token alignment. With identical-length/shape sequences, patch clean activation into a corrupted run at a named hook site, then compute actual change in **one fixed** scalar metric. Repeat across many paired examples and reverse direction where sensible.

Use attribution patching to prioritize sites cheaply, but it is a first-order Taylor approximation `Δa · ∂metric/∂a`, not proof of a causal effect. Strong candidates should be tested with exact interventions. For path-specific claims avoid accidentally patching every downstream reader.

For repeated-token pathologies: compare when repetition begins, shifts in logits between continuation and repeated fragments, per-layer residual contributions, and whether a candidate causal edit consistently reduces looping **without merely forcing termination**. Explicitly preserve unrelated reasoning accuracy, instruction following, diversity and short-form outputs.

## Sparse autoencoders and feature labels

SAE features may offer useful hypotheses about polysemantic activations, but a feature name inferred from top activating examples is an interpretation, not an oracle. Check reconstruction loss, dead features, feature stability, and whether an SAE is trained for the model/layer of interest. Intervene in the **original model** where feasible; report reconstruction/ablation artifacts and uncertainty. Don't claim a feature's label is a ground-truth concept.

## Special cases

- **GQA/MQA:** Q heads and K/V heads may not correspond one-to-one. Verify repetition mapping.
- **LayerNorm/RMSNorm:** residual vectors, attention outputs and direct logit attribution depend on normalization and linearization assumptions.
- **MoE:** log routing and expert selection; do not interpret an inactive expert's raw norm as used.
- **Vision-language:** inspect image preprocessing/crop and vision-token alignment before text-circuit analysis.
- **Quantized models:** edits in packed/int4/int8 weight structures are not equivalent to floating-point tensor assignments. Do diagnostic dequantization or adapter-based overrides only when correctly specified.
- **Large weights:** no arbitrary full-model dumps to the LLM. Use indexed aggregate and trace summaries with provenance.

## Recommended libraries (version-sensitive)

TransformerLens for supported transformer instrumentation/activation and path patching; NNsight for tracing/interventions in PyTorch/HF; SAELens for sparse-autoencoder research; direct PyTorch hooks for architecture-general local analysis. Read library documentation for the actual installed version before writing hooks or converting a checkpoint. A successful conversion must pass output-logit parity tests.

References:
- https://transformerlensorg.github.io/TransformerLens/content/analysis_tools.html
- https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens/patching.html
- https://nnsight.net/getting-started/quickstart/
- https://www.anthropic.com/research/tracing-thoughts-language-model
