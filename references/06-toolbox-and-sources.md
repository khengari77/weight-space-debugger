# Tools, capability selection, and evidence sources

## Choose tools by observability, not trendiness

| Situation | Preferred starting point | When to escalate |
|---|---|---|
| Raw `.safetensors` file, unknown model | `scripts/checkpoint_audit.py`, config/architecture | Need a trusted inference harness for behavioral claims |
| Arbitrary PyTorch MLP/CNN/policy | `scripts/probe.py` and hand-written forward checks | NNsight or custom hooks for surgical interventions |
| Supported transformers | Tokenizer/decode diagnostics; TransformerLens or NNsight | Activation/path patching and SAE on warranted candidates |
| Two checkpoints sharing a base | Metadata and sparse delta maps | Fine-grained functional/causal tests |
| API-only model | Behavioral test suite | No legitimate weight-level inference without internal access |
| Training divergence | Parameter/gradient stats, optimizer/config/data pipeline | Version-specific PyTorch op-level debugging |
| Large sharded/quantized model | Architecture-specific runtime instrumentation | Never use naive `state_dict()` materialization or generic writes |

### PyTorch runtime

`register_forward_hook` sees module outputs and can replace outputs, but names and layout are model-dependent. Always remove hooks in `finally` (a context manager is best), and avoid retaining large activation tensors. Use `torch.inference_mode()` for inference-only probes, not when gradients are required. Beware that autocast, compiled kernels, activation checkpointing and model parallelism affect what a hook sees. `torch.func.functional_call` can use parameter overrides without mutating the model but obeys documented caveats for tied weights, parametrizations and in-place parameter operations.

### Libraries

- **TransformerLens**: activation caches, exact activation patching, attribution patching, path patching; supported models only, conversion parity mandatory. Docs: https://transformerlensorg.github.io/TransformerLens/content/analysis_tools.html
- **NNsight**: trace/activation read-modify/write within model forward pass, generic PyTorch and transformer wrappers. Docs: https://nnsight.net/getting-started/quickstart/
- **SAELens**: SAE representations can suggest features; verify if actual feature is faithfully represented and used. Project: https://github.com/jbloomAus/SAELens
- **PyTorch**: model hooks and functional overrides. Docs: https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html and https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook

## Scientific references

1. Takehi et al., *Sparks of Weight-Space Programming* (Nov 2026): https://rikiyat.github.io/blog/weight-space-programming/ . Published experiment logs: https://github.com/RikiyaT/lm_understand-logs
2. TransformerLens, *Choosing an analysis tool*: https://transformerlensorg.github.io/TransformerLens/content/analysis_tools.html . Notably, attribution patching is a linearized screen; use exact interventions to verify.
3. Geiger et al., *Causal Abstraction* (JMLR 2025): https://www.jmlr.org/papers/v26/23-0058.html . Unifies many intervention perspectives.
4. Anthropic, *Tracing the thoughts of a large language model* (Mar 2025): https://www.anthropic.com/research/tracing-thoughts-language-model . Makes clear that circuit tracing remains partial and may contain methodological artifacts.
5. Gupta et al., *Model Editing at Scale leads to Gradual and Catastrophic Forgetting* (ACL Findings 2024): https://aclanthology.org/2024.findings-acl.902/
6. PyTorch, `torch.func.functional_call`: https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html
7. Agent Skills specification: https://agentskills.io/specification

Do not assume libraries, APIs or model support stay unchanged. Read the installed version's documentation and keep working minimal executable parity tests before analysis.
