# Reasoning correctly about weights

## Matrix orientation matters

For PyTorch `nn.Linear(in_features, out_features)`, stored `weight` has shape `[out_features, in_features]`, and output is `y = x @ weight.T + bias`. **Rows are output features**, columns are input coordinates. For a transformer's up/gate/down projection, embedding table, tied unembedding, convolutional kernel, or quantized block, the semantics are different. Read actual forward code and tensor shapes. Do not copy 'output columns' prescriptions from a paper without verifying that paper's tensor convention.

For an MLP `h = phi(xW1^T + b1)`, `y = hW2^T + b2`, the contribution of hidden unit `j` to output `k` is `h[j] * W2[k,j]`. This is often directly testable. For transformer residual additions, direct logit attribution can be computed with correct layer norm assumptions and unembedding semantics; approximation vs exact must be stated.

## Coordinate invariance and identifiability

A network's weights are not unique. For a two-layer ReLU net, a hidden-unit permutation P and corresponding inverse permutation in the next layer leaves the function unchanged. Positive scaling `W1[j,:] *= c, b1[j] *= c, W2[:,j] /= c` also preserves function for c>0. For consecutive *linear* layers, invertible basis transformations can preserve their composition. Therefore raw unit IDs, cosines between independently trained checkpoints, and pixel heatmaps of weights are not invariant descriptions of function.

Consequences:

- Compare aligned checkpoints only when architecture/order/ancestry makes coordinates meaningful; for independent seeds, consider neuron matching (with its own uncertainty) or functional comparisons.
- Verify feature explanations under natural inputs and counterfactual interventions, not only matrix geometry.
- A matrix's large singular vector can represent an important direction or simply reflect parameterization/scale; check activations and downstream effect.
- High norm does not imply high causal importance; small weights on a high-variance feature can matter substantially.

## What weight-only inspection *can* reveal

- Exact computational structure in small, known architectures: spatial input maps, output-channel selectors, hard-coded feature detectors, attention composition, simple threshold logic.
- Numerical pathologies: corrupt dtype, nonfinite entries, huge parameter drift, accidental constant/zero output, normalization issues and quantization artifacts.
- Changes relative to a **shared base**: sparse deltas, subspace rotations, low-rank directions; these still require function-level verification.
- Falsifiable predictions: which input perturbation or controlled activation insertion should change which output.

It cannot reliably tell an arbitrary large model's behavior just by staring at raw parameter bytes. Even a strong explanation from an LLM is a hypothesis until tested.

## Transformer-specific matrix reasoning

At a simplified single-head layer, attention scores involve `QK^T`; output writes involve `OV` composition. Depending on implementation, projected tensors and axes may be transposed. Correctly account for masking, rotary/positional transformations, scaling, GQA/MQA head sharing, layer norms and residual reads. Weight products are **virtual circuits**, but actual behavior depends on token states, relative positions, gates and downstream computation. A large `W_OV` norm is not evidence of induction or factual recall on its own.

For suspected directions `v`, test:

1. `a·v` distributions across positive/negative examples and nearby controls.
2. Whether a natural activation patch changing the `v` coordinate changes target metric.
3. Whether removing/steering that component also changes unrelated tasks, response length or entropy.
4. Whether a randomized equal-norm direction behaves similarly (null control).
5. Whether normalized and unnormalized representations preserve the effect.

## Efficient exploration of billion-parameter models

Do **metadata first**: module names, shapes/dtypes, total params, tied/shared storage, quantization, finite/norm checks on only candidates. Avoid concatenating parameters. For a suspected layer, retrieve small slices/aggregate quantiles. Capture activations at one chosen token (or short span) and aggregate online; do not store `[batch, seq, layers, d_model]` tensors by default. Use low-rank randomized methods rather than dense SVD of huge matrices; state approximation. Measure inspection memory and time.

## Sparse and low-rank changes

For target tensor W and direction u:

- Rank-one correction: `W' = W + alpha * a b^T` (shape check essential).
- Column removal: `W'[:,J] = W[:,J] - delta[:,J]` for selected input coordinates J.
- Row edit: `W'[I,:] = ...` targets output channels I.
- Projection removal: e.g. `W' = W - (Wv)v^T/(v^Tv)` removes sensitivity along input direction v, but may destroy valuable features. Other conventions use left-side projection; distinguish input from output subspaces.

Record changed *entries*, not just number of tensor rows or columns touched. Compute `|ΔW|_0 / total_parameters`, `||ΔW||_F`, and `||ΔW||_F / (||W||_F + ε)`. Record repeated/shared tensor accounting. Patch metrics cannot establish mechanistic truth by themselves.

Original research: https://rikiyat.github.io/blog/weight-space-programming/
