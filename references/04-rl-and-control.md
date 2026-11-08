# Reinforcement learning, simulators, robotics and control policies

## Inspect the environment as carefully as the policy

Many 'bad weights' problems are observation or action defects. First verify observation order, units, normalization and clipping, frame stacking, time deltas, action bounds/masks, stochastic sampling, termination/truncation, reward function, simulator timestep and physics determinism. When replaying episode traces, include the **exact observation vector** fed to the network, not just video.

Record success rate, return distribution, episode length, action entropy, action histogram, constraint violations, failure stage, and worst-case seeds. Keep evaluation seeds independent from training seeds and from patch-development seeds. Compare deterministic and sampled policies. When possible compare with a simple transparent controller to expose trivial environment bugs.

## State-conditioned hypotheses

A controller may implement a rational local optimum that cannot complete the task (for example, driving forward at an obstacle when backing up first is required). One mechanism hypothesis might be missing state-dependent switching logic: observation(s) x gate a behavior until condition C, then allow the learned controller to continue. Test by plotting and intervening on action logits as functions of state, not merely by inspecting weight magnitudes.

For an actor `logits=πθ(s)`, candidate diagnosis includes:
- Sensitivity of each action logit to chosen input dimensions (`∂logit/∂state` if allowed).
- Local/finite-difference probes over *valid states*, not unconstrained random vectors.
- Hidden-unit response across failure / success trajectories.
- Resampling hidden activations at matched observation states, or targeted policy override, with explicit stochastic-policy handling.
- Compare edited-policy outcomes across unseen scenes, physics parameters and initial conditions.

## Actor-critic and RL-specific constraints

Separate **actor** and **critic**: an edit to value head may not affect action selection, and a critic gradient may explain training instability rather than deployed control. For PPO inspect advantage estimates, GAE, reward scales, clipping ratios, KL divergence, entropy bonus, old-policy log probs, value clipping and training vs evaluation normalization. Model surgery changes the policy distribution: if further PPO training occurs, saved optimizer moments and behavior distributions may be stale.

In robotics, test constraints and emergencies using simulation; do not deploy unverified weight edits on safety-critical equipment. Explicitly separate action selection pathology from dynamics-model error. If physics is non-deterministic, use enough seeded replications and confidence intervals. For sequential decisions a 'fix' can shift failure to a later stage; inspect full rollouts.

## Progressive difficulty ladder

1. Snapshot one failure and successful comparison episode; verify action and observation mapping.
2. Reduce to smallest state/features that explain the choice.
3. Intervene on action logits *in inference* to test whether desired action would rescue the episode.
4. Patch or ablate a candidate intermediate representation to test implementation mechanism.
5. Only then write the smallest reversible weight delta; inspect return and failure modes on independent unseen seeds.

## What the paper did (and didn't)

The weight-space-programming car experiment demonstrated a hand-designed conditional path through a small policy (171,664 weights) that made it reverse before crossing a ramp. This is promising for policies with interpretable state inputs and a clear control-law correction. It is not evidence that arbitrary high-dimensional visual RL policies can be understood or repaired as directly.

Reference: https://rikiyat.github.io/blog/weight-space-programming/
