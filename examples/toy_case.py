#!/usr/bin/env python3
"""A fully transparent, intentionally broken 2-feature policy/classifier.

Goal: predict feature x0 irrespective of nuisance x1. The hand-designed model
mistakenly suppresses positives when x1=1. We test a causal activation
intervention before writing one reversible weight delta. This is a *didactic*
case, not a model trained on real data and not a reproduction of the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from patching import TensorEdit, forward_with_patch, patch_state  # noqa: E402
from probe import ModuleProbe  # noqa: E402


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2, 2, bias=False)
        self.activation = nn.ReLU()
        self.action_head = nn.Linear(2, 1)
        with torch.no_grad():
            self.features.weight.copy_(torch.eye(2))
            self.action_head.weight.copy_(torch.tensor([[4.0, -6.0]]))
            self.action_head.bias.fill_(-1)

    def forward(self, x):
        return self.action_head(self.activation(self.features(x))).squeeze(-1)


def make_dataset(seed: int, count: int, *, nuisance_mode: str = "random"):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randint(0, 2, (count,), generator=g).float()
    if nuisance_mode == "always":
        x1 = torch.ones(count)
    elif nuisance_mode == "never":
        x1 = torch.zeros(count)
    else:
        x1 = torch.randint(0, 2, (count,), generator=g).float()
    x = torch.stack([x0, x1], dim=1)
    return x, x0.bool()


def score(logits, labels) -> float:
    return ((logits > 0) == labels).float().mean().item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, help="optional JSON result output")
    args = ap.parse_args()
    model = TinyPolicy().eval()
    base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    diag_x, diag_y = make_dataset(11, 1000)
    dev_x, dev_y = make_dataset(22, 1000)
    test_x, test_y = make_dataset(33, 1000, nuisance_mode="always")
    preservation_x, preservation_y = make_dataset(44, 1000, nuisance_mode="never")

    with torch.inference_mode(), ModuleProbe(model, ["features", "activation", "action_head"], max_events=2) as probe:
        base_dev = model(dev_x)
        base_diag = model(diag_x)

    # Causal intervention: zero only the nuisance activation during the forward
    # pass; observe if suppression disappears. The original weights do not move.
    def neutralize_nuisance(_mod, _inputs, output):
        safe = output.clone()
        safe[:, 1] = 0
        return safe

    hook = model.activation.register_forward_hook(neutralize_nuisance)
    try:
        with torch.inference_mode():
            causal_logits = model(dev_x)
    finally:
        hook.remove()

    # Hypothesis H1: nuisance feature's downstream weight -6 causes suppression.
    # Patch candidate was selected on diagnostic/development data; test remains sealed.
    overrides, manifest = patch_state(
        model, [TensorEdit("action_head.weight", (0, 1), "set", 0.0)],
        max_changed_fraction=0.15,
    )
    with torch.inference_mode():
        candidate_dev = forward_with_patch(model, overrides, dev_x)
        # Sham vs real control: random equal-norm directions in same 1x2 layer.
        random_results = []
        g = torch.Generator().manual_seed(101)
        for _ in range(20):
            v = torch.randn(2, generator=g)
            v = 6.0 * v / v.norm()
            controls, _ = patch_state(model, [TensorEdit("action_head.weight", (0, slice(None)), "add", v)],
                                      max_changed_fraction=0.4)
            random_results.append(score(forward_with_patch(model, controls, dev_x), dev_y))
        # Exhaust the tiny null set with the repair's exact sparsity and norm.
        # One member is the repair itself, so beating every control is impossible.
        matched_controls = []
        for column in range(2):
            for delta in (-6.0, 6.0):
                controls, control_manifest = patch_state(
                    model, [TensorEdit("action_head.weight", (0, column), "add", delta)],
                    max_changed_fraction=0.15,
                )
                matched_controls.append({
                    "column": column, "delta": delta,
                    "coincides_with_repair": column == 1 and delta == 6.0,
                    "changed_entries": control_manifest["changed_elements"],
                    "delta_l2": control_manifest["edits"][0]["delta_l2"],
                    "development_score": score(forward_with_patch(model, controls, dev_x), dev_y),
                })
        # Independent final evaluation ONLY after the patch was chosen.
        test_base = model(test_x)
        test_patched = forward_with_patch(model, overrides, test_x)
        preserve_base = model(preservation_x)
        preserve_patched = forward_with_patch(model, overrides, preservation_x)
        base_again = model(dev_x)

    output = {
        "description": "synthetic known failure; explanatory demonstration, not blind discovery",
        "hypothesis": "Nuisance feature x1 contributes -6 to logit and causes false negatives",
        "prediction": "Zero x1 activation or remove output weight[0,1], rescuing positives with x1=1",
        "scores": {
            "diagnostic_baseline": score(base_diag, diag_y),
            "development_baseline": score(base_dev, dev_y),
            "development_causal_activation_ablation": score(causal_logits, dev_y),
            "development_weight_patch": score(candidate_dev, dev_y),
            "sealed_test_baseline": score(test_base, test_y),
            "sealed_test_weight_patch": score(test_patched, test_y),
            "preservation_baseline": score(preserve_base, preservation_y),
            "preservation_weight_patch": score(preserve_patched, preservation_y),
            "sham_rerun_dev": score(base_again, dev_y),
            "random_equal_norm_mean_dev": sum(random_results) / len(random_results),
            "random_equal_norm_best_dev": max(random_results),
        },
        "intervention": {"hook_site": "activation", "channel": 1,
                         "target": "action_head.weight[0,1]", "before": -6.0, "after": 0.0},
        "original_model_unchanged": all(
            torch.equal(base_state[name], tensor) for name, tensor in model.state_dict().items()
        ),
        "matched_footprint_controls": matched_controls,
        "control_notes": {
            "dense_random": "20 equal-norm directions changing two entries; not sparsity matched",
            "matched_footprint": "All four one-entry +/-6 edits in this layer; includes the repair",
        },
        "patch_manifest": manifest,
        "probe_events": {name: len(stats["activation"]) for name, stats in probe.records.items()},
        "limitations": ["synthetic, deliberately preconstructed circuit",
                        "binary inputs repeat across splits; independent draws, not unseen tasks",
                        "matched controls include the repair; no superiority claim over all controls",
                        "no generalization to large neural networks implied"]
    }
    payload = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved demo result: {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
