from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from checkpoint_audit import audit
from patching import TensorEdit, forward_with_patch, patch_state
from probe import ModuleProbe, tensor_summary


def test_checkpoint_audit_complete_small_file(tmp_path):
    path = tmp_path / "safe.safetensors"
    save_file({"ok.weight": torch.tensor([[0., 1.], [2., -3.]]),
               "bad.bias": torch.tensor([1.0, float("nan")])}, str(path))
    report = audit(path, limit=32, chunk_elements=16)
    assert report["tensor_count"] == 2
    assert len(report["sha256"]) == 64
    stat = {s["name"]: s for s in report["tensors"]}
    assert stat["ok.weight"]["complete_scan"]
    assert stat["ok.weight"]["zero_fraction_of_sample"] == 0.25
    assert stat["bad.bias"]["nan_in_sample"] == 1
    assert report["warnings"]


def test_checkpoint_partial_sampling_is_labeled(tmp_path):
    path = tmp_path / "huge.safetensors"
    save_file({"tensor": torch.arange(1000, dtype=torch.float32).reshape(100, 10)}, str(path))
    report = audit(path, limit=100, chunk_elements=50)
    info = report["tensors"][0]
    assert info["scanned_elements"] <= 100
    assert info["complete_scan"] is False


def test_checkpoint_reject_pickle_format(tmp_path):
    with pytest.raises(ValueError):
        audit(tmp_path / "x.pt")


def test_patching_does_not_mutate_original():
    model = nn.Linear(2, 1)
    model.register_buffer("sentinel", torch.tensor(17.0))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    base = {name: value.clone() for name, value in model.state_dict().items()}
    grads = {name: value.grad.clone() for name, value in model.named_parameters()}
    modes = {name: module.training for name, module in model.named_modules()}
    before = model(torch.ones(1, 2)).detach().clone()
    overrides, manifest = patch_state(model, [TensorEdit("weight", (0, 1), "add", 5.0)])
    after = forward_with_patch(model, overrides, torch.ones(1, 2))
    assert not torch.allclose(before, after)
    assert all(torch.equal(value, base[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value.grad, grads[name]) for name, value in model.named_parameters())
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert manifest["changed_elements"] == 1
    assert manifest["base_untouched"]


def test_patch_rejects_nonfinite_and_large_footprint():
    model = nn.Linear(3, 2)
    with pytest.raises(ValueError):
        patch_state(model, [TensorEdit("weight", (0, 0), "set", float("nan"))])
    with pytest.raises(ValueError):
        patch_state(model, [TensorEdit("weight", (slice(None), slice(None)), "set", 0.0)],
                    max_changed_fraction=0.01)


def test_patch_rejects_tied_weights():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 2, bias=False)
            self.b = nn.Linear(2, 2, bias=False)
            self.b.weight = self.a.weight
        def forward(self, x):
            return self.b(self.a(x))
    model = Tied()
    with pytest.raises(ValueError, match="tied/shared"):
        patch_state(model, [TensorEdit("a.weight", (0, 1), "set", 2.0)])


def test_hook_cleanup_after_exception():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    try:
        with ModuleProbe(model, ["0", "2"]) as probe:
            _ = model(torch.ones(4, 2))
            assert len(probe.records["0"]["activation"]) == 1
            raise RuntimeError("planned")
    except RuntimeError:
        pass
    assert len(model[0]._forward_hooks) == 0
    assert len(model[2]._forward_hooks) == 0


def test_demo_end_to_end():
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "toy_case.py")],
                            check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    s = report["scores"]
    assert s["development_weight_patch"] > s["development_baseline"]
    assert s["development_causal_activation_ablation"] == s["development_weight_patch"]
    assert s["sealed_test_weight_patch"] == 1.0
    assert s["preservation_weight_patch"] == s["preservation_baseline"]
    assert report["original_model_unchanged"]
    controls = report["matched_footprint_controls"]
    assert {(c["column"], c["delta"]) for c in controls} == {
        (0, -6.0), (0, 6.0), (1, -6.0), (1, 6.0)
    }
    assert all(c["changed_entries"] == 1 and c["delta_l2"] == 6.0 for c in controls)
    coincident = [c for c in controls if c["coincides_with_repair"]]
    assert len(coincident) == 1
    assert coincident[0]["development_score"] == s["development_weight_patch"]


def test_causal_capture_and_patch_changes_only_selected_channel():
    from causal import capture_activation, inject_activation, run_patch_pair
    model = nn.Sequential(nn.Identity(), nn.Linear(2, 1, bias=False)).eval()
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[2.0, -3.0]]))
    clean = torch.tensor([[1.0, 0.0]])
    corrupt = torch.tensor([[1.0, 1.0]])
    with torch.inference_mode():
        original, activation = capture_activation(model, "0", clean)
    assert activation.equal(clean)
    result = run_patch_pair(model, "0", (clean,), (corrupt,), selection=(slice(None), 1))
    assert torch.allclose(result["patched"], result["clean"])
    assert result["intervention"]["calls"] == 1
    assert len(model[0]._forward_hooks) == 0


def test_causal_shape_mismatch_does_not_leak_hook():
    from causal import inject_activation
    model = nn.Sequential(nn.ReLU(), nn.Linear(2, 1)).eval()
    with pytest.raises(ValueError, match="mismatch"):
        with inject_activation(model, "0", torch.zeros(3, 2)):
            model(torch.ones(2, 2))
    assert len(model[0]._forward_hooks) == 0


def test_checkpoint_compare(tmp_path):
    from compare_checkpoints import compare
    base = tmp_path / "a.safetensors"
    patch = tmp_path / "b.safetensors"
    save_file({"weight": torch.zeros(2, 3)}, str(base))
    save_file({"weight": torch.tensor([[0., 0., 0.], [0., 0., 2.]])}, str(patch))
    report = compare(base, patch, limit=20, chunk_elements=12)
    assert report["tensors"][0]["different_sampled_entries"] == 1
    assert report["tensors"][0]["complete_scan"]


def test_paired_bootstrap():
    from metrics import paired_bootstrap_difference
    result = paired_bootstrap_difference([0., 1., 0., 1.], [1., 1., 1., 1.],
                                         n_resamples=300, seed=4)
    assert result["mean_difference_after_minus_before"] == .5
    assert result["percentile_bootstrap_ci_low"] <= .5 <= result["percentile_bootstrap_ci_high"]



def test_scalar_safetensors_are_scanned(tmp_path):
    from compare_checkpoints import compare
    a, b = tmp_path / "scalar1.safetensors", tmp_path / "scalar2.safetensors"
    save_file({"offset": torch.tensor(2.0)}, str(a))
    save_file({"offset": torch.tensor(3.0)}, str(b))
    report = audit(a, limit=16, chunk_elements=8)
    assert report["tensors"][0]["complete_scan"] is True
    assert report["tensors"][0]["finite_mean"] == 2.0
    delta = compare(a, b, limit=16, chunk_elements=8)
    assert delta["tensors"][0]["delta_l2_sample"] == 1.0


@pytest.mark.parametrize("tensor", [
    torch.arange(60.0).reshape(3, 4, 5).transpose(0, 2),
    torch.arange(60.0).reshape(6, 10)[::2, 1::2],
    torch.arange(4.0).expand(7, 4),
])
@pytest.mark.parametrize("budget", [1, 7, 100])
def test_probe_samples_logical_order_with_bounded_count(tensor, budget):
    import math

    expected = tensor.contiguous().view(-1)[::max(1, math.ceil(tensor.numel() / budget))][:budget]
    result = tensor_summary(tensor, budget)
    assert result["sampled"] == expected.numel() <= budget
    assert result["mean_sample"] == pytest.approx(expected.mean().item())
    assert result["l2_sample"] == pytest.approx(expected.norm().item())
    assert result["shape"] == list(tensor.shape)


def test_probe_noncontiguous_allocations_follow_sample_budget():
    tensor = torch.ones(1024, 2048).T
    # The old flatten-before-sample path allocates a full 8 MiB contiguous copy.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                profile_memory=True) as profile:
        result = tensor_summary(tensor, sample_elements=64)
    assert result["sampled"] == 64
    assert max(event.cpu_memory_usage for event in profile.events()) < 65_536


def test_probe_scalar_empty_nonfinite_and_float64():
    assert tensor_summary(torch.tensor(3.0))["mean_sample"] == 3.0
    assert tensor_summary(torch.empty(0, 2))["numel"] == 0
    assert tensor_summary(torch.tensor([float("nan"), float("inf"), 1.0]))["nonfinite_sample"] == 2
    wide = tensor_summary(torch.tensor([1e100], dtype=torch.float64))
    assert wide["nonfinite_sample"] == 0
    assert wide["mean_sample"] == 1e100
    with pytest.raises(ValueError):
        tensor_summary(torch.ones(2), 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_patch_manifest_matches_native_changes(dtype):
    model = nn.Linear(2, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(1.0)
    original = model.weight.detach().clone()
    state, manifest = patch_state(model, [TensorEdit("weight", (0, 1), "add", 0.25)])
    record = manifest["edits"][0]
    assert manifest["changed_elements"] == record["changed_entries"] == 1
    assert record["delta_l2"] == 0.25
    assert record["relative_l2"] == pytest.approx(0.125)
    assert torch.equal(model.weight, original)
    assert state["weight"].dtype == dtype


def test_float64_patch_preserves_sub_float32_change():
    model = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1.0)
    state, manifest = patch_state(model, [TensorEdit("weight", (0, 0), "add", 1e-10)])
    delta = float(state["weight"][0, 0] - model.weight.detach()[0, 0])
    record = manifest["edits"][0]
    assert manifest["changed_elements"] == record["changed_entries"] == 1
    assert record["delta_l2"] == delta > 0
    assert record["relative_l2"] > 0


def test_patch_stats_cross_chunks_and_noncontiguous_parameters():
    from patching import _edit_stats

    original = torch.ones(512, 512, dtype=torch.float64).T
    edited = original.clone()
    edited[0, 0] += 1e-10
    edited[-1, -1] += 0.25
    stats = _edit_stats(original, edited, chunk_elements=1000)
    assert stats["changed_entries"] == 2
    assert stats["delta_l2"] == pytest.approx(0.25)
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(original))
    _, manifest = patch_state(model, [TensorEdit("weight", (1, 1), "add", 0.5)])
    assert manifest["edits"][0]["delta_l2"] == 0.5


def test_checkpoint_only_clis_need_no_model_code_or_dataset(tmp_path):
    base, other = tmp_path / "base.safetensors", tmp_path / "other.safetensors"
    save_file({"weight": torch.zeros(2, 3)}, str(base))
    save_file({"weight": torch.tensor([[0., 0., 0.], [0., 0., 2.]]),
               "new.bias": torch.zeros(1)}, str(other))
    (tmp_path / "model.py").write_text("raise RuntimeError('model code must not run')\n")
    original = {path: path.read_bytes() for path in (base, other)}
    audit_run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/checkpoint_audit.py"), str(base)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    diff_run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/compare_checkpoints.py"), str(base), str(other)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    audit_report, delta_report = json.loads(audit_run.stdout), json.loads(diff_run.stdout)
    assert audit_report["tensor_count"] == 1
    assert delta_report["only_other"] == ["new.bias"]
    assert delta_report["tensors"][0]["different_sampled_entries"] == 1
    assert all(path.read_bytes() == data for path, data in original.items())
