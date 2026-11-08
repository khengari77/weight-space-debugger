from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from package_skill import package_files, source_manifest, validate_package


@pytest.fixture
def package_copy(tmp_path):
    destination = tmp_path / "weight-space-debugger"
    for path in [*package_files(ROOT), ROOT / "MANIFEST.json"]:
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return destination


def test_package_validates_metadata_links_and_hashes():
    assert validate_package(ROOT) == []


def test_package_detects_changed_source(package_copy):
    path = package_copy / "scripts/probe.py"
    path.write_text(path.read_text() + "\n# altered after packaging\n")
    assert any("manifest differs" in error for error in validate_package(package_copy))


def test_package_detects_broken_resource_link_with_fresh_hashes(package_copy):
    path = package_copy / "SKILL.md"
    path.write_text(path.read_text() + "\n[Missing guidance](references/absent.md)\n")
    (package_copy / "MANIFEST.json").write_text(json.dumps(source_manifest(package_copy)))
    assert any("broken or external local link" in error for error in validate_package(package_copy))


def test_package_selection_excludes_private_and_generated_files(package_copy):
    selected_before = set(package_files(package_copy))
    for name in (".env", ".git/config", ".venv/runtime.txt", "reports/private.json",
                 "examples/model.safetensors", "scripts/__pycache__/probe.pyc"):
        path = package_copy / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("must never be packaged")
    assert validate_package(package_copy) == []
    assert set(package_files(package_copy)) == selected_before


def test_package_rejects_disabled_discovery(package_copy):
    path = package_copy / "agents/openai.yaml"
    path.write_text(path.read_text() + "\npolicy:\n  allow_implicit_invocation: false\n")
    assert any("automatic discovery" in error for error in validate_package(package_copy))
