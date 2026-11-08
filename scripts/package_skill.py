#!/usr/bin/env python3
"""Validate the portable skill package; --update refreshes its source manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml


PACKAGE_GLOBS = (
    ".gitignore", "SKILL.md", "README.md", "LICENSE", "requirements.txt",
    "requirements-dev.txt", "pytest.ini", "agents/*.yaml", "scripts/*.py", "references/*.md",
    "assets/*.md", "assets/*.json", "examples/*.md", "examples/*.py",
    "examples/toy-result.json", "tests/*.py", ".github/workflows/*.yml",
)


def package_files(root: Path) -> list[Path]:
    """Select source resources explicitly; never traverse Git or environment trees."""
    return sorted({path for pattern in PACKAGE_GLOBS for path in root.glob(pattern)
                   if path.is_file()})


def source_manifest(root: Path) -> dict:
    records = []
    for path in package_files(root):
        if path.is_symlink():
            raise ValueError(f"package source must not be a symlink: {path.relative_to(root)}")
        data = path.read_bytes()
        records.append({"path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return {"package": "weight-space-debugger", "skill_format": "agentskills.io",
            "files": records,
            "notes": "Synthetic demonstration; no blind real-model reasoning or repair validation."}


def validate_package(root: Path) -> list[str]:
    root = root.resolve()
    errors = []
    for required in ("SKILL.md", "README.md", "LICENSE", "requirements.txt",
                     "requirements-dev.txt", "agents/openai.yaml", "MANIFEST.json",
                     ".github/workflows/validate.yml"):
        if not (root / required).is_file():
            errors.append(f"missing required file: {required}")
    if errors:
        return errors
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", skill, re.DOTALL)
    if not match:
        errors.append("SKILL.md must begin with YAML frontmatter")
    else:
        metadata = yaml.safe_load(match.group(1))
        if not isinstance(metadata, dict):
            errors.append("skill frontmatter must be a mapping")
        else:
            name = metadata.get("name", "")
            if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                    or len(name) > 64 or name != root.name):
                errors.append("skill name must be valid and match its directory")
            description = metadata.get("description", "")
            if not isinstance(description, str) or not 1 <= len(description.strip()) <= 1024:
                errors.append("skill description must be a nonempty string of <=1024 characters")
            if metadata.get("license") != "MIT":
                errors.append("skill license must identify the bundled MIT license")
    ui = yaml.safe_load((root / "agents/openai.yaml").read_text(encoding="utf-8"))
    interface = ui.get("interface", {}) if isinstance(ui, dict) else {}
    short = interface.get("short_description", "")
    if not isinstance(short, str) or not 25 <= len(short) <= 64:
        errors.append("UI short description must have 25-64 characters")
    if not isinstance(interface.get("display_name"), str) or not interface["display_name"].strip():
        errors.append("UI display name is missing")
    prompt = interface.get("default_prompt", "")
    if not isinstance(prompt, str) or "$weight-space-debugger" not in prompt:
        errors.append("UI default prompt must invoke $weight-space-debugger")
    if isinstance(ui, dict) and ui.get("policy", {}).get("allow_implicit_invocation", True) is not True:
        errors.append("automatic discovery must remain enabled")
    for path in package_files(root):
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
            url = urlsplit(target.strip("<>"))
            if url.scheme or not url.path:
                continue
            destination = (path.parent / unquote(url.path)).resolve()
            if not destination.is_relative_to(root) or not destination.is_file():
                errors.append(f"broken or external local link in {relative}: {target}")
    current = source_manifest(root)
    saved = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if saved != current:
        errors.append("source manifest differs; run python scripts/package_skill.py --update")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--update", action="store_true", help="regenerate MANIFEST.json before validation")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.update:
            (root / "MANIFEST.json").write_text(
                json.dumps(source_manifest(root), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        errors = validate_package(root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"Package validation failed: {exc}\n")
    if errors:
        parser.exit(1, "Package validation failed:\n" + "\n".join(f"- {error}" for error in errors) + "\n")
    print(f"Validated skill metadata, local links, JSON and {len(package_files(root))} source hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
