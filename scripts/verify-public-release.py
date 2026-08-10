#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


BLOCKED_SUFFIXES = {
    ".avi",
    ".csv",
    ".jsonl",
    ".key",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".pem",
    ".pyc",
    ".tfplan",
    ".tfstate",
}
BLOCKED_DIRECTORIES = {
    ".pytest_cache",
    "__pycache__",
}
BLOCKED_NAMES = {
    ".env",
    ".secrets",
    "terraform.tfvars",
}
REQUIRED_NOTICE_FILES = {
    "LICENSE",
    "LICENSE-DERIVED-DATA.md",
    "LICENSE-PAPER.md",
    "THIRD_PARTY_NOTICES.md",
    "paper/inputs/README.md",
}
TEXT_SUFFIXES = {
    "",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".txt",
}
BLOCKED_CONTENT = (
    re.compile("/" + "Users" + "/"),
    re.compile("20" + "2511-" + "scene" + "-search", re.IGNORECASE),
    re.compile("scene" + "-search", re.IGNORECASE),
    re.compile("phase" + "2", re.IGNORECASE),
    re.compile("peg" + "asus", re.IGNORECASE),
    re.compile("AK" + "IA[0-9A-Z]{16}"),
    re.compile("AS" + "IA[0-9A-Z]{16}"),
    re.compile("BEGIN " + "(?:RSA |EC |OPENSSH )?PRIVATE KEY"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_text(path: Path, text: str, errors: list[str]) -> None:
    for pattern in BLOCKED_CONTENT:
        if pattern.search(text):
            errors.append(f"blocked content {pattern.pattern!r}: {path}")


def verify_manifest(root: Path, errors: list[str]) -> None:
    paper_dir = root / "paper"
    input_dir = paper_dir / "inputs"
    inputs = sorted(input_dir.glob("*.json"))
    if len(inputs) != 21:
        errors.append(f"expected 21 paper inputs, found {len(inputs)}")
    for path in inputs:
        try:
            with path.open(encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid JSON {path}: {error}")

    manifest_path = paper_dir / "road-paper-artifact-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid artifact manifest: {error}")
        return

    for group in ("inputs", "artifacts"):
        for item in manifest.get(group, []):
            path = paper_dir / item["path"]
            if not path.is_file():
                errors.append(f"manifest file is missing: {path}")
                continue
            actual = sha256_file(path)
            if actual != item["sha256"]:
                errors.append(f"manifest hash mismatch: {path}")


def verify_tree(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in sorted(REQUIRED_NOTICE_FILES):
        if not (root / relative).is_file():
            errors.append(f"required license notice is missing: {relative}")

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if any(part in BLOCKED_DIRECTORIES for part in relative.parts):
            errors.append(f"blocked generated directory: {relative}")
            continue
        if path.is_symlink():
            try:
                path.resolve().relative_to(root)
            except ValueError:
                errors.append(f"external symlink: {relative}")
            continue
        if not path.is_file():
            continue
        if path.name in BLOCKED_NAMES:
            errors.append(f"blocked file name: {relative}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f"blocked file type: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == "Makefile":
            try:
                scan_text(path, path.read_text(encoding="utf-8"), errors)
            except UnicodeDecodeError:
                errors.append(f"non-text data in text file: {relative}")

    pdf = root / "paper" / "ROAD_COSMOS3_SCENE_LABELING_PAPER.pdf"
    if not pdf.is_file():
        errors.append("final PDF is missing")
    else:
        with tempfile.TemporaryDirectory() as temporary:
            text_path = Path(temporary) / "paper.txt"
            try:
                subprocess.run(
                    ["pdftotext", str(pdf), str(text_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                scan_text(
                    pdf,
                    text_path.read_text(encoding="utf-8"),
                    errors,
                )
            except FileNotFoundError:
                errors.append("pdftotext is required for PDF content scanning")
            except subprocess.CalledProcessError as error:
                errors.append(f"PDF text extraction failed: {error.stderr}")

    verify_manifest(root, errors)
    return errors


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    errors = verify_tree(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Publication security verification passed.")


if __name__ == "__main__":
    main()
