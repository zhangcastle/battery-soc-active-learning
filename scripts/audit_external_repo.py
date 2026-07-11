#!/usr/bin/env python
"""Audit an external repository and print a metadata report.

Usage
-----
python scripts/audit_external_repo.py --path external_repos/references/some-repo

# Audit all registered repos
python scripts/audit_external_repo.py --all

# Write audit results to a file
python scripts/audit_external_repo.py --path ... --output docs/external_repo_notes/audit.yaml

Reports:
  - Repository name
  - Origin remote URL
  - Current branch
  - Commit hash (full and short)
  - Dirty working-tree status
  - Detected license file(s)
  - Total size on disk
  - Nested Git repository status (warns if found inside)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit an external repository",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path", help="Path to the external repository to audit")
    group.add_argument("--all", action="store_true", help="Audit all repos in registry")
    parser.add_argument(
        "--registry",
        default=str(_REPO_ROOT / "external_repos" / "repo_registry.local.yaml"),
        help="Registry YAML to use with --all",
    )
    parser.add_argument("--output", default=None, help="Write YAML report to this file")
    return parser.parse_args()


def _git(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path)] + list(args),
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _dir_size_mb(path: Path) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += (Path(dirpath) / f).stat().st_size
            except OSError:
                pass
    return round(total / 1_048_576, 2)


def _find_license(path: Path) -> list[str]:
    candidates = []
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE", "COPYING", "NOTICE"):
        if (path / name).exists():
            candidates.append(name)
    return candidates


def _find_nested_git(path: Path) -> list[str]:
    """Find any .git directories nested inside *path* (excluding the root)."""
    nested = []
    for dirpath, dirnames, _ in os.walk(path):
        dp = Path(dirpath)
        if dp == path:
            continue
        if ".git" in dirnames:
            nested.append(str(dp.relative_to(path)))
        # Do not descend into .git directories
        dirnames[:] = [d for d in dirnames if d != ".git"]
    return nested


def audit_repo(path: Path) -> dict[str, Any]:
    """Collect audit metadata for the repository at *path*."""
    path = Path(path).resolve()
    if not path.exists():
        return {"path": str(path), "error": "Directory does not exist"}

    # Only consider it a git repo if it owns its own .git directory.
    # This prevents the audit from confusingly inheriting the benchmark's parent repo.
    is_git = (path / ".git").exists()

    report: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "is_git_repo": is_git,
    }

    if is_git:
        report["origin_url"] = _git(path, "remote", "get-url", "origin")
        report["branch"] = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
        report["commit_hash_full"] = _git(path, "rev-parse", "HEAD")
        report["commit_hash_short"] = _git(path, "rev-parse", "--short", "HEAD")
        dirty_output = _git(path, "status", "--porcelain")
        report["is_dirty"] = bool(dirty_output)
        if dirty_output:
            report["dirty_files"] = dirty_output.splitlines()[:10]  # first 10
    else:
        report["warning"] = "Not a Git repository — commit hash unavailable"

    report["license_files"] = _find_license(path)
    report["size_mb"] = _dir_size_mb(path)
    report["nested_git_repos"] = _find_nested_git(path)

    if report["nested_git_repos"]:
        report["nested_git_warning"] = (
            "Nested Git repositories found. These should NOT be committed to the "
            "benchmark repo. Add them to .gitignore or remove them."
        )

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable audit report."""
    print("=" * 60)
    print(f"  Repository: {report.get('name', 'unknown')}")
    print("=" * 60)

    if "error" in report:
        print(f"  ERROR: {report['error']}")
        return

    print(f"  Path         : {report['path']}")
    print(f"  Is Git repo  : {report.get('is_git_repo', False)}")

    if report.get("is_git_repo"):
        print(f"  Origin URL   : {report.get('origin_url', 'N/A')}")
        print(f"  Branch       : {report.get('branch', 'N/A')}")
        print(f"  Commit (full): {report.get('commit_hash_full', 'N/A')}")
        print(f"  Commit (short): {report.get('commit_hash_short', 'N/A')}")
        dirty = report.get("is_dirty", False)
        print(f"  Dirty        : {dirty}")
        if dirty and "dirty_files" in report:
            print("  Dirty files  :")
            for f in report["dirty_files"]:
                print(f"    {f}")
    elif "warning" in report:
        print(f"  Warning      : {report['warning']}")

    licenses = report.get("license_files", [])
    print(f"  License files: {licenses if licenses else 'None detected'}")
    print(f"  Size on disk : {report.get('size_mb', 0)} MB")

    nested = report.get("nested_git_repos", [])
    if nested:
        print(f"  NESTED GIT   : {nested}")
        print(f"  WARNING      : {report.get('nested_git_warning', '')}")
    else:
        print("  Nested Git   : None")

    print()


def main() -> None:
    args = parse_args()

    reports = []

    if args.path:
        report = audit_repo(Path(args.path))
        reports.append(report)
        print_report(report)
    else:
        # --all: read registry
        registry_path = Path(args.registry)
        if not registry_path.exists():
            print(f"Registry not found: {registry_path}", file=sys.stderr)
            print("Create it with scripts/register_external_repo.py or copy from repo_registry.example.yaml")
            sys.exit(1)
        try:
            import yaml
            with registry_path.open() as f:
                registry = yaml.safe_load(f) or {}
        except ImportError:
            print("ERROR: pyyaml is required for --all mode.", file=sys.stderr)
            sys.exit(1)

        repos = registry.get("repositories", [])
        if not repos:
            print("No repositories registered.")
            return
        for entry in repos:
            local = _REPO_ROOT / entry.get("local_path", "")
            report = audit_repo(local)
            reports.append(report)
            print_report(report)

    if args.output and reports:
        try:
            import yaml
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                yaml.dump({"audit_reports": reports}, f, default_flow_style=False)
            print(f"Audit report written to: {args.output}")
        except ImportError:
            print("WARNING: pyyaml not available; cannot write YAML output.")


if __name__ == "__main__":
    main()
