#!/usr/bin/env python
"""Register and optionally clone an external repository into the workspace.

Usage
-----
python scripts/register_external_repo.py \\
    --name battery-state-of-charge-estimation \\
    --url  https://github.com/sautee/battery-state-of-charge-estimation.git \\
    --category references

Optional flags:
  --clone          Actually run git clone (default: register metadata only)
  --branch BRANCH  Branch to clone (default: main)
  --force          Overwrite an existing directory
  --registry PATH  Path to registry YAML (default: external_repos/repo_registry.local.yaml)
  --no-registry    Skip writing to registry

Rules enforced:
  - Refuses to overwrite existing destination unless --force.
  - Destination is always inside external_repos/ to prevent accidental nesting.
  - Credentials must not appear in --url; use SSH keys or HTTPS tokens externally.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

VALID_CATEGORIES = ("references", "reproductions", "archived")
VALID_STATUSES = (
    "reference_only",
    "under_review",
    "reproduction_in_progress",
    "code_migration_in_progress",
    "archived",
    "rejected",
)

DEFAULT_REGISTRY = _REPO_ROOT / "external_repos" / "repo_registry.local.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register an external repository in the workspace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="Short repository name (used as directory name)")
    parser.add_argument("--url", required=True, help="Git clone URL (HTTPS or SSH)")
    parser.add_argument(
        "--category",
        required=True,
        choices=VALID_CATEGORIES,
        help="Destination sub-directory",
    )
    parser.add_argument("--branch", default="main", help="Branch to clone")
    parser.add_argument(
        "--status",
        default=None,
        choices=VALID_STATUSES,
        help="Registry status (inferred from category if not given)",
    )
    parser.add_argument("--purpose", default="", help="One-line description of why this repo is here")
    parser.add_argument("--paper", default="", help="DOI or citation key for associated paper")
    parser.add_argument("--license", default="", dest="license_", help="License identifier")
    parser.add_argument("--notes", default="", help="Additional notes")
    parser.add_argument("--clone", action="store_true", help="Run git clone after registering")
    parser.add_argument("--force", action="store_true", help="Overwrite existing destination directory")
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="Path to repo_registry.local.yaml",
    )
    parser.add_argument("--no-registry", action="store_true", help="Skip writing to registry")
    return parser.parse_args()


def infer_status(category: str) -> str:
    return {
        "references": "reference_only",
        "reproductions": "under_review",
        "archived": "archived",
    }[category]


def get_commit_hash(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def detect_license(repo_path: Path) -> str:
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE", "COPYING"):
        if (repo_path / name).exists():
            return name
    return ""


def write_registry_entry(registry_path: Path, entry: dict) -> None:
    """Append or update an entry in the registry YAML."""
    import yaml  # lazy import

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if registry_path.exists():
        with registry_path.open() as f:
            existing = yaml.safe_load(f) or {}

    repos: list = existing.get("repositories", [])

    # Update existing entry with same name, or append
    updated = False
    for i, r in enumerate(repos):
        if r.get("name") == entry["name"]:
            repos[i] = entry
            updated = True
            break
    if not updated:
        repos.append(entry)

    existing["repositories"] = repos
    existing.setdefault("_generated", str(datetime.now().isoformat(timespec="seconds")))
    existing["_generated"] = str(datetime.now().isoformat(timespec="seconds"))

    with registry_path.open("w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    action = "Updated" if updated else "Added"
    print(f"{action} '{entry['name']}' in {registry_path}")


def main() -> None:
    args = parse_args()

    dest = _REPO_ROOT / "external_repos" / args.category / args.name

    # Safety check: destination must be inside external_repos/
    try:
        dest.resolve().relative_to((_REPO_ROOT / "external_repos").resolve())
    except ValueError:
        print(f"ERROR: destination '{dest}' is outside external_repos/. Aborting.", file=sys.stderr)
        sys.exit(1)

    if dest.exists() and not args.force:
        print(
            f"ERROR: '{dest}' already exists. Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.clone:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and args.force:
            import shutil
            shutil.rmtree(dest)
        print(f"Cloning {args.url} → {dest}")
        result = subprocess.run(
            ["git", "clone", "--branch", args.branch, args.url, str(dest)],
            check=False,
        )
        if result.returncode != 0:
            print("ERROR: git clone failed.", file=sys.stderr)
            sys.exit(1)
        print("Clone complete.")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dest}")
        print("(Pass --clone to also clone the repository)")

    commit_hash = get_commit_hash(dest) if dest.exists() else ""
    license_file = detect_license(dest) if dest.exists() else ""

    entry = {
        "name": args.name,
        "url": args.url,
        "local_path": str(dest.relative_to(_REPO_ROOT)),
        "commit_hash": commit_hash,
        "branch": args.branch,
        "license": args.license_ or license_file,
        "paper": args.paper,
        "purpose": args.purpose,
        "status": args.status or infer_status(args.category),
        "notes": args.notes,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }

    if not args.no_registry:
        try:
            import yaml  # noqa: F401
            write_registry_entry(Path(args.registry), entry)
        except ImportError:
            print("WARNING: pyyaml not available; skipping registry write.")
            print("Entry that would have been written:")
            for k, v in entry.items():
                print(f"  {k}: {v!r}")

    print()
    print("Summary:")
    print(f"  name        : {entry['name']}")
    print(f"  category    : {args.category}")
    print(f"  local_path  : {entry['local_path']}")
    print(f"  commit_hash : {entry['commit_hash'] or '(not yet cloned)'}")
    print(f"  status      : {entry['status']}")
    print()
    print("Next steps:")
    print(f"  Inspect code  : ls {dest}")
    print(f"  Audit repo    : python scripts/audit_external_repo.py --path {dest}")
    print(f"  View registry : cat {args.registry}")


if __name__ == "__main__":
    main()
