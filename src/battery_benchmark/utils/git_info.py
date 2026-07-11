"""Retrieve the current Git commit hash for reproducibility logging."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_commit_hash(repo_path: str | Path | None = None) -> str:
    """Return the short Git commit hash of the current HEAD.

    Returns ``"unknown"`` if Git is not available or the directory is not a
    Git repository.
    """
    cwd = str(repo_path) if repo_path is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def get_full_commit_hash(repo_path: str | Path | None = None) -> str:
    """Return the full 40-character Git commit hash."""
    cwd = str(repo_path) if repo_path is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def is_dirty(repo_path: str | Path | None = None) -> bool:
    """Return True if the working tree has uncommitted changes."""
    cwd = str(repo_path) if repo_path is not None else None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
