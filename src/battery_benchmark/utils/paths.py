"""Path resolution utilities.

All benchmark paths are relative to a configurable root.  This module resolves
relative paths against the appropriate root and ensures output directories exist.

Rules:
- Absolute paths are returned unchanged.
- Relative paths are resolved against the provided base directory.
- Never hard-code ``/content/...`` or ``/Users/...`` in source files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_output_path(cfg: dict[str, Any], repo_root: str | Path | None = None) -> Path:
    """Resolve the output directory for an experiment.

    Precedence:
    1. ``cfg["output_path"]`` if it is an absolute path — returned as-is.
    2. ``cfg["output_path"]`` relative to ``repo_root`` (or CWD) — used when
       ``output_path`` is a multi-component path (e.g. ``outputs/smoke_test``).
    3. ``cfg["paths"]["output_root"] / cfg["output_path"]`` — only when
       ``output_path`` is a bare single-level name (e.g. ``smoke_test``).

    This prevents double-nesting when ``output_root: outputs`` and
    ``output_path: outputs/smoke_test`` are both set.
    """
    out = cfg.get("output_path", "outputs/default")
    out = Path(out)
    if out.is_absolute():
        return out

    base = Path(repo_root) if repo_root is not None else Path.cwd()

    # If the path has multiple components it is already rooted at repo_root
    if len(out.parts) > 1:
        return (base / out).resolve()

    # Single-level bare name: prepend output_root if configured
    paths = cfg.get("paths", {})
    if paths.get("output_root"):
        return (base / Path(paths["output_root"]) / out).resolve()

    return (base / out).resolve()


def resolve_paths(cfg: dict[str, Any], repo_root: str | Path | None = None) -> dict[str, Path]:
    """Return a dictionary of resolved, absolute Path objects for all path keys.

    Parameters
    ----------
    cfg:
        Full experiment config dict (including ``cfg["paths"]``).
    repo_root:
        Base directory for relative paths.  Defaults to CWD.

    Returns
    -------
    dict[str, Path]
        Keys from ``cfg["paths"]`` with resolved absolute Path values.
    """
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    paths_cfg = cfg.get("paths", {})
    resolved: dict[str, Path] = {}
    for key, val in paths_cfg.items():
        if val is None:
            continue
        p = Path(val)
        resolved[key] = p if p.is_absolute() else (base / p).resolve()
    return resolved


def ensure_output_dirs(output_root: Path) -> None:
    """Create standard output sub-directories under *output_root*."""
    for sub in ("checkpoints", "logs", "metrics", "predictions", "plots"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk up the directory tree to find the repository root.

    Detection heuristic: presence of ``pyproject.toml`` or ``setup.py``.
    Falls back to CWD if not found.
    """
    current = Path(start).resolve() if start else Path.cwd().resolve()
    for directory in [current, *current.parents]:
        if (directory / "pyproject.toml").exists() or (directory / "setup.py").exists():
            return directory
    return Path.cwd()
