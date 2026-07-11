# External Repositories

This directory is a **temporary workspace** for third-party repositories
inspected, compared, or reproduced during benchmark development.

External repositories are **not part of the benchmark implementation**.
They are excluded from Git (except this README) and must never be imported
directly into `src/`.

---

## Directory layout

```
external_repos/
├── references/      Read-only reference repos (code inspection only)
├── reproductions/   Repos under active testing or comparison
└── archived/        Repos inspected but no longer active
```

---

## Rules

| Rule | Description |
|------|-------------|
| No production imports | `src/` must never import from `external_repos/` |
| No direct copy-paste | Code adopted from external repos must be reviewed, rewritten, documented, tested, and attributed |
| No committed large files | Only README, .gitkeep, and example YAML are tracked |
| No nested submodules | Unless the user explicitly decides to version-pin a dependency |
| No credentials in YAML | `repo_registry.local.yaml` is gitignored |

---

## Cloning a repository into this workspace

```bash
# Clone into references (read-only inspection)
git clone https://github.com/author/repo.git external_repos/references/repo

# Or use the helper script
python scripts/register_external_repo.py \
    --name repo \
    --url https://github.com/author/repo.git \
    --category references
```

---

## Recording URL and commit hash

After cloning, capture metadata:

```bash
python scripts/audit_external_repo.py --path external_repos/references/repo
```

Or write directly to `external_repos/repo_registry.local.yaml` using the
format in `external_repos/repo_registry.example.yaml`.

---

## Colab usage

In Colab, clone external repositories into **local Colab storage**, not Drive:

```python
import subprocess
subprocess.run([
    "git", "clone",
    "https://github.com/author/repo.git",
    "/content/external_repos/references/repo"
], check=True)
```

Configure the root in `configs/paths/colab_drive.yaml`:

```yaml
external_repo_root: /content/external_repos
```

Large or temporary repos are **not** saved to Drive unless local modifications
need to be preserved (see below).

---

## Preserving experimental modifications

If you modify an external repository experimentally, save:

1. A patch file:
   ```bash
   git -C external_repos/references/repo diff > docs/external_repo_notes/repo_changes.patch
   ```

2. The commit hash (from `audit_external_repo.py`).

3. Migration notes in `docs/external_repo_notes/`.

Do **not** commit an entire modified third-party repository into the benchmark
Git history.

---

## Removing an external repository

```bash
rm -rf external_repos/references/repo

# Remove the corresponding entry from repo_registry.local.yaml
```

---

## Files tracked by Git

| File | Tracked |
|------|---------|
| `external_repos/README.md` | Yes |
| `external_repos/.gitkeep` | Yes |
| `external_repos/repo_registry.example.yaml` | Yes |
| `external_repos/repo_registry.local.yaml` | **No** (gitignored) |
| `external_repos/references/*` | **No** |
| `external_repos/reproductions/*` | **No** |
| `external_repos/archived/*` | **No** |
