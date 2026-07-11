# External Repository Notes

This directory contains migration notes, patch files, and inspection logs
for third-party repositories examined during benchmark development.

These notes **are** committed to Git.  The external repositories themselves
are **not** committed (they live in `external_repos/` which is gitignored).

---

## Directory contents

| File pattern | Purpose |
|--------------|---------|
| `<repo_name>_inspection.md` | Free-form notes from code inspection |
| `<repo_name>_migration.md` | Decisions made about migrating code into `src/` |
| `<repo_name>_changes.patch` | Git diff of experimental changes to the external repo |
| `audit_<date>.yaml` | Snapshot from `scripts/audit_external_repo.py` |

---

## How to create a patch file

If you modify a cloned external repository experimentally, save the diff
before discarding or re-cloning it:

```bash
# From the repo root:
git -C external_repos/references/<repo> diff > docs/external_repo_notes/<repo>_changes.patch

# Record the base commit hash:
git -C external_repos/references/<repo> rev-parse HEAD
# → write this into the corresponding migration notes file
```

---

## Code migration checklist

Before moving any code from an external repository into `src/`:

- [ ] Inspection notes written in `<repo>_inspection.md`
- [ ] License identified and recorded in `repo_registry.local.yaml`
- [ ] License compatible with benchmark's MIT licence (or noted if not)
- [ ] Code reviewed and rewritten (not copy-pasted verbatim)
- [ ] Attribution comment added in the target `src/` module
- [ ] Unit test written for the migrated functionality
- [ ] Migration notes written in `<repo>_migration.md`
- [ ] External repository entry status updated to `code_migration_in_progress` or `archived`

---

## Important: do not commit entire modified repos

Do **not** copy a modified third-party repository into the benchmark Git history.
Only save:
1. The patch file
2. The base commit hash
3. These markdown notes

The external repository can always be re-cloned and the patch re-applied.

---

## Current notes

*(Add files to this directory as external repositories are inspected.)*
