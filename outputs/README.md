# Generated output namespaces

Everything below `outputs/` is generated and ignored by Git. New producers use:

- `datasets/<dataset>/<pipeline>/` for canonical data, labels, QC, and audits;
- `reproductions/<paper>/<case>/` for paper-faithful experiment artifacts;
- `benchmarks/<benchmark>/<method>/` for unified comparisons.

Each formal producer must write its resolved protocol/config, generator identity,
input hashes, generation time, and Git revision/dirty state into run metadata.
Historical top-level directories are legacy artifacts and are not the target for
new runs.
