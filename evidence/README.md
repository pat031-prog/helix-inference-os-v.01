# HeliX Public Evidence Layer

`evidence/` is the **canonical public curation layer** for HeliX.

Use this directory first when you need public-facing claims, tiering, falsifiers, and bounded explanations of what a cited artifact does or does not prove.

## Start Here

- [index.json](index.json)
- [../CLAIMS.md](../CLAIMS.md)
- [../THREAT_MODEL.md](../THREAT_MODEL.md)
- [../NULL_RESULTS.md](../NULL_RESULTS.md)
- [../REPRODUCING.md](../REPRODUCING.md)

## Directory Meaning

- `rigorous/`: strongest mechanics and trust artifacts
- `empirical-observations/`: narrow real-world observations that are auditable but not overinterpreted
- `experimental/`: supportive or research-facing evidence kept with explicit boundaries
- `demos/`: product-ish or showcase lanes
- `null-results/`: failed, blocked, or falsification-preserved lanes
- `manifests/`: batch snapshots and public-layer support manifests

## Relationship to `verification/`

`verification/` remains the raw lab/archive tree:

- historical run outputs
- preregistrations
- suite outputs
- viewer assets
- scratch or pre-hardening snapshots

The public rule is simple:

- cite `evidence/` for public claims
- use `verification/` for raw inspection, replay, and historical context

## Legacy Curated Copies

The older `academic/`, `infrastructure/`, and `product/` subdirectories are kept as legacy curated copies from the pre-reframe layout. They are no longer the preferred public navigation surface; `index.json` and the stable buckets above are.
