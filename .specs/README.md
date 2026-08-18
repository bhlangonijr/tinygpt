# .specs

Experiment specifications for the research variants in this repo. These are the
documents the code is written against, kept in-tree so an implementation and its
specification can be reviewed together — and so drift between them is visible in
a diff rather than discovered later.

| Spec | Subject | Status |
|---|---|---|
| [spectral-optimizer-state.md](spectral-optimizer-state.md) | Dynamic spectral (DCT/STFT) compression of AdamW and SGD-momentum optimizer state | MVP implemented (DCT, steps 1–9); not yet run |

## Conventions

* **Section numbers are an API.** Code comments cite them (`# spec section 15`),
  so sections are never renumbered — new material is appended.
* **Every spec ends with a reconciliation block**: an implementation map, an
  explicit list of deviations with reasons, and a verification status saying
  which claims have actually been checked and which have only been asserted.
* **Empty result tables stay empty** until a number is measured on real hardware.
  A specification is not a place to record what we expect to find.
