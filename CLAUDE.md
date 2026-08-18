# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

tinyGPT is a small (~10M-param) decoder-only transformer trained from scratch on
TinyStories. The goal is to study **how much language-model quality you can get
at a fixed, tiny parameter count**, and which levers move it. A full run is cheap
(a couple of hours on one consumer GPU), so end-to-end experiments are the unit
of work. Read `README.md` for the framing and reproduction details.

There are three trainers over the **same** model and data — the only variable is
the learning rule:

- `train.py` — backprop + AdamW (the baseline).
- `train_sa.py` — gradient-free **directional simulated annealing** (research
  variant; forward passes only, no `loss.backward()`).
- `train_spectral.py` — backprop, but the optimizer's persistent state (SGD
  momentum / Adam `m`, `v`) is stored as `K` DCT coefficients per weight matrix
  instead of one fp32 per parameter. Library lives in `spectral/`.

## Architecture (one model, shared everywhere)

`model.py` holds the **single** model definition (`RMSNorm`, `Block`, `GPT`),
plus `build_model()`, `generate()`, and `save_checkpoint()` / `load_checkpoint()`.
`train.py`, `train_sa.py`, `infer.py`, and `bench.py` all import it — never fork
the model into another file. The transformer is standard: byte-level BPE → token
+ position embeddings → `n_layer` × (RMSNorm → causal SDPA → RMSNorm → GELU MLP)
with residuals → final RMSNorm → tied linear head. No biases. Requires
`n_embd % n_head == 0`.

Data flow worth knowing:

1. `config.py` is the single source of truth for architecture + training knobs
   and the `clean()` text normalizer. SA-specific knobs live at the top of
   `train_sa.py` on purpose (only that script reads them).
2. `data.py` trains/caches the byte-level BPE tokenizer, encodes the training file
   and a **separate held-out validation file**, and returns CPU-resident int16
   token streams + a `get_batch` closure. Token streams stay on CPU; each `(B, T)`
   window ships to the GPU per step (keeps VRAM free at large corpus scale).
3. `<|endoftext|>` is a reserved BPE special token (story boundary). The model
   learns to emit it; `generate()` stops there. `clean()` preserves it.
4. Checkpoints embed `n_head` (not recoverable from raw weights). `load_checkpoint`
   rebuilds the exact architecture from tensor shapes, so inference/bench adapt to
   any size, and it also reads a legacy raw state-dict (falling back to
   `config.n_head`).

## Commands

```bash
pip install -r requirements.txt
python download_tinystories.py          # -> tinystories.txt (valid file auto-downloads on first train)
python train.py                         # -> microgpt_best.pt / microgpt_last.pt
python train_sa.py                      # -> microgpt_sa_best.pt / microgpt_sa_last.pt
python infer.py "once upon a time "     # complete a story (any checkpoint via first .pt arg)
python bench.py                         # TinyStories eval-prompt benchmark
python bench.py --judge                 # + LLM-judge grades (needs ANTHROPIC_API_KEY)

# spectral optimizer-state experiment (see README for the full story)
python spectral_selftest.py             # correctness gate -- run this FIRST
python spectral_probe.py --scale micro  # is optimizer state compressible at all? (go/no-go)
python train_spectral.py --opt adamw --state dense --scale micro --name adamw_dense
python train_spectral.py --sweep experiments/mvp_screen.jsonl
python spectral_report.py runs/ --plots
```

There is no local torch-free validation path; changes to model/training logic are
verified by actually running a short training job (lower `max_steps` in
`config.py` and/or cap the corpus with `TARGET_CHARS` on the download).

## Conventions

- **Keep `model.py` the only model definition.** If a change touches the
  architecture, it happens there and every script inherits it.
- **New knobs go in `config.py`** (or, for optimizer-specific ones, the top of the
  relevant trainer) — never hard-code them mid-script.
- **Any quality claim needs a `bench.py` number.** bits/char (or the `--judge`
  grades) before vs. after; bits/char is only comparable within the same corpus.
- **When the corpus changes, delete the tokenizer cache** (`tok.json`,
  `bpe_cache.npz`, `bpe_cache_val.npz`) so it rebuilds; otherwise the old tokens
  are silently reused. Changing dropout/width does not require this.
- **bf16/TF32 autocast** enable automatically on capable GPUs and are a no-op
  elsewhere — don't gate them on a specific device.
- The repo is public and Apache-2.0 licensed; keep it self-contained (no
  environment-specific or private references) and update `NOTICE` if you add a
  third-party derivation.

## The spectral experiment (`spectral/`, `*_spectral*.py`)

- **The spec is `.specs/spectral-optimizer-state.md`** and its section numbers are
  cited in code comments (`# spec section 15`). Never renumber sections — append.
  When behaviour deviates from the spec, record it in the spec's deviations
  section (41) rather than letting the two drift apart.
- **`spectral_selftest.py` is the gate.** It asserts that with every coefficient
  retained (`K = N`) the spectral optimizers reproduce `torch.optim.SGD` /
  `torch.optim.AdamW` exactly. If that breaks, a compressed run is measuring a
  bug rather than compression — fix it before trusting any quality number.
- **Instrumented mode must never change the trajectory.** Shadow dense states
  exist only to measure; the capacity controller runs on the same proxy spectrum
  in both modes, so an instrumented run and its perf twin take the same path.
  Instrumented runs are not valid memory benchmarks and are flagged as such.
- **The second moment uses `T(g²)`, not `T(g)²`.** The latter would redefine Adam
  in coefficient space instead of approximating real Adam state.
- Transform-agnostic by construction: STFT (spec sections 7/10/11) drops in
  behind the `Transform` interface in `spectral/transforms.py`. Cross-transform
  comparisons must be in **bytes**, not coefficient counts — a complex STFT
  coefficient costs two fp32s.
- Spectral knobs live in `SpectralConfig` (`spectral/optim.py`) and are surfaced
  as `train_spectral.py` flags — the same "experiment knobs stay with the
  experiment" rule as `train_sa.py`.
