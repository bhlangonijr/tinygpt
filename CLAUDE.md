# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

tinyGPT is a small (~10M-param) decoder-only transformer trained from scratch on
TinyStories. The goal is to study **how much language-model quality you can get
at a fixed, tiny parameter count**, and which levers move it. A full run is cheap
(a couple of hours on one consumer GPU), so end-to-end experiments are the unit
of work. Read `README.md` for the framing and reproduction details.

There are two trainers over the **same** model and data — the only variable is
the learning rule:

- `train.py` — backprop + AdamW (the baseline).
- `train_sa.py` — gradient-free **directional simulated annealing** (research
  variant; forward passes only, no `loss.backward()`).

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
