# tinyGPT

**A tiny, trainable-from-scratch GPT for squeezing coherent English out of the
fewest parameters possible.** tinyGPT is a small decoder-only transformer that
learns from the [TinyStories](https://arxiv.org/abs/2305.07759) corpus and
completes short children's stories. At ~10M parameters it trains from scratch in
a couple of hours on a single consumer GPU (and runs, slowly, on CPU), which
makes it a fast, honest sandbox for one question:

> **How much quality can you get at a fixed, tiny parameter count — and what
> levers move it?**

Because a full training run is cheap, you can change *anything* — the data, the
regularization, the optimizer, even the learning rule itself — and measure the
effect end-to-end. tinyGPT ships with two trainers built for exactly that:

- **`train.py`** — the standard backprop + AdamW trainer. The baseline you tune against.
- **`train_sa.py`** — a **gradient-free** trainer that replaces backprop with directional
  **simulated annealing**. A research variant: same model, no `loss.backward()` at all.

The point of the small scale is reproducibility: an idea that helps (or doesn't)
shows up clearly, and anyone can rerun it. Contributions of new levers are
welcome — see [Contributing](#contributing).

The model is a clean decoder-only transformer: byte-level BPE → token + position
embeddings → a stack of (RMSNorm → causal self-attention → RMSNorm → GELU MLP)
blocks with residual connections → a tied linear head. No biases. Attention uses
PyTorch's fused `scaled_dot_product_attention` (flash when available). It
descends from Andrej Karpathy's nanoGPT/minGPT line (see [Credits](#credits)).

---

## Quick start

This is the **complete pipeline** — the defaults run the *full* thing (the
entire TinyStories corpus, the full training schedule). No extra flags needed.

```bash
# 1. install
pip install -r requirements.txt

# 2. download the FULL training corpus (~2.2 GB, the whole TinyStories V2 GPT-4 set).
#    The held-out validation file auto-downloads on the first training run.
python download_tinystories.py

# 3. train the full model with the defaults in config.py
#    (~9.6M params, whole corpus, 50k steps -> microgpt_best.pt + microgpt_last.pt)
python train.py                        # uses the GPU if there is one, else CPU
CUDA_VISIBLE_DEVICES=0 python train.py # pick a specific GPU

# 4. complete stories from the trained model (prompts are auto-normalized to
#    match training: lowercased, charset-filtered, whitespace-trimmed)
python infer.py "once upon a time"
python infer.py "the little robot"
python infer.py                        # interactive: type a prompt, get a story

# 5. benchmark it on the official TinyStories eval prompts
python bench.py
```

**What to expect.** Step 2 downloads ~2.2 GB. On the first `train.py` run the
tokenizer is trained and the corpus is encoded + cached (`tok.json`,
`bpe_cache*.npz`) — a few minutes on the full corpus, then instant on later runs.
The full 50k-step run takes roughly **2.7 hours on a single bf16-capable consumer
GPU** (much slower on CPU). You'll see an eval line every `eval_interval` steps,
and `*` marks a new best-val checkpoint. A real run looks like:

```
vocab: 1024 | eot id: 0 | train tokens: 653,798,038 | val tokens: 6,597,993 | chars/token: 3.32 | device: cuda | bf16: True
params: 9,585,300
--- training ---
step      0 | train 6.9601 | val 6.9586 | 3.020 bits/char | 12s *      # ~ln(vocab): untrained
step    250 | train 2.8836 | val 2.8880 | 1.253 bits/char | 61s *
step  20000 | train 1.2745 | val 1.2769 | 0.554 bits/char | 3953s *
step  49500 | train 1.1676 | val 1.1662 | 0.506 bits/char | 9757s *    # best-val checkpoint
step  49999 | train 1.1648 | val 1.1721 | 0.509 bits/char | 9855s
```

(The initial loss ≈ `ln(vocab)` ≈ 6.93 by construction. Exact numbers depend on
the corpus and hardware; the figures above are from one full run — see
[Results](#results).)

### Fast smoke test

To verify the pipeline end-to-end in minutes instead of hours, cap the corpus and
shorten the run:

```bash
TARGET_CHARS=20000000 python download_tinystories.py   # ~20M-char slice instead of 2.2 GB
# then lower max_steps in config.py (e.g. 2000) and run:
python train.py
```

---

## What's happening (the pipeline)

The code is small and split into focused modules. Reading them in this order is
the fastest way in:

1. **Corpus** (`data.py`, `config.py:clean`) — load TinyStories, split on the
   `<|endoftext|>` story marker, lowercase + filter each story to a small ASCII
   subset + collapse whitespace, and **keep the marker** so story boundaries
   survive. Training uses the whole training file; validation uses a **separate
   held-out file**, so val loss is a true out-of-sample estimate.
2. **Byte-level BPE** (`data.py`) — train a byte-level BPE tokenizer (Rust
   `tokenizers`) to `VOCAB_SIZE`, with `<|endoftext|>` reserved as an atomic
   special token so no token ever straddles a story boundary. Encoded ids are
   cached (`tok.json`, `bpe_cache*.npz`) so later runs start instantly.
3. **Model** (`model.py`) — the transformer described above, processing a whole
   `(batch, sequence)` in one forward pass. One definition, shared by every script.
4. **Train** (`train.py` / `train_sa.py`) — sample random windows, compute
   cross-entropy, update the weights (AdamW, or gradient-free SA), report train +
   val loss and bits/char, and keep the best-val checkpoint.
5. **Sample** (`infer.py`, `model.generate`) — encode a prompt and
   autoregressively extend it, stopping at the `<|endoftext|>` boundary so you get
   one clean story.
6. **Benchmark** (`bench.py`) — score a checkpoint on the official TinyStories
   evaluation prompts (perplexity, bits/char, generation-quality heuristics, and
   an optional LLM judge).

Story boundaries are learned, not hard-coded: the model emits `<|endoftext|>`
when it thinks a story is done, and generation stops there.

---

## The two trainers

Same model, same data, same evaluation — only the **learning rule** differs.

| | `train.py` | `train_sa.py` |
|---|---|---|
| **Rule** | backprop + AdamW | directional **simulated annealing** (gradient-free) |
| **Uses gradients?** | yes (`loss.backward()`) | **no** — forward passes only |
| **Checkpoints** | `microgpt_best.pt` / `microgpt_last.pt` | `microgpt_sa_best.pt` / `microgpt_sa_last.pt` |
| **Status** | production baseline | research experiment |

### How the gradient-free trainer works

`train_sa.py` never computes a gradient. Each step:

1. Picks **one** weight tensor (a "local layer"). Probing one tensor at a time
   keeps the measured loss change attributable to *that* tensor, so the
   signal-to-noise ratio stays usable — a global all-tensor perturbation smears
   one scalar Δloss across millions of parameters and collapses to a plateau.
2. Evaluates the loss at **+noise** and **−noise** on that tensor (an *antithetic*
   pair on the same batch) and forms a zeroth-order gradient estimate
   `g = (L₊ − L₋) / (2·var)·noise`. It is unbiased and stays nonzero even on flat
   regions, so the walk never freezes.
3. Feeds that estimate into a per-tensor **credit** momentum (the descent
   direction), takes a step along it, and **Metropolis-accepts** it against a
   temperature `T` that **cools** over training.

It converges far more slowly than backprop and is not expected to match it — it
is here to make *the optimizer itself* a variable you can experiment with. Sample
or benchmark it exactly like the backprop model (`python infer.py
microgpt_sa_best.pt`, `python bench.py microgpt_sa_best.pt`) for a head-to-head.

---

## Experiment: spectral optimizer state

A third learning-rule experiment, in `spectral/` plus four root scripts. The
model, corpus and schedule are untouched — what changes is **where the optimizer
keeps its memory**. The full experiment specification — research questions,
metrics, hypotheses, failure criteria, and what has actually been verified —
lives in [`.specs/spectral-optimizer-state.md`](.specs/spectral-optimizer-state.md).

AdamW stores two fp32 values per parameter (`m` and `v`), so its state is `8N`
bytes — for this model, more memory than the model itself. The question is
whether those `N` numbers are really `N` independent numbers, or whether they
carry enough structure to be described by `K << N` coefficients of a **fixed**
basis (a 2D DCT over each weight matrix). Fixed matters: unlike a learned or
SVD-derived subspace, the basis itself costs nothing to store.

Because the DCT is linear, the state recursions carry over exactly:

```
SGD    M_t = β·M_{t-1} + T(g_t)
AdamW  M_t = β₁·M_{t-1} + (1-β₁)·T(g_t)
       V_t = β₂·V_{t-1} + (1-β₂)·T(g_t²)      ← T(g²), never T(g)²
```

so with **every** coefficient kept, these optimizers are algebraically identical
to `torch.optim.SGD` / `torch.optim.AdamW` — `spectral_selftest.py` asserts it in
fp64. Truncation to `K` coefficients is then the only *algorithmic* difference
between a compressed run and its dense control, which is what makes a loss delta
mean something. (In fp32 even `K = N` is not bit-exact AdamW: coefficient rounding
is set by a tensor's largest values, and AdamW divides each element by its own
`sqrt(v)`. Measured in the spec, section 45.5.)

```bash
# 1. correctness gate: transforms, controller, and K=N equivalence with torch's optimizers
python spectral_selftest.py

# 2. the go/no-go measurement: how many coefficients does the state actually need?
python spectral_probe.py --scale micro --steps 3000 --probe-interval 250

# 3. controls and compressed runs (same seed -> same init, same batch order)
python train_spectral.py --opt adamw --state dense --scale micro --name adamw_dense
python train_spectral.py --opt adamw --state dct --select topk --kmax-frac 0.05 \
    --mode instrumented --scale micro --name adamw_dct_topk05

# 4. a whole sweep, then its tables. One output folder per sweep: every delta is
#    against that sweep's own dense control, and the report refuses a folder that
#    mixes commits or GPUs (each summary.json records both).
python train_spectral.py --sweep experiments/mvp_screen.jsonl --out runs/mvp_screen
python spectral_report.py runs/mvp_screen --plots
```

**Result: no-go** (micro scale, one seed, all numbers from one commit on one GPU).
Optimizer state is *not* usefully compressible in a fixed orthonormal basis. The
first moment needs more DCT coefficients than raw coordinates (0.73N vs 0.70N at
tau=0.99), and shuffling a tensor's entries barely changes that — the DCT finds no
spatial structure to exploit. The second moment *is* energy-concentrated (86% of
its energy in 1% of coefficients), but AdamW divides by `sqrt(v)`, so the update
depends on `v`'s small values, exactly where an energy-optimal truncation errs.
The one variant that trains within 1% of dense AdamW — spectral `v`, dense `m` —
lands +0.25% to +0.44% with no trend in K, saves at most 1.77x of optimizer state
(less as K grows, since `m` stays dense) and costs 24-27% of throughput.
Compressing both moments, the first point within 5% of dense is DCT-Low K/N = 75%
(+2.84% val loss): 14% less optimizer state, 32% less throughput, and slightly
*higher* peak GPU memory; within 1% no point saves memory at all. Full numbers,
the defects found and fixed on the way, and the surprises (DCT-Low beats DCT-TopK
for training even though TopK reconstructs better; fp32 spectral state is not
exactly AdamW even at K = N) are in
[`.specs/spectral-optimizer-state.md`](.specs/spectral-optimizer-state.md),
sections 44-45.

**Start with the probe.** It trains with an ordinary dense AdamW and only
*measures*, comparing four bases — best-K DCT, low-frequency DCT, the state's own
coordinates, and DCT after shuffling the entries. The last two are the controls
that matter: if raw top-K is as compact as DCT top-K, or if shuffling the entries
doesn't hurt, then the DCT is finding sparsity that was already there rather than
spatial structure, and there is nothing here worth compressing.

Two execution modes, and they follow the *same* optimization path:

| Mode | What it does | Valid for |
|---|---|---|
| `--mode perf` | no dense state anywhere; capacity is sized from a proxy spectrum the optimizer can actually see | memory + throughput benchmarks |
| `--mode instrumented` | additionally keeps shadow dense states, **never used to update a parameter**, to measure oracle `K_required`, reconstruction error and update-fidelity | understanding what happened |

Instrumented runs deliberately allocate dense state for measurement, so they are
*not* memory benchmarks; `summary.json` flags them and the report marks them `(i)`.

Only the block projection matrices (`qkv`, `proj`, `fc`, `out`) get spectral
state — embeddings, the tied head and the RMSNorm gains stay dense — so every run
reports its **spectral coverage**, without which a whole-model memory number
can't be read.

---

## Files

| File | Role |
|---|---|
| `config.py` | Shared knobs: architecture + training hyper-parameters + the `clean()` normalizer. One source of truth. |
| `model.py` | The GPT model (RMSNorm, attention block, tied head), plus `generate()` and checkpoint save/load. Imported by everything. |
| `data.py` | Corpus loading, byte-level BPE training/caching, held-out validation, batching. |
| `train.py` | Backprop + AdamW trainer. |
| `train_sa.py` | Gradient-free simulated-annealing trainer. |
| `infer.py` | Story completion from any checkpoint (backprop or SA). |
| `bench.py` | TinyStories eval-prompt benchmark: perplexity, bits/char, quality heuristics, optional LLM judge. |
| `spectral/` | Spectral optimizer-state module: DCT transforms, sparse state + dynamic-K controller, metrics, `SpectralSGD` / `SpectralAdamW`, telemetry. |
| `spectral_selftest.py` | Correctness gate — run before anything else (K=N equivalence with torch's optimizers). |
| `spectral_probe.py` | Measures how compressible optimizer state is under a *dense* run. The experiment's go/no-go. |
| `train_spectral.py` | Trainer for the spectral experiment: dense controls and compressed runs, single or swept. |
| `spectral_report.py` | Aggregates runs into the result tables and figures. |
| `download_tinystories.py` | Fetch the TinyStories corpus from Hugging Face. |

Checkpoints embed `n_head` (the one hyper-parameter not recoverable from the raw
weights); `infer.py` / `bench.py` rebuild the exact architecture from the tensor
shapes, so they adapt to whatever size you trained.

---

## Config knobs

All in `config.py` (SA-specific knobs are at the top of `train_sa.py`).

| Knob | Default | Controls |
|---|---|---|
| `VOCAB_SIZE` | `1024` | BPE vocab size (byte alphabet is 256, so go well above it). |
| `n_layer` / `n_embd` / `n_head` | `6` / `356` / `4` | Depth / width / heads (`n_embd % n_head == 0`). These + `block_size` + `VOCAB_SIZE` fix the ~9.6M param count. |
| `block_size` | `256` | Context length in tokens (~4–5 sentences). |
| `dropout` | `0.1` | Regularization; raise to fight overfitting. Param-free. |
| `batch_size` / `max_steps` | `128` / `50000` | Sequences per step / total steps. |
| `lr` / `min_lr` / `warmup_steps` | `1e-3` / `1e-4` / `200` | Warmup → cosine-decay schedule. |
| `weight_decay` | `0.1` | AdamW decay (matrices only; norms undecayed). Param-free. |
| `MAX_CHARS` | `None` | Cap the corpus (chars); `None` = use it all. |
| `eval_interval` / `eval_iters` | `250` / `50` | Steps between evals / batches per val estimate. |

**Whenever you change the corpus, delete the tokenizer cache** so it rebuilds:

```bash
rm -f tok.json bpe_cache.npz bpe_cache_val.npz
```

(Changing `dropout` or model width does *not* require deleting the cache.)

---

## Reading the training output

Each eval line prints `train`, `val`, and `bits/char`. Bits/char is normalized
per character, so it is comparable across runs **on the same corpus**.

- **Both falling, gap small and steady** → healthy; keep training or scale up.
- **Train falling, val flat or rising, gap widening** → overfitting. Your best
  model is the val minimum (captured automatically in `microgpt_best.pt`); steps
  past it just memorize.
- **Train and val flat together, small gap** → capacity-limited; add params.

This is the loop the project runs on: push quality at a fixed size with the
param-free levers (more data, dropout, weight decay, LR schedule, batch size),
then decide whether the remaining gap is worth more parameters.

---

## Benchmarking

`bench.py` scores a checkpoint on Eldan & Li's official TinyStories evaluation
prompts (auto-downloaded on first run) and reports three things:

```bash
python bench.py                          # microgpt_best.pt, all prompts (offline metrics)
python bench.py microgpt_sa_best.pt      # benchmark the SA model
python bench.py --limit 8 --show 4       # quick look, print 4 completions
python bench.py --judge                  # add the LLM-judge grades (needs ANTHROPIC_API_KEY)
python bench.py --json scores.json       # also dump full per-prompt results
```

1. **Language modeling** — perplexity and bits/char of the prompt text under the
   model. The principled, comparable number (lower is better). Prompts are
   normalized with the trainer's `clean()` first so they match the training
   distribution; pass `--raw` to skip that.
2. **Generation quality** — the model continues each prompt and the continuation
   is scored with offline heuristics aimed at this model's known failure modes:
   repetition (`rep-n`), diversity (`distinct-n`), longest repeated-token run,
   whether it emitted `<|endoftext|>` and stopped, and whether it ends on sentence
   punctuation.
3. **LLM judge** (`--judge`) — the paper-comparable signal. The official
   TinyStories eval grades completions with a strong LLM on
   grammar/creativity/consistency/plot; `--judge` reproduces that with Claude via
   the `anthropic` SDK. The offline metrics are local proxies; this is the real one.

---

## Results

**Reference (backprop, `train.py` at the default 9,585,300-param config):**
trained on the full TinyStories corpus (~654M training tokens, vocab 1024,
`block_size=256`, 50k steps at batch 128), a full run reaches a best held-out
**val loss of 1.166 → 0.506 bits/char** in ~2.7 h on one bf16-capable consumer
GPU. It produces fluent, coherent short stories — named characters, simple
dialogue, and a setup/conflict/resolution arc, stopping cleanly on
`<|endoftext|>`. That coherence emerges precisely because the *data's* complexity
matches the model's *capacity* (the central TinyStories result). Exact numbers
depend on corpus size, steps, and hardware; treat this as a ballpark to
reproduce, not a fixed target.

Expected small-model artifacts remain at this scale: local fluency with
occasional semantic slips (an object used out of context, a coreference that
drifts across sentences). These shrink with scale/data, not with sampling
tweaks.

**Benchmark (`bench.py`, all 44 official TinyStories eval prompts).** Scoring
that same checkpoint (`python bench.py --judge`, sampling `T=0.7 top_k=40
max_new=200`):

| Metric | Value | Notes |
|---|---|---|
| perplexity | 5.54 | on the eval-prompt text |
| bits/char | 0.682 | > the 0.506 val above — expected: hand-written, out-of-distribution openings, normalized over short prompts |
| distinct-1 / -2 | 0.51 / 0.79 | diversity (higher is better) |
| repetition 2 / 3 / 4 | 0.21 / 0.11 / 0.06 | looping (lower is better) |
| mean longest run | 1.0 tokens | no token-level loops |
| stopped on `<|endoftext|>` | 86% | learned the story boundary |
| ends cleanly | 86% | ends on sentence punctuation |
| heuristic quality (offline proxy) | 86.3 / 100 | — |

LLM-judge grades (Claude, 1–10, all 44 completions graded):

| grammar | creativity | consistency | plot |
|---|---|---|---|
| 7.45 | 3.95 | 4.91 | 4.55 |

This is the characteristic tiny-model profile: **grammar is essentially solved**
(fluent surface, no repetition loops, clean story stops), while
creativity/consistency/plot sit around 4–5 — the global-coherence ceiling of a
~10M-param model (the judge flags things like an object used out of context or a
coreference that drifts). Those three axes are capacity/data-bound, not
sampling-bound, so they are the interesting targets to move.

**Simulated annealing (`train_sa.py`):** a research variant. Gradient-free
training converges much more slowly and is not expected to match backprop at
equal steps; it is provided so the learning rule is something you can study and
improve, not a competitive baseline. Benchmark it head-to-head with `bench.py`.

---

## Things to try

- **Feed it your own text.** Point `INPUT_FILE` at any plain-text file with
  `<|endoftext|>` between documents, delete the tokenizer cache, and retrain.
- **Improve quality at fixed params.** More data, dropout, weight decay, a longer
  cosine schedule, and a bigger batch all raise quality without changing the param
  count — that is the core game here.
- **Make it tiny / make it big.** Drop `n_layer`/`n_embd` for a bigram-soup model
  in minutes, or scale up and watch the train-vs-val gap.
- **Change the temperature.** Low (`0.3`) parrots common phrases; high (`1.2`)
  goes surreal.
- **Beat backprop's efficiency with the gradient-free trainer** — or find where it
  plateaus and why.

## Contributing

tinyGPT is meant to be forked and experimented on. Good contributions:

- A new **param-free quality lever** with a before/after `bench.py` comparison.
- Improvements to the **gradient-free trainer** (better probes, schedules, credit rules).
- New **evaluation metrics** or a stronger judging setup.
- Reproductions at different sizes / corpora, with numbers.

Keep the model definition in `model.py` singular and shared; keep new
hyper-parameters in `config.py` (or, for optimizer-specific knobs, at the top of
the relevant trainer). Include a `bench.py` result with any quality claim.

---

## Caveats

- **Small-model failure modes are expected:** entity/coreference breakdown across
  sentences, occasional repetition loops, and rare-word drift. They improve with
  scale, more data, and sampling tweaks.
- **CPU works but is slow.** The project assumes a single consumer GPU for a full
  run; bf16/TF32 autocast enable automatically when the hardware supports them.
- **bits/char is only comparable within a corpus.** Different corpora are different
  rulers (TinyStories is intrinsically low-entropy).
- **The offline generation-quality metrics are proxies.** Use `--judge` for the
  paper-comparable grammar/creativity/consistency/plot signal.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Credits

The model and training/inference scaffolding descend from
[Andrej Karpathy](https://github.com/karpathy)'s nanoGPT / minGPT teaching GPTs
(MIT), reworked with a PyTorch loop, byte-level BPE, a TinyStories pipeline, and a
gradient-free training variant. Attributions are recorded in [`NOTICE`](NOTICE).

## References

- **DCT via FFT** — Makhoul, J. *A Fast Cosine Transform in One and Two
  Dimensions.* IEEE Trans. ASSP 28(1), 27–34 (1980). The even-extension identity
  behind `spectral/transforms.py`; the implementation here is derived from the
  identity directly and checked against an explicit DCT matrix in
  `spectral_selftest.py`.
- **Memory-efficient optimizer state** (comparison points for the spectral
  experiment, not dependencies) — GaLore (arXiv:2403.03507) and COAP
  (arXiv:2412.00071) reduce optimizer-state memory by projecting gradients into
  a *learned* low-rank subspace. The experiment here asks the narrower question
  of whether a **fixed** basis, which costs nothing to store, is enough.
- **TinyStories** — Eldan, R. & Li, Y. *TinyStories: How Small Can Language Models
  Be and Still Speak Coherent English?* arXiv:2305.07759 (2023).
  https://arxiv.org/abs/2305.07759 (`roneneldan/TinyStories` on Hugging Face)

```bibtex
@article{eldan2023tinystories,
  author  = {Eldan, Ronen and Li, Yuanzhi},
  title   = {TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  journal = {arXiv preprint arXiv:2305.07759},
  year    = {2023},
  url     = {https://arxiv.org/abs/2305.07759}
}
```
