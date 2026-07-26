"""
Shared configuration for tinyGPT.

Every knob the model and training loop need lives here, in ONE place, so the
trainer (`train.py`), the simulated-annealing trainer (`train_sa.py`), the
inference script (`infer.py`), and the benchmark (`bench.py`) can never drift
apart.

Most architecture sizes (vocab, n_embd, n_layer, block_size) are recoverable
from a saved checkpoint, so `infer.py` / `bench.py` read those from the weights.
`n_head` is the exception -- it is NOT recoverable from the weights -- so it is
saved *inside* the checkpoint by the trainers and only falls back to the value
here for a raw state-dict. Keeping it here means the trainers and everything
downstream agree by construction.
"""

import re

# ---------------- end-of-text token (shared) ----------------
EOT = '<|endoftext|>'                 # story/document boundary; registered as a BPE special token


def clean(text):
    """Normalize text to the model's training distribution.

    `<|endoftext|>` story boundaries are PRESERVED as a single special token;
    each chunk between them is lowercased, charset-filtered (a small ASCII
    subset), and whitespace-collapsed. Shared by the trainer and the benchmark
    so a prompt is always normalized exactly like the training corpus.
    """
    parts = text.split(EOT)
    out = []
    for p in parts:
        p = p.lower()
        p = re.sub(r"[^a-z .,;:?!'\n-]", '', p)
        p = re.sub(r'\s+', ' ', p).strip()
        out.append(p)
    return EOT.join(out)[:MAX_CHARS]   # MAX_CHARS=None -> no truncation, keep the whole corpus


# ---------------- files ----------------
INPUT_FILE     = 'tinystories.txt'                              # training corpus (used whole, no split)
VAL_FILE       = 'TinyStoriesV2-GPT4-valid.txt'                 # held-out validation set (separate file)
VAL_URL        = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
                  "resolve/main/TinyStoriesV2-GPT4-valid.txt")  # auto-download source for VAL_FILE
TOK_FILE       = 'tok.json'                                     # trained tokenizer
CACHE_FILE     = 'bpe_cache.npz'                                # cached train token ids + chars/token
VAL_CACHE_FILE = 'bpe_cache_val.npz'                            # cached val token ids + chars/token

# ---------------- tokenizer / corpus ----------------
VOCAB_SIZE         = 1024            # byte-level alphabet is 256, so go well above it
MAX_CHARS          = None            # truncate the cleaned corpus to this many chars; None = use it all
ENCODE_CHUNK_CHARS = 1_000_000       # encode/train the BPE in chunks this size (keeps the Rust encoder memory-friendly)

# ---------------- model architecture (shared) ----------------
n_layer    = 6
n_embd     = 356                     # head_dim = n_embd / n_head = 89
n_head     = 4                       # not stored in raw weights -- trainers embed it in the checkpoint
block_size = 256                     # context length, ~4-5 sentences
dropout    = 0.1                     # regularization; raises train loss, narrows the train/val gap

# ---------------- training (train.py) ----------------
batch_size    = 128                  # the token stream lives on CPU, so VRAM is free for bigger batches
max_steps     = 50_000
warmup_steps  = 200
lr            = 1e-3
min_lr        = 1e-4
weight_decay  = 0.1
eval_interval = 250
eval_iters    = 50
seed          = 42
