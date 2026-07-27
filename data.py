"""
Corpus + tokenizer + batching -- shared by `train.py` and `train_sa.py`.

`load_data()` trains (or loads the cache for) a byte-level BPE tokenizer on the
training corpus, encodes both the training file and a separate held-out
validation file, and returns everything the training loop needs. The token
streams stay on the CPU as int16 (vocab << 32767) and each `(B, T)` window is
shipped to the GPU per step -- at full-TinyStories scale the training stream is
hundreds of millions of tokens, which as a device-resident int64 tensor would be
several GB and could OOM a small card before the model even allocates.

Everything is cached on the first run (`tok.json`, `bpe_cache.npz`,
`bpe_cache_val.npz`); later runs -- and the two trainers -- reuse it. Delete
those three files whenever you change the corpus.
"""

import os
import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from config import (clean, EOT, INPUT_FILE, VAL_FILE, VAL_URL, TOK_FILE,
                    CACHE_FILE, VAL_CACHE_FILE, VOCAB_SIZE, ENCODE_CHUNK_CHARS)


def _encode_corpus(tok, text, label='corpus'):
    """Encode `text` in chunks so the Rust encoder stays memory-friendly.

    One small int32 array per chunk, concatenated at the end -- a single Python
    list of every id would be ~28 B/int and blow up to tens of GB at multi-GB
    corpus scale; int32 arrays are ~4 B/id. Prints a running char counter so a
    multi-GB encode never looks stalled.
    """
    parts = []
    n = len(text)
    for i in range(0, n, ENCODE_CHUNK_CHARS):
        parts.append(np.asarray(tok.encode(text[i:i + ENCODE_CHUNK_CHARS]).ids, dtype=np.int32))
        print(f"  encoding {label}: {min(i + ENCODE_CHUNK_CHARS, n):,} / {n:,} chars", end='\r', flush=True)
    if n:
        print()   # finish the \r progress line
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)


def load_data():
    """Returns a dict with: tok, encode, decode, vocab_size, eot_id,
    chars_per_token, train_data, val_data (both CPU int16 tensors)."""
    # --- training tokens + tokenizer, trained on the training text only ---
    if os.path.exists(TOK_FILE) and os.path.exists(CACHE_FILE):
        print("loading cached tokenizer + train ids...", flush=True)
        tok  = Tokenizer.from_file(TOK_FILE)
        blob = np.load(CACHE_FILE)
        train_ids, train_cpt = blob['ids'], float(blob['cpt'])
    else:
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(
                f"Expected training corpus at {INPUT_FILE!r}. "
                f"Run `python download_tinystories.py` first.")
        # First-run, one-time work (all cached afterward). Each step is announced
        # BEFORE it starts, with flush=True, so the multi-GB corpus never looks stalled.
        print(f"first run: preparing tokenizer + token cache (one-time; several "
              f"minutes on the full corpus, then reused).", flush=True)
        print(f"  reading + normalizing {INPUT_FILE!r} ...", flush=True)
        text = clean(open(INPUT_FILE, encoding='utf-8', errors='ignore').read())
        print(f"  corpus: {len(text):,} chars | training byte-level BPE to vocab "
              f"{VOCAB_SIZE} (this is the slow one-time step) ...", flush=True)
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder       = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=[EOT],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 show_progress=True)
        chunks = (text[i:i + ENCODE_CHUNK_CHARS] for i in range(0, len(text), ENCODE_CHUNK_CHARS))
        tok.train_from_iterator(chunks, tr)
        print("  BPE trained. encoding the corpus to token ids ...", flush=True)
        train_ids = _encode_corpus(tok, text, label='train')
        train_cpt = len(text.replace(EOT, '')) / len(train_ids)   # exclude marker chars -> bits/char stays comparable
        tok.save(TOK_FILE)
        np.savez(CACHE_FILE, ids=train_ids, cpt=train_cpt)
        print(f"  cached: {TOK_FILE} + {CACHE_FILE} (next run skips all of the above).", flush=True)

    # --- held-out validation tokens (separate file, never seen in training) ---
    if os.path.exists(VAL_CACHE_FILE):
        print("loading cached val ids...", flush=True)
        vblob = np.load(VAL_CACHE_FILE)
        val_ids, val_cpt = vblob['ids'], float(vblob['cpt'])
    else:
        if not os.path.exists(VAL_FILE):
            print(f"val file {VAL_FILE!r} not found -- downloading...", flush=True)
            import urllib.request
            urllib.request.urlretrieve(VAL_URL, VAL_FILE)
        print(f"  reading + normalizing {VAL_FILE!r} ...", flush=True)
        vtext = clean(open(VAL_FILE, encoding='utf-8', errors='ignore').read())
        print(f"  val corpus: {len(vtext):,} chars | encoding ...", flush=True)
        val_ids = _encode_corpus(tok, vtext, label='val')
        val_cpt = len(vtext.replace(EOT, '')) / len(val_ids)
        np.savez(VAL_CACHE_FILE, ids=val_ids, cpt=val_cpt)

    return {
        'tok': tok,
        'encode': lambda s: tok.encode(s).ids,
        'decode': lambda ids: tok.decode(ids),
        'vocab_size': tok.get_vocab_size(),
        'eot_id': tok.token_to_id(EOT),          # story-boundary token; sampling stops here
        'chars_per_token': val_cpt,              # bits/char is reported on the val stream
        'train_data': torch.tensor(train_ids, dtype=torch.int16).pin_memory(),
        'val_data':   torch.tensor(val_ids,   dtype=torch.int16).pin_memory(),
    }


def make_get_batch(train_data, val_data, block_size, batch_size, device):
    """Build a `get_batch(split)` closure that draws random (B, T) windows.

    Indices are drawn on the CPU (the streams live there); each window is cast to
    long (embedding / cross_entropy need int64) and moved to the GPU per step.
    """
    def get_batch(split):
        src = train_data if split == 'train' else val_data
        ix = torch.randint(len(src) - block_size - 1, (batch_size,))
        x = torch.stack([src[i:i + block_size] for i in ix])
        y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
        x = x.to(device, dtype=torch.long, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        return x, y
    return get_batch
