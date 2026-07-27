#!/usr/bin/env python3
"""
Inference / story completion for tinyGPT. Loads a saved checkpoint and generates
text -- no training, no corpus needed (just the tokenizer `tok.json`). Works with
any checkpoint: backprop (`microgpt_best.pt`) or simulated-annealing
(`microgpt_sa_best.pt`).

    python infer.py                                 # interactive, uses microgpt_best.pt
    python infer.py "the king " "she said "         # one completion per prompt
    python infer.py microgpt_sa_best.pt "one day "  # first .pt arg overrides the checkpoint
    python infer.py --no-stop "the king "           # keep generating past <|endoftext|>

Generation stops at the `<|endoftext|>` story boundary by default (clean single
story); pass `--no-stop` to run the full max_new tokens. Architecture sizes are
read from the checkpoint, so this adapts to whatever size you trained; `n_head`
is read from the checkpoint too (falling back to config.py for a raw state-dict).
"""

import sys

import torch
from tokenizers import Tokenizer

from config import TOK_FILE, EOT, n_head as CFG_N_HEAD, clean
from model import load_checkpoint

# ---------------- inference knobs ----------------
MAX_NEW     = 200
TEMPERATURE = 0.7
TOP_K       = 40

# --no-stop keeps generating past the end-of-text token (default: stop on it)
args = sys.argv[1:]
STOP_AT_EOT = '--no-stop' not in args
args = [a for a in args if a != '--no-stop']

# first .pt argument (if any) overrides the checkpoint; the rest are prompts
CKPT = 'microgpt_best.pt'
if args and args[0].endswith('.pt'):
    CKPT = args.pop(0)
prompts = args

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
dev_type = 'cuda' if device == 'cuda' else 'cpu'
use_amp  = (device == 'cuda') and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32

# ---------------- load ----------------
tok = Tokenizer.from_file(TOK_FILE)
eot_id = tok.token_to_id(EOT)                  # None for a pre-EOT tokenizer -> no early stop
model, cfg = load_checkpoint(CKPT, device, n_head_fallback=CFG_N_HEAD)
print(f"loaded {CKPT}: vocab {cfg.vocab_size} | n_embd {cfg.n_embd} | n_layer {cfg.n_layer} "
      f"| block {cfg.block_size} | n_head {cfg.n_head} | device {device}", file=sys.stderr)


def sample(prompt, max_new=MAX_NEW, temperature=TEMPERATURE, top_k=TOP_K, stop=STOP_AT_EOT):
    # Normalize the prompt exactly like the training corpus (lowercase, charset
    # filter, strip). This matches the trainer/bench and, crucially, removes any
    # TRAILING space: with byte-level BPE a leading space attaches to the NEXT
    # word, so a dangling trailing space is an out-of-distribution token that
    # glitches the first generated token (a stray "-", a "-year-old" fragment...).
    ids = tok.encode(clean(prompt)).ids
    if not ids:                                # prompt normalized to nothing -> nothing to condition on
        return ''
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
        out = model.generate(idx, max_new, temperature=temperature, top_k=top_k,
                             eot_id=eot_id, stop=stop)
    return tok.decode(out[0].tolist())         # decode drops EOT (skip_special_tokens=True)


if prompts:
    for p in prompts:
        print(f"\n[{p!r}]\n{sample(p)}")
else:
    print("Enter a prompt (blank line or Ctrl-D to quit):", file=sys.stderr)
    while True:
        try:
            p = input("> ")
        except EOFError:
            break
        if not p.strip():
            break
        print(sample(p))
