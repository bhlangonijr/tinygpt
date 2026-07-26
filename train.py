"""
tinyGPT trainer -- standard backprop with AdamW.

Trains the model in `model.py` from scratch on TinyStories: sample random (B, T)
windows from the tokenized corpus, forward + cross-entropy + backward, clip the
global grad norm, take an AdamW step on a warmup->cosine learning-rate schedule.
Trains on the WHOLE training file and validates on a SEPARATE held-out file, so
the reported val loss is a true out-of-sample estimate.

    python train.py                        # train on tinystories.txt
    CUDA_VISIBLE_DEVICES=0 python train.py # pick a GPU

Writes `microgpt_best.pt` (lowest val loss -- the one you usually want) and
`microgpt_last.pt`. TF32 + bf16 autocast turn on automatically when the GPU
supports them; on CPU everything runs in fp32.

All knobs live in `config.py`. The first run builds + caches the tokenizer and
token ids (`tok.json`, `bpe_cache*.npz`); delete those three if you change the
corpus.
"""

import math
import time

import torch

from config import (seed, block_size, batch_size, dropout, n_layer, n_embd,
                    n_head, max_steps, warmup_steps, lr, min_lr, weight_decay,
                    eval_interval, eval_iters)
from data import load_data, make_get_batch
from model import build_model, save_checkpoint

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
dev_type = 'cuda' if device == 'cuda' else 'cpu'
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True   # free on Ampere+, ignored elsewhere
torch.backends.cudnn.allow_tf32 = True
use_amp   = (device == 'cuda') and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32

# ---------------- data ----------------
d = load_data()
encode, decode = d['encode'], d['decode']
vocab_size, eot_id, chars_per_token = d['vocab_size'], d['eot_id'], d['chars_per_token']
get_batch = make_get_batch(d['train_data'], d['val_data'], block_size, batch_size, device)
print(f"vocab: {vocab_size} | eot id: {eot_id} | train tokens: {len(d['train_data']):,} "
      f"| val tokens: {len(d['val_data']):,} | chars/token: {chars_per_token:.2f} "
      f"| device: {device} | bf16: {use_amp}")

# ---------------- model ----------------
model = build_model(vocab_size, n_layer, n_embd, n_head, block_size, dropout).to(device)
print(f"params: {model.num_params():,}")

# ---------------- optimizer ----------------
decay   = [p for p in model.parameters() if p.dim() >= 2]   # weight-decay matrices only
nodecay = [p for p in model.parameters() if p.dim() < 2]    # norms/embeddings gains stay undecayed
opt = torch.optim.AdamW(
    [{'params': decay, 'weight_decay': weight_decay},
     {'params': nodecay, 'weight_decay': 0.0}],
    lr=lr, betas=(0.9, 0.95))


def lr_at(step):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    r = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * r))


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------- train ----------------
print("--- training ---")
t0 = time.time()
best_val = float('inf')
for step in range(max_steps):
    for g in opt.param_groups:
        g['lr'] = lr_at(step)
    x, y = get_batch('train')
    with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % eval_interval == 0 or step == max_steps - 1:
        m = evaluate()
        bpc = m['val'] / math.log(2) / chars_per_token        # bits/char, comparable across runs
        tag = ""
        if m['val'] < best_val:                               # keep the best-val checkpoint
            best_val = m['val']
            save_checkpoint(model, 'microgpt_best.pt', n_head)
            tag = " *"
        print(f"step {step:6d} | train {m['train']:.4f} | val {m['val']:.4f} "
              f"| {bpc:.3f} bits/char | {time.time() - t0:.0f}s{tag}")

save_checkpoint(model, 'microgpt_last.pt', n_head)

# ---------------- sample ----------------
print("--- inference ---")
for p in ["once upon a time ", "the little ", "one day ", "lily and tom ", "the big dog "]:
    idx = torch.tensor([encode(p)], dtype=torch.long, device=device)
    with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
        out = model.generate(idx, max_new=120, temperature=0.8, top_k=40, eot_id=eot_id)
    print(f"[{p!r}] -> {decode(out[0].tolist())}")
