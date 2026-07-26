"""
tinyGPT-SA -- a gradient-free trainer: train the SAME model as `train.py`, but
with NO backprop. The AdamW step is replaced by directional SIMULATED ANNEALING.

This is an experiment in whether a tiny language model can be trained without
gradients at all. Everything except the learning rule -- the model, corpus,
tokenizer, evaluation, and sampling -- is identical to `train.py`, so the only
variable under study is how the weights get updated:

  * No `loss.backward()`, no gradients -- forward passes only.
  * Each step probes ONE weight tensor (the "local layer") with an ANTITHETIC
    pair: it evaluates the loss at +noise and -noise on that tensor and forms the
    zeroth-order gradient estimate  g = (L+ - L-) / (2 * var) * noise. Probing
    one tensor at a time keeps the loss change attributable to THAT tensor (local
    credit), so the signal-to-noise ratio stays usable -- a global all-tensor move
    smears one scalar delta-loss across millions of params and collapses to the
    uniform-output plateau. The antithetic difference is unbiased and, unlike a
    one-sided probe, stays nonzero on flat regions, so the walk doesn't freeze.
  * That estimate drives a per-tensor "credit" momentum (the descent direction);
    a step `bias_gain * credit` is applied and Metropolis-accepted.
  * A cooling schedule lowers the temperature T over training, shrinking the
    perturbation magnitude and tightening Metropolis acceptance.

    python train_sa.py                          # train the SA model on tinystories.txt
    CUDA_VISIBLE_DEVICES=0 python train_sa.py    # pick a GPU

Writes `microgpt_sa_best.pt` / `microgpt_sa_last.pt` (distinct names so the
backprop checkpoints are never clobbered). Sample or benchmark it with
`infer.py` / `bench.py` for a head-to-head against the backprop model. The
tokenizer + token caches are shared with `train.py` and reused as-is.

NOTE: this is a research toy. SA converges far slower than backprop and is not
expected to match it -- it is here to make the learning rule itself a variable
you can experiment with.
"""

import math
import time

import torch

from config import (seed, block_size, batch_size, dropout, n_layer, n_embd,
                    n_head, max_steps, eval_interval, eval_iters)
from data import load_data, make_get_batch
from model import build_model, save_checkpoint

# ---------------- SA hyper-parameters ----------------
# Deliberately local to this experiment (NOT in the shared config.py, which the
# backprop trainer reads). Tune these first.
T0          = 1.0e-3     # initial temperature; low enough that improving moves stick
T_min       = 1.0e-5     # floor temperature
sigma0      = 0.02       # antithetic probe std, in units of each tensor's own std
credit_beta = 0.9        # momentum for the credit (descent-direction) memory
bias_gain   = 0.5        # step size along the credit estimate (acts like an LR)

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
dev_type = 'cuda' if device == 'cuda' else 'cpu'
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
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
# dropout=0: SA probes the loss difference from a weight change, so the +noise vs
# -noise batches must share the SAME (deterministic) forward -- a dropout mask
# would inject noise that swamps the signal.
model = build_model(vocab_size, n_layer, n_embd, n_head, block_size, dropout=0.0).to(device)
print(f"params: {model.num_params():,}")

# ---------------- per-tensor SA state ----------------
# The tied head.weight/wte.weight appears ONCE in .parameters(), so perturbing it
# correctly moves both. `credit` is the momentum of directions that lowered loss;
# `scale` makes sigma0 relative to each tensor's own magnitude (embeddings vs.
# norm gains differ by orders of magnitude).
params = [p for p in model.parameters()]
credit = [torch.zeros_like(p) for p in params]
scale  = [(p.detach().std().item() or 1.0) for p in params]


def temp_at(step):                                          # cosine cooling
    r = step / max(1, max_steps - 1)
    return T_min + 0.5 * (T0 - T_min) * (1 + math.cos(math.pi * r))


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
    return out


@torch.no_grad()
def sa_loss(x, y):                                          # forward-only loss probe (eval mode)
    model.eval()
    with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
        _, loss = model(x, y)
    return loss.item()


# ---------------- train: directional simulated annealing (NO backprop) ----------------
print("--- training (simulated annealing, gradient-free) ---")
t0 = time.time()
best_val = float('inf')
n_accept = 0                                                # accepted moves since the last eval print
for step in range(max_steps):
    with torch.no_grad():
        T     = temp_at(step)
        sigma = sigma0 * (T / T0)                           # perturbations shrink as it cools
        x, y  = get_batch('train')

        # LOCAL move: probe ONE tensor so the loss change is attributable to it.
        j     = int(torch.randint(len(params), ()))
        var   = (sigma * scale[j]) ** 2 + 1e-12
        noise = torch.randn_like(params[j]) * (sigma * scale[j])

        # antithetic pair on the SAME batch: L+ at +noise, L- at -noise.
        params[j].add_(noise);     lp = sa_loss(x, y)       # L+
        params[j].sub_(2 * noise); lm = sa_loss(x, y)       # L-  (now at -noise)
        params[j].add_(noise)                               # back to the unperturbed point
        loss0 = 0.5 * (lp + lm)                             # free baseline for the Metropolis test
        grad  = (lp - lm) / (2 * var) * noise               # local zeroth-order gradient estimate

        # credit = momentum of the DESCENT direction (-grad); updated every step
        # (the probe is informative whether or not the move below is accepted).
        credit[j].mul_(credit_beta).add_(-grad, alpha=(1 - credit_beta))

        # apply the descent step and Metropolis-accept it (SA character + brake)
        step_delta = bias_gain * credit[j]
        params[j].add_(step_delta)
        loss1 = sa_loss(x, y)
        delta = loss1 - loss0
        accept = (delta < 0) or (float(torch.rand(())) < math.exp(-delta / max(T, 1e-8)))
        if accept:
            n_accept += 1
        else:
            params[j].sub_(step_delta)                      # roll back the step

    if step % eval_interval == 0 or step == max_steps - 1:
        m = evaluate()
        bpc = m['val'] / math.log(2) / chars_per_token
        denom = eval_interval if step else 1
        acc_rate = n_accept / denom
        n_accept = 0
        tag = ""
        if m['val'] < best_val:
            best_val = m['val']
            save_checkpoint(model, 'microgpt_sa_best.pt', n_head)
            tag = " *"
        print(f"step {step:6d} | train {m['train']:.4f} | val {m['val']:.4f} "
              f"| {bpc:.3f} bits/char | T {T:.2e} | acc {acc_rate:.2f} | {time.time() - t0:.0f}s{tag}")

save_checkpoint(model, 'microgpt_sa_last.pt', n_head)

# ---------------- sample ----------------
print("--- inference ---")
for p in ["once upon a time ", "the little ", "one day ", "lily and tom ", "the big dog "]:
    idx = torch.tensor([encode(p)], dtype=torch.long, device=device)
    with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
        out = model.generate(idx, max_new=120, temperature=0.8, top_k=40, eot_id=eot_id)
    print(f"[{p!r}] -> {decode(out[0].tolist())}")
