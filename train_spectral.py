#!/usr/bin/env python3
"""
tinyGPT trainer for the spectral-optimizer-state experiment.

Same model, same corpus, same schedule as `train.py` -- the ONLY variable is
where the optimizer keeps its persistent state:

    --state dense    control: ordinary SGD-M / AdamW state, one fp32 per param
    --state dct      state lives as K DCT coefficients per weight matrix

Both paths run through the same optimizer classes (`spectral/optim.py`), so the
control and the treatment share their arithmetic order, their weight-decay
handling, and their bias correction. Compression is the only difference, which
is what makes a loss delta attributable.

    # controls
    python train_spectral.py --opt adamw --state dense --scale micro --name adamw_dense
    python train_spectral.py --opt sgd   --state dense --scale micro --name sgd_dense

    # compressed state, fully instrumented (shadow dense states -> fidelity metrics)
    python train_spectral.py --opt adamw --state dct --select topk --kmax-frac 0.05 \
        --mode instrumented --scale micro --name adamw_dct_topk05

    # a whole screening matrix, benchmark mode
    python train_spectral.py --sweep experiments/mvp_screen.jsonl

Every run writes `runs/<name>/{summary.json, telemetry.csv, best.pt, last.pt}`.
Checkpoints use the shared `save_checkpoint`, so `infer.py` and `bench.py` read
them with no changes.

Fairness (spec section 29): model init, batch order, LR schedule, clipping,
weight decay, precision and token budget are pinned by `--seed` and are
identical across variants. Batches come from a private `torch.Generator`, so a
run that consumes extra randomness for instrumentation still sees exactly the
same data in the same order.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

import config as C
from data import load_data, make_get_batch
from model import build_model, save_checkpoint
from spectral.optim import SpectralAdamW, SpectralConfig, SpectralSGD
from spectral.telemetry import Telemetry, write_summary

# ---------------- experiment defaults ----------------
# Spectral knobs live here, not in config.py: only this experiment reads them
# (same convention as the SA trainer's hyper-parameters).
MIN_SPECTRAL_NUMEL = 4096          # smaller matrices keep dense state; the overhead is not worth it
SGD_DEFAULT_LR     = 0.1           # SGD-M needs its own scale; AdamW uses config.lr
SPIKE_FACTOR       = 1.5           # train loss > factor * EMA counts as a spike (stability metric)

SCALES = {
    # Micro: correctness + instrumentation (spec section 26). ~2M params.
    'micro': dict(n_layer=4, n_embd=192, n_head=4, block_size=128, batch_size=32),
    # Small: the repo's own configuration -- the primary research result.
    'small': dict(n_layer=C.n_layer, n_embd=C.n_embd, n_head=C.n_head,
                  block_size=C.block_size, batch_size=C.batch_size),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="spectral optimizer-state experiment")
    p.add_argument('--opt', choices=['adamw', 'sgd'], default='adamw')
    p.add_argument('--state', choices=['dense', 'dct'], default='dense')
    p.add_argument('--select', choices=['low', 'topk'], default='topk')
    p.add_argument('--kmax-frac', type=float, default=0.05, help='K_max / N')
    p.add_argument('--kmin-frac', type=float, default=0.005, help='K_min / N')
    p.add_argument('--tau', type=float, default=0.99, help='target retained energy')
    p.add_argument('--refresh', type=int, default=10, help='R: steps between active-set updates')
    p.add_argument('--fixed-k', action='store_true', help='pin K at K_max (no dynamic sizing)')
    p.add_argument('--shrink-ratio', type=float, default=0.75)
    p.add_argument('--shrink-patience', type=int, default=3)
    p.add_argument('--m-only', action='store_true', help='AdamW: spectral m, dense v')
    p.add_argument('--v-only', action='store_true', help='AdamW: spectral v, dense m')
    p.add_argument('--v-floor', type=float, default=0.0,
                   help='floor for the reconstructed second moment (0 = spec section 15 clamp)')
    p.add_argument('--mode', choices=['perf', 'instrumented'], default='perf')
    p.add_argument('--telemetry-interval', type=int, default=200)
    p.add_argument('--profile', action='store_true', help='sampled per-phase timing (synchronizes)')
    p.add_argument('--torch-baseline', action='store_true',
                   help='dense control via torch.optim.AdamW instead of the in-repo dense path')

    p.add_argument('--scale', choices=list(SCALES), default='micro')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--min-lr', type=float, default=None)
    p.add_argument('--warmup', type=int, default=C.warmup_steps)
    p.add_argument('--weight-decay', type=float, default=C.weight_decay)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--dropout', type=float, default=C.dropout)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=C.seed)
    p.add_argument('--eval-interval', type=int, default=250)
    p.add_argument('--eval-iters', type=int, default=C.eval_iters)
    p.add_argument('--target-val-loss', type=float, default=None,
                   help='record tokens/wall-clock to first reach this val loss')

    p.add_argument('--name', default=None)
    p.add_argument('--out', default='runs')
    p.add_argument('--no-ckpt', action='store_true')
    p.add_argument('--sweep', default=None, help='jsonl of configs; runs each as a subprocess')
    return p.parse_args(argv)


def run_name(a):
    if a.name:
        return a.name
    if a.state == 'dense':
        return f"{a.opt}_dense_{a.scale}_s{a.seed}"
    bits = [a.opt, a.state, a.select, f"k{a.kmax_frac:g}", f"t{a.tau:g}", f"r{a.refresh}"]
    if a.fixed_k:
        bits.append('fixedk')
    if a.m_only:
        bits.append('monly')
    if a.v_only:
        bits.append('vonly')
    return "_".join(bits + [a.scale, f"s{a.seed}"])


# ---------------- parameter routing (spec section 27) ----------------
def build_param_groups(model, weight_decay, spectral_enabled):
    """Eligible: the block projection matrices. Everything else keeps dense state.

    Embeddings, the tied head and the RMSNorm gains are deliberately excluded --
    the spec compresses attention/MLP matrices first, and reporting the coverage
    fraction is what makes a whole-model memory number interpretable.
    """
    spec, dense_decay, nodecay = [], [], []
    spec_n, dense_n, nod_n = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        eligible = (spectral_enabled and p.dim() == 2 and name.startswith('blocks.')
                    and p.numel() >= MIN_SPECTRAL_NUMEL)
        if eligible:
            spec.append(p), spec_n.append(name)
        elif p.dim() >= 2:
            dense_decay.append(p), dense_n.append(name)
        else:
            nodecay.append(p), nod_n.append(name)
    groups = []
    if spec:
        groups.append({'params': spec, 'names': spec_n, 'spectral': True,
                       'weight_decay': weight_decay})
    if dense_decay:
        groups.append({'params': dense_decay, 'names': dense_n, 'spectral': False,
                       'weight_decay': weight_decay})
    if nodecay:
        groups.append({'params': nodecay, 'names': nod_n, 'spectral': False,
                       'weight_decay': 0.0})
    covered = sum(p.numel() for p in spec)
    total = sum(p.numel() for p in model.parameters())
    return groups, covered / max(1, total)


def main(argv=None):
    a = parse_args(argv)
    if a.sweep:
        return run_sweep(a)

    name = run_name(a)
    out_dir = os.path.join(a.out, name)
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev_type = 'cuda' if device == 'cuda' else 'cpu'
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = (device == 'cuda') and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    scale = SCALES[a.scale]
    lr = a.lr if a.lr is not None else (C.lr if a.opt == 'adamw' else SGD_DEFAULT_LR)
    min_lr = a.min_lr if a.min_lr is not None else lr * 0.1

    # ---------------- data (private RNG stream -> identical batch order per seed) ----------------
    d = load_data()
    batch_gen = torch.Generator().manual_seed(a.seed)
    get_batch = make_get_batch(d['train_data'], d['val_data'], scale['block_size'],
                               scale['batch_size'], device, gen=batch_gen)
    # Evaluation draws from its OWN stream, re-seeded at every evaluation: the
    # training batch order is then independent of the eval schedule, and every
    # evaluation of every run scores the same windows (so differences between
    # runs are not partly differences between eval samples).
    eval_gen = torch.Generator()
    get_eval_batch = make_get_batch(d['train_data'], d['val_data'], scale['block_size'],
                                    scale['batch_size'], device, gen=eval_gen)
    chars_per_token = d['chars_per_token']

    # ---------------- model ----------------
    model = build_model(d['vocab_size'], scale['n_layer'], scale['n_embd'], scale['n_head'],
                        scale['block_size'], a.dropout).to(device)
    n_params = model.num_params()

    # ---------------- optimizer ----------------
    groups, coverage = build_param_groups(model, a.weight_decay, a.state != 'dense')
    cfg = SpectralConfig(state=a.state, selection=a.select, kmax_frac=a.kmax_frac,
                         kmin_frac=a.kmin_frac, tau=a.tau, refresh=a.refresh,
                         shrink_ratio=a.shrink_ratio, shrink_patience=a.shrink_patience,
                         dynamic=not a.fixed_k, m_only=a.m_only, v_only=a.v_only,
                         v_floor=a.v_floor, mode=a.mode,
                         telemetry_interval=a.telemetry_interval, profile=a.profile)
    tel = Telemetry(os.path.join(out_dir, 'telemetry.csv'), enabled=(a.state != 'dense'))
    if a.torch_baseline:
        assert a.state == 'dense' and a.opt == 'adamw', \
            "--torch-baseline cross-checks the dense AdamW path only"
        opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != 'names' and k != 'spectral'}
                                 for g in groups], lr=lr, betas=(0.9, 0.95))
    elif a.opt == 'adamw':
        opt = SpectralAdamW(groups, lr=lr, betas=(0.9, 0.95), weight_decay=a.weight_decay,
                            cfg=cfg, telemetry=tel)
    else:
        opt = SpectralSGD(groups, lr=lr, momentum=a.momentum, weight_decay=0.0,
                          cfg=cfg, telemetry=tel)

    print(f"run {name} | device {device} | bf16 {use_amp} | params {n_params:,} "
          f"| spectral coverage {coverage:.1%} | vocab {d['vocab_size']}")

    def lr_at(step):
        if step < a.warmup:
            return lr * (step + 1) / a.warmup
        r = (step - a.warmup) / max(1, a.steps - a.warmup)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * r))

    @torch.no_grad()
    def evaluate():
        model.eval()
        eval_gen.manual_seed(a.seed + 1)
        out = {}
        for split in ('train', 'val'):
            losses = torch.zeros(a.eval_iters)
            for i in range(a.eval_iters):
                x, y = get_eval_batch(split)
                with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---------------- train ----------------
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    tokens_per_step = scale['batch_size'] * scale['block_size']
    hist = {'step': [], 'train': [], 'val': [], 'bpc': [], 'time': []}
    stability = {'nan_count': 0, 'spikes': 0, 'grad_norm_mean': 0.0, 'update_norm_mean': 0.0,
                 'grad_norm_samples': 0, 'update_norm_samples': 0}
    best_val, ema, target_hit = float('inf'), None, None
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        x, y = get_batch('train')
        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))

        sample = (step % a.telemetry_interval == 0)
        pre = torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()]) if sample else None
        opt.step()
        if sample:
            post = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            stability['update_norm_mean'] += float(torch.linalg.vector_norm(post - pre))
            stability['update_norm_samples'] += 1
            del pre, post
        stability['grad_norm_mean'] += gnorm
        stability['grad_norm_samples'] += 1

        lval = loss.item()
        if not math.isfinite(lval):
            stability['nan_count'] += 1
        else:
            if ema is not None and lval > SPIKE_FACTOR * ema:
                stability['spikes'] += 1
            ema = lval if ema is None else 0.98 * ema + 0.02 * lval

        if step % a.eval_interval == 0 or step == a.steps - 1:
            m = evaluate()
            bpc = m['val'] / math.log(2) / chars_per_token
            elapsed = time.time() - t0
            hist['step'].append(step), hist['train'].append(m['train'])
            hist['val'].append(m['val']), hist['bpc'].append(bpc), hist['time'].append(elapsed)
            tag = ""
            if m['val'] < best_val:
                best_val = m['val']
                if not a.no_ckpt:
                    save_checkpoint(model, os.path.join(out_dir, 'best.pt'), scale['n_head'])
                tag = " *"
            if a.target_val_loss and target_hit is None and m['val'] <= a.target_val_loss:
                target_hit = {'step': step, 'seconds': elapsed,
                              'tokens': (step + 1) * tokens_per_step}
            print(f"step {step:6d} | train {m['train']:.4f} | val {m['val']:.4f} "
                  f"| {bpc:.3f} bits/char | {elapsed:.0f}s{tag}", flush=True)

    wall = time.time() - t0
    if not a.no_ckpt:
        save_checkpoint(model, os.path.join(out_dir, 'last.pt'), scale['n_head'])
    tel.flush()

    # ---------------- summary ----------------
    final_val = hist['val'][-1]
    mem = opt.memory_report() if hasattr(opt, 'memory_report') else {}
    spec = opt.spectral_summary() if hasattr(opt, 'spectral_summary') else {}
    for k in ('grad_norm', 'update_norm'):
        n = stability.pop(f'{k}_samples')
        stability[f'{k}_mean'] = stability[f'{k}_mean'] / max(1, n)
    summary = {
        'name': name,
        'config': {**vars(a), 'lr': lr, 'min_lr': min_lr, 'device': device, 'bf16': use_amp,
                   'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                   'spectral': cfg.dict(), 'scale_dims': scale},
        'model': {'params': n_params, 'spectral_coverage': coverage,
                  'vocab_size': d['vocab_size'], 'chars_per_token': chars_per_token},
        'memory': {**mem,
                   'peak_cuda_bytes': (torch.cuda.max_memory_allocated() if device == 'cuda' else None)},
        'timing': {'wall_seconds': wall, 'tokens': a.steps * tokens_per_step,
                   'tokens_per_sec': a.steps * tokens_per_step / max(wall, 1e-9),
                   'time_to_target': target_hit},
        # Spec section 24: the token-level numbers (PPL, bits/token) are exact for a
        # tokenized model; bits/char additionally divides by measured chars/token on the
        # validation stream, which is what makes it comparable across tokenizers.
        'quality': {'final_val_loss': final_val, 'best_val_loss': best_val,
                    'final_ppl': math.exp(min(final_val, 30)), 'best_ppl': math.exp(min(best_val, 30)),
                    'final_bits_per_token': final_val / math.log(2),
                    'best_bits_per_token': best_val / math.log(2),
                    'final_bpc': hist['bpc'][-1], 'best_bpc': min(hist['bpc']),
                    'final_train_loss': hist['train'][-1]},
        'stability': stability,
        'spectral_behaviour': spec,
        'history': hist,
    }
    write_summary(os.path.join(out_dir, 'summary.json'), summary)
    print(f"done: val {final_val:.4f} | best {best_val:.4f} | {hist['bpc'][-1]:.3f} bits/char "
          f"| {summary['timing']['tokens_per_sec']:.0f} tok/s | state "
          f"{mem.get('persistent_state_bytes', 0) / 2**20:.1f} MiB "
          f"({mem.get('state_ratio', 1):.3f} of dense) -> {out_dir}")
    return summary


def run_sweep(a):
    """Run every config in a jsonl file as its own process (clean CUDA peak per run)."""
    with open(a.sweep) as f:
        rows = [json.loads(line) for line in f if line.strip() and not line.startswith('#')]
    print(f"sweep: {len(rows)} configs from {a.sweep}")
    for i, row in enumerate(rows, 1):
        argv = []
        for k, v in row.items():
            flag = '--' + k.replace('_', '-')
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
            else:
                argv += [flag, str(v)]
        name = row.get('name') or f"sweep{i}"
        if os.path.exists(os.path.join(a.out, name, 'summary.json')):
            print(f"[{i}/{len(rows)}] {name}: already done, skipping")
            continue
        print(f"[{i}/{len(rows)}] {' '.join(argv)}", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), *argv], check=False)


if __name__ == '__main__':
    main()
