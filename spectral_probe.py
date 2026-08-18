#!/usr/bin/env python3
"""
Offline measurement: how compressible is optimizer state, really?

This is spec section 37 step 4, and it is the experiment's GO/NO-GO. It trains
the model with an ORDINARY dense AdamW -- nothing is compressed, nothing is
approximated -- and periodically asks, for every eligible weight matrix:

    how many coefficients does it take to hold `tau` of the energy of
    the gradient / of Adam's m / of Adam's v?

measured in four bases:

    dct_topk       best K-term 2D-DCT approximation (largest coefficients)
    dct_low        deterministic low-frequency prefix
    identity_topk  the state's own coordinates, largest entries first
    dct_shuffled   2D DCT after a fixed random permutation of the entries

The last two are controls, and they are the point. `identity_topk` says how much
of the apparent compressibility is just heavy-tailed magnitudes rather than
spatial structure -- if raw top-K is as compact as DCT top-K, the transform
bought nothing. `dct_shuffled` destroys spatial adjacency while preserving the
value distribution: if `dct_shuffled ~ dct_topk`, the DCT is not finding
structure, it is finding sparsity that was already there (spec section 34).

    python spectral_probe.py --scale micro --steps 3000 --probe-interval 250

Writes `runs/<name>/probe.csv` (one row per step/tensor/state/basis/tau) and
prints the summary table that decides whether compressed training is worth
running at all.
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import torch

import config as C
from data import load_data, make_get_batch
from model import build_model
from spectral.metrics import energy_profile, k_required, retained_energy, spectral_entropy
from spectral.transforms import DCT2D
from train_spectral import SCALES, build_param_groups

TAUS = (0.95, 0.99, 0.999)
RESID_FRACS = (0.01, 0.05, 0.10, 0.25)     # K_max/N values the report cares about
BASES = ('dct_topk', 'dct_low', 'identity_topk', 'dct_shuffled')
HEADER = (['step', 'tensor', 'state', 'basis', 'n', 'tau', 'k_req', 'k_req_over_n',
           'spectral_entropy', 'k_eff_over_n']
          + [f'resid_at_{int(f * 100)}pct' for f in RESID_FRACS])

_PERM = {}


def perm_for(shape, device):
    """Fixed random permutation of a tensor's entries (the structure control)."""
    key = (tuple(shape), str(device))
    if key not in _PERM:
        g = torch.Generator(device='cpu').manual_seed(1234 + shape[0] * 31 + shape[1])
        _PERM[key] = torch.randperm(shape[0] * shape[1], generator=g).to(device)
    return _PERM[key]


def coefficients(s, basis, dct):
    """Coefficient vector of state `s` in `basis`, plus its selection order."""
    if basis == 'identity_topk':
        return s.reshape(-1), None
    if basis == 'dct_shuffled':
        shuffled = s.reshape(-1)[perm_for(s.shape, s.device)].reshape(s.shape)
        return dct.analyze(shuffled).reshape(-1), None
    c = dct.analyze(s).reshape(-1)
    return c, (dct.default_order(s.shape, s.device) if basis == 'dct_low' else None)


def analyze(s, dct):
    """All bases x all taus for one state tensor. Returns a list of row dicts."""
    n = s.numel()
    rows = []
    for basis in BASES:
        c, order = coefficients(s, basis, dct)
        cum, total = energy_profile(c, order)
        if total == 0.0:
            continue
        ent, keff = spectral_entropy(c)
        resid = {f'resid_at_{int(f * 100)}pct': 1.0 - retained_energy(cum, max(1, int(f * n)))
                 for f in RESID_FRACS}
        for tau in TAUS:
            k = k_required(None, tau, cum=cum)
            rows.append({'basis': basis, 'n': n, 'tau': tau, 'k_req': k, 'k_req_over_n': k / n,
                         'spectral_entropy': ent, 'k_eff_over_n': keff / n, **resid})
    return rows


def summarize(path):
    """Slice an existing probe.csv -- no training, no GPU.

    The headline table the probe prints at the end is one number per
    (state, basis) at tau=0.99. That is enough to decide go/no-go, but not
    enough to understand a result, so this reads the whole file back and asks
    the three follow-up questions the spec cares about: how sensitive is the
    answer to tau (section 9), does required capacity move during training
    (Q2 / section 31), and does it vary by tensor?

    The last table is the one that decides anything, because coefficient counts
    are not comparable across policies: DCT-Low stores 4 bytes per retained
    coefficient and needs no indices, while any TopK policy stores a 4-byte
    value AND a 4-byte index. Sections 7 and 22 require the comparison to be in
    bytes, against the 4 bytes per parameter a dense fp32 state costs.
    """
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"{path}: empty")
        return
    steps = sorted({int(r['step']) for r in rows})
    mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')

    def sel(**kw):
        out = rows
        for k, v in kw.items():
            out = [r for r in out if r[k] == str(v)]
        return out

    states = [s for s in ('g', 'm', 'v') if sel(state=s)]

    print(f"\n== K_required / N by reconstruction target ==")
    print(f"  {'state':<6} {'tau':>6} " + " ".join(f"{b:>15}" for b in BASES))
    for st in states:
        for tau in TAUS:
            cells = [f"{mean([float(r['k_req_over_n']) for r in sel(state=st, basis=b, tau=tau)]):>15.4f}"
                     for b in BASES]
            print(f"  {st:<6} {tau:>6} " + " ".join(cells))

    print(f"\n== evolution of K_required / N (tau=0.99, dct_topk) -- does capacity move? ==")
    print(f"  {'state':<6} " + " ".join(f"{s:>8}" for s in steps))
    for st in states:
        cells = [f"{mean([float(r['k_req_over_n']) for r in sel(state=st, basis='dct_topk', tau=0.99, step=s)]):>8.3f}"
                 for s in steps]
        print(f"  {st:<6} " + " ".join(cells))

    print(f"\n== K_required / N by tensor kind (tau=0.99, dct_topk) ==")
    kinds = sorted({r['tensor'].split('.')[-2] for r in rows})
    print(f"  {'state':<6} " + " ".join(f"{k:>10}" for k in kinds))
    for st in states:
        cells = [f"{mean([float(r['k_req_over_n']) for r in sel(state=st, basis='dct_topk', tau=0.99) if r['tensor'].split('.')[-2] == k]):>10.3f}"
                 for k in kinds]
        print(f"  {st:<6} " + " ".join(cells))

    print(f"\n== residual energy left unrepresented at a fixed budget (tau-free) ==")
    cols = [f'resid_at_{int(f * 100)}pct' for f in RESID_FRACS]
    print(f"  {'state':<6} {'basis':<15} " + " ".join(f"{c.replace('resid_at_', 'K='):>10}" for c in cols))
    for st in states:
        for b in BASES:
            got = sel(state=st, basis=b, tau=0.99)
            print(f"  {st:<6} {b:<15} " + " ".join(f"{mean([float(r[c]) for r in got]):>10.4f}" for c in cols))

    print(f"\n== bytes per parameter of persistent state (dense fp32 = 4.00) ==")
    print(f"  {'state':<6} {'basis':<15} {'K/N':>8} {'B/param':>9} {'vs dense':>9}")
    for st in states:
        for b in BASES:
            k = mean([float(r['k_req_over_n']) for r in sel(state=st, basis=b, tau=0.99)])
            per = 4 if b == 'dct_low' else 8      # deterministic prefix needs no stored indices
            print(f"  {st:<6} {b:<15} {k:>8.3f} {k * per:>9.2f} {k * per / 4:>8.2f}x")
    print("\n  A ratio >= 1.00 means the compressed representation costs more than the dense\n"
          "  state it replaces. Spec section 35 wants <= 0.50.")


def main():
    p = argparse.ArgumentParser(description="offline optimizer-state compressibility probe")
    p.add_argument('--analyze', default=None,
                   help='summarize an existing probe.csv and exit (no training)')
    p.add_argument('--scale', choices=list(SCALES), default='micro')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--probe-interval', type=int, default=250)
    p.add_argument('--lr', type=float, default=C.lr)
    p.add_argument('--weight-decay', type=float, default=C.weight_decay)
    p.add_argument('--dropout', type=float, default=C.dropout)
    p.add_argument('--seed', type=int, default=C.seed)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--name', default='probe')
    p.add_argument('--out', default='runs')
    a = p.parse_args()
    if a.analyze:
        return summarize(a.analyze)

    out_dir = os.path.join(a.out, a.name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'probe.csv')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev_type = 'cuda' if device == 'cuda' else 'cpu'
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = (device == 'cuda') and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    scale = SCALES[a.scale]
    d = load_data()
    gen = torch.Generator().manual_seed(a.seed)
    get_batch = make_get_batch(d['train_data'], d['val_data'], scale['block_size'],
                               scale['batch_size'], device, gen=gen)
    model = build_model(d['vocab_size'], scale['n_layer'], scale['n_embd'], scale['n_head'],
                        scale['block_size'], a.dropout).to(device)
    groups, coverage = build_param_groups(model, a.weight_decay, True)
    eligible = {}
    for g in groups:
        if g.get('spectral'):
            eligible = dict(zip(g['names'], g['params']))
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k not in ('names', 'spectral')}
                             for g in groups], lr=a.lr, betas=(0.9, 0.95))
    dct = DCT2D()
    print(f"probe | device {device} | params {model.num_params():,} | "
          f"{len(eligible)} eligible tensors | coverage {coverage:.1%}", flush=True)

    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, HEADER).writeheader()

    agg = defaultdict(list)
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        x, y = get_batch('train')
        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)

        if step % a.probe_interval == 0:
            rows = []
            with torch.no_grad():
                for name, param in eligible.items():
                    st = opt.state.get(param, {})
                    states = {'g': param.grad.float()}
                    if 'exp_avg' in st:                     # after the first step
                        states['m'] = st['exp_avg'].float()
                        states['v'] = st['exp_avg_sq'].float()
                    for sname, s in states.items():
                        for r in analyze(s, dct):
                            rows.append({'step': step, 'tensor': name, 'state': sname, **r})
                            if r['tau'] == 0.99:
                                agg[(sname, r['basis'])].append(r['k_req_over_n'])
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, HEADER).writerows(rows)
            print(f"step {step:6d} | loss {loss.item():.4f} | {time.time() - t0:.0f}s "
                  f"| probed {len(rows)} rows", flush=True)
        opt.step()

    # ---------------- verdict ----------------
    print(f"\nK_required / N at tau=0.99, mean over tensors and probe steps "
          f"({a.scale}, {a.steps} steps):\n")
    print(f"  {'state':<6} " + " ".join(f"{b:>15}" for b in BASES))
    for sname in ('g', 'm', 'v'):
        cells = []
        for basis in BASES:
            vals = agg.get((sname, basis), [])
            cells.append(f"{sum(vals) / len(vals):>15.4f}" if vals else f"{'-':>15}")
        print(f"  {sname:<6} " + " ".join(cells))
    print(f"\nwrote {csv_path}")
    print("read it like this: dct_topk << identity_topk means the DCT is finding real spatial\n"
          "structure; dct_topk ~ dct_shuffled means it is not, and the premise is weak (section 34).")


if __name__ == '__main__':
    main()
