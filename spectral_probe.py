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
import json
import os
import time
from collections import defaultdict

import torch

import config as C
from data import load_data, make_get_batch
from model import build_model
from spectral.metrics import (energy_profile, k_required, negative_v_stats, retained_energy,
                              spectral_entropy, state_error, update_fidelity)
from spectral.telemetry import provenance
from spectral.transforms import DCT2D
from train_spectral import SCALES, build_param_groups

TAUS = (0.95, 0.99, 0.999)
RESID_FRACS = (0.01, 0.05, 0.10, 0.25)     # K_max/N values the report cares about
BASES = ('dct_topk', 'dct_low', 'identity_topk', 'dct_shuffled')
HEADER = (['step', 'tensor', 'state', 'basis', 'n', 'tau', 'k_req', 'k_req_over_n',
           'spectral_entropy', 'k_eff_over_n']
          + [f'resid_at_{int(f * 100)}pct' for f in RESID_FRACS])

FID_BUDGETS = (0.01, 0.05, 0.10, 0.25)
FID_HEADER = ['step', 'tensor', 'n', 'budget', 'variant', 'floor', 'update_cosine',
              'update_rel_norm', 'update_rel_err', 'm_recon_err', 'v_recon_err', 'v_neg_frac',
              'v_at_floor']
# How a reconstructed second moment is made non-negative before 1/sqrt(v) sees it.
# 'clamp0' is spec section 15 as written; the others exist because clamping at exactly
# zero does not produce a bad approximation, it produces a singular one -- the
# denominator becomes eps and that element's step is ~1e5x oversized, which swamps
# every fidelity metric regardless of how good the rest of the reconstruction is.
V_FLOORS = ('clamp0', 'abs1e-8', 'rel1e-3')

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

    fid_path = os.path.join(os.path.dirname(path) or '.', 'probe_fidelity.csv')
    if os.path.exists(fid_path):
        rows = []
        for r in csv.DictReader(open(fid_path)):
            r['budget'] = float(r['budget'])
            r['update_cosine'] = float(r['update_cosine'])
            r['update_rel_norm'] = float(r['update_rel_norm'])
            rows.append(r)
        print_fidelity(rows)


def truncate_topk(x, k, dct):
    """Best K-term DCT approximation of `x` (the DCT-TopK policy, applied offline)."""
    c = dct.analyze(x).reshape(-1)
    keep = torch.zeros_like(c)
    i = c.abs().topk(k).indices
    keep[i] = c[i]
    return dct.synthesize(keep.view_as(x))


def floored(v_raw, policy):
    """Make a reconstructed second moment non-negative under one policy.

    `rel1e-3` is floored at a thousandth of the reconstruction's own mean, which
    is deployable: for an orthonormal DCT the mean IS the DC coefficient over
    sqrt(mn), so it is known without keeping anything extra.
    """
    if policy == 'clamp0':
        return v_raw.clamp_min(0.0)
    if policy == 'abs1e-8':
        return v_raw.clamp_min(1e-8)
    return v_raw.clamp_min(max(1e-30, 1e-3 * float(v_raw.mean())))


def fidelity_rows(m, v, t, dct, betas=(0.9, 0.95), eps=1e-8):
    """How good is the UPDATE when the state is truncated? (spec section 20)

    Retained energy answers a question about the state; it does not answer the
    question that matters, because AdamW does not use `v` -- it uses
    `1/sqrt(v)`. A second moment can hold 86% of its energy in 1% of its DCT
    coefficients and still produce a useless update, because the reciprocal
    square root amplifies exactly the small values a low-order reconstruction
    gets wrong, and any element driven below zero has to be made positive first.

    That last step is not a detail. DCT ringing puts a few percent of elements
    below zero at EVERY budget -- the fraction is set by the transform, not by
    K -- so clamping them to exactly zero hands those elements a denominator of
    `eps` and an update ~1e5 times too large. They then dominate the update norm,
    and the measured cosine reports the singularity rather than the
    approximation, never improving as K grows. Each floor policy is therefore
    measured separately: `clamp0` is the spec-literal rule and shows the damage,
    the floored variants show what the spectral approximation is actually worth.

    `update_rel_norm` is reported alongside the cosine precisely so this failure
    is visible: a value far above 1 means a handful of elements have taken over.
    """
    b1, b2 = betas
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    u_dense = (m / bc1) / ((v / bc2).sqrt() + eps)
    rows = []
    for f in FID_BUDGETS:
        k = max(1, int(f * m.numel()))
        m_h = truncate_topk(m, k, dct)
        v_raw = truncate_topk(v, k, dct)
        neg = negative_v_stats(v_raw)
        m_err = state_error(m, m_h)

        cos, rnorm, rerr = update_fidelity(u_dense, (m_h / bc1) / ((v / bc2).sqrt() + eps))
        rows.append({'n': m.numel(), 'budget': f, 'variant': 'm_only', 'floor': 'n/a',
                     'update_cosine': cos, 'update_rel_norm': rnorm, 'update_rel_err': rerr,
                     'm_recon_err': m_err, 'v_recon_err': '', 'v_neg_frac': '', 'v_at_floor': ''})

        for policy in V_FLOORS:
            v_h = floored(v_raw, policy)
            at_floor = float((v_h == v_h.min()).float().mean()) if policy != 'clamp0' \
                else float((v_h == 0).float().mean())
            v_err = state_error(v, v_h)
            # Two controls, and they are the point of this table. `v_scalar` replaces the
            # whole second moment with one number (its mean) -- if that scores as well as
            # a K-coefficient DCT, the transform is earning nothing at the update level,
            # exactly as identity_topk showed for the energy criterion. `v_floor_only`
            # applies the floor WITHOUT truncating, isolating what the floor alone costs.
            v_scalar = floored(torch.full_like(v, float(v.mean())), policy)
            v_nofloor = floored(v, policy)
            for variant, mm, vv in (('v_only', m, v_h), ('both', m_h, v_h),
                                    ('v_scalar', m, v_scalar), ('v_floor_only', m, v_nofloor)):
                cos, rnorm, rerr = update_fidelity(
                    u_dense, (mm / bc1) / ((vv / bc2).sqrt() + eps))
                rows.append({'n': m.numel(), 'budget': f, 'variant': variant, 'floor': policy,
                             'update_cosine': cos, 'update_rel_norm': rnorm, 'update_rel_err': rerr,
                             'm_recon_err': m_err if variant == 'both' else '',
                             'v_recon_err': v_err if variant in ('v_only', 'both') else '',
                             'v_neg_frac': neg['neg_frac'], 'v_at_floor': at_floor})
    return rows


def print_fidelity(rows):
    """Summary of the section-20 metrics -- the numbers that decide viability."""
    if not rows:
        return
    mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')

    def pick(budget, variant, floor=None):
        return [r for r in rows if r['budget'] == budget and r['variant'] == variant
                and (floor is None or r['floor'] == floor)]

    print("\n== AdamW update fidelity under DCT-TopK truncation ==")
    print("   cosine with the dense update; |u|/|u*| is the relative update norm\n")
    print(f"  {'K/N':>5} {'m_only':>9} {'m |u|/|u*|':>11} | "
          + " | ".join(f"{'v_only ' + fl:>16}" for fl in V_FLOORS))
    for f in FID_BUDGETS:
        m_rows = pick(f, 'm_only')
        cells = []
        for fl in V_FLOORS:
            r = pick(f, 'v_only', fl)
            cells.append(f"{mean([x['update_cosine'] for x in r]):>7.4f}"
                         f" ({mean([x['update_rel_norm'] for x in r]):>7.1e})")
        print(f"  {f * 100:>4.0f}% {mean([x['update_cosine'] for x in m_rows]):>9.4f} "
              f"{mean([x['update_rel_norm'] for x in m_rows]):>11.2f} | " + " | ".join(cells))

    print(f"\n  {'K/N':>5} " + " ".join(f"{'both ' + fl:>18}" for fl in V_FLOORS)
          + f" {'v<0 frac':>10} {'v at floor':>11}")
    for f in FID_BUDGETS:
        cells = [f"{mean([x['update_cosine'] for x in pick(f, 'both', fl)]):>18.4f}"
                 for fl in V_FLOORS]
        r = pick(f, 'v_only', 'clamp0')
        print(f"  {f * 100:>4.0f}% " + " ".join(cells)
              + f" {mean([float(x['v_neg_frac']) for x in r]):>10.4f}"
              + f" {mean([float(x['v_at_floor']) for x in r]):>11.4f}")

    ctrl_floor = 'abs1e-8'
    print(f"\n  controls at floor={ctrl_floor} -- is the DCT earning anything at the update level?")
    print(f"  {'K/N':>5} {'v_only (K coeffs)':>19} {'v_scalar (1 number)':>21} "
          f"{'v_floor_only (no trunc)':>25}")
    for f in FID_BUDGETS:
        vals = [mean([x['update_cosine'] for x in pick(f, var, ctrl_floor)])
                for var in ('v_only', 'v_scalar', 'v_floor_only')]
        print(f"  {f * 100:>4.0f}% {vals[0]:>19.4f} {vals[1]:>21.4f} {vals[2]:>25.4f}")
    print("  v_scalar does not depend on K. If v_only never beats it, the spectral")
    print("  representation of the second moment is decoration.")

    print("\n  clamp0 is spec section 15 as written. If its cosine is far below the floored\n"
          "  variants and its |u|/|u*| is huge, the metric is reporting the eps singularity\n"
          "  (a few percent of elements with a denominator of eps), not the approximation.\n"
          "  Compare floored columns across K: those must improve with K, or the state\n"
          "  genuinely does not reconstruct.")


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

    fid_path = os.path.join(out_dir, 'probe_fidelity.csv')
    with open(os.path.join(out_dir, 'probe_meta.json'), 'w') as f:
        json.dump({**vars(a), **provenance(), 'device': device,
                   'gpu': torch.cuda.get_device_name() if device == 'cuda' else None}, f, indent=2)
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, HEADER).writeheader()
    with open(fid_path, 'w', newline='') as f:
        csv.DictWriter(f, FID_HEADER).writeheader()

    agg, fid_all = defaultdict(list), []
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
            rows, fid = [], []
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
                    if 'm' in states:
                        t_adam = int(float(st['step'])) if 'step' in st else step + 1
                        for r in fidelity_rows(states['m'], states['v'], max(1, t_adam), dct):
                            fid.append({'step': step, 'tensor': name, **r})
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, HEADER).writerows(rows)
            with open(fid_path, 'a', newline='') as f:
                csv.DictWriter(f, FID_HEADER).writerows(fid)
            fid_all.extend(fid)
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
    print_fidelity(fid_all)
    print(f"\nwrote {csv_path} and {fid_path}")
    print("read it like this: dct_topk << identity_topk means the DCT is finding real spatial\n"
          "structure; dct_topk ~ dct_shuffled means it is not, and the premise is weak (section 34).")


if __name__ == '__main__':
    main()
