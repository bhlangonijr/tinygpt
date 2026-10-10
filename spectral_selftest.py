#!/usr/bin/env python3
"""
Correctness gate for the spectral optimizer module. Run this first, always.

    python spectral_selftest.py

The load-bearing test is the equivalence pair: with EVERY coefficient retained
(`K = N`), `SpectralSGD` must reproduce `torch.optim.SGD(momentum=...)` and
`SpectralAdamW` must reproduce `torch.optim.AdamW` to floating-point tolerance,
step for step. If that holds, then in a compressed run the only thing that
differs from dense training is truncation -- which is the entire premise of the
experiment. If it fails, every downstream quality number is measuring a bug
instead of measuring compression.

The first sections run on tiny tensors in float64 on the CPU. The last ones run
at the micro model's real shapes in fp32 -- the precision training actually uses
-- on the GPU when one is present, and check what fp64 cannot: rounding at the
training dtype, that the reported memory is the memory actually held, and how
long a step takes. Under a minute, no data.
"""

import math
import sys
import time

import torch

from spectral.metrics import (energy_profile, jaccard_churn, k_required, spectral_entropy,
                             state_error, update_fidelity)
from spectral.optim import SpectralAdamW, SpectralConfig, SpectralSGD
from spectral.state import KController, SpectralState
from spectral.transforms import DCT2D, dct, dct2, idct, idct2

FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-30))


# ---------------- transforms ----------------
def dct_matrix(n, dtype=torch.float64):
    """Explicit orthonormal DCT-II matrix -- the independent reference."""
    k = torch.arange(n, dtype=dtype).view(n, 1)
    i = torch.arange(n, dtype=dtype).view(1, n)
    d = torch.cos(math.pi * k * (2 * i + 1) / (2 * n)) * math.sqrt(2.0 / n)
    d[0] = d[0] / math.sqrt(2.0)
    return d


def test_transforms():
    print("transforms")
    torch.manual_seed(0)
    n, m = 24, 16
    x = torch.randn(7, n, dtype=torch.float64)
    d = dct_matrix(n)
    check("DCT matrix is orthonormal", rel(d @ d.T, torch.eye(n, dtype=torch.float64)) < 1e-12)
    check("FFT DCT-II == matrix DCT-II", rel(dct(x), x @ d.T) < 1e-11,
          f"max rel {rel(dct(x), x @ d.T):.2e}")
    check("IDCT inverts DCT", rel(idct(dct(x)), x) < 1e-12)

    w = torch.randn(m, n, dtype=torch.float64)
    c = dct2(w)
    check("2D DCT == D_m W D_n^T", rel(c, dct_matrix(m) @ w @ dct_matrix(n).T) < 1e-11)
    check("2D round trip", rel(idct2(c), w) < 1e-12)
    check("Parseval (orthonormal)", abs(float(c.pow(2).sum() - w.pow(2).sum())) < 1e-18 * float(w.numel()) + 1e-9)

    g = torch.randn(m, n, dtype=torch.float64)
    check("T(g^2) != T(g)^2 (the second-moment trap)", rel(dct2(g * g), dct2(g) ** 2) > 1e-3)

    # truncation error must fall monotonically with K, and match the Parseval prediction
    flat = c.reshape(-1)
    order = flat.abs().argsort(descending=True)
    errs = []
    for k in (10, 50, 100, 200):
        keep = torch.zeros_like(flat)
        keep[order[:k]] = flat[order[:k]]
        errs.append(float((idct2(keep.view_as(c)) - w).norm() / w.norm()))
    check("truncation error decreases with K", all(a > b for a, b in zip(errs, errs[1:])),
          " > ".join(f"{e:.3f}" for e in errs))
    cum, _ = energy_profile(flat)
    predicted = math.sqrt(max(0.0, 1.0 - float(cum[199])))
    check("energy criterion predicts L2 error", abs(predicted - errs[-1]) < 1e-9,
          f"predicted {predicted:.6f} vs actual {errs[-1]:.6f}")

    dct2d = DCT2D()
    o = dct2d.default_order((4, 5), torch.device('cpu'))
    check("low-frequency order starts at DC and is a permutation",
          int(o[0]) == 0 and sorted(o.tolist()) == list(range(20)))
    pre = dct2d.low_prefix((4, 5), 7, torch.device('cpu'))
    check("low prefix == first K of the order, stored as int32",
          pre.dtype == torch.int32 and pre.tolist() == o[:7].tolist())

    # the 1D transform along a non-last axis, and batching over leading axes
    # (both Adam moments go through ONE batched call) must be the same transform
    check("DCT along dim -2 == matrix DCT", rel(dct(w, -2), dct_matrix(m) @ w) < 1e-11)
    check("IDCT along dim -2 inverts it", rel(idct(dct(w, -2), -2), w) < 1e-12)
    stacked = dct2(torch.stack([w, g]))
    check("batched 2D DCT == per-matrix 2D DCT",
          rel(stacked[0], dct2(w)) < 1e-14 and rel(stacked[1], dct2(g)) < 1e-14)


# ---------------- metrics + controller ----------------
def test_metrics():
    print("metrics and controller")
    c = torch.zeros(100, dtype=torch.float64)
    c[:4] = torch.tensor([3.0, 4.0, 0.0, 0.0])          # energy 9 + 16 = 25
    check("K_required at tau=0.99", k_required(c, 0.99) == 2)
    check("K_required at tau=0.5", k_required(c, 0.5) == 1)

    flat = torch.ones(64, dtype=torch.float64)
    h, keff = spectral_entropy(flat)
    check("white spectrum -> H_norm=1, K_eff=N", abs(h - 1.0) < 1e-12 and abs(keff - 64) < 1e-6)
    spike = torch.zeros(64, dtype=torch.float64); spike[3] = 1.0
    h2, keff2 = spectral_entropy(spike)
    check("delta spectrum -> H_norm=0, K_eff=1", abs(h2) < 1e-12 and abs(keff2 - 1) < 1e-9)

    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([3, 4, 5, 6])
    churn, n_new, n_gone = jaccard_churn(a, b)
    check("Jaccard churn", abs(churn - (1 - 2 / 6)) < 1e-12 and n_new == 2 and n_gone == 2)
    grow = jaccard_churn(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2, 3, 4]))
    check("pure growth evicts nothing", grow[1] == 2 and grow[2] == 0,
          f"activated {grow[1]}, evicted {grow[2]}")

    ctrl = KController(k_min=10, k_max=100, shrink_ratio=0.75, patience=3)
    check("grows immediately", ctrl.update(60) == 60)
    check("holds while requirement is close", ctrl.update(50) == 60)
    seq = [ctrl.update(20) for _ in range(3)]
    check("shrinks only after patience", seq == [60, 60, 20], f"{seq}")
    check("clamps at K_max", ctrl.update(10 ** 9) == 100)
    check("cap hits counted", ctrl.cap_hit_rate > 0)


# ---------------- optimizer equivalence at K = N ----------------
def _groups(params, names, spectral, wd):
    return [{'params': params, 'names': names, 'spectral': spectral, 'weight_decay': wd}]


def _grad_stream(shapes, steps, seed=7):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(*s, generator=g, dtype=torch.float64) * (1.0 + 0.1 * t)
             for s in shapes] for t in range(steps)]


def _run(opt_factory, shapes, grads, spectral, wd, seed=3):
    g = torch.Generator().manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(*s, generator=g, dtype=torch.float64) * 0.02)
              for s in shapes]
    names = [f'blocks.{i}.fc.weight' for i in range(len(shapes))]
    opt = opt_factory(_groups(params, names, spectral, wd))
    for step_grads in grads:
        for p, gr in zip(params, step_grads):
            p.grad = gr.clone()
        opt.step()
    return params


def full_k_cfg(**kw):
    """K = N: keep every coefficient, so truncation is the only thing switched off."""
    kw.setdefault('selection', 'topk')
    return SpectralConfig(state='dct', kmax_frac=1.0, kmin_frac=1.0,
                          dynamic=False, refresh=1, mode='perf', telemetry_interval=10 ** 9, **kw)


def test_equivalence():
    print("optimizer equivalence (K = N)")
    shapes = [(16, 24), (8, 8)]
    steps = 40
    grads = _grad_stream(shapes, steps)
    lr, wd, betas, eps, mom = 0.01, 0.1, (0.9, 0.95), 1e-8, 0.9

    ref = _run(lambda g: torch.optim.SGD([{'params': g[0]['params']}], lr=lr, momentum=mom),
               shapes, grads, True, 0.0)
    dense = _run(lambda g: SpectralSGD(g, lr=lr, momentum=mom,
                                       cfg=SpectralConfig(state='dense')), shapes, grads, True, 0.0)
    spec = _run(lambda g: SpectralSGD(g, lr=lr, momentum=mom, cfg=full_k_cfg()),
                shapes, grads, True, 0.0)
    check("dense SGD-M path == torch.optim.SGD",
          max(rel(a, b) for a, b in zip(dense, ref)) < 1e-12)
    err = max(rel(a, b) for a, b in zip(spec, ref))
    check("spectral SGD-M at K=N == torch.optim.SGD", err < 1e-9, f"max rel {err:.2e}")

    ref = _run(lambda g: torch.optim.AdamW([{'params': g[0]['params'], 'weight_decay': wd}],
                                           lr=lr, betas=betas, eps=eps), shapes, grads, True, wd)
    dense = _run(lambda g: SpectralAdamW(g, lr=lr, betas=betas, eps=eps, weight_decay=wd,
                                         cfg=SpectralConfig(state='dense')), shapes, grads, True, wd)
    spec = _run(lambda g: SpectralAdamW(g, lr=lr, betas=betas, eps=eps, weight_decay=wd,
                                        cfg=full_k_cfg()), shapes, grads, True, wd)
    monly = _run(lambda g: SpectralAdamW(g, lr=lr, betas=betas, eps=eps, weight_decay=wd,
                                         cfg=full_k_cfg(m_only=True)), shapes, grads, True, wd)
    check("dense AdamW path == torch.optim.AdamW",
          max(rel(a, b) for a, b in zip(dense, ref)) < 1e-12)
    err = max(rel(a, b) for a, b in zip(spec, ref))
    check("spectral AdamW at K=N == torch.optim.AdamW", err < 1e-8, f"max rel {err:.2e}")
    err = max(rel(a, b) for a, b in zip(monly, ref))
    check("spectral AdamW (m-only) at K=N == torch.optim.AdamW", err < 1e-8, f"max rel {err:.2e}")
    vonly = _run(lambda g: SpectralAdamW(g, lr=lr, betas=betas, eps=eps, weight_decay=wd,
                                         cfg=full_k_cfg(v_only=True)), shapes, grads, True, wd)
    err = max(rel(a, b) for a, b in zip(vonly, ref))
    check("spectral AdamW (v-only) at K=N == torch.optim.AdamW", err < 1e-8, f"max rel {err:.2e}")
    low = _run(lambda g: SpectralAdamW(g, lr=lr, betas=betas, eps=eps, weight_decay=wd,
                                       cfg=full_k_cfg(selection='low')), shapes, grads, True, wd)
    err = max(rel(a, b) for a, b in zip(low, ref))
    check("spectral AdamW (DCT-Low) at K=N == torch.optim.AdamW", err < 1e-8, f"max rel {err:.2e}")

    # non-eligible params must fall back to the exact dense rule inside the same optimizer
    mixed = _run(lambda g: SpectralAdamW(g, lr=lr, betas=betas, eps=eps, weight_decay=wd,
                                         cfg=full_k_cfg()), shapes, grads, False, wd)
    check("dense fallback for non-eligible tensors == torch.optim.AdamW",
          max(rel(a, b) for a, b in zip(mixed, ref)) < 1e-12)


# ---------------- truncated behaviour ----------------
def test_truncation():
    print("truncated behaviour")
    shapes = [(32, 32)]
    grads = _grad_stream(shapes, 30, seed=11)

    # SGD first: no second-moment denominator, so this isolates truncation alone.
    ref = _run(lambda g: torch.optim.SGD([{'params': g[0]['params']}], lr=0.01, momentum=0.9),
               shapes, grads, True, 0.0)
    errs = []
    for frac in (0.01, 0.1, 0.5, 1.0):
        out = _run(lambda g, f=frac: SpectralSGD(
            g, lr=0.01, momentum=0.9,
            cfg=SpectralConfig(state='dct', selection='topk', kmax_frac=f, kmin_frac=f,
                               dynamic=False, refresh=1, telemetry_interval=10 ** 9)),
            shapes, grads, True, 0.0)
        errs.append(rel(out[0], ref[0]))
    check("SGD-M: more coefficients -> closer to dense", all(a > b for a, b in zip(errs, errs[1:])),
          " > ".join(f"{e:.2e}" for e in errs))

    # AdamW is tested on the quantity the experiment actually measures (sections
    # 19-20): reconstruction error of the state, and fidelity of the update it
    # produces, at a realistic state. A 30-step trajectory would instead be
    # dominated by the section-15 denominator collapse, whose blow-up is checked
    # separately below and is not monotone in K.
    ref = _run(lambda g: torch.optim.AdamW([{'params': g[0]['params'], 'weight_decay': 0.0}],
                                           lr=0.01, betas=(0.9, 0.95)), shapes, grads, True, 0.0)
    dense_opt_params = [torch.nn.Parameter(torch.randn(32, 32, dtype=torch.float64) * 0.02)]
    dense_opt = SpectralAdamW(_groups(dense_opt_params, ['blocks.0.fc.weight'], True, 0.0),
                              lr=0.01, betas=(0.9, 0.95),
                              cfg=SpectralConfig(state='dense', telemetry_interval=10 ** 9))
    for step_grads in grads:
        dense_opt_params[0].grad = step_grads[0].clone()
        dense_opt.step()
    st = dense_opt.state[dense_opt_params[0]]
    m_d, v_d, t = st['m'], st['v'], st['t']
    bc1, bc2, eps = 1 - 0.9 ** t, 1 - 0.95 ** t, 1e-8
    u_dense = (m_d / bc1) / ((v_d / bc2).sqrt() + eps)

    def truncate(x, k):
        c = dct2(x).reshape(-1)
        keep = torch.zeros_like(c)
        i = c.abs().topk(k).indices
        keep[i] = c[i]
        return idct2(keep.view_as(x))

    m_err, v_err, cos = [], [], []
    for frac in (0.01, 0.05, 0.25, 1.0):
        k = max(1, int(frac * m_d.numel()))
        m_h, v_h = truncate(m_d, k), truncate(v_d, k).clamp_min(0.0)
        m_err.append(state_error(m_d, m_h))
        v_err.append(state_error(v_d, v_h))
        cos.append(update_fidelity(u_dense, (m_h / bc1) / ((v_h / bc2).sqrt() + eps))[0])
    check("AdamW: m reconstruction error falls with K",
          all(a > b for a, b in zip(m_err, m_err[1:])), " > ".join(f"{e:.3f}" for e in m_err))
    check("AdamW: v reconstruction error falls with K",
          all(a > b for a, b in zip(v_err, v_err[1:])), " > ".join(f"{e:.3f}" for e in v_err))
    check("AdamW: update cosine rises with K", all(a < b for a, b in zip(cos, cos[1:])),
          " < ".join(f"{c:.4f}" for c in cos))
    print(f"  [--] H2 observation at K/N=5%: m error {m_err[1]:.3f} vs v error {v_err[1]:.3f} "
          f"({'m' if m_err[1] < v_err[1] else 'v'} compresses better here)")

    # DCT-Low's real invariant: the active set is always a PREFIX of the fixed
    # low-frequency order. Growing K therefore adds coefficients without ever
    # displacing one, so no history is destroyed -- but churn is still nonzero
    # while K grows, because the set genuinely changed.
    def churn_run(selection, **kw):
        params = [torch.nn.Parameter(torch.randn(32, 32, dtype=torch.float64) * 0.02)]
        opt = SpectralSGD(_groups(params, ['blocks.0.fc.weight'], True, 0.0),
                          lr=0.01, momentum=0.9,
                          cfg=SpectralConfig(state='dct', selection=selection, kmax_frac=0.25,
                                             refresh=1, telemetry_interval=10 ** 9, **kw))
        g = torch.Generator().manual_seed(5)
        for _ in range(30):
            params[0].grad = torch.randn(32, 32, generator=g, dtype=torch.float64)
            opt.step()
        return opt, opt.state[params[0]]['m']

    _, low = churn_run('low', kmin_frac=0.01)
    prefix = DCT2D().default_order((32, 32), torch.device('cpu'))[:low.k]
    check("DCT-Low active set is always a prefix of the fixed order",
          low.idx.tolist() == prefix.tolist() and low.n_evicted == 0,
          f"K={low.k}, activated {low.n_activated}, evicted {low.n_evicted}")

    _, fixed = churn_run('low', kmin_frac=0.25)
    check("DCT-Low at fixed K never churns at all",
          fixed.mean_churn == 0.0 and fixed.n_activated == 0 and fixed.n_evicted == 0,
          f"mean churn {fixed.mean_churn:.3f}")

    _, top = churn_run('topk', kmin_frac=0.25)
    check("TopK at fixed K does churn, and evicts history", top.n_evicted > 0,
          f"mean churn {top.mean_churn:.3f}, evicted {top.n_evicted} vs Low 0")
    check("dynamic K stayed within [K_min, K_max]", low.ctrl.k_min <= low.k <= low.ctrl.k_max,
          f"K={low.k} in [{low.ctrl.k_min}, {low.ctrl.k_max}]")
    opt, _ = churn_run('low', kmin_frac=0.01)
    mem = opt.memory_report()
    check("DCT-Low stores no indices", mem['persistent_state_bytes'] < mem['dense_equivalent_bytes'],
          f"{mem['persistent_state_bytes']}B vs dense {mem['dense_equivalent_bytes']}B "
          f"(ratio {mem['state_ratio']:.3f})")

    # truncation genuinely can break v >= 0; the clamp is a patch, not a fix.
    # Use a spiky-but-strictly-positive state (like Adam's v = g^2: mostly small,
    # a few large): its energy spreads across high frequencies, so keeping only the
    # top coefficients rings below zero. (A smooth random positive field usually
    # survives top-K truncation, which is why this needs a deterministic spike.)
    m = torch.full((32, 32), 0.01, dtype=torch.float64)
    m[16, 16] += 1.0
    c = dct2(m).reshape(-1)
    keep = torch.zeros_like(c)
    idx = c.abs().topk(20).indices
    keep[idx] = c[idx]
    check("truncated reconstruction of a positive state can go negative",
          bool((idct2(keep.view(32, 32)) < 0).any()))


# ---------------- the truncated path, exactly ----------------
def test_truncated_exact():
    """K < N, checked against an independent reference rather than a trend.

    With DCT-Low at fixed K the active set never moves, so by linearity the held
    coefficients must be EXACTLY the first K low-frequency coefficients of the
    dense moments, and every parameter update must be EXACTLY AdamW run on their
    truncated reconstructions. This is the truncated arithmetic itself -- the
    K = N gate cannot see a bug that only bites when coefficients are dropped.
    """
    print("truncated path == independent truncated-AdamW reference")
    shape, steps, frac = (24, 40), 25, 0.1
    lr, wd, (b1, b2), eps, floor = 0.01, 0.1, (0.9, 0.95), 1e-8, 1e-6
    grads = _grad_stream([shape], steps, seed=21)
    for mode in ('both', 'v_only', 'm_only'):
        out = _run(lambda g: SpectralAdamW(
            g, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd,
            cfg=SpectralConfig(state='dct', selection='low', kmax_frac=frac, kmin_frac=frac,
                               dynamic=False, refresh=1, v_floor=floor,
                               v_only=(mode == 'v_only'), m_only=(mode == 'm_only'),
                               telemetry_interval=10 ** 9)), [shape], grads, True, wd)
        k = int(round(frac * shape[0] * shape[1]))
        keep = DCT2D().default_order(shape, torch.device('cpu'))[:k]

        def trunc(x):
            c = dct2(x).reshape(-1)
            kept = torch.zeros_like(c)
            kept[keep] = c[keep]
            return idct2(kept.view(shape))

        p = torch.randn(*shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 0.02
        m = torch.zeros(shape, dtype=torch.float64)
        v = torch.zeros(shape, dtype=torch.float64)
        for t, (gr,) in enumerate(grads, start=1):
            m = b1 * m + (1 - b1) * gr
            v = b2 * v + (1 - b2) * gr * gr
            m_hat = trunc(m) if mode != 'v_only' else m
            v_hat = trunc(v).clamp_min(floor) if mode != 'm_only' else v
            p = p * (1 - lr * wd)
            p = p - lr * (m_hat / (1 - b1 ** t)) / ((v_hat / (1 - b2 ** t)).sqrt() + eps)
        err = rel(out[0].detach(), p)
        check(f"DCT-Low fixed K=10% ({mode}) == truncated-AdamW reference", err < 1e-10,
              f"max rel {err:.2e}")


def test_reselect():
    """Active-set moves: survivors keep their history, newcomers get this step,
    evictions are counted -- checked element by element."""
    print("active-set re-selection")
    cpu = torch.device('cpu')
    t = DCT2D()
    ctrl = KController(4, 4)
    s = SpectralState((4, 4), t, 'topk', ctrl, None, cpu, dtype=torch.float64)
    s.idx = torch.tensor([1, 3, 5, 7], dtype=torch.int32)
    s.coef = torch.tensor([10., 30., 50., 70.], dtype=torch.float64)
    ref = torch.zeros(16, dtype=torch.float64)
    ref[[3, 7, 9, 11]] = torch.tensor([4., 3., 2., 1.], dtype=torch.float64)
    seed = torch.arange(16, dtype=torch.float64) * -1.0
    churn = s.reselect(ref, 4, seed=seed)
    held = dict(zip(s.idx.tolist(), s.coef.tolist()))
    check("TopK keeps survivors' accumulated value", held[3] == 30. and held[7] == 70.)
    check("TopK seeds newcomers with this step's contribution", held[9] == -9. and held[11] == -11.)
    check("TopK counts evictions and Jaccard churn",
          s.n_evicted == 2 and s.n_activated == 2 and abs(churn - (1 - 2 / 6)) < 1e-12)
    check("TopK indices stay sorted int32",
          s.idx.dtype == torch.int32 and s.idx.tolist() == sorted(s.idx.tolist()))

    ctrl = KController(2, 6)
    pre = t.low_prefix((4, 4), 6, cpu)
    s = SpectralState((4, 4), t, 'low', ctrl, pre, cpu, dtype=torch.float64)
    s.coef = torch.tensor([1., 2.], dtype=torch.float64)
    s.reselect(None, 5, seed=seed)
    check("DCT-Low growth keeps the prefix in place and seeds only the tail",
          s.idx.tolist() == pre[:5].tolist() and s.coef[:2].tolist() == [1., 2.]
          and s.coef[2:].tolist() == [float(-i) for i in pre[2:5].tolist()] and s.n_evicted == 0)
    s.reselect(None, 3, seed=seed)
    check("DCT-Low shrink drops only the tail",
          s.idx.tolist() == pre[:3].tolist() and s.coef.tolist()[:2] == [1., 2.] and s.n_evicted == 2)
    check("DCT-Low holds no per-state index storage",
          s.idx.data_ptr() == pre.data_ptr() and s.persistent_bytes() == 3 * 8 + 16)


def test_controller_proxy():
    """Regression for the under-provisioning bug (spec section 43, correction).

    Under a persistent gradient an EMA converges to the gradient itself, so the
    true state's `K_required` is that of `T(g)` (for `m`) and `T(g^2)` (for `v`).
    The controller must end at or above it -- the old proxy compared retained
    EMAs with a one-step `(1-b).T(g)` and settled at roughly half. It may sit
    above by up to the hysteresis band (shrink only below 0.75 . K).
    """
    print("capacity controller sizes K from the steady-state spectrum")
    shape, tau = (32, 32), 0.99
    gen = torch.Generator().manual_seed(9)
    g = torch.randn(*shape, generator=gen, dtype=torch.float64) * torch.linspace(0.1, 3, 32).view(32, 1)
    params = [torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float64))]
    opt = SpectralAdamW(_groups(params, ['blocks.0.fc.weight'], True, 0.0), lr=1e-3,
                        cfg=SpectralConfig(state='dct', selection='topk', kmax_frac=1.0,
                                           kmin_frac=0.01, tau=tau, refresh=1, v_floor=1e-12,
                                           telemetry_interval=10 ** 9))
    for _ in range(300):
        params[0].grad = g.clone()
        opt.step()
    st = opt.state[params[0]]
    for label, target in (('m', g), ('v', g * g)):
        need = k_required(dct2(target), tau)
        k = st[label].k
        check(f"controller K for {label} covers the steady-state requirement",
              need <= k <= math.ceil(need / 0.75) + 1, f"K={k}, K_required(steady)={need}")


# ---------------- training shapes, training dtype ----------------
MICRO_SHAPES = [(576, 192), (192, 192), (768, 192), (192, 768)]   # qkv, proj, fc, out


def _device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_training_precision():
    """fp32 at the shapes the optimizer actually sees (fp64 above hides rounding)."""
    dev = _device()
    print(f"fp32 at training shapes ({dev})")
    worst = {'fwd': 0.0, 'rt': 0.0, 'parseval': 0.0, 'batch': 0.0}
    for shape in MICRO_SHAPES:
        x = torch.randn(*shape, device=dev)
        ref = (dct_matrix(shape[0]).to(dev) @ x.double() @ dct_matrix(shape[1]).to(dev).T)
        c = dct2(x)
        worst['fwd'] = max(worst['fwd'], rel(c.double(), ref))
        worst['rt'] = max(worst['rt'], rel(idct2(c), x))
        worst['parseval'] = max(worst['parseval'],
                                abs(float(c.double().pow(2).sum() / x.double().pow(2).sum()) - 1))
        worst['batch'] = max(worst['batch'], rel(dct2(torch.stack([x, x * x]))[1], dct2(x * x)))
    check("fp32 DCT == fp64 matrix DCT at every training shape", worst['fwd'] < 2e-6,
          f"max rel {worst['fwd']:.2e}")
    check("fp32 round trip at every training shape", worst['rt'] < 2e-6, f"max rel {worst['rt']:.2e}")
    check("fp32 Parseval at every training shape", worst['parseval'] < 1e-5,
          f"max rel {worst['parseval']:.2e}")
    check("batched == separate, bit for bit", worst['batch'] == 0.0, f"max rel {worst['batch']:.2e}")

    # K = N in fp32 against torch.optim.AdamW in fp32. Gradients are linear-layer
    # shaped (row x column scales) at a realistic magnitude.
    def grad_stream(steps, spread, bounded=False):
        """`bounded` keeps |g| away from zero (v spans ~6 decades), so the
        representational limit below cannot fire and K = N must be exact."""
        gen = torch.Generator(device=dev).manual_seed(2)
        out = []
        for _ in range(steps):
            row = []
            for sh in MICRO_SHAPES:
                r = (torch.randn(sh[0], 1, device=dev, generator=gen) * spread).exp()
                c = (torch.randn(1, sh[1], device=dev, generator=gen) * spread).exp()
                z = torch.randn(*sh, device=dev, generator=gen)
                if bounded:
                    z = z.sign() * (0.5 + z.abs().clamp(max=2.0))
                row.append(z * r * c * 1e-3)
            out.append(row)
        return out

    def run(factory, grads, dtype=torch.float32):
        gen = torch.Generator(device=dev).manual_seed(1)
        params = [torch.nn.Parameter((torch.randn(*sh, device=dev, generator=gen) * 0.02).to(dtype))
                  for sh in MICRO_SHAPES]
        opt = factory(_groups(params, [f'blocks.{i}.w' for i in range(len(params))], True, 0.1))
        for row in grads:
            for p, gr in zip(params, row):
                p.grad = gr.to(dtype).clone()
            opt.step()
        return [p.detach().double() for p in params]

    gen = torch.Generator(device=dev).manual_seed(1)
    init = [(torch.randn(*sh, device=dev, generator=gen) * 0.02).double() for sh in MICRO_SHAPES]

    def drift(a, b):          # worst element error, relative to that tensor's total movement
        return max(float((x - y).abs().max() / (y - y0).abs().max()) for x, y, y0 in zip(a, b, init))

    def off(a, b):            # fraction of elements whose error exceeds 1% of their own movement
        n = sum(x.numel() for x in a)
        return sum(int(((x - y).abs() > 0.01 * (y - y0).abs()).sum()) for x, y, y0 in zip(a, b, init)) / n

    torch_adamw = lambda g: torch.optim.AdamW(g[0]['params'], lr=1e-3, betas=(0.9, 0.95),
                                              weight_decay=0.1)
    spectral = lambda **kw: (lambda g: SpectralAdamW(g, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1,
                                                      cfg=full_k_cfg(**kw)))

    # The algorithm at the training shapes: in fp64, K = N must be AdamW exactly
    # (on gradients bounded away from zero -- see the limit measured next).
    grads = grad_stream(100, 0.5, bounded=True)
    ref64 = run(torch_adamw, grads, torch.float64)
    for name, kw in (('m+v topk', {}), ('m+v low', dict(selection='low')),
                     ('m-only', dict(m_only=True)), ('v-only', dict(v_only=True))):
        d = drift(run(spectral(**kw), grads, torch.float64), ref64)
        check(f"fp64 K=N {name} == torch AdamW at training shapes", d < 1e-9,
              f"worst element error {d:.2e} of its movement")

    # Gaussian gradients have entries arbitrarily close to zero, and then even
    # fp64 is not exact: the worst element's step-1 g^2 sits ~14 decades under
    # the tensor's maximum, at the edge of fp64 resolution.
    grads = grad_stream(100, 0.5)
    d = drift(run(spectral(), grads, torch.float64), run(torch_adamw, grads, torch.float64))
    print(f"  [--] fp64 K=N m+v, Gaussian gradients (unbounded below): worst element error {d:.2e}")

    # fp32 is NOT exact even at K = N, for either moment, and this is measured
    # rather than gated. Each fp32 coefficient carries ~1e-7 relative rounding,
    # and that error lands on every element at the scale of the tensor's
    # LARGEST values, while AdamW divides each element by its OWN sqrt(v). An
    # element ~7 decades under the maximum is therefore swamped: for `v` its
    # reconstruction reaches zero and the step is set by the floor (or by eps,
    # at floor 0). No transform precision fixes it -- only fp64 coefficients (2x
    # the memory) or a floor (a different update rule). Dense fp32 rounds each
    # element relative to ITSELF, which is the baseline printed first. The effect
    # on real micro-model training is measured in spec section 43.
    for spread, label in ((0.5, 'moderate'), (1.0, 'heavy-tailed')):
        grads = grad_stream(100, spread)
        ref64 = run(torch_adamw, grads, torch.float64)
        base = run(torch_adamw, grads)
        print(f"  [--] {label} gradients, dense torch AdamW fp32 vs fp64: "
              f"{off(base, ref64):.2%} of elements off by >1%, worst {drift(base, ref64):.2e}")
        for name, kw in (('m-only', dict(m_only=True)), ('m+v floor 0', {}),
                         ('m+v floor 1e-8', dict(v_floor=1e-8))):
            out = run(spectral(**kw), grads)
            print(f"  [--] {label} gradients, spectral K=N {name} fp32 vs fp64 AdamW: "
                  f"{off(out, ref64):.2%} of elements off by >1%, worst {drift(out, ref64):.2e}")

    # nothing goes NaN/inf under heavy truncation, a floor, or an all-zero gradient
    params = [torch.nn.Parameter(torch.randn(*sh, device=dev) * 0.02) for sh in MICRO_SHAPES]
    opt = SpectralAdamW(_groups(params, [f'blocks.{i}.w' for i in range(4)], True, 0.1), lr=1e-3,
                        cfg=SpectralConfig(state='dct', selection='topk', kmax_frac=0.01,
                                           v_floor=1e-8, refresh=10, telemetry_interval=10 ** 9))
    for t, row in enumerate(grad_stream(50, 0.5)):
        for p, gr in zip(params, row):
            p.grad = torch.zeros_like(gr) if t % 10 == 3 else gr.clone()
        opt.step()
    check("K/N=1% TopK with floor: finite after 50 steps incl. all-zero gradients",
          all(bool(torch.isfinite(p).all()) for p in params))


def test_memory_accounting():
    """The reported persistent bytes must be the bytes actually held.

    Walks every tensor the optimizer keeps between steps and sums its real
    storage. A shared DCT-Low prefix is counted once. Nothing per-state may be
    dense-sized except a moment that is deliberately dense (m-only / v-only).
    """
    dev = _device()
    print(f"memory accounting == bytes actually held ({dev})")
    for name, kw in (('m+v low 5%', dict(selection='low')),
                     ('m+v topk 5%', dict(selection='topk')),
                     ('v-only topk 5%', dict(selection='topk', v_only=True)),
                     ('m-only low 5%', dict(selection='low', m_only=True))):
        params = [torch.nn.Parameter(torch.randn(*sh, device=dev) * 0.02)
                  for sh in MICRO_SHAPES for _ in range(4)]
        opt = SpectralAdamW(_groups(params, [f'blocks.{i}.w' for i in range(len(params))], True, 0.1),
                            lr=1e-3, cfg=SpectralConfig(state='dct', kmax_frac=0.05, kmin_frac=0.01,
                                                        refresh=1, v_floor=1e-8,
                                                        telemetry_interval=10 ** 9, **kw))
        for _ in range(3):
            for p in params:
                p.grad = torch.randn_like(p) * 1e-3
            opt.step()
        held, shared, n_states, dense_sized = 0, {}, 0, 0
        for p in params:
            for key in ('m', 'v'):
                s = opt.state[p][key]
                if isinstance(s, SpectralState):
                    n_states += 1
                    held += s.coef.numel() * s.coef.element_size()
                    if s.prefix is None:
                        held += s.idx.numel() * s.idx.element_size()
                    else:
                        shared[s.prefix.data_ptr()] = s.prefix.numel() * s.prefix.element_size()
                    dense_sized += int(max(s.coef.numel(), s.idx.numel()) >= p.numel())
                else:
                    held += s.numel() * s.element_size()
        held += sum(shared.values())
        rep = opt.memory_report()
        reported = rep['persistent_state_bytes'] - 16 * n_states
        check(f"{name}: reported state == tensors held", reported == held,
              f"reported {reported} B vs held {held} B, ratio to dense {rep['state_ratio']:.3f}")
        check(f"{name}: no dense-sized spectral buffer", dense_sized == 0)


def test_throughput():
    """Not a gate -- hardware dependent. Measured so that efficiency claims rest
    on a number: transform cost at training shapes and whole optimizer steps."""
    dev = _device()
    print(f"throughput, informational ({dev}; {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'})")

    def timed(fn, reps=50):
        for _ in range(5):
            fn()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    x = torch.randn(768, 192, device=dev)
    xx = torch.stack([x, x])
    print(f"  [--] dct2 {timed(lambda: dct2(x)) * 1e6:.0f} us, idct2 {timed(lambda: idct2(x)) * 1e6:.0f} us, "
          f"batched pair dct2 {timed(lambda: dct2(xx)) * 1e6:.0f} us  (768x192 fp32)")

    def step_time(factory):
        params = [torch.nn.Parameter(torch.randn(*sh, device=dev) * 0.02)
                  for sh in MICRO_SHAPES for _ in range(4)]
        opt = factory(_groups(params, [f'blocks.{i}.w' for i in range(len(params))], True, 0.1))
        for p in params:
            p.grad = torch.randn_like(p) * 1e-3
        return timed(opt.step, reps=30)

    base = step_time(lambda g: torch.optim.AdamW(g[0]['params'], lr=1e-3, betas=(0.9, 0.95)))
    print(f"  [--] torch AdamW step over the 16 micro matrices: {base * 1e3:.2f} ms")
    for name, kw in (('m+v low 5%', dict(selection='low')), ('m+v topk 5%', dict(selection='topk')),
                     ('v-only topk 5%', dict(selection='topk', v_only=True))):
        t = step_time(lambda g, kw=kw: SpectralAdamW(
            g, lr=1e-3, betas=(0.9, 0.95),
            cfg=SpectralConfig(state='dct', kmax_frac=0.05, kmin_frac=0.05, dynamic=False,
                               refresh=10, v_floor=1e-8, telemetry_interval=10 ** 9, **kw)))
        print(f"  [--] spectral {name}: {t * 1e3:.2f} ms  ({t / base:.1f}x torch AdamW)")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_transforms()
    test_metrics()
    test_equivalence()
    test_truncation()
    test_truncated_exact()
    test_reselect()
    test_controller_proxy()
    test_training_precision()
    test_memory_accounting()
    test_throughput()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")
