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

Everything runs on tiny tensors in float64 on the CPU: seconds, no GPU, no data.
"""

import math
import sys

import torch

from spectral.metrics import jaccard_churn, k_required, spectral_entropy, energy_profile
from spectral.optim import SpectralAdamW, SpectralConfig, SpectralSGD
from spectral.state import KController
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
    churn, n_new = jaccard_churn(a, b)
    check("Jaccard churn", abs(churn - (1 - 2 / 6)) < 1e-12 and n_new == 2)

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
    return SpectralConfig(state='dct', selection='topk', kmax_frac=1.0, kmin_frac=1.0,
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
    ref = _run(lambda g: torch.optim.AdamW([{'params': g[0]['params'], 'weight_decay': 0.0}],
                                           lr=0.01, betas=(0.9, 0.95)), shapes, grads, True, 0.0)
    errs = []
    for frac in (0.01, 0.1, 0.5, 1.0):
        out = _run(lambda g, f=frac: SpectralAdamW(
            g, lr=0.01, betas=(0.9, 0.95), weight_decay=0.0,
            cfg=SpectralConfig(state='dct', selection='topk', kmax_frac=f, kmin_frac=f,
                               dynamic=False, refresh=1, telemetry_interval=10 ** 9)),
            shapes, grads, True, 0.0)
        errs.append(rel(out[0], ref[0]))
    check("more coefficients -> closer to dense AdamW", all(a > b for a, b in zip(errs, errs[1:])),
          " > ".join(f"{e:.2e}" for e in errs))

    # DCT-Low keeps a deterministic prefix, so its active set never churns
    params = [torch.nn.Parameter(torch.randn(32, 32, dtype=torch.float64) * 0.02)]
    opt = SpectralSGD(_groups(params, ['blocks.0.fc.weight'], True, 0.0), lr=0.01, momentum=0.9,
                      cfg=SpectralConfig(state='dct', selection='low', kmax_frac=0.25,
                                         kmin_frac=0.01, refresh=1, telemetry_interval=10 ** 9))
    for t in range(30):
        params[0].grad = torch.randn(32, 32, dtype=torch.float64)
        opt.step()
    s = opt.state[params[0]]['m']
    check("DCT-Low active set never churns", s.mean_churn == 0.0 and s.n_reset_coeff == 0,
          f"mean churn {s.mean_churn:.3f}, reactivated {s.n_reset_coeff}")
    check("dynamic K stayed within [K_min, K_max]", s.ctrl.k_min <= s.k <= s.ctrl.k_max,
          f"K={s.k} in [{s.ctrl.k_min}, {s.ctrl.k_max}]")
    mem = opt.memory_report()
    check("DCT-Low stores no indices", mem['persistent_state_bytes'] < mem['dense_equivalent_bytes'],
          f"{mem['persistent_state_bytes']}B vs dense {mem['dense_equivalent_bytes']}B "
          f"(ratio {mem['state_ratio']:.3f})")

    # truncation genuinely can break v >= 0; the clamp is a patch, not a fix
    m = torch.randn(32, 32, dtype=torch.float64).abs()
    c = dct2(m).reshape(-1)
    keep = torch.zeros_like(c)
    idx = c.abs().topk(20).indices
    keep[idx] = c[idx]
    check("truncated reconstruction of a positive state can go negative",
          bool((idct2(keep.view(32, 32)) < 0).any()))


if __name__ == '__main__':
    torch.manual_seed(0)
    test_transforms()
    test_metrics()
    test_equivalence()
    test_truncation()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")
