"""
Fixed orthonormal transforms for optimizer-state compression.

The experiment asks whether optimizer state (SGD momentum, Adam's `m` and `v`)
is spectrally redundant -- i.e. whether `K << N` coefficients of a FIXED basis
reconstruct it well enough to train with. Everything here is that basis: no
learned matrices, no SVD, nothing to store (which is the whole systems argument
against GaLore/COAP-style adaptive subspaces).

`DCT2D` is the transform used by the MVP. It is orthonormal, so Parseval holds:

    ||s||_F^2 == ||T(s)||_F^2

which makes retained-coefficient energy an EXACT reconstruction criterion -- the
relative L2 error of a truncated reconstruction is `sqrt(1 - retained_energy)`,
with no inverse transform needed to measure it. That is what makes the
`K_required` telemetry cheap enough to run every refresh.

Implementation is FFT-based (O(N log N)), via the even-extension identity

    DCT-II(x)_k = Re( exp(-i.pi.k / 2N) * FFT_2N([x, reverse(x)])_k )

so it runs on the GPU and needs no scipy and no stored basis matrix. The inverse
is the adjoint (the transform is orthonormal), evaluated with `irfft` by the
same route. `spectral_selftest.py` checks both directions against an explicit
DCT matrix, in fp64 and at the training shapes in fp32.

An STFT transform (spec sections 7/10) drops in behind the same `Transform`
interface: `analyze` / `synthesize` / `default_order` / byte accounting. The
only thing the rest of the code assumes is that `analyze` returns a real
coefficient tensor shaped like its input -- an STFT would return complex
coefficients and must therefore report `coeff_bytes = 8`, which is exactly why
all cross-transform comparisons in the report are in BYTES, not coefficients.
"""

import math

import torch

_ORDER_CACHE = {}       # (shape, k, device) -> int32 low-frequency prefix (persistent, counted)
_CONST_CACHE = {}       # (n, axis-from-end, device, dtype) -> twiddles + mirror permutation


# ---------------- 1D orthonormal DCT-II / DCT-III along one axis ----------------
# The transform runs every step on every eligible tensor, at sizes (~1e5 values)
# where the GPU is bound by kernel launches, not FLOPs. So everything that does
# not depend on the data -- twiddles, the orthonormal scale, the mirror
# permutation -- is built once and cached, and both directions work along any
# axis in place of transposing: a forward transform is 4 kernels, an inverse 2.
def _consts(n, axis_from_end, device, dtype):
    key = (n, axis_from_end, str(device), dtype)
    hit = _CONST_CACHE.get(key)
    if hit is not None:
        return hit
    k = torch.arange(n, device=device, dtype=torch.float64)
    # forward: orthonormal scale folded into the twiddle exp(-i.pi.k / 2n)
    scale = torch.full((n,), math.sqrt(1.0 / (2.0 * n)), device=device, dtype=torch.float64)
    scale[0] = math.sqrt(1.0 / (4.0 * n))
    fwd = torch.polar(scale, -math.pi * k / (2 * n))
    # inverse: basis weights, the conjugate twiddle, the half-spectrum 0.5 (DC is
    # not doubled) and irfft's 1/2n normalization, all in one complex vector
    s = torch.full((n,), math.sqrt(2.0 / n), device=device, dtype=torch.float64)
    s[0] = math.sqrt(1.0 / n)
    inv = torch.polar(s * (0.5 * 2 * n), math.pi * k / (2 * n))
    inv[0] = s[0] * (2 * n)
    cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
    shape = [n] + [1] * (axis_from_end - 1)
    mirror = torch.cat([torch.arange(n, device=device), torch.arange(n - 1, -1, -1, device=device)])
    hit = (fwd.to(cdt).view(shape), inv.to(cdt).view(shape), mirror)
    _CONST_CACHE[key] = hit
    return hit


def dct(x, dim=-1):
    """Orthonormal DCT-II along `dim` (scipy's `norm='ortho'`)."""
    d = dim % x.dim()
    n = x.shape[d]
    fwd, _, mirror = _consts(n, x.dim() - d, x.device, x.dtype)
    spec = torch.fft.rfft(x.index_select(d, mirror), dim=d).narrow(d, 0, n)   # even extension
    return (spec * fwd).real


def idct(c, dim=-1):
    """Inverse of `dct` (orthonormal DCT-III) along `dim`."""
    d = dim % c.dim()
    n = c.shape[d]
    _, inv, _ = _consts(n, c.dim() - d, c.device, c.dtype)
    # irfft zero-pads the missing Nyquist bin itself; keep the first half of the period
    return torch.fft.irfft(c * inv, n=2 * n, dim=d).narrow(d, 0, n)


# ---------------- separable 2D (over the last two axes; leading axes batch) ----------------
def dct2(x):
    """Orthonormal 2D DCT of a matrix: C = D_m . X . D_n^T."""
    return dct(dct(x, -1), -2)


def idct2(c):
    return idct(idct(c, -1), -2)


def low_frequency_order(shape, device):
    """Low-frequency-first ordering of the 2D coefficient grid (flat int64 indices).

    Sorted by `i + j` (total frequency), ties broken by `max(i, j)` so the
    frontier grows as a rounded corner rather than along an axis. Computed on
    demand and NOT cached: a full order is one int per coefficient, so keeping it
    around would be a dense-sized buffer per shape -- exactly what the experiment
    is trying not to store.
    """
    m, n = shape
    i = torch.arange(m, device=device).view(m, 1).expand(m, n)
    j = torch.arange(n, device=device).view(1, n).expand(m, n)
    rank = (i + j).long() * (max(m, n) + 1) + torch.maximum(i, j).long()
    return rank.reshape(-1).argsort(stable=True)


# ---------------- transform interface ----------------
class Transform:
    """A fixed linear basis for optimizer state.

    `coeff_bytes` is the persistent cost of ONE retained coefficient value and
    `index_bytes` the cost of remembering WHICH coefficient it is (zero for
    deterministic selection policies, which need only `K` itself).
    """

    name = 'abstract'
    coeff_bytes = 4          # fp32 amplitude
    index_bytes = 4          # int32 flat index

    def analyze(self, x):
        raise NotImplementedError

    def synthesize(self, c):
        raise NotImplementedError

    def default_order(self, shape, device):
        """Deterministic coefficient priority (flat indices), best-first."""
        raise NotImplementedError

    def low_prefix(self, shape, k, device):
        """The first `k` entries of `default_order`, cached as shared metadata."""
        raise NotImplementedError


class DCT2D(Transform):
    """Orthonormal separable 2D DCT over a weight-shaped state matrix."""

    name = 'dct'

    def analyze(self, x):
        return dct2(x)

    def synthesize(self, c):
        return idct2(c)

    def default_order(self, shape, device):
        return low_frequency_order(shape, device)

    def low_prefix(self, shape, k, device):
        """The first `k` entries of the low-frequency order, as int32 -- the
        DCT-Low active set at its largest. Shared by every state with the same
        shape and `K_max` (so `m` and `v`, and equal-shaped layers, hold one
        copy), and counted ONCE by the optimizer's `memory_report()` as
        persistent metadata. Growing or shrinking `K` is a slice of it, so
        DCT-Low stores no per-state indices at all.
        """
        key = (tuple(shape), int(k), str(device))
        hit = _ORDER_CACHE.get(key)
        if hit is None:
            hit = low_frequency_order(shape, device)[:k].to(torch.int32).contiguous()
            _ORDER_CACHE[key] = hit
        return hit


def build_transform(name):
    if name == 'dct':
        return DCT2D()
    raise ValueError(f"unknown transform {name!r} (STFT is spec section 7, not in this MVP)")
