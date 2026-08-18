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

so it runs on the GPU and needs no scipy. The inverse is the adjoint (the
transform is orthonormal), evaluated with `irfft` by the same route.
`spectral_selftest.py` checks both directions against an explicit DCT matrix.

An STFT transform (spec sections 7/10) drops in behind the same `Transform`
interface: `analyze` / `synthesize` / `default_order` / byte accounting. The
only thing the rest of the code assumes is that `analyze` returns a real
coefficient tensor shaped like its input -- an STFT would return complex
coefficients and must therefore report `coeff_bytes = 8`, which is exactly why
all cross-transform comparisons in the report are in BYTES, not coefficients.
"""

import math

import torch

_ORDER_CACHE = {}


# ---------------- 1D orthonormal DCT-II / DCT-III over the last axis ----------------
def dct(x):
    """Orthonormal DCT-II along the last dimension (scipy's `norm='ortho'`)."""
    n = x.shape[-1]
    mirror = torch.cat([x, x.flip(-1)], dim=-1)                 # even extension, length 2n
    spec = torch.fft.rfft(mirror, dim=-1)[..., :n]              # bins k = 0 .. n-1
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    twiddle = torch.polar(torch.ones_like(k), -math.pi * k / (2 * n))
    c = (spec * twiddle).real                                   # unnormalized DCT-II
    scale = x.new_full((n,), math.sqrt(1.0 / (2.0 * n)))
    scale[0] = math.sqrt(1.0 / (4.0 * n))
    return c * scale


def idct(c):
    """Inverse of `dct` (orthonormal DCT-III) along the last dimension."""
    n = c.shape[-1]
    s = c.new_full((n,), math.sqrt(2.0 / n))
    s[0] = math.sqrt(1.0 / n)
    cp = c * s                                                  # basis-weighted coefficients
    k = torch.arange(n, device=c.device, dtype=c.dtype)
    twiddle = torch.polar(torch.ones_like(k), math.pi * k / (2 * n))
    w = cp * twiddle * 0.5
    w[..., 0] = cp[..., 0]                                      # DC term is not doubled
    w = torch.cat([w, torch.zeros_like(w[..., :1])], dim=-1)    # Nyquist bin -> 0, length n+1
    u = torch.fft.irfft(w, n=2 * n, dim=-1)
    return u[..., :n] * (2 * n)


# ---------------- separable 2D ----------------
def dct2(x):
    """Orthonormal 2D DCT of a matrix: C = D_m . X . D_n^T."""
    return dct(dct(x).transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()


def idct2(c):
    return idct(idct(c).transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()


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


class DCT2D(Transform):
    """Orthonormal separable 2D DCT over a weight-shaped state matrix."""

    name = 'dct'

    def analyze(self, x):
        return dct2(x)

    def synthesize(self, c):
        return idct2(c)

    def default_order(self, shape, device):
        """Low-frequency-first ordering of the 2D coefficient grid.

        Sorted by `i + j` (total frequency), ties broken by `max(i, j)` so the
        frontier grows as a rounded corner rather than along an axis. The result
        is a permutation of flat indices, cached per (shape, device): retaining
        the first K of them is the DCT-Low policy, and because it is a
        deterministic prefix, growing or shrinking K never disturbs the
        coefficients already held (zero active-set churn, zero index storage).
        """
        m, n = shape
        key = (m, n, str(device))
        hit = _ORDER_CACHE.get(key)
        if hit is not None:
            return hit
        i = torch.arange(m, device=device).view(m, 1).expand(m, n)
        j = torch.arange(n, device=device).view(1, n).expand(m, n)
        rank = (i + j).long() * (max(m, n) + 1) + torch.maximum(i, j).long()
        order = rank.reshape(-1).argsort(stable=True).to(torch.int64)
        _ORDER_CACHE[key] = order
        return order


def build_transform(name):
    if name == 'dct':
        return DCT2D()
    raise ValueError(f"unknown transform {name!r} (STFT is spec section 7, not in this MVP)")
