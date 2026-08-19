"""
Spectral compression of optimizer state (research module).

The question: is the persistent state of AdamW / SGD-with-momentum actually
`N` independent numbers, or does it carry enough structure to be described by
`K << N` coefficients of a FIXED basis? If the latter, the 8N bytes AdamW
spends on `m` and `v` can shrink with no learned projection to store.

The model weights, gradients and activations stay dense throughout -- only the
optimizer's memory is compressed. See `README.md` (spectral section) for the
commands and `spectral_probe.py` for the measurement that decides whether any of
this is worth running.
"""

from .optim import SpectralAdamW, SpectralConfig, SpectralSGD
from .state import KController, SpectralState
from .telemetry import RunTimer, Telemetry, write_summary
from .transforms import DCT2D, build_transform, dct, dct2, idct, idct2

__all__ = [
    'SpectralAdamW', 'SpectralSGD', 'SpectralConfig',
    'SpectralState', 'KController',
    'Telemetry', 'RunTimer', 'write_summary',
    'DCT2D', 'build_transform', 'dct', 'idct', 'dct2', 'idct2',
]
