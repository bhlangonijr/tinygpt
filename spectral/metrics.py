"""
Measurement primitives for the spectral-compression experiment.

Nothing here changes training -- these are the numbers the experiment exists to
produce (spec sections 9, 10, 15, 17-20). They are deliberately separate from
the optimizer so the same functions serve the online telemetry
(`spectral/optim.py`) and the offline probe (`spectral_probe.py`), and so a
metric can never be defined twice with two different conventions.

Two conventions worth stating once:

* `K_required` is always measured UNDER A SELECTION ORDER. For DCT-TopK that
  order is descending coefficient magnitude (the best K-term approximation an
  orthonormal basis admits); for DCT-Low it is the fixed low-frequency prefix.
  The same spectrum therefore has two different `K_required` values, which is
  the whole point of testing both policies: if TopK is compact but Low is not,
  the state is DCT-sparse but NOT low-frequency.

* Retained energy `R_K` is an exact reconstruction criterion only because the
  DCT is orthonormal (Parseval). The relative L2 reconstruction error of the
  truncation is `sqrt(1 - R_K)`, so `E_K <= eps` and `R_K >= 1 - eps^2` are the
  same test. A redundant transform (STFT with overlapping windows) breaks that
  identity and must measure `E_K` by actually reconstructing -- which is why
  `state_error()` below exists and is reported alongside.
"""

import math

import torch

EPS = 1e-12


# ---------------- capacity (spec sections 9, 10) ----------------
def energy_profile(coeff, order=None):
    """Cumulative energy fraction along a selection order.

    Returns `(cum, total)` where `cum[k]` is the fraction of total coefficient
    energy captured by the first k+1 coefficients in the order. `order=None`
    means "sort by descending magnitude" (the TopK policy).
    """
    e = coeff.reshape(-1).float().pow(2)
    if order is None:
        e, _ = torch.sort(e, descending=True)
    else:
        e = e[order]
    total = float(e.sum())
    if total <= EPS:
        return torch.zeros_like(e), 0.0
    return torch.cumsum(e, 0) / total, total


def k_required(coeff, tau, order=None, cum=None):
    """Smallest K whose retained energy reaches `tau` (spec section 9).

    Allowed to exceed `K_max`: over-subscription is a first-class output (spec
    section 17.5), so this never clamps.
    """
    if cum is None:
        cum, total = energy_profile(coeff, order)
        if total == 0.0:
            return 1
    idx = int(torch.searchsorted(cum, torch.tensor(float(tau), device=cum.device)))
    return min(idx + 1, cum.numel())


def retained_energy(cum, k):
    """Energy fraction actually captured by the first `k` coefficients."""
    if cum.numel() == 0 or k <= 0:
        return 0.0
    return float(cum[min(k, cum.numel()) - 1])


# ---------------- spectral complexity (spec section 18) ----------------
def spectral_entropy(coeff):
    """Returns `(H_norm, K_eff)` for the coefficient energy distribution.

    `H_norm -> 0` is a concentrated spectrum, `-> 1` is white. `K_eff = e^H` is
    a threshold-free effective coefficient count, so it says how many modes
    matter without committing to a particular `tau`.
    """
    e = coeff.reshape(-1).float().pow(2)
    total = float(e.sum())
    n = e.numel()
    if total <= EPS or n < 2:
        return 0.0, 1.0
    p = e / total
    h = float(-(p * torch.log(p.clamp_min(EPS))).sum())
    return h / math.log(n), math.exp(min(h, 60.0))


# ---------------- fidelity (spec sections 19, 20) ----------------
def state_error(s, s_hat):
    """Relative L2 state reconstruction error `||s - s_hat|| / ||s||`."""
    num = float(torch.linalg.vector_norm(s.reshape(-1) - s_hat.reshape(-1)))
    den = float(torch.linalg.vector_norm(s.reshape(-1))) + 1e-12
    return num / den


def update_fidelity(u, u_hat):
    """Cosine similarity, relative norm and relative error of the update.

    The spec's central fidelity triple: reconstruction error of the STATE is not
    enough, because what actually moves the weights is the UPDATE.
    """
    a, b = u.reshape(-1).float(), u_hat.reshape(-1).float()
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    cos = float(torch.dot(a, b)) / (na * nb + 1e-12)
    return cos, nb / (na + 1e-12), float(torch.linalg.vector_norm(a - b)) / (na + 1e-12)


# ---------------- second-moment validity (spec section 15) ----------------
def negative_v_stats(v_hat):
    """How badly a truncated reconstruction violates `v >= 0`.

    A clamp makes training survive; these numbers say whether it should have.
    Large negative energy means the spectral approximation of `v` is
    structurally wrong even when its global reconstruction error looks fine.
    """
    flat = v_hat.reshape(-1)
    neg = flat < 0
    frac = float(neg.float().mean())
    if frac == 0.0:
        return {'neg_frac': 0.0, 'neg_energy': 0.0, 'max_neg': 0.0}
    negs = flat[neg]
    return {
        'neg_frac': frac,
        'neg_energy': float(negs.pow(2).sum() / (flat.pow(2).sum() + EPS)),
        'max_neg': float(negs.min()),
    }


# ---------------- active-set stability (spec section 16) ----------------
def jaccard_churn(old_sorted, new_sorted):
    """`1 - |A n B| / |A u B|` for two sorted index sets.

    High churn means retained modes keep changing, so the optimizer history
    stored in dropped coefficients is being thrown away and rebuilt from zero.
    """
    if old_sorted.numel() == 0 or new_sorted.numel() == 0:
        return 1.0, 0
    pos = torch.searchsorted(old_sorted, new_sorted).clamp_max(old_sorted.numel() - 1)
    hit = old_sorted[pos] == new_sorted
    inter = int(hit.sum())
    union = old_sorted.numel() + new_sorted.numel() - inter
    return 1.0 - inter / max(union, 1), int(new_sorted.numel() - inter)
