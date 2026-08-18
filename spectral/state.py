"""
Sparse spectral optimizer state + the dynamic-K controller.

`SpectralState` is what replaces one dense `(m, n)` optimizer-state tensor: a
sorted index buffer and a coefficient buffer, both of length `K`. It never
materializes a dense copy of itself -- growing, shrinking and re-selecting the
active set are all done sparsely (`searchsorted` against the previous sorted
indices), because a dense shadow buffer per state would silently give back the
memory the experiment is trying to save.

The controller implements spec section 11: grow immediately when capacity is
short, shrink only after the requirement has stayed well below the current size
for `patience` consecutive refreshes. The asymmetry is deliberate -- symmetric
resizing oscillates (K=100, 180, 100, 180...) and every oscillation destroys
optimizer history, since a coefficient that leaves the active set comes back
carrying only the current step, not its accumulated past (spec section 16).
"""

import torch

from .metrics import jaccard_churn


class KController:
    """Hysteretic capacity controller for one spectral state."""

    def __init__(self, k_min, k_max, shrink_ratio=0.75, patience=3):
        self.k_min = max(1, int(k_min))
        self.k_max = max(self.k_min, int(k_max))
        self.shrink_ratio = shrink_ratio
        self.patience = patience
        self.k_cur = self.k_min
        self.low_streak = 0
        self.cap_hits = 0
        self.refreshes = 0

    def update(self, k_req):
        """Feed the measured requirement, return the new capacity."""
        self.refreshes += 1
        target = min(max(int(k_req), self.k_min), self.k_max)
        if target >= self.k_max:
            self.cap_hits += 1
        if target > self.k_cur:                       # grow at once
            self.k_cur, self.low_streak = target, 0
        elif target < self.shrink_ratio * self.k_cur:  # shrink only if it persists
            self.low_streak += 1
            if self.low_streak >= self.patience:
                self.k_cur, self.low_streak = max(target, self.k_min), 0
        else:
            self.low_streak = 0
        return self.k_cur

    @property
    def cap_hit_rate(self):
        return self.cap_hits / max(1, self.refreshes)


class SpectralState:
    """One optimizer-state tensor held as `K` spectral coefficients.

    `idx` is kept SORTED at all times: the sorted invariant is what makes
    carry-over on resize an O(K log K) `searchsorted` instead of a dense
    scatter, and it makes churn measurable for free.
    """

    def __init__(self, shape, transform, selection, ctrl, order, device, dtype=torch.float32):
        self.shape = tuple(shape)
        self.numel = int(shape[0] * shape[1])
        self.transform = transform
        self.selection = selection
        self.ctrl = ctrl
        self.order = order                       # deterministic priority (DCT-Low)
        init = order[:ctrl.k_cur] if selection == 'low' else torch.arange(ctrl.k_cur, device=device)
        self.idx = init.sort().values.contiguous()
        self.coef = torch.zeros(self.idx.numel(), device=device, dtype=dtype)
        self.total_churn = 0.0
        self.n_refresh = 0
        self.n_reset_coeff = 0                   # coefficients that came back with no history

    # ---------------- the linear recursion (spec sections 13, 14) ----------------
    def accumulate(self, coeff_dense_flat, decay, alpha):
        """`c <- decay * c + alpha * T(g)` restricted to the active set."""
        self.coef.mul_(decay).add_(coeff_dense_flat[self.idx], alpha=alpha)

    def scatter_into(self, buf_flat):
        """Write the sparse coefficients into a zeroed dense buffer."""
        buf_flat.zero_()
        buf_flat[self.idx] = self.coef
        return buf_flat

    def reconstruct(self, buf):
        """Dense state estimate `s_hat = T^-1(S_K(C))` (uses `buf` as scratch)."""
        self.scatter_into(buf.reshape(-1))
        return self.transform.synthesize(buf)

    # ---------------- active-set maintenance ----------------
    def reselect(self, ref_dense_flat, k, seed=None):
        """Move to a `k`-coefficient active set chosen from `ref_dense_flat`.

        Surviving coefficients keep their accumulated value. A newly activated
        coefficient has no history -- that loss is real and is counted in
        `n_reset_coeff` and in the churn metric. What it does NOT have to lose
        is the CURRENT step: `seed` (this step's transformed gradient
        contribution) initializes new coefficients to the value they would have
        had if they had been active for exactly this one step. Without it, a
        re-selection silently discards fresh gradient information as well as old
        history, and the churn measurement would be confounded by an
        implementation choice.
        """
        if self.selection == 'low':
            new_idx = self.order[:k]
        else:
            new_idx = torch.topk(ref_dense_flat.abs(), k, sorted=False).indices
        new_idx = new_idx.sort().values.contiguous()
        churn, n_new = jaccard_churn(self.idx, new_idx)
        if k == self.idx.numel() and n_new == 0:
            self.n_refresh += 1
            return 0.0
        new_coef = (seed[new_idx].clone() if seed is not None
                    else torch.zeros(k, device=self.coef.device, dtype=self.coef.dtype))
        if self.idx.numel():
            pos = torch.searchsorted(self.idx, new_idx).clamp_max(self.idx.numel() - 1)
            hit = self.idx[pos] == new_idx
            new_coef[hit] = self.coef[pos[hit]]
        self.idx, self.coef = new_idx, new_coef
        self.total_churn += churn
        self.n_refresh += 1
        self.n_reset_coeff += n_new
        return churn

    # ---------------- accounting (spec section 22) ----------------
    def persistent_bytes(self):
        """Bytes this state actually needs to keep between steps.

        DCT-Low stores no indices: the active set is the first K entries of a
        deterministic order, so K (one int) is the entire metadata. TopK must
        store the index of every retained coefficient, which is the honest
        reason a TopK coefficient costs 8 bytes where a Low coefficient costs 4.
        """
        k = self.idx.numel()
        idx_bytes = 0 if self.selection == 'low' else k * self.transform.index_bytes
        return k * self.transform.coeff_bytes + idx_bytes + 16   # +16: K and controller scalars

    def dense_bytes(self):
        return self.numel * 4

    @property
    def k(self):
        return self.idx.numel()

    @property
    def mean_churn(self):
        return self.total_churn / max(1, self.n_refresh)
