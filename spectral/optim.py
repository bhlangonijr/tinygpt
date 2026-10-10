"""
Spectral-state optimizers: SGD with momentum and AdamW.

The model weights stay dense. What changes is the PERSISTENT optimizer state:
instead of one fp32 value per parameter, each eligible weight matrix keeps `K`
coefficients of a fixed orthonormal basis. Because the transform is linear, the
state recursions carry over exactly (spec sections 13, 14):

    SGD    M_t = beta . M_{t-1} + T(g_t)
    AdamW  M_t = b1 . M_{t-1} + (1-b1) . T(g_t)
           V_t = b2 . V_{t-1} + (1-b2) . T(g_t^2)      <-- T(g^2), NOT T(g)^2

so with every coefficient retained these optimizers are ALGEBRAICALLY IDENTICAL
to their dense counterparts (`spectral_selftest.py` asserts exactly that). The
only approximation introduced anywhere is truncation to `K` coefficients --
which is what makes the experiment interpretable: any quality difference is
attributable to compression alone.

`T(g^2)` deserves its own note. `T(g)^2 != T(g^2)`, and using the former would
quietly redefine Adam in coefficient space rather than approximate real Adam's
state. It costs a second forward transform per step; that cost is the honest
price of the second moment and shows up in the timing block.

Two execution modes (spec section 21), and they follow the SAME optimization
path -- the mode must not change the trajectory, or instrumented runs would tell
us nothing about the runs we actually benchmark:

Either moment can be spectral independently (`m_only`, `v_only`), which matters
because they are not equally compressible -- see section 45.1 of the spec for the
measured asymmetry.

* `perf` -- nothing dense is kept for measurement. The capacity controller runs
  on a PROXY spectrum (retained coefficients, plus the current gradient's
  contribution wherever the active set does not reach). This is the only signal
  a deployed implementation could have.
* `instrumented` -- additionally maintains shadow dense states, NEVER used to
  update a parameter. They measure the oracle `K_required`, state
  reconstruction error, update fidelity, and how well the deployable proxy
  tracks the oracle. These runs are not valid memory benchmarks and say so in
  their summary.
"""

import time
from dataclasses import dataclass, asdict

import torch

from .metrics import (energy_profile, k_required, retained_energy, spectral_entropy,
                      state_error, update_fidelity, negative_v_stats)
from .state import KController, SpectralState
from .transforms import build_transform


@dataclass
class SpectralConfig:
    """Every knob of the compressed-state rule (spec sections 8-12, 21)."""

    state: str = 'dct'              # 'dense' (control) | 'dct'
    selection: str = 'topk'         # 'low' | 'topk'
    kmax_frac: float = 0.05         # K_max / N
    kmin_frac: float = 0.005        # K_min / N
    tau: float = 0.99               # target retained energy
    refresh: int = 10               # R: steps between active-set re-evaluations
    shrink_ratio: float = 0.75
    shrink_patience: int = 3        # H consecutive low refreshes before shrinking
    dynamic: bool = True            # False -> K pinned at K_max (fixed-K control)
    m_only: bool = False            # AdamW: spectral m, dense v (spec step 7)
    v_only: bool = False            # AdamW: spectral v, dense m (the mirror; see section 45.1)
    seed_new_from_grad: bool = True  # newly activated coefficients keep this step's gradient
    v_floor: float = 0.0            # floor for the reconstructed second moment; see note below
    mode: str = 'perf'              # 'perf' | 'instrumented'
    telemetry_interval: int = 200
    profile: bool = False           # sampled per-phase timing (synchronizes; costs throughput)
    profile_every: int = 50

    # `v_floor` note: spec section 15 says to clamp a negative reconstructed v at
    # ZERO and measure the damage, which is the default here. Be aware of what
    # that does numerically -- where v_hat lands at exactly 0 the AdamW
    # denominator collapses to `eps`, so the update becomes |m_hat| / 1e-8 and a
    # heavily truncated run can diverge on its first steps. That divergence is a
    # real property of the approximation, not a bug, so it is not hidden; if you
    # want to keep such a run alive long enough to study it, raise this floor
    # (e.g. 1e-12) and say so when reporting, because it changes the rule.

    def dict(self):
        return asdict(self)


class _SpectralBase(torch.optim.Optimizer):
    """Shared machinery: state creation, refresh, accounting, telemetry."""

    _states_per_param = 1

    def __init__(self, param_groups, defaults, cfg, telemetry=None):
        super().__init__(param_groups, defaults)
        self.cfg = cfg
        self.transform = build_transform('dct') if cfg.state != 'dense' else None
        self.tel = telemetry
        self.instrumented = (cfg.mode == 'instrumented')
        self._scratch = {}
        self._names = {}
        self.phase = {'transform': 0.0, 'inverse': 0.0, 'refresh': 0.0, 'step': 0.0}
        self.phase_samples = 0
        for g in self.param_groups:
            for p, n in zip(g['params'], g.get('names', [''] * len(g['params']))):
                self._names[id(p)] = n

    # ---------------- sampled phase timing (spec section 23) ----------------
    def _profiling(self, t):
        return self.cfg.profile and (t % self.cfg.profile_every == 0)

    @staticmethod
    def _now(device):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _add_phase(self, name, dt):
        self.phase[name] += dt

    # ---------------- setup ----------------
    def _scratch_for(self, p, count=1):
        """Shared dense buffer for reconstruction; `count` stacked copies when
        both Adam moments are synthesized in one batched inverse transform."""
        key = (tuple(p.shape), p.device, count)
        buf = self._scratch.get(key)
        if buf is None:
            buf = torch.zeros(count * p.numel(), device=p.device, dtype=p.dtype)
            self._scratch[key] = buf
        return buf.view(count, *p.shape) if count > 1 else buf.view_as(p)

    def _new_spectral(self, p):
        n, cfg = p.numel(), self.cfg
        k_max = max(1, min(n, int(round(cfg.kmax_frac * n))))
        k_min = k_max if not cfg.dynamic else max(1, min(k_max, int(round(cfg.kmin_frac * n))))
        ctrl = KController(k_min, k_max, cfg.shrink_ratio, cfg.shrink_patience)
        prefix = (self.transform.low_prefix(p.shape, k_max, p.device)
                  if cfg.selection == 'low' else None)
        return SpectralState(p.shape, self.transform, cfg.selection, ctrl, prefix,
                             p.device, dtype=p.dtype)

    def _is_spectral(self, p, group):
        return bool(group.get('spectral', False)) and self.cfg.state != 'dense' and p.dim() == 2

    def _init_common(self, p, st, group):
        st['t'] = 0
        st['spectral'] = self._is_spectral(p, group)
        st['name'] = self._names.get(id(p), '')

    # ---------------- active-set maintenance ----------------
    def _refresh(self, s, contrib, ref_contrib=None):
        """Size and re-select the active set. Returns (k_required_proxy, churn).

        Two different quantities are needed here, and conflating them
        under-provisions `K` badly.

        `contrib` is this step's actual contribution (e.g. `(1-b1).T(g)`), and it
        SEEDS newly activated coefficients -- that is genuinely what one step
        adds.

        `ref_contrib` is what the coefficient would grow to if it stayed active,
        and it is what the spectrum ESTIMATE must use. For an EMA
        `m <- b.m + (1-b).g` a persistent gradient converges to `g` itself, not
        to `(1-b).g`, so comparing retained coefficients (which hold accumulated
        EMAs) against a one-step contribution makes the out-of-set energy look
        10-20x smaller than it is. The controller then believes it needs fewer
        coefficients than it does and sits at roughly half its cap. Defaults to
        `contrib` when the two coincide.

        This proxy is the controller's input in BOTH modes -- the shadow states
        of instrumented mode never touch the update path, so an instrumented run
        and its perf twin follow the same trajectory.
        """
        ref = (contrib if ref_contrib is None else ref_contrib).abs()
        ref[s.idx] = s.coef.abs()
        order = self._full_order(s)
        cum, total = energy_profile(ref, order)
        k_req = k_required(None, self.cfg.tau, cum=cum) if total > 0 else s.ctrl.k_min
        k = s.ctrl.update(k_req) if self.cfg.dynamic else s.ctrl.k_max
        seed = contrib if self.cfg.seed_new_from_grad else None
        return k_req, s.reselect(ref, k, seed=seed)

    def _full_order(self, s):
        """The full selection order, built transiently: `K_required` under DCT-Low
        is measured along the whole low-frequency order, which is one int per
        coefficient and is therefore never kept between steps."""
        return self.transform.default_order(s.shape, s.coef.device) if s.selection == 'low' else None

    # ---------------- telemetry ----------------
    def _row(self, st, s, label, shadow=None, hat=None, fid=None, extras=None):
        """One section-31 record. Oracle columns are filled only when a shadow
        dense state exists (instrumented mode); performance mode emits the
        cheap columns so `K(t)` is still plottable from a benchmark run."""
        if self.tel is None:
            return
        n, kmax = s.numel, s.ctrl.k_max
        proxy = st.get(f'k_req_proxy_{label}')
        row = {
            'step': st['t'], 'tensor': st.get('name', ''), 'state': label, 'n': n,
            'k_current': s.k, 'k_max': kmax,
            'k_cur_over_n': s.k / n, 'k_cur_over_kmax': s.k / kmax,
            'churn': st.get(f'churn_{label}', ''), 'evicted_cum': s.n_evicted,
            'k_req_proxy': proxy if proxy is not None else '',
        }
        if shadow is not None:
            c = self.transform.analyze(shadow).reshape(-1)
            cum, _ = energy_profile(c, self._full_order(s))
            k_req = k_required(None, self.cfg.tau, cum=cum)
            ent, keff = spectral_entropy(c)
            row.update({
                'k_required': k_req, 'k_req_over_n': k_req / n, 'k_req_over_kmax': k_req / kmax,
                'retained_energy': retained_energy(cum, s.k),
                'residual_energy_at_kmax': 1.0 - retained_energy(cum, kmax),
                'excess_over_n': max(0, k_req - kmax) / n,
                'spectral_entropy': ent, 'k_eff_over_n': keff / n,
                'proxy_over_oracle': (proxy / k_req) if proxy and k_req else '',
            })
            if hat is not None:
                row['recon_error'] = state_error(shadow, hat)
        if fid is not None:
            row['update_cosine'], row['update_rel_norm'], row['update_rel_err'] = fid
        if extras:
            row.update(extras)
        self.tel.record(row)

    # ---------------- accounting (spec section 22) ----------------
    def memory_report(self):
        """Persistent optimizer-state bytes, spectral vs dense-equivalent."""
        spec_bytes = dense_equiv = fallback = 0
        n_spec = n_total = 0
        prefixes = {}                       # shared DCT-Low prefixes, counted once each
        for group in self.param_groups:
            for p in group['params']:
                st = self.state.get(p, {})
                if not st:
                    continue
                n_total += p.numel()
                dense_equiv += self._states_per_param * p.numel() * 4
                spectral = st.get('spectral')
                if spectral:
                    n_spec += p.numel()
                for key in ('m', 'v'):
                    s = st.get(key)
                    if isinstance(s, SpectralState):
                        spec_bytes += s.persistent_bytes()
                        if s.prefix is not None:
                            prefixes[s.prefix.data_ptr()] = s.prefix.numel() * s.prefix.element_size()
                    elif torch.is_tensor(s):
                        b = s.numel() * s.element_size()
                        spec_bytes += b if spectral else 0        # m_only's dense v
                        fallback += 0 if spectral else b
        prefix_bytes = sum(prefixes.values())
        spec_bytes += prefix_bytes
        scratch = sum(b.numel() * b.element_size() for b in self._scratch.values())
        total = spec_bytes + fallback
        return {
            'persistent_state_bytes': total,
            'spectral_state_bytes': spec_bytes,
            'low_prefix_bytes': prefix_bytes,
            'dense_fallback_state_bytes': fallback,
            'dense_equivalent_bytes': dense_equiv,
            'state_ratio': total / max(1, dense_equiv),
            'scratch_bytes': scratch,
            'spectral_coverage': n_spec / max(1, n_total),
            'instrumented_shadow_bytes': self._shadow_bytes(),
            'valid_memory_benchmark': not self.instrumented,
        }

    def _shadow_bytes(self):
        b = 0
        for group in self.param_groups:
            for p in group['params']:
                for key in ('m_shadow', 'v_shadow'):
                    s = self.state.get(p, {}).get(key)
                    if torch.is_tensor(s):
                        b += s.numel() * s.element_size()
        return b

    def spectral_summary(self):
        """Aggregate active-set behaviour over the whole run."""
        acc = {'k_over_n': [], 'occupancy': [], 'churn': [], 'cap_hit': []}
        activated = evicted = 0
        for group in self.param_groups:
            for p in group['params']:
                st = self.state.get(p, {})
                for key in ('m', 'v'):
                    s = st.get(key)
                    if not isinstance(s, SpectralState):
                        continue
                    acc['k_over_n'].append(s.k / s.numel)
                    acc['occupancy'].append(s.k / s.ctrl.k_max)
                    acc['churn'].append(s.mean_churn)
                    acc['cap_hit'].append(s.ctrl.cap_hit_rate)
                    activated += s.n_activated
                    evicted += s.n_evicted
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None
        out = {f'{k}_mean': mean(v) for k, v in acc.items()}
        out['activated_coefficients'] = activated
        out['evicted_coefficients'] = evicted      # the section-16 history loss
        out['phase_seconds_sampled'] = dict(self.phase)
        out['phase_sample_steps'] = self.phase_samples
        return out


class SpectralSGD(_SpectralBase):
    """SGD with momentum: `m_t = beta.m_{t-1} + g_t`, `p -= lr.m_t` (spec 4.1)."""

    _states_per_param = 1

    def __init__(self, param_groups, lr, momentum=0.9, weight_decay=0.0,
                 cfg=None, telemetry=None):
        super().__init__(param_groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay),
                         cfg or SpectralConfig(), telemetry)

    @torch.no_grad()
    def step(self, closure=None):
        cfg = self.cfg
        for group in self.param_groups:
            lr, beta, wd = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    self._init_common(p, st, group)
                    st['m'] = self._new_spectral(p) if st['spectral'] else torch.zeros_like(p)
                    if st['spectral'] and self.instrumented:
                        st['m_shadow'] = torch.zeros_like(p)
                st['t'] += 1
                t = st['t']
                g = p.grad.to(p.dtype)
                if wd:
                    p.mul_(1 - lr * wd)                        # decoupled, on the dense parameter
                if not st['spectral']:
                    st['m'].mul_(beta).add_(g)
                    p.add_(st['m'], alpha=-lr)
                    continue

                prof = self._profiling(t)
                m = st['m']
                t0 = self._now(p.device) if prof else 0.0
                cg = self.transform.analyze(g).reshape(-1)
                if prof:
                    t1 = self._now(p.device)
                    self._add_phase('transform', t1 - t0)
                m.accumulate(cg, beta, 1.0)
                if t == 1 or t % cfg.refresh == 0:
                    t1 = self._now(p.device) if prof else 0.0
                    # steady state of `m <- beta.m + g` is g/(1-beta)
                    st['k_req_proxy_momentum'], st['churn_momentum'] = self._refresh(
                        m, cg, cg / max(1e-6, 1.0 - beta))
                    if prof:
                        self._add_phase('refresh', self._now(p.device) - t1)
                t2 = self._now(p.device) if prof else 0.0
                m_hat = m.reconstruct(self._scratch_for(p))
                if prof:
                    self._add_phase('inverse', self._now(p.device) - t2)
                    self.phase_samples += 1
                p.add_(m_hat, alpha=-lr)

                if self.instrumented:
                    st['m_shadow'].mul_(beta).add_(g)
                if t % cfg.telemetry_interval == 0:
                    shadow = st.get('m_shadow')
                    fid = update_fidelity(shadow, m_hat) if shadow is not None else None
                    self._row(st, m, 'momentum', shadow=shadow, hat=m_hat, fid=fid)


class SpectralAdamW(_SpectralBase):
    """AdamW with spectral first/second moments (spec sections 4.2, 14, 15)."""

    _states_per_param = 2

    def __init__(self, param_groups, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 cfg=None, telemetry=None):
        super().__init__(param_groups, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
                         cfg or SpectralConfig(), telemetry)

    @torch.no_grad()
    def step(self, closure=None):
        cfg = self.cfg
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    self._init_adam(p, st, group)
                st['t'] += 1
                t = st['t']
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                g = p.grad.to(p.dtype)
                if wd:
                    p.mul_(1 - lr * wd)                        # decoupled weight decay
                if not st['spectral']:
                    st['m'].mul_(b1).add_(g, alpha=1 - b1)
                    st['v'].mul_(b2).addcmul_(g, g, value=1 - b2)
                    p.addcdiv_(st['m'] / bc1, (st['v'] / bc2).sqrt() + eps, value=-lr)
                    continue

                prof = self._profiling(t)
                m, v = st['m'], st['v']
                # Each moment is independently spectral or dense, so `m_only` and
                # `v_only` are the same code path with a different allocation --
                # and a dense moment costs no transform at all.
                m_spec, v_spec = isinstance(m, SpectralState), isinstance(v, SpectralState)
                both = m_spec and v_spec
                t0 = self._now(p.device) if prof else 0.0
                if both:                                        # one batched transform for g and g^2
                    cg, cg2 = self.transform.analyze(torch.stack([g, g * g])).reshape(2, -1).unbind(0)
                else:
                    cg = self.transform.analyze(g).reshape(-1) if m_spec else None
                    cg2 = self.transform.analyze(g * g).reshape(-1) if v_spec else None
                if prof:
                    self._add_phase('transform', self._now(p.device) - t0)

                if m_spec:
                    m.accumulate(cg, b1, 1 - b1)
                else:
                    m.mul_(b1).add_(g, alpha=1 - b1)
                if v_spec:
                    v.accumulate(cg2, b2, 1 - b2)               # T(g^2), never T(g)^2
                else:
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)

                if t == 1 or t % cfg.refresh == 0:
                    t1 = self._now(p.device) if prof else 0.0
                    if m_spec:
                        st['k_req_proxy_m'], st['churn_m'] = self._refresh(m, cg * (1 - b1), cg)
                    if v_spec:
                        st['k_req_proxy_v'], st['churn_v'] = self._refresh(v, cg2 * (1 - b2), cg2)
                    if prof:
                        self._add_phase('refresh', self._now(p.device) - t1)

                t2 = self._now(p.device) if prof else 0.0
                if both:                                        # one batched inverse for m and v
                    buf = self._scratch_for(p, 2)
                    m.scatter_into(buf[0].view(-1))
                    v.scatter_into(buf[1].view(-1))
                    m_hat, v_raw = self.transform.synthesize(buf).unbind(0)
                else:
                    buf = self._scratch_for(p)
                    m_hat = m.reconstruct(buf) if m_spec else m
                    v_raw = v.reconstruct(buf) if v_spec else None
                if prof:
                    self._add_phase('inverse', self._now(p.device) - t2)
                    self.phase_samples += 1
                if v_spec:
                    clamp_frac = float((v_raw < cfg.v_floor).float().mean()) if self.instrumented else 0.0
                    v_hat = v_raw.clamp_min(cfg.v_floor)        # spec section 15
                else:
                    v_hat, clamp_frac = v, 0.0
                p.addcdiv_(m_hat / bc1, (v_hat / bc2).sqrt() + eps, value=-lr)

                if self.instrumented:
                    st['m_shadow'].mul_(b1).add_(g, alpha=1 - b1)
                    st['v_shadow'].mul_(b2).addcmul_(g, g, value=1 - b2)
                if t % cfg.telemetry_interval == 0:
                    self._log(st, m_hat, v_hat, v_raw, clamp_frac, bc1, bc2, eps)

    def _init_adam(self, p, st, group):
        self._init_common(p, st, group)
        if st['spectral']:
            assert not (self.cfg.m_only and self.cfg.v_only), "m_only and v_only are exclusive"
            st['m'] = torch.zeros_like(p) if self.cfg.v_only else self._new_spectral(p)
            st['v'] = torch.zeros_like(p) if self.cfg.m_only else self._new_spectral(p)
            if self.instrumented:
                st['m_shadow'] = torch.zeros_like(p)
                st['v_shadow'] = torch.zeros_like(p)
        else:
            st['m'] = torch.zeros_like(p)
            st['v'] = torch.zeros_like(p)

    def _log(self, st, m_hat, v_hat, v_raw, clamp_frac, bc1, bc2, eps):
        md, vd = st.get('m_shadow'), st.get('v_shadow')
        fid = None
        if md is not None:
            u_dense = (md / bc1) / ((vd / bc2).sqrt() + eps)
            u_spec = (m_hat / bc1) / ((v_hat / bc2).sqrt() + eps)
            fid = update_fidelity(u_dense, u_spec)
        extras = None
        if v_raw is not None and self.instrumented:
            s = negative_v_stats(v_raw)
            extras = {'neg_v_frac': s['neg_frac'], 'neg_v_energy': s['neg_energy'],
                      'max_neg_v': s['max_neg'], 'v_clamp_frac': clamp_frac}
        for label, shadow, s, hat, ex in (('m', md, st['m'], m_hat, None),
                                          ('v', vd, st['v'], v_hat, extras)):
            if isinstance(s, SpectralState):
                self._row(st, s, label, shadow=shadow, hat=hat, fid=fid, extras=ex)
