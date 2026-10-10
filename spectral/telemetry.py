"""
Run outputs: the per-tensor telemetry stream and the run summary.

Two artifacts per run, both written under `runs/<name>/`:

* `telemetry.csv` -- one row per (step, tensor, state), the schema of spec
  section 31. This is what `K(t)` plots and the saturation-vs-fidelity
  diagnostic are built from, so the columns are fixed and never reordered.
* `summary.json` -- the run-level result: config, memory accounting (section
  22), timing (section 23), quality (section 24) and the aggregate spectral
  behaviour that fills the two tables in section 38.

Rows are buffered and flushed periodically: telemetry is written at a step
interval, not every step, and in performance mode only the cheap fields are
populated (the expensive ones need shadow dense state, which performance mode
does not allocate).
"""

import csv
import json
import os
import subprocess
import time

COLUMNS = [
    'step', 'tensor', 'state', 'n',
    'k_current', 'k_required', 'k_max',
    'k_cur_over_n', 'k_req_over_n', 'k_cur_over_kmax', 'k_req_over_kmax',
    'retained_energy', 'residual_energy_at_kmax', 'excess_over_n',
    'spectral_entropy', 'k_eff_over_n', 'churn', 'evicted_cum',
    'recon_error',
    'update_cosine', 'update_rel_norm', 'update_rel_err',
    'neg_v_frac', 'neg_v_energy', 'max_neg_v', 'v_clamp_frac',
    'k_req_proxy', 'proxy_over_oracle',
]


class Telemetry:
    """Buffered CSV writer for the section 31 schema."""

    def __init__(self, path, enabled=True, flush_every=200):
        self.enabled = enabled
        self.rows = []
        self.flush_every = flush_every
        self.path = path
        if enabled:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w', newline='') as f:
                csv.DictWriter(f, COLUMNS).writeheader()

    def record(self, row):
        if not self.enabled:
            return
        self.rows.append({k: row.get(k, '') for k in COLUMNS})
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.enabled or not self.rows:
            return
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, COLUMNS).writerows(self.rows)
        self.rows.clear()


class RunTimer:
    """Coarse phase timing (spec section 23).

    Deliberately sync-free on the hot path: a `torch.cuda.synchronize()` per
    phase per step would itself cost more than the transform being measured and
    would make tokens/sec meaningless. Phases are accumulated as CPU wall time
    and are only interpretable in aggregate; the throughput number that matters
    (tokens/sec, time-to-target-loss) is measured end-to-end.
    """

    def __init__(self):
        self.acc = {}
        self._t = {}

    def start(self, name):
        self._t[name] = time.perf_counter()

    def stop(self, name):
        t0 = self._t.pop(name, None)
        if t0 is not None:
            self.acc[name] = self.acc.get(name, 0.0) + (time.perf_counter() - t0)

    def as_dict(self):
        return dict(self.acc)


def write_summary(path, blob):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(blob, f, indent=2, sort_keys=False)


def provenance():
    """Which code produced a result: the git commit, and whether tracked files
    differed from it. Every summary carries this, so a number in a report can be
    traced to the exact implementation -- and results from different code are
    never mixed by accident (`spectral_report.py` refuses to)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git('status', '--porcelain', '--untracked-files=no')
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(status) if status is not None else None}
