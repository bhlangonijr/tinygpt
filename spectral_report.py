#!/usr/bin/env python3
"""
Aggregate spectral runs into the experiment's result tables and plots.

    python spectral_report.py runs/                 # tables to stdout + report.md
    python spectral_report.py runs/ --plots         # also the section-32 figures

Reads every `<root>/*/summary.json` (plus its `telemetry.csv`) and emits the two
tables of spec section 38: the headline resource/quality table, and the
diagnostic table of reconstruction and update fidelity. Quality and throughput
deltas are computed against the DENSE run of the same optimizer, at the same
scale and seed, in the same root, so "1.2% worse" always means "worse than its
own control". One root is one sweep: it must hold exactly one such control.

Every table is stamped with the commit, tree state and GPU its runs came from,
and the report refuses a root that mixes them (`--allow-mixed` overrides, and
says so in the output): numbers from different implementations or hardware are
not comparable, and a table that silently combined them would be wrong.

The plot that matters most is the last one: `K_required/K_max` against update
cosine similarity. That is the direct test of whether spectral saturation is a
usable early-warning signal for quality degradation (hypothesis H4) -- if
saturation does not predict fidelity loss, the dynamic-K machinery is
measuring nothing.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

MIB = 1024 ** 2


def load_runs(root):
    runs = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, 'summary.json')
        if os.path.exists(path):
            with open(path) as f:
                blob = json.load(f)
            blob['_dir'] = os.path.join(root, name)
            runs.append(blob)
    return runs


def telemetry_stats(run_dir):
    """Mean of the numeric telemetry columns, split by state (m / v / momentum)."""
    path = os.path.join(run_dir, 'telemetry.csv')
    if not os.path.exists(path):
        return {}
    acc = defaultdict(lambda: defaultdict(list))
    pairs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            state = row.get('state', '')
            for k, v in row.items():
                if k in ('step', 'tensor', 'state') or v == '':
                    continue
                try:
                    acc[state][k].append(float(v))
                except ValueError:
                    pass
            if row.get('k_req_over_kmax') and row.get('update_cosine'):
                pairs.append((float(row['k_req_over_kmax']), float(row['update_cosine'])))
    out = {s: {k: sum(v) / len(v) for k, v in cols.items()} for s, cols in acc.items()}
    out['_pairs'] = pairs
    return out


def pearson(pairs):
    n = len(pairs)
    if n < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def method_label(cfg):
    if cfg['state'] == 'dense':
        return f"{cfg['opt'].upper()} dense"
    bits = f"{cfg['opt'].upper()} {cfg['state'].upper()}-{cfg['select']}"
    bits += f" K{cfg['kmax_frac']:g}"
    bits += " fixedK" if cfg.get('fixed_k') else f" tau{cfg['tau']:g} R{cfg['refresh']}"
    if cfg.get('m_only'):
        bits += " m-only"
    if cfg.get('v_only'):
        bits += " v-only"
    if cfg['opt'] == 'adamw':
        bits += f" floor{cfg.get('v_floor', 0.0):g}"
    if cfg.get('steps') != 3000:
        bits += f" {cfg.get('steps')}steps"
    return bits


def provenance_of(runs):
    """{(commit, dirty, gpu): [run names]} -- one key means one implementation on one GPU."""
    groups = defaultdict(list)
    for r in runs:
        c = r['config']
        groups[(c.get('commit'), c.get('dirty'), c.get('gpu'))].append(r['name'])
    return groups


def fmt(x, spec='.3f', pct=False):
    if x is None or x == '':
        return '-'
    return f"{x * 100:{spec}}%" if pct else f"{x:{spec}}"


def build_tables(runs):
    baselines = {}
    for r in runs:
        c = r['config']
        if c['state'] == 'dense':
            key = (c['opt'], c['scale'], c['seed'])
            if key in baselines:
                sys.exit(f"two dense controls for {key} in one root "
                         f"({baselines[key]['name']}, {r['name']}): one root is one sweep")
            baselines[key] = r

    main_rows, diag_rows = [], []
    for r in runs:
        c, q, mem = r['config'], r['quality'], r['memory']
        tel = telemetry_stats(r['_dir'])
        base = baselines.get((c['opt'], c['scale'], c['seed']))
        delta = ((q['final_val_loss'] - base['quality']['final_val_loss'])
                 / base['quality']['final_val_loss']) if base and base is not r else None
        tok_delta = (r['timing']['tokens_per_sec'] / base['timing']['tokens_per_sec'] - 1
                     if base and base is not r else None)
        beh = r.get('spectral_behaviour', {})
        main_rows.append({
            'run': r['name'],
            'tok_delta': tok_delta,
            'profiled': bool(c.get('profile')),
            'method': method_label(c),
            'state_mib': mem.get('persistent_state_bytes', 0) / MIB,
            'ratio': mem.get('state_ratio'),
            'peak_mib': (mem.get('peak_cuda_bytes') or 0) / MIB,
            'k_over_n': beh.get('k_over_n_mean'),
            'cap_hit': beh.get('cap_hit_mean'),
            'tok_s': r['timing']['tokens_per_sec'],
            'val': q['final_val_loss'],
            'ppl': q['final_ppl'],
            'bpc': q['final_bpc'],
            'delta': delta,
            'valid_mem': mem.get('valid_memory_benchmark', True),
            'nan': r.get('stability', {}).get('nan_count', 0),
        })
        if c['state'] == 'dense':
            continue
        m, v = tel.get('m', tel.get('momentum', {})), tel.get('v', {})
        diag_rows.append({
            'run': r['name'],
            'method': method_label(c),
            'm_err': m.get('recon_error'),
            'v_err': v.get('recon_error'),
            'cos': m.get('update_cosine'),
            'rel_norm': m.get('update_rel_norm'),
            'entropy': m.get('spectral_entropy'),
            'k_req_kmax': m.get('k_req_over_kmax'),
            'v_clamp': v.get('v_clamp_frac'),
            'neg_v': v.get('neg_v_energy'),
            'churn': m.get('churn'),
            'proxy': m.get('proxy_over_oracle'),
            'sat_corr': pearson(tel.get('_pairs', [])),
        })
    return main_rows, diag_rows


def render(main_rows, diag_rows, prov):
    out = []
    out.append("## Provenance\n")
    out.append("| Commit | Tree | GPU | Runs |")
    out.append("| --- | --- | --- | ---: |")
    for (commit, dirty, gpu), names in sorted(prov.items(), key=lambda kv: str(kv[0])):
        tree = {True: 'DIRTY', False: 'clean', None: 'unknown'}[dirty]
        out.append(f"| `{(commit or 'unrecorded')[:12]}` | {tree} | {gpu or 'unrecorded'} | {len(names)} |")
    if len(prov) > 1:
        out.append("\n**WARNING: this root mixes implementations or GPUs (`--allow-mixed`); "
                   "rows from different provenance are NOT comparable.**")
    out.append("")
    out.append("## Resources and quality (spec section 38)\n")
    out.append("| Run | Method | State MiB | vs dense | Peak MiB | K/N avg | Cap hit | tok/s | "
               "d tok/s | Val loss | PPL | BPC | dVal | NaN |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in main_rows:
        mark = '' if r['valid_mem'] else ' (i)'
        mark += ' (p)' if r['profiled'] else ''
        out.append(
            f"| {r['run']} | {r['method']}{mark} | {r['state_mib']:.2f} | {fmt(r['ratio'])} | "
            f"{r['peak_mib']:.0f} | {fmt(r['k_over_n'], '.2f', pct=True)} | "
            f"{fmt(r['cap_hit'], '.0f', pct=True)} | {r['tok_s']:.0f} | "
            f"{fmt(r['tok_delta'], '+.1f', pct=True)} | {r['val']:.4f} | "
            f"{r['ppl']:.2f} | {r['bpc']:.3f} | {fmt(r['delta'], '+.2f', pct=True)} | {r['nan']} |")
    out.append("\n`(i)` = instrumented run: shadow dense states are allocated for measurement, "
               "so its memory and throughput columns are NOT a valid benchmark. "
               "`(p)` = sampled phase timing on (synchronizes): throughput is not comparable. "
               "`K/N avg` and `Cap hit` are the final K and the cap-hit rate, meaned over "
               "spectral states.\n")

    out.append("\n## Fidelity diagnostics (spec section 38, second table)\n")
    out.append("| Run | m error | v error | Update cos | Rel norm | Entropy | K_req/K_max | "
               "v clamp | neg-v energy | Churn | Proxy/oracle | Saturation r |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in diag_rows:
        out.append(
            f"| {r['run']} | {fmt(r['m_err'], '.4f')} | {fmt(r['v_err'], '.4f')} | "
            f"{fmt(r['cos'], '.4f')} | {fmt(r['rel_norm'], '.3f')} | {fmt(r['entropy'], '.3f')} | "
            f"{fmt(r['k_req_kmax'], '.2f')} | {fmt(r['v_clamp'], '.1f', pct=True)} | "
            f"{fmt(r['neg_v'], '.4f')} | {fmt(r['churn'], '.3f')} | {fmt(r['proxy'], '.2f')} | "
            f"{fmt(r['sat_corr'], '.2f')} |")
    out.append("\n`Saturation r` is the correlation between `K_required/K_max` and update cosine "
               "over the run: strongly negative means saturation predicts fidelity loss (H4).\n")
    return "\n".join(out)


def make_plots(runs, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots", file=sys.stderr)
        return
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        h = r.get('history', {})
        if h.get('step'):
            tokens = [(s + 1) * r['config']['scale_dims']['batch_size']
                      * r['config']['scale_dims']['block_size'] for s in h['step']]
            ax.plot(tokens, h['val'], label=method_label(r['config']))
    ax.set_xlabel('training tokens'), ax.set_ylabel('validation loss')
    ax.legend(fontsize=7), ax.grid(alpha=.3)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, 'val_loss_vs_tokens.png'), dpi=140)
    plt.close(fig)

    # K(t) and update fidelity, straight from the telemetry stream
    for col, fname, ylab in (('k_cur_over_n', 'k_over_n_vs_step.png', 'K / N'),
                             ('k_cur_over_kmax', 'occupancy_vs_step.png', 'K / K_max'),
                             ('update_cosine', 'update_cosine_vs_step.png', 'update cosine'),
                             ('spectral_entropy', 'entropy_vs_step.png', 'spectral entropy'),
                             ('residual_energy_at_kmax', 'residual_vs_step.png', 'residual energy')):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        drew = False
        for r in runs:
            path = os.path.join(r['_dir'], 'telemetry.csv')
            if not os.path.exists(path):
                continue
            series = defaultdict(list)
            with open(path) as f:
                for row in csv.DictReader(f):
                    if row.get(col):
                        series[(row['state'])].append((int(row['step']), float(row[col])))
            for state, pts in series.items():
                agg = defaultdict(list)
                for s, v in pts:
                    agg[s].append(v)
                xs = sorted(agg)
                ax.plot(xs, [sum(agg[x]) / len(agg[x]) for x in xs],
                        label=f"{method_label(r['config'])} [{state}]")
                drew = True
        if drew:
            ax.set_xlabel('step'), ax.set_ylabel(ylab), ax.legend(fontsize=6), ax.grid(alpha=.3)
            fig.tight_layout(), fig.savefig(os.path.join(out_dir, fname), dpi=140)
        plt.close(fig)

    # the section-32 headline diagnostic
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in runs:
        pairs = telemetry_stats(r['_dir']).get('_pairs', [])
        if pairs:
            ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=8, alpha=.5,
                       label=method_label(r['config']))
    ax.axvline(1.0, color='k', lw=1, ls='--')
    ax.set_xlabel('K_required / K_max  (>1 = saturated)'), ax.set_ylabel('update cosine similarity')
    ax.legend(fontsize=7), ax.grid(alpha=.3)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, 'saturation_vs_fidelity.png'), dpi=140)
    plt.close(fig)
    print(f"plots -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="aggregate spectral experiment runs")
    ap.add_argument('root', nargs='?', default='runs')
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--out', default=None, help='markdown output (default <root>/report.md)')
    ap.add_argument('--allow-mixed', action='store_true',
                    help='report a root that mixes commits / dirty trees / GPUs (flagged in output)')
    a = ap.parse_args()

    runs = load_runs(a.root)
    if not runs:
        print(f"no summary.json found under {a.root}", file=sys.stderr)
        sys.exit(1)
    prov = provenance_of(runs)
    if not a.allow_mixed:
        if len(prov) > 1:
            sys.exit(f"{a.root} mixes provenance {sorted(map(str, prov))}; "
                     f"rerun on one commit/GPU or pass --allow-mixed")
        (commit, dirty, gpu), = prov
        if commit is None or dirty is not False:
            sys.exit(f"{a.root}: runs have no recorded commit or came from a dirty tree; "
                     f"pass --allow-mixed to report them anyway")
    md = render(*build_tables(runs), prov)
    print(md)
    out = a.out or os.path.join(a.root, 'report.md')
    with open(out, 'w') as f:
        f.write(md + "\n")
    print(f"\nwrote {out}")
    if a.plots:
        make_plots(runs, os.path.join(a.root, 'figures'))


if __name__ == '__main__':
    main()
