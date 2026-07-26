#!/usr/bin/env python3
"""
TinyStories benchmark + prompt-test suite for tinyGPT.

Loads a checkpoint (`microgpt_best.pt` by default), runs it over the official
TinyStories evaluation prompts (Eldan & Li's "Evaluation prompts.yaml",
auto-downloaded on first run), and reports:

  1. Language-modeling score -- perplexity and bits/char of the prompt text
     under the model. The principled, comparable number (same bits/char yardstick
     the trainer prints). Lower = the model predicts TinyStories text better.

  2. Generation-quality metrics -- the model continues each prompt and the
     continuation is scored with offline heuristics aimed at this model's known
     failure modes: repetition collapse (rep-n), low diversity (distinct-n),
     story runover (stopped_eot -- did it emit <|endoftext|> and stop?), and
     whether it ends on sentence punctuation.

  3. LLM judge (--judge) -- the official TinyStories eval grades completions with
     a strong LLM on grammar / creativity / consistency / plot. With --judge,
     each continuation is graded the same way by Claude via the `anthropic` SDK
     (pip install anthropic; set ANTHROPIC_API_KEY). The offline metrics in (2)
     are local proxies; this is the real, paper-comparable signal.

Usage:
    python bench.py                       # microgpt_best.pt, all prompts (offline metrics)
    python bench.py microgpt_sa_best.pt   # benchmark the SA model instead
    python bench.py --limit 8 --show 4    # quick look, print 4 completions
    python bench.py --judge               # also LLM-grade (needs ANTHROPIC_API_KEY)
    python bench.py --json scores.json    # also dump full per-prompt results

Head-to-head backprop vs. simulated annealing:
    python bench.py microgpt_best.pt      # backprop baseline
    python bench.py microgpt_sa_best.pt   # gradient-free model
"""

import os
import sys
import math
import json
import argparse
import urllib.request

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import TOK_FILE, EOT, n_head as CFG_N_HEAD, clean
from model import load_checkpoint

PROMPTS_URL  = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/Evaluation%20prompts.yaml"
PROMPTS_FILE = "eval_prompts.yaml"

# ---------------- CLI ----------------
ap = argparse.ArgumentParser(description="TinyStories benchmark for tinyGPT")
ap.add_argument("checkpoint", nargs="?", default="microgpt_best.pt", help="checkpoint .pt (default: microgpt_best.pt)")
ap.add_argument("--prompts", default=PROMPTS_FILE, help="eval prompts yaml")
ap.add_argument("--limit", type=int, default=0, help="only evaluate the first N prompts (0 = all)")
ap.add_argument("--max-new", type=int, default=200, help="tokens to generate per prompt")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top-k", type=int, default=40)
ap.add_argument("--show", type=int, default=3, help="print this many full sample completions")
ap.add_argument("--json", default="", help="write the full per-prompt results to this path")
ap.add_argument("--raw", action="store_true", help="feed prompts verbatim (skip the trainer's lowercase/charset clean)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--judge", action="store_true", help="LLM-grade completions on grammar/creativity/consistency/plot")
ap.add_argument("--judge-model", default="claude-opus-4-8", help="anthropic model id for grading")
ap.add_argument("--judge-workers", type=int, default=4, help="concurrent grading requests")
args = ap.parse_args()
torch.manual_seed(args.seed)

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
dev_type = 'cuda' if device == 'cuda' else 'cpu'
use_amp  = (device == 'cuda') and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32


# ---------------- prompts ----------------
def load_prompts(path):
    if not os.path.exists(path):
        print(f"prompts file {path!r} not found -- downloading...", file=sys.stderr)
        urllib.request.urlretrieve(PROMPTS_URL, path)
    raw = open(path, encoding='utf-8').read()
    try:
        import yaml
        prompts = yaml.safe_load(raw)
    except ModuleNotFoundError:                       # minimal fallback for the "- |-" block-scalar list
        prompts, cur, indent = [], None, None
        for line in raw.splitlines():
            if line.startswith('- '):
                if cur is not None:
                    prompts.append('\n'.join(cur).strip())
                cur, indent = [], None
                rest = line[2:].strip()
                if rest not in ('|-', '|', '>-', '>'):
                    cur.append(rest)
            elif cur is not None and line.strip() == '':
                cur.append('')
            elif cur is not None and (line.startswith('  ') or line.startswith('\t')):
                if indent is None:
                    indent = len(line) - len(line.lstrip())
                cur.append(line[indent:])
        if cur is not None:
            prompts.append('\n'.join(cur).strip())
    return [p.strip() for p in prompts if p and p.strip()]


# ---------------- metrics ----------------
def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

def rep_n(seq, n):                                    # fraction of n-grams that are repeats (0=none, 1=collapse)
    g = ngrams(seq, n)
    return 1.0 - len(set(g)) / len(g) if g else 0.0

def distinct_n(seq, n):                               # unique n-grams / total (higher = more diverse)
    g = ngrams(seq, n)
    return len(set(g)) / len(g) if g else 0.0

def max_run(seq):                                     # longest run of one repeated token
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if seq else 0

def ends_clean(text):
    return text.rstrip().endswith(('.', '!', '?', '"', '”', '’'))


@torch.no_grad()
def text_loss(ids):
    """Total NLL (nats) and token count for predicting `ids`, windowed at block_size."""
    total, ntok = 0.0, 0
    for s in range(0, len(ids) - 1, block_size):
        chunk = ids[s:s + block_size + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor([chunk[1:]], device=device)
        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(x)
        nll = F.cross_entropy(logits[0].float(), y[0], reduction='sum')
        total += nll.item(); ntok += y.numel()
    return total, ntok


@torch.no_grad()
def generate(ids, max_new, temperature, top_k):
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stopped = False
    for _ in range(max_new):
        ctx = idx[:, -block_size:]
        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(ctx)
        logits = logits[:, -1, :].float() / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        if eot_id is not None and nxt.item() == eot_id:   # model ended the story
            stopped = True
            break
    gen = idx[0, len(ids):].tolist()                  # generated continuation only
    if stopped and gen and gen[-1] == eot_id:         # drop the trailing EOT so it skips the metrics
        gen = gen[:-1]
    return gen, stopped


# ---------------- LLM judge (official-style GPT-Eval) ----------------
JUDGE_SYSTEM = (
    "You are a strict but fair elementary-school teacher grading a student's completion "
    "of a short story. The STORY BEGINNING was given to the student; they wrote the "
    "CONTINUATION. Grade ONLY the continuation, on four axes, each an integer from 1 "
    "(terrible) to 10 (excellent):\n"
    "  - grammar: grammatical and spelling correctness. The text is lowercased and uses "
    "limited punctuation BY CONSTRUCTION (the training corpus is normalized that way), so "
    "do NOT penalize missing capitalization or punctuation -- judge only genuine grammar errors.\n"
    "  - creativity: originality and imagination of the continuation.\n"
    "  - consistency: how logically and coherently it follows from the beginning (characters, "
    "setting, and stated facts stay consistent).\n"
    "  - plot: whether the continuation forms a sensible, coherent narrative.\n"
    "Also estimate age_group: the age the writing reads like (e.g. '3-4', '5-7', '8-10', "
    "'teen', 'adult'), and give a one-sentence justification. Calibrate: a fluent but bland "
    "continuation is roughly 8 grammar / 4 creativity; word-salad is 1-2 across the board."
)
JUDGE_SCHEMA = {                                      # structured output: no min/max, so clamp in code
    "type": "object",
    "properties": {
        "grammar":       {"type": "integer"},
        "creativity":    {"type": "integer"},
        "consistency":   {"type": "integer"},
        "plot":          {"type": "integer"},
        "age_group":     {"type": "string"},
        "justification": {"type": "string"},
    },
    "required": ["grammar", "creativity", "consistency", "plot", "age_group", "justification"],
    "additionalProperties": False,
}


def grade_one(client, model_id, prompt, completion):
    resp = client.messages.create(
        model=model_id, max_tokens=1024, system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"STORY BEGINNING (the prompt):\n{prompt}\n\n"
                   f"STUDENT'S CONTINUATION (grade this):\n{completion}"}],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    score = json.loads(next(b.text for b in resp.content if b.type == "text"))
    for k in ("grammar", "creativity", "consistency", "plot"):
        score[k] = max(1, min(10, int(score[k])))     # clamp defensively to 1..10
    return score


def judge_all(results, model_id, workers):
    import anthropic                                   # lazy: only --judge needs the SDK + key
    from concurrent.futures import ThreadPoolExecutor
    client = anthropic.Anthropic()                     # reads ANTHROPIC_API_KEY from the env

    def work(r):
        if not r['completion'].strip():
            return r['idx'], {"error": "empty completion"}
        try:
            return r['idx'], grade_one(client, model_id, r['prompt'], r['completion'])
        except Exception as e:
            return r['idx'], {"error": f"{type(e).__name__}: {e}"}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, score in ex.map(work, results):       # map preserves order; idx == position
            results[idx]['judge'] = score


# ---------------- load ----------------
if not os.path.exists(args.checkpoint):
    sys.exit(f"checkpoint not found: {args.checkpoint}")
tok = Tokenizer.from_file(TOK_FILE)
eot_id = tok.token_to_id(EOT)                  # None for pre-EOT tokenizers -> generation never stops early
encode = lambda s: tok.encode(s).ids
decode = lambda i: tok.decode(i)

model, cfg = load_checkpoint(args.checkpoint, device, n_head_fallback=CFG_N_HEAD)
block_size = cfg.block_size
nparams = model.num_params()

prompts = load_prompts(args.prompts)
if not args.raw:
    prompts = [clean(p) for p in prompts]
if args.limit:
    prompts = prompts[:args.limit]

print(f"checkpoint : {args.checkpoint}  ({nparams/1e6:.1f}M params)")
print(f"arch       : vocab {cfg.vocab_size} | n_embd {cfg.n_embd} | n_layer {cfg.n_layer} "
      f"| n_head {cfg.n_head} | block {cfg.block_size}")
print(f"device     : {device} (bf16 {use_amp}) | sampling: T={args.temperature} top_k={args.top_k} max_new={args.max_new}")
print(f"prompts    : {len(prompts)} from {args.prompts} ({'raw' if args.raw else 'cleaned to match training'})\n")

# ---------------- run ----------------
results = []
print(f"{'#':>3}  {'ppl':>7}  {'b/char':>6}  {'gen':>4}  {'rep4':>5}  {'dist2':>5}  {'run':>3}  {'eot':>3}  {'end':>3}")
print("-" * 58)
for i, prompt in enumerate(prompts):
    pids = encode(prompt)
    nats, ntok = text_loss(pids)
    ppl = math.exp(nats / ntok) if ntok else float('nan')
    bpc = nats / math.log(2) / len(prompt) if prompt else float('nan')

    gids, stopped = generate(pids, args.max_new, args.temperature, args.top_k)
    gtext = decode(gids)
    r2, r3, r4 = rep_n(gids, 2), rep_n(gids, 3), rep_n(gids, 4)
    d1, d2 = distinct_n(gids, 1), distinct_n(gids, 2)
    run = max_run(gids)
    ends = ends_clean(gtext)

    results.append(dict(idx=i, prompt=prompt, completion=gtext, ppl=ppl, bits_per_char=bpc,
                        gen_tokens=len(gids), rep2=r2, rep3=r3, rep4=r4,
                        distinct1=d1, distinct2=d2, max_run=run, stopped_eot=stopped, ends_clean=ends))
    print(f"{i:>3}  {ppl:>7.2f}  {bpc:>6.3f}  {len(gids):>4}  {r4:>5.2f}  {d2:>5.2f}  {run:>3}  "
          f"{'Y' if stopped else 'n':>3}  {'Y' if ends else 'n':>3}")

# ---------------- LLM grading (optional) ----------------
if args.judge:
    print(f"\njudging {len(results)} completions with {args.judge_model} "
          f"(workers={args.judge_workers})...")
    judge_all(results, args.judge_model, args.judge_workers)

# ---------------- aggregate ----------------
def mean(key):
    return sum(r[key] for r in results) / len(results)
clean_rate = sum(r['ends_clean'] for r in results) / len(results)
eot_rate   = sum(r['stopped_eot'] for r in results) / len(results)

# heuristic 0-100 generation-quality score: reward diversity + clean stops, punish repetition.
qual = 100 * (0.4 * mean('distinct2') + 0.4 * (1 - mean('rep4')) + 0.2 * clean_rate)

print("\n" + "=" * 52)
print("AGGREGATE SCORES")
print("=" * 52)
print(f"  Language modeling (lower is better)")
print(f"    perplexity        : {mean('ppl'):.2f}")
print(f"    bits/char         : {mean('bits_per_char'):.3f}")
print(f"  Generation quality")
print(f"    distinct-1 / -2   : {mean('distinct1'):.2f} / {mean('distinct2'):.2f}   (higher = more diverse)")
print(f"    repetition 2/3/4  : {mean('rep2'):.2f} / {mean('rep3'):.2f} / {mean('rep4'):.2f}   (lower = less looping)")
print(f"    mean longest run  : {mean('max_run'):.1f} tokens")
print(f"    stopped on EOT    : {eot_rate*100:.0f}% of completions   (emitted <|endoftext|> before max_new)")
print(f"    ends cleanly      : {clean_rate*100:.0f}% of completions")
print(f"    mean gen length   : {mean('gen_tokens'):.0f} tokens")
print(f"  Heuristic quality score (offline proxy): {qual:.1f} / 100")

# LLM-judge aggregate (the paper-comparable signal); only populated with --judge
graded = [r['judge'] for r in results if 'judge' in r and 'error' not in r['judge']]
if args.judge:
    def jmean(k):
        return sum(g[k] for g in graded) / len(graded) if graded else float('nan')
    nfail = len(results) - len(graded)
    print(f"  LLM judge ({args.judge_model}, 1-10, higher is better)")
    print(f"    grammar           : {jmean('grammar'):.2f}")
    print(f"    creativity        : {jmean('creativity'):.2f}")
    print(f"    consistency       : {jmean('consistency'):.2f}")
    print(f"    plot              : {jmean('plot'):.2f}")
    print(f"    graded            : {len(graded)}/{len(results)} completions"
          + (f"  ({nfail} failed -- see per-prompt 'judge.error')" if nfail else ""))
else:
    print(f"\n  (Run with --judge to add the paper's grammar/creativity/consistency/plot")
    print(f"   grades via {args.judge_model}; the metrics above are local proxies.)")

# ---------------- sample completions ----------------
if args.show:
    print("\n" + "=" * 52)
    print(f"SAMPLE COMPLETIONS (first {min(args.show, len(results))})")
    print("=" * 52)
    for r in results[:args.show]:
        print(f"\n[{r['idx']}] PROMPT: {r['prompt'][:160]}{'...' if len(r['prompt'])>160 else ''}")
        print(f"    GEN: {r['completion']}")
        j = r.get('judge')
        if j and 'error' not in j:
            print(f"    JUDGE: gram {j['grammar']} | creat {j['creativity']} | consist {j['consistency']} "
                  f"| plot {j['plot']} | age {j['age_group']} -- {j['justification']}")
        elif j:
            print(f"    JUDGE: failed ({j['error']})")

# ---------------- json ----------------
if args.json:
    jagg = None
    if args.judge:
        jagg = dict(model=args.judge_model, graded=len(graded),
                    grammar=jmean('grammar'), creativity=jmean('creativity'),
                    consistency=jmean('consistency'), plot=jmean('plot'))
    with open(args.json, 'w') as f:
        json.dump(dict(checkpoint=args.checkpoint, params=nparams,
                       arch=dict(vocab=cfg.vocab_size, n_embd=cfg.n_embd, n_layer=cfg.n_layer,
                                 n_head=cfg.n_head, block_size=cfg.block_size),
                       sampling=dict(temperature=args.temperature, top_k=args.top_k, max_new=args.max_new),
                       aggregate=dict(perplexity=mean('ppl'), bits_per_char=mean('bits_per_char'),
                                      distinct1=mean('distinct1'), distinct2=mean('distinct2'),
                                      rep2=mean('rep2'), rep3=mean('rep3'), rep4=mean('rep4'),
                                      stopped_eot_rate=eot_rate, ends_clean_rate=clean_rate,
                                      quality_score=qual, judge=jagg),
                       results=results), f, indent=2)
    print(f"\nwrote {args.json}")
