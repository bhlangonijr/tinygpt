#!/usr/bin/env python3
"""
download_tinystories.py -- fetch the TinyStories corpus for tinyGPT.

TinyStories (Eldan & Li, 2023, "TinyStories: How Small Can Language Models Be
and Still Speak Coherent English?") is a synthetic dataset of very short, simple
stories written with the vocabulary of a 3-4 year old. The paper's point: shrink
the *data's* complexity to match a tiny model's *capacity* and models of only a
few-to-tens of millions of parameters start producing fluent, coherent English.
That is exactly tinyGPT's regime -- so this is the corpus to reproduce it on.

This grabs the raw .txt straight from the Hugging Face hub and streams it into a
plain corpus file. By default it takes the WHOLE file; set TARGET_CHARS to an int
to cap it for a quick test or to stay memory-safe on a small box. The raw
`<|endoftext|>` story separators are KEPT: the trainer registers `<|endoftext|>`
as a BPE special token, so the model learns a real story boundary and can stop
generating there.

    pip install huggingface_hub
    python download_tinystories.py          # -> tinystories.txt (the whole GPT-4 V2 set)

Then train:

    python train.py

The held-out validation file (TinyStoriesV2-GPT4-valid.txt) auto-downloads on the
first training run, so you only need this script for the training corpus.

Environment overrides (so you never edit this file):
    DATASET_FILE   which file in the dataset repo (default: TinyStoriesV2-GPT4-train.txt)
    OUT_FILE       output path (default: tinystories.txt, what config.py reads)
    TARGET_CHARS   cap the corpus to this many chars (default: unset = whole file)
"""

import os
import sys

# Files in the roneneldan/TinyStories dataset repo (switch DATASET_FILE to pick):
#   TinyStoriesV2-GPT4-train.txt   <- recommended: GPT-4 only, cleanest
#   TinyStoriesV2-GPT4-valid.txt
#   TinyStories-train.txt          <- V1 (GPT-3.5 + GPT-4 mix), smaller
#   TinyStories-valid.txt
REPO_ID      = os.environ.get("REPO_ID", "roneneldan/TinyStories")
DATASET_FILE = os.environ.get("DATASET_FILE", "TinyStoriesV2-GPT4-train.txt")
OUT_FILE     = os.environ.get("OUT_FILE", "tinystories.txt")
_tc          = os.environ.get("TARGET_CHARS")            # cap the corpus to this many chars
TARGET_CHARS = int(_tc) if _tc else None                 # unset/empty = take the WHOLE file
EOT          = "<|endoftext|>"                            # story separator in the raw file

PROGRESS_EVERY_STORIES = 50_000     # how often to print the running stories/chars count
CHARS_PER_TOKEN_EST    = 4          # ~4 chars/token for byte-BPE on simple text (closing estimate only)


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not found -- install it first: pip install huggingface_hub")

    print(f"downloading {DATASET_FILE} from {REPO_ID}  (cached + resumable)...")
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=DATASET_FILE, repo_type="dataset")
    except Exception as e:
        msg = str(e).lower()
        print(f"\ndownload failed: {e}\n")
        if any(s in msg for s in ("resolution", "resolve", "getaddr", "name or service")):
            print("Looks like a DNS / network failure reaching huggingface.co -- check your connection and retry.")
        elif "404" in msg or "not found" in msg:
            print(f"Couldn't find {DATASET_FILE!r}. Check the filename list at the top of this script.")
        sys.exit(1)

    cap = "no limit (whole file)" if TARGET_CHARS is None else f"cap {TARGET_CHARS:,} chars"
    print(f"got: {path}\nstreaming into {OUT_FILE}  ({cap})...")
    written = stories = 0
    with open(path, encoding="utf-8", errors="ignore") as fin, \
         open(OUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            if EOT in line:                       # story boundary -> KEEP the marker so the
                stories += line.count(EOT)        # model learns a real end-of-text token to stop on
            fout.write(line)
            written += len(line)
            if stories and stories % PROGRESS_EVERY_STORIES == 0:
                print(f"  {stories:,} stories | {written:,} chars", end="\r")
            if TARGET_CHARS is not None and written >= TARGET_CHARS:
                break

    approx_tokens = written // CHARS_PER_TOKEN_EST
    print(f"\ndone: {OUT_FILE} | ~{stories:,} stories | {written:,} chars "
          f"| ~{approx_tokens:,} tokens")
    print("\nnext:\n  python train.py")


if __name__ == "__main__":
    main()
