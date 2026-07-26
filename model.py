"""
The tinyGPT model -- one definition, shared by every script.

A standard decoder-only transformer: token + position embeddings -> a stack of
(RMSNorm -> causal self-attention -> RMSNorm -> GELU MLP) blocks with residual
connections -> a final RMSNorm -> a tied linear head to vocab logits. No biases.
Attention runs through PyTorch's fused `scaled_dot_product_attention` (flash when
available), so memory is O(N.d), not O(N^2).

`train.py`, `train_sa.py`, `infer.py`, and `bench.py` all import `GPT` /
`build_model` from here, so there is exactly one model definition to keep
correct. Checkpoints are saved with `save_checkpoint()` (which embeds `n_head`,
the one hyper-parameter not recoverable from the weights) and reloaded with
`load_checkpoint()`, which rebuilds the architecture straight from the tensor
shapes -- so inference adapts to whatever size you trained.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_embd: int
    n_head: int
    block_size: int
    dropout: float = 0.0

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.n1, self.n2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.out  = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)   # residual dropout after attention
        self.mlp_drop  = nn.Dropout(cfg.dropout)   # residual dropout after MLP

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # dropout on the attention weights, only while training
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_drop(self.proj(a))
        x = x + self.mlp_drop(self.out(F.gelu(self.fc(self.n2(x)))))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte    = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe    = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop   = nn.Dropout(cfg.dropout)                     # embedding dropout
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.nf     = RMSNorm(cfg.n_embd)
        self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight                        # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # tied head shares wte.weight, so it is counted once
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.nf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=40, eot_id=None, stop=True):
        """Autoregressively extend `idx` (B=1) by up to `max_new` tokens.

        Context is truncated to the last `block_size` tokens each step. If `stop`
        and the model emits `eot_id`, generation halts there (the marker is kept
        in the returned ids; the tokenizer drops it on decode).
        """
        self.eval()
        for _ in range(max_new):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float() / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop and eot_id is not None and nxt.item() == eot_id:
                break
        return idx


def build_model(vocab_size, n_layer, n_embd, n_head, block_size, dropout=0.0):
    return GPT(GPTConfig(vocab_size, n_layer, n_embd, n_head, block_size, dropout))


def save_checkpoint(model, path, n_head):
    """Save a checkpoint that embeds `n_head` (the one hyper-parameter not
    recoverable from the raw weights). `load_checkpoint` reads it back."""
    torch.save({'model': model.state_dict(), 'n_head': n_head}, path)


def load_checkpoint(path, device, n_head_fallback, dropout=0.0):
    """Load a checkpoint and rebuild the model straight from the tensor shapes.

    Handles both the wrapped format from `save_checkpoint` (carries `n_head`)
    and a raw state-dict (falls back to `n_head_fallback`, e.g. `config.n_head`).
    Returns `(model, GPTConfig)`.
    """
    blob = torch.load(path, map_location=device, weights_only=True)
    if isinstance(blob, dict) and 'model' in blob and 'wte.weight' not in blob:
        sd = blob['model']
        n_head = int(blob.get('n_head', n_head_fallback))
    else:                                          # raw state-dict
        sd = blob
        n_head = n_head_fallback
    vocab_size, n_embd = sd['wte.weight'].shape
    block_size = sd['wpe.weight'].shape[0]
    n_layer = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
    cfg = GPTConfig(vocab_size, n_layer, n_embd, n_head, block_size, dropout)
    model = GPT(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg
