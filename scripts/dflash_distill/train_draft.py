"""LoRA fine-tune of a DFlash draft on target-side distillation data.

Inputs
  --base DIR      HF DFlash draft (config.json + model.safetensors, bf16)
  --tables DIR    embed.f16 [vocab x dim], lmhead_t.f16 [dim x vocab] (dflash_distill_dump tables)
  --data DIR      index.jsonl + h_*.f16 (dflash_distill_dump hidden), rows in --tokens order
  --tokens FILE   HIPFIRE_SPEC_TOKEN_LOG jsonl ({"prompt","output"} per line)
Output
  --out DIR       merged bf16 model.safetensors + config.json (feed to dflash_convert)

The forward mirrors hipfire's draft_forward: fc -> hidden_norm over the context rows, per
layer q/k/v with per-head RMSNorm, half-split RoPE at absolute positions, attention of the
block queries over [context rows < anchor ; own block], sliding layers windowed (and causal
with --swa-causal 1), final norm, target lm_head. Blocks are [seed, mask x (B-1)] anchored
at response positions; loss is CE on positions 1..B-1 weighted exp(-(k-1)/gamma).
"""
import argparse, json, math, os, pathlib, random, time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--tables", required=True)
p.add_argument("--data", required=True)
p.add_argument("--tokens", required=True)
p.add_argument("--out", required=True)
p.add_argument("--rank", type=int, default=64)
p.add_argument("--alpha", type=float, default=128.0)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--epochs", type=float, default=2.0)
p.add_argument("--blocks", type=int, default=32, help="anchors per sequence per step")
p.add_argument("--gamma", type=float, default=7.0)
p.add_argument("--val-frac", type=float, default=0.05)
p.add_argument("--swa-causal", type=int, default=1)
p.add_argument("--eval-only", action="store_true")
p.add_argument("--max-ctx", type=int, default=6144)
p.add_argument("--seed", type=int, default=20260926)
args = p.parse_args()

dev = torch.device("cuda")
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
cfg = json.load(open(os.path.join(args.base, "config.json")))
H = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]; HD = cfg["head_dim"]
EPS = cfg["rms_norm_eps"]; THETA = cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 1e7))
DC = cfg["dflash_config"]; BLOCK = DC["block_size"]; MASK = DC["mask_token_id"]; NL_T = len(DC["target_layer_ids"])
LAYER_TYPES = cfg.get("layer_types", ["full_attention"] * cfg["num_hidden_layers"])
WINDOW = cfg.get("sliding_window", None)
NLAYER = cfg["num_hidden_layers"]
print(f"draft: {NLAYER} layers {LAYER_TYPES} W={WINDOW} block={BLOCK} mask={MASK} target_layers={NL_T}", flush=True)

# ---------------- tables ----------------
tj = json.load(open(os.path.join(args.tables, "tables.json")))
VOCAB = tj["vocab"]
embed = torch.from_numpy(np.fromfile(os.path.join(args.tables, "embed.f16"), dtype=np.float16).reshape(VOCAB, H))
lm_t = torch.from_numpy(np.fromfile(os.path.join(args.tables, "lmhead_t.f16"), dtype=np.float16).reshape(H, VOCAB))
lm_t = lm_t.to(dev, torch.bfloat16)
mask_emb = embed[MASK].to(dev, torch.bfloat16)

# ---------------- model ----------------
base = load_file(os.path.join(args.base, "model.safetensors"))


class LoRA(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        out_f, in_f = w.shape
        self.register_buffer("w", w.to(dev, torch.bfloat16), persistent=False)
        self.a = torch.nn.Parameter(torch.randn(args.rank, in_f, device=dev) * (1.0 / math.sqrt(in_f)))
        self.b = torch.nn.Parameter(torch.zeros(out_f, args.rank, device=dev))
        self.scale = args.alpha / args.rank

    def forward(self, x):
        y = F.linear(x, self.w)
        return y + F.linear(F.linear(x.float(), self.a), self.b).to(y.dtype) * self.scale

    def merged(self):
        return (self.w.float() + (self.b @ self.a) * self.scale).to(torch.bfloat16)


class Norm(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w.to(dev, torch.float32))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * self.w).to(torch.bfloat16)


class Layer(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        g = lambda n: base[f"layers.{i}.{n}"]
        self.inorm = Norm(g("input_layernorm.weight")); self.pnorm = Norm(g("post_attention_layernorm.weight"))
        self.q = LoRA(g("self_attn.q_proj.weight")); self.k = LoRA(g("self_attn.k_proj.weight"))
        self.v = LoRA(g("self_attn.v_proj.weight")); self.o = LoRA(g("self_attn.o_proj.weight"))
        self.qn = Norm(g("self_attn.q_norm.weight")); self.kn = Norm(g("self_attn.k_norm.weight"))
        self.gate = LoRA(g("mlp.gate_proj.weight")); self.up = LoRA(g("mlp.up_proj.weight")); self.down = LoRA(g("mlp.down_proj.weight"))
        self.sliding = LAYER_TYPES[i] == "sliding_attention"


class Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = LoRA(base["fc.weight"]); self.hnorm = Norm(base["hidden_norm.weight"]); self.norm = Norm(base["norm.weight"])
        self.layers = torch.nn.ModuleList([Layer(i) for i in range(NLAYER)])


model = Draft()
del base
inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))


def rope(x, pos):  # x [n, heads, hd], pos [n]
    ang = pos.float()[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    xf = x.float()
    x1, x2 = xf[..., : HD // 2], xf[..., HD // 2:]
    rot = torch.cat([-x2, x1], -1)
    return (xf * cos + rot * sin).to(torch.bfloat16)


def layer_fwd(L, state, ctx_proj, ctx_pos, q_pos, masks):
    h = L.inorm(state)
    nq, nc = h.shape[0], ctx_proj.shape[0]
    q = L.qn(L.q(h).view(nq, NH, HD))
    k = L.kn(torch.cat([L.k(ctx_proj), L.k(h)], 0).view(nc + nq, NKV, HD))
    v = torch.cat([L.v(ctx_proj), L.v(h)], 0).view(nc + nq, NKV, HD)
    q = rope(q, q_pos)
    k = rope(k, torch.cat([ctx_pos, q_pos]))
    rep = NH // NKV
    kk = k.repeat_interleave(rep, 1).transpose(0, 1)[None]
    vv = v.repeat_interleave(rep, 1).transpose(0, 1)[None]
    att = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kk, vv,
                                         attn_mask=masks[1 if L.sliding else 0])[0].transpose(0, 1)
    state = state + L.o(att.reshape(nq, NH * HD))
    h = L.pnorm(state)
    return state + L.down(F.silu(L.gate(h)) * L.up(h))


def build_masks(anchors, nc):
    """[nq, nc + nq] visibility for full and sliding layers."""
    nb = len(anchors)
    nq = nb * BLOCK
    a = torch.tensor(anchors, device=dev)
    qb = torch.arange(nq, device=dev) // BLOCK
    qi = torch.arange(nq, device=dev) % BLOCK
    qpos = a[qb] + qi
    cpos = torch.arange(nc, device=dev)
    ctx_vis = cpos[None, :] < a[qb][:, None]
    kb = torch.arange(nq, device=dev) // BLOCK
    ki = torch.arange(nq, device=dev) % BLOCK
    same = qb[:, None] == kb[None, :]
    full = torch.cat([ctx_vis, same], 1)
    if WINDOW is None:
        return full, full, qpos
    kpos = torch.cat([cpos, a[kb] + ki])
    win = (qpos[:, None] - kpos[None, :]) < WINDOW
    blk = same & ((ki[None, :] <= qi[:, None]) if args.swa_causal else torch.ones_like(same))
    slide = torch.cat([ctx_vis, blk], 1) & win
    return full, slide, qpos


def forward(hidden, tokens, anchors):
    """hidden [T, NL_T*H] f16 (cpu), tokens [T] long; returns final states [nb*BLOCK, H]."""
    nc = max(anchors)
    ctx = hidden[:nc].to(dev, torch.bfloat16, non_blocking=True)
    ctx_proj = model.hnorm(model.fc(ctx))
    ctx_pos = torch.arange(nc, device=dev)
    seeds = tokens[torch.tensor(anchors)]
    noise = mask_emb[None].repeat(len(anchors) * BLOCK, 1)
    noise[0::BLOCK] = embed[seeds].to(dev, torch.bfloat16)
    full, slide, qpos = build_masks(anchors, nc)
    state = noise
    for L in model.layers:
        if model.training:
            state = torch.utils.checkpoint.checkpoint(layer_fwd, L, state, ctx_proj, ctx_pos, qpos, (full, slide), use_reentrant=False)
        else:
            state = layer_fwd(L, state, ctx_proj, ctx_pos, qpos, (full, slide))
    return model.norm(state)


wk = torch.tensor([math.exp(-(k - 1) / args.gamma) for k in range(1, BLOCK)], device=dev)


def loss_fn(states, tokens, anchors):
    nb = len(anchors)
    s = states.view(nb, BLOCK, H)[:, 1:]  # predict positions 1..B-1
    tgt = torch.stack([tokens[a + 1: a + BLOCK] for a in anchors]).to(dev)
    logits = (s.reshape(-1, H) @ lm_t).float()
    ce = F.cross_entropy(logits, tgt.reshape(-1), reduction="none").view(nb, BLOCK - 1)
    return (ce * wk).sum() / (wk.sum() * nb)


@torch.no_grad()
def accept_len(states, tokens, anchors):
    nb = len(anchors)
    s = states.view(nb, BLOCK, H)[:, 1:]
    pred = (s.reshape(-1, H) @ lm_t).argmax(-1).view(nb, BLOCK - 1).cpu()
    tgt = torch.stack([tokens[a + 1: a + BLOCK] for a in anchors])
    ok = (pred == tgt).int().cumprod(1).sum(1)
    return ok.float().tolist()


# ---------------- data ----------------
tok_lines = open(args.tokens, encoding="utf-8").read().splitlines()
index = [json.loads(l) for l in open(os.path.join(args.data, "index.jsonl"), encoding="utf-8")]
samples = []
for r in index:
    t = json.loads(tok_lines[r["line"]])
    seq = (t["prompt"] + t["output"])[: r["len"]]
    if r["len"] - r["prompt_len"] < BLOCK + 1 or r["len"] > args.max_ctx:
        continue
    samples.append((r, torch.tensor(seq, dtype=torch.long)))
rng.shuffle(samples)
nval = max(8, int(len(samples) * args.val_frac))
val, train = samples[:nval], samples[nval:]
print(f"{len(train)} train / {len(val)} val sequences, {sum(r['len'] - r['prompt_len'] for r, _ in train)} train response tokens", flush=True)


def load_hidden(r):
    a = np.fromfile(os.path.join(args.data, r["file"]), dtype=np.float16).reshape(r["len"], NL_T * H)
    return torch.from_numpy(a)


def anchors_for(r, n, stride_rng):
    lo, hi = r["prompt_len"], r["len"] - BLOCK  # seed at a, labels a+1..a+B-1 must exist
    cand = list(range(lo, hi))
    if len(cand) <= n:
        return cand
    return sorted(stride_rng.sample(cand, n))


def evaluate():
    model.eval()
    lens = []
    erng = random.Random(1)
    for r, toks in val:
        h = load_hidden(r)
        a = anchors_for(r, 48, erng)
        lens += accept_len(forward(h, toks, a), toks, a)
    model.train()
    return sum(lens) / max(1, len(lens)), len(lens)


params = [p for p in model.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
acc0, n0 = evaluate()
print(f"val offline accept length before: {acc0:.3f} over {n0} blocks", flush=True)
if args.eval_only:
    raise SystemExit
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
steps = int(len(train) * args.epochs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
model.train()
t0 = time.time(); run = 0.0
for step in range(steps):
    r, toks = train[step % len(train)]
    if step % len(train) == 0 and step:
        rng.shuffle(train)
    h = load_hidden(r)
    a = anchors_for(r, args.blocks, rng)
    loss = loss_fn(forward(h, toks, a), toks, a)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
    if step % 50 == 0:
        print(f"step {step}/{steps} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
    if (step + 1) % 500 == 0 or step + 1 == steps:
        acc, n = evaluate()
        print(f"step {step + 1}: val offline accept length {acc:.3f} (before {acc0:.3f})", flush=True)

# ---------------- export merged ----------------
out = {}
out["fc.weight"] = model.fc.merged().cpu()
out["hidden_norm.weight"] = model.hnorm.w.detach().to(torch.bfloat16).cpu()
out["norm.weight"] = model.norm.w.detach().to(torch.bfloat16).cpu()
for i, L in enumerate(model.layers):
    pre = f"layers.{i}."
    for name, mod in [("self_attn.q_proj", L.q), ("self_attn.k_proj", L.k), ("self_attn.v_proj", L.v), ("self_attn.o_proj", L.o),
                      ("mlp.gate_proj", L.gate), ("mlp.up_proj", L.up), ("mlp.down_proj", L.down)]:
        out[pre + name + ".weight"] = mod.merged().cpu()
    for name, mod in [("input_layernorm", L.inorm), ("post_attention_layernorm", L.pnorm), ("self_attn.q_norm", L.qn), ("self_attn.k_norm", L.kn)]:
        out[pre + name + ".weight"] = mod.w.detach().to(torch.bfloat16).cpu()
os.makedirs(args.out, exist_ok=True)
save_file(out, os.path.join(args.out, "model.safetensors"))
json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
print("saved", args.out, flush=True)
