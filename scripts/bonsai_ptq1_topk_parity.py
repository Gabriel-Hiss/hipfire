#!/usr/bin/env python3
"""Top-20 parity of hipfire batched prefill against the PrismML llama.cpp fork.

Developer measurement script. The fork greedily continues one prompt of
benchmarks/prompts/bonsai_ptq1_prefill_parity.json for N tokens with
n_probs=20; hipfire then prefills prompt + continuation once through the
batched path (prefill_topk_dump) and reports its top-20 at each of the N
positions, so both engines score the same contexts. The fork's run is cached
as JSON, so later hipfire builds and formats need no fork server.

Needs, the first time per (prompt, N), a fork server on the same GGUF:

    llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -c 8192 -fa on --port 8091

and a hipfire build of the dumper:

    cargo build --release -p saddle-lab --features arch-qwen35,deltanet --example prefill_topk_dump

Usage: bonsai_ptq1_topk_parity.py <converted .ptq1 model> <cache dir> [--n 1024]
       [--prompt harness_mid_raw] [--tag name] [--server URL] [--dumper EXE]
Environment variables (e.g. HIPFIRE_PTQ1_ACT) pass through to the dumper.
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
K = 20

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("cache", type=pathlib.Path)
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--prompt", default="harness_mid_raw")
ap.add_argument("--tag", default="hipfire")
ap.add_argument("--server", default="http://127.0.0.1:8091")
ap.add_argument("--dumper", default=str(ROOT / "target/release/examples/prefill_topk_dump"))
args = ap.parse_args()
args.cache.mkdir(parents=True, exist_ok=True)
prompts = json.loads((ROOT / "benchmarks/prompts/bonsai_ptq1_prefill_parity.json").read_text(encoding="utf-8"))


def http(path, body):
    req = urllib.request.Request(args.server + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


fork_path = args.cache / f"fork_top{K}_{args.prompt}_{args.n}.json"
if fork_path.exists():
    fork = json.loads(fork_path.read_text())
else:
    ids = http("/tokenize", {"content": prompts[args.prompt], "parse_special": True})["tokens"]
    res = http("/completion", {"prompt": ids, "n_predict": args.n, "temperature": 0, "top_k": 1, "n_probs": K,
                               "cache_prompt": False, "samplers": ["top_k"], "return_tokens": True})
    fork = {"ids": ids, "gen": res["tokens"],
            "top": [{str(q["id"]): q["logprob"] for q in p["top_logprobs"]} for p in res["completion_probabilities"]]}
    fork_path.write_text(json.dumps(fork))

seq_path = args.cache / f"seq_{args.prompt}_{args.n}.txt"
seq_path.write_text(" ".join(map(str, fork["ids"] + fork["gen"][: args.n - 1])))
out = args.cache / f"hip_{args.tag}_{args.prompt}_{args.n}.csv"
r = subprocess.run([args.dumper, args.model, str(seq_path), str(len(fork["ids"])), str(K), str(out)],
                   capture_output=True, text=True, timeout=7200)
if r.returncode:
    raise SystemExit(r.stderr[-2000:])

gap, overlap, pos_worst, flips = [], [], [], []
same_order = argmax_ok = 0
for pos, row in enumerate(out.read_text().splitlines()[: args.n]):
    f = row.split(",")
    h = {f[1 + 2 * i]: float(f[2 + 2 * i]) for i in range(K)}
    fk = fork["top"][pos]
    h1, f1 = max(h.values()), max(fk.values())
    ho = sorted(h, key=h.get, reverse=True)
    fo = sorted(fk, key=fk.get, reverse=True)
    shared = set(h) & set(fk)
    overlap.append(len(shared))
    same_order += ho == fo
    g = [abs((h[t] - h1) - (fk[t] - f1)) for t in shared]
    gap += g
    pos_worst.append(max(g))
    if ho[0] == str(fork["gen"][pos]):
        argmax_ok += 1
    else:
        flips.append((pos, round(fk[fo[0]] - fk.get(fo[1], -99.0), 4), round(h[ho[0]] - h[ho[1]], 4)))
gap.sort()
pos_worst.sort()
n = len(overlap)
print(f"[{args.tag}] {args.prompt}: prompt {len(fork['ids'])} tokens, {n} positions")
print(f"  argmax == fork greedy token: {argmax_ok}/{n}; flips (pos, fork top1-top2 margin, hipfire margin): {flips}")
print(f"  top-{K} sets identical: {sum(o == K for o in overlap)}/{n}; order identical: {same_order}/{n}; "
      f"overlap mean {statistics.mean(overlap):.2f} worst {min(overlap)}")
print(f"  |logit-gap diff| over shared top-{K} (n={len(gap)}): mean {statistics.mean(gap):.4f} "
      f"median {gap[len(gap) // 2]:.4f} p99 {gap[int(0.99 * len(gap))]:.4f} worst {gap[-1]:.4f} best {gap[0]:.6f}")
print(f"  per-position worst gap: mean {statistics.mean(pos_worst):.4f} best {pos_worst[0]:.4f} worst {pos_worst[-1]:.4f}")
