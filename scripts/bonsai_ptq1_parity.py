#!/usr/bin/env python3
"""Greedy parity of hipfire against the PrismML llama.cpp fork on Ternary-Bonsai.

Developer measurement script. Needs a running fork server on the same GGUF:

    llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -c 4096 -fa on --port 8091

and a hipfire build of the top-5 dumper:

    cargo build --release -p saddle-lab --features arch-qwen35,deltanet --example greedy_dump_top5

For each prompt in benchmarks/prompts/bonsai_ptq1_parity.json it compares
prompt token counts (hipfire tokenizes the text itself), the greedy
continuations token by token, and the gap of every shared top-5 candidate to
the top-1 logit (hipfire raw logits vs fork log-probs; the gaps are equal when
the logits agree). A divergence is printed with both engines' top-5 so a
near-tie can be told apart from a real error.

Usage: bonsai_ptq1_parity.py <converted .ptq1 model> [n_tokens] [--server URL] [--dumper EXE]
"""
import argparse
import json
import os
import pathlib
import subprocess
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("n_tokens", nargs="?", type=int, default=256)
ap.add_argument("--server", default="http://127.0.0.1:8091")
ap.add_argument("--dumper", default=str(ROOT / "target/release/examples/greedy_dump_top5"))
ap.add_argument("--prompts", default=str(ROOT / "benchmarks/prompts/bonsai_ptq1_parity.json"))
args = ap.parse_args()


def http(path, body):
    req = urllib.request.Request(args.server + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1200).read())


def hipfire_greedy(text, n, out_prefix):
    env = dict(os.environ, PROMPT_MODE="raw")
    r = subprocess.run([args.dumper, args.model, out_prefix, "--max-gen", str(n - 1), text],
                       env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-2000:])
    prompt_tokens = next(int(l.split()[1]) for l in r.stderr.splitlines() if l.startswith("prompt:"))
    tokens = [int(t) for t in pathlib.Path(out_prefix + ".tokens").read_text().split()]
    rows = pathlib.Path(out_prefix + ".top5.csv").read_text().splitlines()[1:]
    top5 = []
    for row in rows:
        f = row.split(",")
        top5.append({int(f[1 + 2 * k]): float(f[2 + 2 * k]) for k in range(5)})
    return prompt_tokens, tokens, top5


def fork_greedy(ids, n):
    res = http("/completion", {"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "n_probs": 5,
                               "cache_prompt": False, "samplers": ["top_k"], "return_tokens": True})
    probs = [{q["id"]: q["logprob"] for q in p["top_logprobs"]} for p in res["completion_probabilities"]]
    return res["tokens"], probs


prompts = json.loads(pathlib.Path(args.prompts).read_text(encoding="utf-8"))
gap_err = []
identical = 0
with tempfile.TemporaryDirectory() as tmp:
    for name, text in prompts.items():
        ids = http("/tokenize", {"content": text, "parse_special": True})["tokens"]
        hp, ht, h5 = hipfire_greedy(text, args.n_tokens, os.path.join(tmp, name))
        ft, f5 = fork_greedy(ids, args.n_tokens)
        n = min(len(ht), len(ft))
        div = next((i for i, (a, b) in enumerate(zip(ht, ft)) if a != b), None)
        identical += div is None
        for pos in range(n if div is None else div + 1):
            h1, l1 = max(h5[pos].values()), max(f5[pos].values())
            gap_err += [abs((h5[pos][t] - h1) - (f5[pos][t] - l1)) for t in h5[pos].keys() & f5[pos].keys()]
        print(f"{name}: prompt tokens hipfire {hp} / fork {len(ids)}, compared {n}, first divergence {div}")
        if div is not None:
            print("   hipfire top-5", sorted(h5[div].items(), key=lambda kv: -kv[1]))
            print("   fork    top-5", sorted(f5[div].items(), key=lambda kv: -kv[1]))
gap_err.sort()
print(f"identical greedy sequences: {identical}/{len(prompts)}")
print(f"top-5 logit-gap |hipfire - fork|: n={len(gap_err)} mean={sum(gap_err) / len(gap_err):.4f} "
      f"p99={gap_err[int(0.99 * len(gap_err))]:.4f} max={gap_err[-1]:.4f}")
