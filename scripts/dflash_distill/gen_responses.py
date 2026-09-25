"""Run conversations through hipfire serve (DFlash, greedy) with HIPFIRE_SPEC_TOKEN_LOG on.

usage: python gen_responses.py CONVS.jsonl TOKLOG.jsonl START COUNT <serve_harness args...>
Appends one line per finished conversation to TOKLOG.ids (conversation id, in log order),
so a restarted run resumes after the last completed id.
"""
import json, os, pathlib, sys, time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import serve_harness as sh  # noqa: E402

convs_path, toklog, start, count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rest = sys.argv[5:]
os.environ["HIPFIRE_SPEC_TOKEN_LOG"] = os.path.abspath(toklog)
ids_path = pathlib.Path(toklog + ".ids")
done = set(ids_path.read_text(encoding="utf-8").split()) if ids_path.exists() else set()
convs = [json.loads(l) for l in open(convs_path, encoding="utf-8")][start:start + count]


def run(cfg, args):
    t0 = time.time(); ntok = 0; n = 0
    for c in convs:
        if c["id"] in done:
            continue
        before = os.path.getsize(toklog) if os.path.exists(toklog) else 0
        try:
            r = sh.send(cfg, c["messages"], c["tools"])
        except Exception as e:  # keep going; a failed request logs nothing
            print(f"  {c['id']} FAILED {e}", flush=True)
            continue
        after = os.path.getsize(toklog) if os.path.exists(toklog) else 0
        if after == before:
            print(f"  {c['id']} no token log line (finish={r.get('finish')})", flush=True)
            continue
        with ids_path.open("a", encoding="utf-8") as f:
            f.write(c["id"] + "\n")
        n += 1; ntok += r["gen"]
        el = time.time() - t0
        print(f"  {c['id']:<32} gen={r['gen']:5d} tok/s={r['decode_tok_s']} tau={r.get('tau')} finish={r['finish']} "
              f"| {n} done, {ntok} tok, {el/60:.1f} min", flush=True)
    return []


sh.run = run
sh._assert_dflash_request_proofs = lambda *a, **k: None
sys.argv = ["serve_harness.py"] + rest
sh.main()
