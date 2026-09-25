"""Build agentic coding conversations (OpenAI messages + tools) from real source files.

The assistant turns are produced later by the target model itself (self-distillation);
this script only builds the contexts. Output: JSONL, one {"id","messages","tools"} per line.
"""
import json, os, random, re, sys, pathlib, hashlib

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=1600)
ap.add_argument("--seed", type=int, default=20260926)
ap.add_argument("--root", action="append", required=True,
                help="source tree to draw files from (repeatable); every known code extension is used")
cli = ap.parse_args()
OUT = pathlib.Path(cli.out)
N = cli.n
rng = random.Random(cli.seed)
CODE_EXTS = (".rs", ".py", ".sh", ".hip", ".cpp", ".h", ".c", ".js", ".ts", ".swift", ".go", ".java", ".kt")
ROOTS = [(r, CODE_EXTS) for r in cli.root]
LANG = {".rs": "rust", ".py": "python", ".sh": "bash", ".hip": "cpp", ".cpp": "cpp", ".h": "cpp", ".c": "c",
        ".js": "javascript", ".ts": "typescript", ".swift": "swift", ".go": "go", ".java": "java", ".kt": "kotlin"}

files = []
for root, exts in ROOTS:
    for dp, dn, fn in os.walk(root):
        if any(x in dp for x in ("\\target", "/target", "node_modules", "\\.git", "/.git", "__pycache__", "build-hip", "vendor")):
            continue
        for f in fn:
            p = os.path.join(dp, f)
            if not f.endswith(exts):
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if 1200 <= sz <= 7000:
                files.append(p)
rng.shuffle(files)
print(f"{len(files)} candidate files", file=sys.stderr)


def read(p):
    try:
        t = pathlib.Path(p).read_text(encoding="utf-8")
    except Exception:
        return None
    if "\x00" in t or t.count("\n") < 25:
        return None
    return t


def rel(p):
    """Path relative to the --root it came from, as an agent would see it."""
    p = p.replace("\\", "/")
    for root, _ in ROOTS:
        r = root.replace("\\", "/").rstrip("/") + "/"
        if p.startswith(r):
            return p[len(r):]
    return os.path.basename(p)


IDENT = re.compile(r"\b(?:fn|def|function|func|class|struct|enum|impl|trait|const|let|static|void|int|float|pub fn)\s+([A-Za-z_][A-Za-z0-9_]{3,})")


def symbols(text):
    seen = []
    for m in IDENT.finditer(text):
        s = m.group(1)
        if s not in seen and s not in ("self", "main", "None", "True", "False"):
            seen.append(s)
    return seen


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


S = {"type": "string"}
TOOLSETS = [
    [fn("read_file", "Read a UTF-8 text file from the workspace.", {"path": S}, ["path"]),
     fn("write_file", "Create or overwrite a file with the complete new content.", {"path": S, "content": S}, ["path", "content"]),
     fn("edit_file", "Replace one exact occurrence of old_string with new_string in a file.", {"path": S, "old_string": S, "new_string": S}, ["path", "old_string", "new_string"]),
     fn("list_dir", "List the entries of a directory.", {"path": S}, ["path"]),
     fn("run_command", "Run a shell command in the workspace root and return stdout, stderr and the exit code.", {"command": S, "timeout_secs": {"type": "integer"}}, ["command"]),
     fn("search", "Search the workspace for a regular expression; returns path:line:text matches.", {"pattern": S, "path": S}, ["pattern"])],
    [fn("view", "Show the contents of a file with line numbers.", {"file_path": S}, ["file_path"]),
     fn("create", "Write a new file, replacing any existing one.", {"file_path": S, "file_text": S}, ["file_path", "file_text"]),
     fn("str_replace", "Replace old_str with new_str in file_path; old_str must be unique.", {"file_path": S, "old_str": S, "new_str": S}, ["file_path", "old_str", "new_str"]),
     fn("bash", "Execute a bash command.", {"command": S}, ["command"])],
    [fn("Read", "Reads a file from the local filesystem.", {"file_path": S, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["file_path"]),
     fn("Write", "Writes a file to the local filesystem, overwriting it.", {"file_path": S, "content": S}, ["file_path", "content"]),
     fn("Edit", "Performs exact string replacement in a file.", {"file_path": S, "old_string": S, "new_string": S, "replace_all": {"type": "boolean"}}, ["file_path", "old_string", "new_string"]),
     fn("Glob", "Fast file pattern matching.", {"pattern": S, "path": S}, ["pattern"]),
     fn("Grep", "Search file contents with a regex.", {"pattern": S, "path": S, "glob": S}, ["pattern"]),
     fn("Bash", "Executes a bash command.", {"command": S, "description": S}, ["command"])],
]
# (read tool, path arg, write tool, content arg, edit tool, old, new, run tool, cmd arg, list tool, list arg)
TOOLMAP = [
    ("read_file", "path", "write_file", "content", "edit_file", "old_string", "new_string", "run_command", "command", "list_dir", "path"),
    ("view", "file_path", "create", "file_text", "str_replace", "old_str", "new_str", "bash", "command", "bash", "command"),
    ("Read", "file_path", "Write", "content", "Edit", "old_string", "new_string", "Bash", "command", "Glob", "pattern"),
]
SYSTEMS = [
    "You are a coding agent working inside a git repository. Use the tools to inspect and change files. "
    "When you change a file, prefer write_file with the complete file for rewrites and edit_file for small, local fixes. "
    "Do not explain what you are going to do; call the tools.",
    "You are an autonomous software engineer. You can read, create and edit files and run shell commands through tools. "
    "Work step by step, one tool call at a time, and keep changes minimal and correct.",
    "You are a helpful coding assistant with access to the user's workspace. Use the available tools to read code "
    "before changing it. Answer briefly when no change is needed.",
]
TEST_CMD = {"rust": "cargo test", "python": "pytest -q", "javascript": "npm test", "typescript": "npm test", "go": "go test ./...",
            "cpp": "make test", "c": "make test", "bash": "bash -n", "swift": "swift test", "java": "gradle test", "kotlin": "gradle test"}

cid = 0


def call(name, args):
    global cid
    cid += 1
    return {"id": f"call_{cid}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def read_turn(tm, path, content):
    c = call(tm[0], {tm[1]: path})
    return [{"role": "assistant", "content": "", "tool_calls": [c]},
            {"role": "tool", "tool_call_id": c["id"], "content": content}]


def make(i):
    p = files[i % len(files)]
    text = read(p)
    if text is None:
        return None
    ext = os.path.splitext(p)[1]
    lang = LANG.get(ext, "text")
    path = rel(p)
    syms = symbols(text)
    k = rng.randrange(len(TOOLSETS))
    tools, tm = TOOLSETS[k], TOOLMAP[k]
    system = SYSTEMS[rng.randrange(len(SYSTEMS))]
    kind = rng.choices(
        ["rename", "add_fn", "docs", "edit_const", "tests", "explain", "multi_read", "after_write", "refactor", "bugfix"],
        weights=[14, 14, 10, 10, 10, 8, 8, 8, 10, 8])[0]
    hist = read_turn(tm, path, text)
    s1 = syms[0] if syms else "the main function"
    s2 = syms[1] if len(syms) > 1 else s1
    new = (s1 + "_v2") if rng.random() < 0.3 else re.sub(r"([a-z])([A-Z])", r"\1_\2", s1).lower() + "_impl"
    if kind == "rename":
        user = f"In {path}, rename `{s1}` to `{new}` everywhere and write the whole file back."
    elif kind == "add_fn":
        user = (f"Add a small helper next to `{s2}` in {path} that validates its inputs and returns early on invalid "
                f"ones, use it from `{s2}`, and write the complete file.")
    elif kind == "docs":
        user = f"Add concise doc comments to every public item in {path} that lacks one. Write the whole file back."
    elif kind == "edit_const":
        user = f"In {path}, make the smallest change needed to log a debug message at the start of `{s1}`. Use an edit, not a rewrite."
    elif kind == "tests":
        user = f"Write unit tests for `{s1}` and `{s2}` in {path} in a new {lang} test file next to it."
    elif kind == "explain":
        user = f"Explain what {path} does and point out anything that looks fragile. No changes."
    elif kind == "multi_read":
        others = [rel(files[(i + d) % len(files)]) for d in (1, 2, 3)]
        user = f"I need an overview before we refactor. Read {path}, then {', '.join(others)}."
        hist = []
    elif kind == "after_write":
        user = f"Rename `{s1}` to `{new}` in {path}, then run the tests."
        c = call(tm[2], {tm[1]: path, tm[3]: text.replace(s1, new)})
        hist = hist + [{"role": "assistant", "content": "", "tool_calls": [c]},
                       {"role": "tool", "tool_call_id": c["id"], "content": f"Wrote {len(text)} bytes to {path}."}]
    elif kind == "refactor":
        user = f"Refactor {path} to reduce duplication without changing behaviour, then write the whole file."
    else:
        user = (f"`{s1}` in {path} fails on empty input. Fix it so empty input is handled gracefully and write the complete file.")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}] + hist
    if kind == "after_write":
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}] + hist
    return {"id": f"{i:05d}-{kind}-{hashlib.md5(path.encode()).hexdigest()[:6]}", "kind": kind, "source": p,
            "messages": msgs, "tools": tools}


out = []
i = 0
while len(out) < N and i < len(files) * 2:
    c = make(i)
    i += 1
    if c:
        out.append(c)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8", newline="\n") as f:
    for c in out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
from collections import Counter
print(len(out), Counter(c["kind"] for c in out), file=sys.stderr)
