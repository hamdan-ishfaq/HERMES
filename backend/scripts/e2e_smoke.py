#!/usr/bin/env python3
"""
HERMES end-to-end test harness. Runs the REAL system, not mocks.

What it does
  1. Starts throwaway Postgres / Redis / Qdrant containers on separate ports
     (your real hermes data is never touched).
  2. Boots the real FastAPI server against them (real BGE-m3, reranker, LLM).
  3. Calls the real HTTP APIs and checks behaviour.
  4. Tears everything down and writes e2e-report.md / e2e-report.json.

Usage (from ~/HERMES-clean/backend, with the venv active via uv):
  uv run python scripts/e2e_smoke.py
  uv run python scripts/e2e_smoke.py --keep          # leave containers running
  uv run python scripts/e2e_smoke.py --skip-quality  # only API/security checks

Needs: docker (daemon running), httpx, and an LLM configured in backend/.env
(the server reads it). The first run downloads the models and can take minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parent.parent
REPORT_DIR = Path.home() / "audit-logs"
PW = "E2eTest!pass123"

PG_PORT, REDIS_PORT, QDRANT_PORT, API_PORT = 55432, 56379, 56333, 8765
CONTAINERS = {
    "hermes-e2e-pg": [
        "-p", f"{PG_PORT}:5432", "-e", "POSTGRES_USER=hermes",
        "-e", "POSTGRES_PASSWORD=hermes_pass", "-e", "POSTGRES_DB=hermes_e2e",
        "postgres:15",
    ],
    "hermes-e2e-redis": ["-p", f"{REDIS_PORT}:6379", "redis:7-alpine"],
    "hermes-e2e-qdrant": ["-p", f"{QDRANT_PORT}:6333", "qdrant/qdrant"],
}

# ---- invented facts: a model cannot know these, so correct answers prove retrieval
DOC_A = [
    "Quillfen Research Station - Operations Handbook",
    "The Zorblax-7 reactor was commissioned in 1987 at the Quillfen Research Station.",
    "Its coolant is a mixture of liquid helium and purified brine called Marnite.",
    "The station director is Dr. Ottoline Brandvik, who joined the station in 2004.",
    "Maintenance shutdowns occur every 14 months and last 9 days.",
    "The emergency vent valve is labelled V-312 and opens at 41 bar.",
    "All operators must complete the Level-3 safety course before entering Hall C.",
]
DOC_B = [
    "Project Heliotrope - Confidential Brief",
    "The Heliotrope budget is 8.2 million credits, approved by committee Yarrow.",
    "The Heliotrope launch window opens on 3 March 2031 from the Ostrava pad.",
    "The lead engineer for Heliotrope is Matteo Szabo.",
]
ANSWERABLE = [
    ("In what year was the Zorblax-7 reactor commissioned?", ["1987"]),
    ("What is the Zorblax-7 coolant called?", ["marnite"]),
    ("Who is the director of the Quillfen Research Station?", ["brandvik"]),
    ("How often do maintenance shutdowns occur at Quillfen?", ["14"]),
    ("What is the emergency vent valve labelled?", ["v-312", "v312"]),
]
UNANSWERABLE = [
    "What is the favourite colour of the station cat at Quillfen?",
    "How many employees work at the Quillfen Research Station?",
    "What was the total construction cost of the Zorblax-7 reactor?",
    "Which company supplies the Marnite coolant?",
    "What is the phone number of the Quillfen reception desk?",
]
ABSTAIN_MARKERS = [
    "no relevant information", "not found", "don't know", "do not know",
    "cannot find", "can't find", "not mentioned", "not specified",
    "no information", "does not", "doesn't", "unable to", "not provided",
    "not contain", "isn't", "is not",
]

RESULTS: list[dict] = []
METRICS: dict = {}


def rec(cat: str, name: str, ok: bool, detail: str = "", info: bool = False):
    status = "INFO" if info else ("PASS" if ok else "FAIL")
    RESULTS.append({"category": cat, "name": name, "status": status, "detail": detail})
    print(f"  [{status}] {cat}: {name}" + (f"  -- {detail}" if detail else ""))


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def make_pdf(lines: list[str]) -> bytes:
    content = ["BT", "/F1 11 Tf", "14 TL", "50 750 Td"]
    for ln in lines:
        esc = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content.append(f"({esc}) Tj T*")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", "replace")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return out


# ------------------------------------------------------------------ infra
def start_infra():
    for name in CONTAINERS:
        sh(["docker", "rm", "-f", "-v", name])
    for name, args in CONTAINERS.items():
        r = sh(["docker", "run", "-d", "--name", name, *args])
        if r.returncode != 0:
            raise SystemExit(f"Could not start {name}: {r.stderr.strip()}")
    print("Waiting for containers ...")
    deadline = time.time() + 120
    while time.time() < deadline:
        pg = sh(["docker", "exec", "hermes-e2e-pg", "pg_isready", "-U", "hermes"])
        rd = sh(["docker", "exec", "hermes-e2e-redis", "redis-cli", "ping"])
        try:
            qd = httpx.get(f"http://localhost:{QDRANT_PORT}/healthz", timeout=2).status_code == 200
        except Exception:
            qd = False
        if pg.returncode == 0 and "PONG" in rd.stdout and qd:
            time.sleep(2)  # postgres restarts once during init
            return
        time.sleep(2)
    raise SystemExit("Containers did not become ready in 120 s")


def stop_infra():
    for name in CONTAINERS:
        sh(["docker", "rm", "-f", "-v", name])


def server_env() -> dict:
    env = dict(os.environ)
    env.update({
        "DATABASE_URL": f"postgresql+asyncpg://hermes:hermes_pass@localhost:{PG_PORT}/hermes_e2e",
        "REDIS_URL": f"redis://localhost:{REDIS_PORT}",
        "QDRANT_URL": f"http://localhost:{QDRANT_PORT}",
        "ENV": "development",
        "SECRET_KEY": secrets.token_urlsafe(48),
        "EVAL_ADMIN_EMAILS": "",
    })
    env.pop("HERMES_MCP_USER_ID", None)
    return env


def start_server(env: dict) -> subprocess.Popen:
    # Create tables if the app does not do it on startup (harmless if it does).
    sh([sys.executable, "-c",
        "import asyncio; from src import db; asyncio.run(db.create_tables())"],
       cwd=BACKEND, env=env)
    log = open(REPORT_DIR / "e2e-server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app", "--port", str(API_PORT)],
        cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("Server exited early; see ~/audit-logs/e2e-server.log")
        try:
            if httpx.get(f"http://localhost:{API_PORT}/health", timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    raise SystemExit("Server did not become healthy; see ~/audit-logs/e2e-server.log")


# ------------------------------------------------------------------ API discovery
class Api:
    def __init__(self, client: httpx.Client):
        self.c = client
        spec = client.get("/openapi.json").json()
        self.spec = spec
        self.paths = spec.get("paths", {})

    def _ref(self, obj):
        while isinstance(obj, dict) and "$ref" in obj:
            node = self.spec
            for part in obj["$ref"].lstrip("#/").split("/"):
                node = node[part]
            obj = node
        return obj

    def body_props(self, path: str, method="post"):
        op = self.paths.get(path, {}).get(method, {})
        content = op.get("requestBody", {}).get("content", {})
        for ctype, val in content.items():
            schema = self._ref(val.get("schema", {}))
            return ctype, {k: self._ref(v) for k, v in schema.get("properties", {}).items()}
        return None, {}

    def find(self, suffix: str):
        for p in self.paths:
            if p.rstrip("/").endswith(suffix):
                return p
        return None


def pick(props: dict, candidates: list[str], default: str) -> str:
    for c in candidates:
        if c in props:
            return c
    return default


# ------------------------------------------------------------------ user helpers
class User:
    def __init__(self, api: Api, label: str):
        self.api, self.label = api, label
        self.email = f"{label}-{uuid.uuid4().hex[:8]}@e2e.example.com"
        self.token = ""
        self.calls = 0

    def register(self):
        p = self.api.find("/auth/register")
        return self.api.c.post(p, json={"email": self.email, "password": PW})

    def login(self, password=PW):
        p = self.api.find("/auth/login")
        ctype, props = self.api.body_props(p)
        if ctype and "x-www-form-urlencoded" in ctype:
            r = self.api.c.post(p, data={"username": self.email, "password": password})
        else:
            r = self.api.c.post(p, json={"email": self.email, "password": password})
        if r.status_code == 200:
            self.token = r.json().get("access_token", "")
        return r

    @property
    def h(self):
        return {"Authorization": f"Bearer {self.token}"}

    def ingest_pdf(self, lines: list[str]):
        p = self.api.find("/ingest/pdf")
        ctype, props = self.api.body_props(p)
        field = next((k for k, v in props.items() if v.get("format") == "binary"), "file")
        return self.api.c.post(p, headers=self.h,
                               files={field: ("doc.pdf", make_pdf(lines), "application/pdf")})

    def ingest_url(self, url: str):
        p = self.api.find("/ingest/url")
        _, props = self.api.body_props(p)
        return self.api.c.post(p, headers=self.h, json={pick(props, ["url"], "url"): url})

    def ask(self, q: str):
        p = self.api.find("/research")
        _, props = self.api.body_props(p)
        field = pick(props, ["query", "question", "q", "message"], "query")
        t0 = time.time()
        r = self.api.c.post(p, headers=self.h, json={field: q})
        dt = time.time() - t0
        self.calls += 1
        j = r.json() if r.status_code == 200 else {}
        ans = (j.get("answer") or j.get("final_answer") or "") if isinstance(j, dict) else ""
        return r, j, ans, dt

    def stream(self, q: str):
        p = self.api.find("/research/stream")
        _, props = self.api.body_props(p)
        field = pick(props, ["query", "question", "q", "message"], "query")
        text, n_events, first = "", 0, None
        t0 = time.time()
        with self.api.c.stream("POST", p, headers=self.h, json={field: q}) as r:
            if r.status_code != 200:
                return r.status_code, "", 0, None
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                n_events += 1
                if first is None:
                    first = time.time() - t0
                data = line[5:].strip()
                try:
                    obj = json.loads(data)
                    if isinstance(obj, dict):
                        for k in ("token", "text", "content", "delta", "answer"):
                            if isinstance(obj.get(k), str):
                                text += obj[k]
                                break
                    elif isinstance(obj, str):
                        text += obj
                except Exception:
                    text += data
        self.calls += 1
        return 200, text, n_events, first


_PUNCT_MAP = {
    "‑": "-",  # non-breaking hyphen
    "‐": "-",  # hyphen
    "―": "-",  # horizontal bar
    "–": "-",  # en dash
    "—": "-",  # em dash
    " ": " ",  # nbsp
}


def normalize(text: str) -> str:
    """Lowercase and fold Unicode punctuation that models emit instead of ASCII."""
    t = text.lower()
    for src, dst in _PUNCT_MAP.items():
        t = t.replace(src, dst)
    return t


def has_any(text: str, needles: list[str]) -> bool:
    t = normalize(text)
    return any(normalize(n) in t for n in needles)


# ------------------------------------------------------------------ test groups
def test_auth(api: Api, a: User):
    cat = "auth"
    r = a.register()
    rec(cat, "register new user", r.status_code in (200, 201), f"HTTP {r.status_code}")
    r = a.login("wrong-password")
    rec(cat, "wrong password rejected", r.status_code in (400, 401, 403), f"HTTP {r.status_code}")
    r = a.login()
    rec(cat, "login returns token", r.status_code == 200 and bool(a.token), f"HTTP {r.status_code}")
    p = api.find("/research")
    r = api.c.post(p, json={"query": "hello"})
    rec(cat, "research without token is rejected", r.status_code in (401, 403), f"HTTP {r.status_code}")
    weak = User(api, "weak")
    weak.email = f"weak-{uuid.uuid4().hex[:6]}@e2e.example.com"
    rp = api.find("/auth/register")
    r = api.c.post(rp, json={"email": weak.email, "password": "a"})
    rec(cat, "one-character password accepted (no password policy)",
        r.status_code not in (200, 201), f"HTTP {r.status_code}", info=True)


def test_ssrf(api: Api, a: User):
    cat = "ssrf"
    probes = [
        f"http://127.0.0.1:{QDRANT_PORT}/", "http://localhost/",
        "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
        "http://[::ffff:127.0.0.1]/", "file:///etc/passwd", "ftp://example.com/x",
    ]
    c = api.c
    old = c.timeout
    c.timeout = httpx.Timeout(25.0)
    try:
        for url in probes:
            try:
                r = a.ingest_url(url)
                rec(cat, f"blocked {url}", r.status_code in (400, 403, 422), f"HTTP {r.status_code}")
            except httpx.TimeoutException:
                rec(cat, f"blocked {url}", False, "no rejection (request hung/timeout)")
    finally:
        c.timeout = old


def test_ingest_and_quality(api: Api, a: User, b: User, args):
    cat = "ingest"
    ra, rb = a.ingest_pdf(DOC_A), b.ingest_pdf(DOC_B)
    rec(cat, "user A ingests PDF", ra.status_code in (200, 201, 202), f"HTTP {ra.status_code} {ra.text[:120]}")
    rec(cat, "user B ingests PDF", rb.status_code in (200, 201, 202), f"HTTP {rb.status_code} {rb.text[:120]}")
    if ra.status_code not in (200, 201, 202):
        return

    q0, must0 = ANSWERABLE[0]
    deadline, ready = time.time() + args.ingest_wait, False
    while time.time() < deadline:
        r, j, ans, _ = a.ask(q0)
        if r.status_code == 200 and has_any(ans, must0):
            ready = True
            break
        time.sleep(5)
    rec(cat, "ingested content becomes retrievable", ready, f"waited up to {args.ingest_wait}s")
    if not ready:
        return

    cat = "quality"
    hits, lat = 0, []
    for q, must in ANSWERABLE:
        r, j, ans, dt = a.ask(q)
        ok = r.status_code == 200 and has_any(ans, must)
        cites = (j.get("citations") or []) if isinstance(j, dict) else []
        hits += ok
        if not (j.get("cache_hit") if isinstance(j, dict) else False):
            lat.append(dt)
        rec(cat, f"answers: {q[:48]}", ok, f"citations={len(cites)} ans={ans[:70]!r}", info=False)
    METRICS["answerable_hit_rate"] = round(hits / len(ANSWERABLE), 2)
    rec(cat, f"answerable hit rate >= {args.min_hit}", hits / len(ANSWERABLE) >= args.min_hit,
        f"{hits}/{len(ANSWERABLE)}")

    abst = 0
    for q in UNANSWERABLE:
        r, j, ans, dt = a.ask(q)
        abst += has_any(ans, ABSTAIN_MARKERS)
        if r.status_code == 200 and not (j.get("cache_hit") if isinstance(j, dict) else False):
            lat.append(dt)
    METRICS["abstention_rate"] = round(abst / len(UNANSWERABLE), 2)
    rec(cat, f"abstains on unanswerable questions >= {args.min_abstain}",
        abst / len(UNANSWERABLE) >= args.min_abstain, f"{abst}/{len(UNANSWERABLE)} (keyword heuristic)")
    if len(lat) >= 3:
        lat.sort()
        METRICS["latency_p50_s"] = round(statistics.median(lat), 2)
        METRICS["latency_p95_s"] = round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2)
        rec(cat, "end-to-end latency", True,
            f"p50={METRICS['latency_p50_s']}s p95={METRICS['latency_p95_s']}s n={len(lat)}", info=True)

    cat = "cache"
    q1 = ANSWERABLE[1][0]
    a.ask(q1)
    r, j, ans, dt2 = a.ask(q1)
    hit = bool(j.get("cache_hit")) if isinstance(j, dict) else False
    rec(cat, "identical question served from cache", hit, f"second call {dt2:.2f}s")
    r, j, ans, _ = a.ask("What is the name of the Zorblax-7 coolant mixture?")
    rec(cat, "paraphrase hits cache (informational)",
        bool(j.get("cache_hit")) if isinstance(j, dict) else False, info=True)

    cat = "stream"
    code, text, n, first = a.stream(ANSWERABLE[0][0])
    rec(cat, "SSE stream returns tokens", code == 200 and n > 0, f"HTTP {code}, {n} events")
    rec(cat, "streamed answer contains the fact", has_any(text, ANSWERABLE[0][1]),
        f"first event after {first:.2f}s" if first is not None else "")

    cat = "isolation"
    r, j, ans, _ = b.ask(ANSWERABLE[0][0])
    leaked = has_any(ans, ["1987", "zorblax", "quillfen"]) and "no relevant" not in ans.lower()
    cache_leak = bool(j.get("cache_hit")) if isinstance(j, dict) else False
    rec(cat, "user B cannot retrieve user A's document", not leaked, ans[:100])
    rec(cat, "user B does not receive user A's cached answer", not cache_leak)
    r, j, ans, _ = a.ask("What is the Heliotrope budget?")
    rec(cat, "user A cannot retrieve user B's document",
        not has_any(ans, ["8.2 million", "yarrow", "szabo"]), ans[:100])
    code, text, _, _ = b.stream("In what year was the Zorblax-7 reactor commissioned?")
    rec(cat, "stream endpoint also isolated", not has_any(text, ["1987"]), f"HTTP {code}")


def test_eval_endpoints(api: Api, a: User, b: User):
    cat = "eval"
    p = api.find("/eval/run")
    r = api.c.post(p, headers=a.h)
    rec(cat, "ordinary user cannot start an eval run", r.status_code in (401, 403), f"HTTP {r.status_code}")
    p = api.find("/eval/dashboard")
    r = api.c.get(p, headers=b.h)
    if r.status_code == 200:
        total = r.json().get("total_queries")
        rec(cat, "dashboard counts only the caller's queries",
            isinstance(total, int) and total <= b.calls, f"total_queries={total}, B made {b.calls} calls")
    else:
        rec(cat, "dashboard reachable", False, f"HTTP {r.status_code}")


def test_mcp(env: dict):
    cat = "mcp"
    code = ("from src.mcp.server import hermes_search; print(hermes_search('x'))")
    try:
        r = sh([sys.executable, "-c", code], cwd=BACKEND, env=env, timeout=240)
        out = (r.stdout or "") + (r.stderr or "")
        rec(cat, "hermes_search refuses to run without HERMES_MCP_USER_ID",
            '"error"' in out or "error" in out.lower(), out.strip()[-120:])
    except Exception as e:
        rec(cat, "hermes_search without user id", False, f"could not run: {e}", info=True)


# ------------------------------------------------------------------ report
def write_report(args):
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "e2e-report.json").write_text(
        json.dumps({"results": RESULTS, "metrics": METRICS}, indent=2))
    n = {s: sum(1 for r in RESULTS if r["status"] == s) for s in ("PASS", "FAIL", "INFO")}
    lines = [f"# HERMES end-to-end report ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"**{n['PASS']} passed, {n['FAIL']} failed, {n['INFO']} informational**", "",
             "## Metrics", ""]
    lines += [f"- {k}: {v}" for k, v in METRICS.items()] or ["- (none collected)"]
    lines += ["", "| Category | Check | Result | Detail |", "|---|---|---|---|"]
    for r in RESULTS:
        d = r["detail"].replace("|", "/").replace("\n", " ")[:110]
        lines.append(f"| {r['category']} | {r['name']} | {r['status']} | {d} |")
    (REPORT_DIR / "e2e-report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{n['PASS']} passed, {n['FAIL']} failed, {n['INFO']} informational")
    print(f"Report: {REPORT_DIR / 'e2e-report.md'}")
    return n["FAIL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="leave containers running")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--ingest-wait", type=int, default=180)
    ap.add_argument("--min-hit", type=float, default=0.8)
    ap.add_argument("--min-abstain", type=float, default=0.6)
    args = ap.parse_args()
    REPORT_DIR.mkdir(exist_ok=True)

    if sh(["docker", "info"]).returncode != 0:
        raise SystemExit("Docker daemon not reachable. Try: sudo service docker start")
    if not (BACKEND / "src" / "main.py").exists():
        raise SystemExit("Run this from ~/HERMES-clean/backend (src/main.py not found)")

    proc = None
    try:
        start_infra()
        env = server_env()
        proc = start_server(env)
        client = httpx.Client(base_url=f"http://localhost:{API_PORT}", timeout=httpx.Timeout(300.0))
        api = Api(client)
        a, b = User(api, "alice"), User(api, "bob")
        print("\nRunning checks ...")
        rec("server", "health endpoint", client.get("/health").status_code == 200)
        test_auth(api, a)
        b.register(); b.login()
        test_ssrf(api, a)
        if not args.skip_quality:
            test_ingest_and_quality(api, a, b, args)
        test_eval_endpoints(api, a, b)
        test_mcp(env)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        if not args.keep:
            stop_infra()
    sys.exit(1 if write_report(args) else 0)


if __name__ == "__main__":
    main()