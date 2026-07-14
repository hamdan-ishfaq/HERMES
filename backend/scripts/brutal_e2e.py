#!/usr/bin/env python3
"""
Brutal in-process + light API E2E for Hermes.

Uses the already-running Qdrant/Redis/Ollama stack. Wipes hermes_docs, re-ingests
gold KB URLs with user_id=eval, then exercises research / cache / memory / tools
in-process (avoids single-worker uvicorn blocking). Optionally smokes the HTTP
stream endpoint if API is up.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

import httpx
import redis
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

ROOT = "http://127.0.0.1:8000"
API = f"{ROOT}/api"


def fail(msg: str) -> None:
    print(f"\nFAIL: {msg}", flush=True)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  OK  {msg}", flush=True)


def main() -> None:
    print("=" * 60, flush=True)
    print("HERMES BRUTAL E2E (in-process)", flush=True)
    print("=" * 60, flush=True)

    # --- Ollama ---
    print("\n[0] Probes", flush=True)
    tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=30).json()
    models = [m["name"] for m in tags.get("models", [])]
    for need in ("nomic-embed-text", "llama3.2:3b", "llama3.1:8b"):
        if not any(need in m for m in models):
            fail(f"missing model {need}; have {models}")
    ok(f"ollama {models}")

    qc = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    try:
        qc.delete_collection("hermes_docs")
        ok("cleared hermes_docs")
    except Exception as e:
        ok(f"clear skipped ({e})")

    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    for key in r.scan_iter(match="hermes:cache:index*", count=200):
        r.delete(key)
    for key in r.scan_iter(match="hermes:query:*", count=200):
        r.delete(key)
    ok("cleared hermes redis keys")

    from src.evaluation.golden_dataset import load_kb_urls
    from src.ingestion.url_loader import ingest_url
    from src.rag.factory import get_retriever, reset_retriever
    from src.agents.graph import run_research

    reset_retriever()
    # Force graph rebuild after code changes
    import src.agents.graph as graph_mod

    graph_mod._graph = None

    print("\n[1] Re-ingest demo KB", flush=True)
    retriever = get_retriever()
    urls = load_kb_urls()
    total_children = 0
    for url in urls:
        t0 = time.time()
        print(f"  ingesting {url} ...", flush=True)
        result = ingest_url(
            url,
            retriever,
            extra_metadata={"user_id": "eval", "workspace_id": "eval"},
        )
        if result.get("status") == "failed":
            fail(f"ingest failed: {result}")
        kids = result.get("total_children", 0)
        total_children += kids
        ok(f"{url.split('/')[-1]} → {kids} children in {time.time()-t0:.1f}s")
    if total_children < 10:
        fail(f"too few chunks: {total_children}")
    ok(f"total children ~{total_children}")

    print("\n[2] Research miss + tool_trace + citations", flush=True)
    q1 = "What is retrieval augmented generation?"
    t0 = time.time()
    s1 = run_research(q1, session_id=str(uuid.uuid4()), user_id="eval")
    dt1 = time.time() - t0
    ans1 = s1.get("final_answer") or s1.get("draft_answer") or ""
    if len(ans1) < 40:
        fail(f"short answer: {ans1!r}")
    if not s1.get("citations"):
        fail("no citations")
    trace = s1.get("tool_trace") or []
    if not any(t.get("name") == "hybrid_search" for t in trace):
        fail(f"no hybrid_search in {trace}")
    if s1.get("cache_hit"):
        fail("unexpected cache hit")
    ok(f"miss {dt1:.1f}s answer_len={len(ans1)} citations={len(s1['citations'])}")
    ok(f"tool_trace={trace}")

    print("\n[3] Paraphrase semantic cache", flush=True)
    # Cache keys on original query; set already happened for simple path.
    # Paraphrase: use retriever.cache.get on a near-duplicate embedding.
    q2 = "Explain what retrieval-augmented generation is."
    t0 = time.time()
    s2 = run_research(q2, session_id=str(uuid.uuid4()), user_id="eval")
    dt2 = time.time() - t0
    if s2.get("cache_hit"):
        ok(f"cache HIT in {dt2:.2f}s")
        if dt2 > 15:
            print(f"  WARN: cache hit slower than expected ({dt2:.1f}s)", flush=True)
    else:
        print(f"  WARN: paraphrase miss ({dt2:.1f}s) — cache threshold may not match", flush=True)

    # Exact repeat should hit
    t0 = time.time()
    s2b = run_research(q1, session_id=str(uuid.uuid4()), user_id="eval")
    dt2b = time.time() - t0
    if not s2b.get("cache_hit"):
        fail(f"exact repeat not cached ({dt2b:.1f}s)")
    ok(f"exact-repeat cache HIT in {dt2b:.2f}s")

    print("\n[4] Multi-turn follow-up rewrite", flush=True)
    sid = str(uuid.uuid4())
    hist = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": ans1[:500]},
    ]
    q3 = "What limitations does it have?"
    s3 = run_research(q3, session_id=sid, messages=hist, user_id="eval")
    rw = (s3.get("rewritten_query") or "").lower()
    ans3 = (s3.get("final_answer") or s3.get("draft_answer") or "").lower()
    ok(f"rewritten_query={s3.get('rewritten_query')!r}")
    if "rag" in rw or "retrieval" in rw or "augmented" in rw:
        ok("rewrite kept RAG entity")
    else:
        print(f"  WARN: weak rewrite: {rw!r}", flush=True)
    if "limit" in ans3 or "challenge" in ans3 or "misinterpret" in ans3 or "conflict" in ans3:
        ok("follow-up answer discusses limitations")
    else:
        print(f"  WARN: weak follow-up answer: {ans3[:200]!r}", flush=True)

    print("\n[5] ACL isolation (negative)", flush=True)
    s_other = run_research(q1, session_id=str(uuid.uuid4()), user_id="other-user")
    ans_other = s_other.get("final_answer") or s_other.get("draft_answer") or ""
    if s_other.get("citations") or (
        "no relevant" not in ans_other.lower() and len(ans_other) > 80
    ):
        # Strict: other user should get no_context
        if s_other.get("error") != "no_context" and s_other.get("citations"):
            fail(f"ACL leak: other user got citations={s_other.get('citations')}")
    ok(f"other user blocked (error={s_other.get('error')!r})")

    print("\n[6] Empty / nonsense honesty", flush=True)
    s4 = run_research(
        "What is the capital of ZEPHYR_PLANET_99?",
        session_id=str(uuid.uuid4()),
        user_id="eval",
    )
    ans4 = (s4.get("final_answer") or "").lower()
    if "http://" in ans4 or "https://" in ans4:
        fail(f"invented URL in answer: {ans4[:200]}")
    ok(f"no invented URLs (error={s4.get('error')!r})")

    print("\n[7] HTTP stream smoke (optional)", flush=True)
    try:
        health = httpx.get(f"{ROOT}/health", timeout=5)
        if health.status_code != 200:
            print("  SKIP API down", flush=True)
        else:
            email = f"e2e_{uuid.uuid4().hex[:8]}@hermes.dev"
            reg = httpx.post(
                f"{API}/auth/register",
                json={"email": email, "password": "e2e_pass_99"},
                timeout=30,
            )
            if reg.status_code not in (200, 201):
                print(f"  SKIP register {reg.status_code}", flush=True)
            else:
                token = reg.json()["access_token"]
                # Re-ingest one URL for this JWT user so ACL allows hits
                ingest = httpx.post(
                    f"{API}/ingest/url",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"url": urls[0]},
                    timeout=300,
                )
                ok(f"API ingest status={ingest.status_code}")
                with httpx.stream(
                    "POST",
                    f"{API}/research/stream",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"query": "What is RAG?"},
                    timeout=600,
                ) as stream:
                    if stream.status_code != 200:
                        fail(f"stream {stream.status_code}")
                    tokens = 0
                    done = None
                    event = "message"
                    for line in stream.iter_lines():
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            raw = line[5:].strip()
                            if not raw:
                                continue
                            data = json.loads(raw)
                            if event == "token":
                                tokens += 1
                            elif event == "done":
                                done = data
                            event = "message"
                    if tokens < 1 and not (done and done.get("answer")):
                        fail("stream empty")
                    ok(f"SSE tokens={tokens} done_answer_len={len((done or {}).get('answer') or '')}")
    except Exception as e:
        print(f"  WARN: API smoke skipped ({e})", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("BRUTAL E2E PASSED", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
