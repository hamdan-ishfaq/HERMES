"""
Standalone connectivity checker for Hermes infrastructure dependencies.

What this file does:
    Probes Ollama, Qdrant, Redis, Postgres, and Groq with a minimal request
    each, printing pass/fail lines to the terminal. Useful when setting up a
    new machine or debugging "connection refused" errors.

Where it sits in the HERMES pipeline:
    Outside the runtime application — this is a developer utility, not part
    of the FastAPI server or agent graph. Run it manually before starting the
    backend to confirm all external services are reachable.

What calls this:
    Executed directly: ``python -m src.verify`` or ``python src/verify.py``.

What this calls:
    - Ollama HTTP API (model list)
    - Qdrant client (list collections)
    - Redis (PING)
    - asyncpg (Postgres connect)
    - LiteLLM → Groq (one-token completion smoke test)
"""

import asyncio, os, httpx
from dotenv import load_dotenv
load_dotenv()


async def check_all():
    """
    Run all dependency checks sequentially and print human-readable results.

    Each block is wrapped in try/except so one failure does not stop the
    rest of the report from printing.

    Returns:
        None — output goes to stdout only.
    """
    print("=== Hermes Connection Check ===\n")

    # Ollama — local LLM and embedding server used by agents and retriever.
    try:
        r = httpx.get(f"{os.getenv('OLLAMA_API_BASE')}/api/tags")
        models = [m["name"] for m in r.json()["models"]]
        print(f"✅ Ollama — models: {models}")
    except Exception as e:
        print(f"❌ Ollama — {e}")

    # Qdrant — vector database storing document chunk embeddings.
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        info = client.get_collections()
        print(f"✅ Qdrant — {len(info.collections)} collections")
    except Exception as e:
        print(f"❌ Qdrant — {e}")

    # Redis — backing store for the semantic query cache.
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL"))
        r.ping()
        print("✅ Redis — connected")
    except Exception as e:
        print(f"❌ Redis — {e}")

    # Postgres — SQL metadata (users, jobs, query logs).
    # asyncpg expects a DSN without the "+asyncpg" SQLAlchemy driver suffix.
    try:
        import asyncpg
        conn = await asyncpg.connect(os.getenv("DATABASE_URL").replace("+asyncpg", ""))
        await conn.close()
        print("✅ Postgres — connected")
    except Exception as e:
        print(f"❌ Postgres — {e}")

    # Groq — cloud LLM used for "complex" queries via LiteLLM fallback chain.
    try:
        from litellm import completion
        resp = completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say OK only"}],
            max_tokens=5
        )
        print(f"✅ Groq — {resp.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"❌ Groq — {e}")

    print("\n=== Done ===")


# Script entry point — asyncio.run drives the single async check function.
asyncio.run(check_all())
