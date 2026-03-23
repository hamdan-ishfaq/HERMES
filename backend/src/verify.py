import asyncio, os, httpx
from dotenv import load_dotenv
load_dotenv()

async def check_all():
    print("=== Hermes Connection Check ===\n")

    # Ollama
    try:
        r = httpx.get(f"{os.getenv('OLLAMA_API_BASE')}/api/tags")
        models = [m["name"] for m in r.json()["models"]]
        print(f"✅ Ollama — models: {models}")
    except Exception as e:
        print(f"❌ Ollama — {e}")

    # Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        info = client.get_collections()
        print(f"✅ Qdrant — {len(info.collections)} collections")
    except Exception as e:
        print(f"❌ Qdrant — {e}")

    # Redis
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL"))
        r.ping()
        print("✅ Redis — connected")
    except Exception as e:
        print(f"❌ Redis — {e}")

    # Postgres
    try:
        import asyncpg
        conn = await asyncpg.connect(os.getenv("DATABASE_URL").replace("+asyncpg", ""))
        await conn.close()
        print("✅ Postgres — connected")
    except Exception as e:
        print(f"❌ Postgres — {e}")

    # Groq
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

asyncio.run(check_all())
