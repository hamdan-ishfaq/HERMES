import sys
import json
sys.path.insert(0, '/home/mhamd/hermes/backend')
from src.rag.cache import SemanticCache

cache = SemanticCache()
index = cache.redis.lrange(cache._index_key(), 0, -1)
print("TOTAL ENTRIES IN REDIS:", len(index))
if index:
    for i, e in enumerate(index):
        entry = json.loads(e)
        keys = list(entry.get("result", {}).keys())
        print(f"Entry {i} Result Keys:", keys)
