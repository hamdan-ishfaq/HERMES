import sys
sys.path.insert(0, '/home/mhamd/hermes/backend')
from src.rag.cache import SemanticCache

cache = SemanticCache()
cache.redis.delete(cache._index_key())
print("Semantic Cache Cleared Successfully!")
