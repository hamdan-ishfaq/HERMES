import sys
import json
sys.path.insert(0, '/Ubuntu/home/mhamd/hermes/backend')
from src.agents.graph import run_research

state = run_research('What is retrieval augmented generation?')
print("STATE_KEYS:", state.keys())
print("CACHE_HIT_VALUE:", state.get("cache_hit"))
