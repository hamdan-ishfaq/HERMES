# ⚡ Hermes: Architectural Deep-Dive & System Documentation

This document serves as the internal technical post-mortem and living architectural blueprint for **Hermes**, our Enterprise-Grade Agentic RAG system. It is designed to act as a deep technical reference for precisely how our multi-stage retrieval, agentic routing, and persistent memory pipelines orchestrate together under the hood.

---

## 1. The 'Life of a Query' (Data Flow)

When a user submits a question via the React UI, Hermes executes a highly sequenced, deterministic data flow across multiple layers of the stack:

1. **Frontend Trigger (`ResearchView.jsx`)**: The user types a query. The React component packages the `query` along with a locally stored `session_id` (from `sessionStorage`) and fires a POST request to `/api/research`.
2. **FastAPI Router (`routers/research.py`)**: The backend receives the payload. If no `session_id` is provided, a fresh UUID4 is generated. The router invokes the `run_research()` orchestrator.
3. **LangGraph Initialization (`agents/graph.py`)**: The initial `ResearchState` dictionary is constructed. LangGraph's `MemorySaver` checkpointer loads any episodic history mapped to the incoming `session_id`.
4. **Supervisor Routing (`supervisor_node`)**: The query is evaluated for complexity (`simple`, `multi_hop`, `synthesis`). It is dynamically routed to the appropriate execution node (e.g., `research_node`).
5. **Semantic Cache Intercept (`rag/cache.py`)**: Before incurring LLM or Vector DB payloads, the query is embedded via `nomic-embed-text`. Upstash Redis is queried. If the cosine similarity matches an existing high-entropy entry (`>= 0.95`), the system instantly bypasses all subsequent steps and returns the fully drafted answer.
6. **Hybrid Retrieval (`rag/retriever.py`)**: Upon a cache miss, Hermes simultaneously calculates a dense embedding (`nomic-embed-text`) and a sparse keyword matrix (`BM25`). Qdrant executes a fused database search using mathematically combined Reciprocal Rank Fusion (RRF).
7. **Architectural Reranking (`rag/reranker.py`)**: The raw Qdrant chunks are passed through a heavy neural cross-encoder (`ms-marco-MiniLM-L-6-v2`) which simultaneously scores the user query against every retrieved passage to strip mathematical vector noise and establish a definitive Top-K ranking.
8. **Organic Synthesis (`agents/research.py`)**: The `research_node` pulls the conversation history natively from `state["messages"]`, prepends the last 4 messages to the system prompt (simulating memory), bundles the parent contexts, and invokes the LLM (Llama-3.1 via Ollama) to generate a clean, citation-free synthesis. 
9. **Final Output & Persistence**: The synthesized answer is safely cached locally in Redis alongside its dense vector. The LangGraph state resolves, and FastAPI streams the text, the visual citation mappings, and the active `session_id` back to the UI.

---

## 2. Deep-Dive: Advanced RAG Modules

Hermes completely abandons "Naive RAG" to circumvent widespread enterprise hallucination issues.

### 🔹 Hybrid Search (Qdrant + RRF)
Standard dense vectors capture broader semantics but frequently fail at exact keyword/serial number lookups. We solve this by bridging two isolated embedding techniques:
- **Dense Vectors**: `nomic-embed-text` parses contextual meaning natively.
- **Sparse Vectors**: Fastembed generates lightning-fast `BM25` word-frequency vectors.
Qdrant executes both searches asynchronously and algorithmically fuses the results via **Reciprocal Rank Fusion (RRF)**, securing exact semantic and syntax matches simultaneously.

### 🔹 Parent-Child Hierarchical Chunking
Large text blocks dilute vector embeddings; tiny text blocks deprive LLMs of critical surrounding context. We implemented a decoupled extraction strategy:
- We chunk documents into extremely tiny "Child" blocks for highly-isolated, pinpoint Qdrant vector retrieval.
- Once matched, the retriever natively traverses a relational map to grab the much larger surrounding "Parent" block, feeding the LLM the broader contextual environment.

### 🔹 Cross-Encoder Reranking
We observed that pure mathematical cosine similarities can create false-positive rankings based purely on linguistic proximity. We deployed `ms-marco-MiniLM-L-6-v2` as a terminal filter. Instead of comparing two isolated vectors, the cross-encoder feeds the query and the document into the neural network *jointly*, generating an aggressively accurate relevance classification right before the context window is constructed.

### 🔹 Semantic Caching (Redis Optimization)
To radically suppress repetitive GPU cycles, we built a mathematically tolerant Semantic Cache inside Redis. 
- **The Float-Noise Bug**: Initially, minor float-point noise in embedding generation caused exact requests to misalign under identical thresholds, breaking early intercepts. Furthermore, the early architecture only cached 'contexts', leaving the LLM generation active.
- **The Fix**: We tightened the semantic fallback boundary to exclusively trigger at `>= 0.95` similarity. We refactored the pipeline to seamlessly extract and cache the final **Answer String** natively on resolution. Now, any identical or conceptually isomorphic query successfully bypasses the LLM execution block entirely.

---

## 3. Agentic & Memory Architecture

### 🔹 LangGraph Orchestration
Instead of deterministic chains, the backend logic relies on cyclic state compilation natively powered by LangGraph:
- **`supervisor_node`**: Acts as the primary router, classifying query complexities on arrival.
- **`research_node`**: The workhorse node that handles retrieval, reranking, and natural language synthesis. 

### 🔹 Conversational Memory (Checkpointers)
To defeat basic single-turn amnesia, we implemented a persistent conversational episodic memory utilizing LangGraph's `MemorySaver`. 
- **Backend Threading**: When the FastAPI router intercepts an empty `session_id`, it generates a unique `UUID4`. This UUID maps cleanly across the state history under `config = {"configurable": {"thread_id": session_id}}`.
- **Frontend Hydration**: The React component pulls the returning `session_id` natively out of the API payload, locks it locally inside browser `sessionStorage`, and securely passes it inside every subsequent `POST /api/research` body, effectively unlocking perpetual narrative threads across user refreshes.

---

## 4. Post-Mortem & Future Architecture Trade-Offs

During development, critical architectural concessions were made exclusively to balance high-efficiency processing loops against theoretical capability limits.

### 📉 Unimplemented Concept: Core Self-RAG
While implementing a Self-RAG Reflection loop (forcing the agent to grade its own context retrieval and loop autonomously if relevance is too low) radically increases reliability, we consciously omitted it due to severe latency aggregation. Injecting an isolated evaluation LLM jump prior to final execution breached our time-to-first-byte (TTFB) targets for enterprise users.

### 📉 Unimplemented Concept: Query Expansion
We opted against upfront Query Rewriting / Expansion (e.g., having an LLM rewrite the user prompt into multiple hypothetical variations). Because we successfully optimized Qdrant's Hybrid Vector space (RRF + BM25), the dense semantic fallback natively covers varying terminology vectors perfectly, rendering the latency bump of Query Expansion computationally redundant.

### 🚀 Scale Strategy: The '10x Simplification'
If we encounter severe vertical scaling events (e.g., pushing from 100 to 10,000 concurrent enterprise users), the most immediate architectural simplification point is the **Cross-Encoder Reranker**. 
Currently, the `ms-marco` integration represents the most demanding CPU/GPU chokepoint alongside the core LLM execution block. Detaching the Cross-Encoder and relying purely on the Hybrid Qdrant RRF layer is the fastest immediate way to drop total pipeline latency by roughly ~40% under extreme throughput stress without abandoning our core architecture.
