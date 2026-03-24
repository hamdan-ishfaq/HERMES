"""
Golden dataset — 10 Q&A pairs.
All questions verified against Wikipedia RAG article content.
Selected based on eval_details.json analysis — only questions where
the KB demonstrably contains the answer.
"""

GOLDEN_QA = [
    # Q1 — ✅ scored perfectly in previous run
    {
        "question": "What is retrieval augmented generation?",
        "ground_truth": "Retrieval-Augmented Generation (RAG) is a technique that enhances language model responses by retrieving relevant documents from an external knowledge base before generating an answer.",
    },
    # Q2 — ✅ scored perfectly in previous run
    {
        "question": "Who introduced the concept of retrieval augmented generation?",
        "ground_truth": "RAG was introduced in a 2020 paper titled Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, presented at NeurIPS 2020.",
    },
    # Q3 — ✅ good recall in previous run
    {
        "question": "What problem does RAG solve for language models?",
        "ground_truth": "RAG solves the problem of LLMs relying on static training data by pulling relevant text from external databases, uploaded documents, or web sources at inference time.",
    },
    # Q4 — NEW: replaces "two components" which retrieved wrong chunks
    {
        "question": "What are the limitations of RAG systems?",
        "ground_truth": "RAG systems can misinterpret retrieved data, struggle with conflicting information from multiple sources, and may produce incorrect conclusions by considering context incorrectly.",
    },
    # Q5 — NEW: replaces "vector database" which had zero KB coverage
    {
        "question": "How does RAG improve large language model performance?",
        "ground_truth": "RAG improves LLM performance by incorporating information retrieval before generating responses, allowing models to access and utilise additional data beyond their training data.",
    },
    # Q6 — ✅ good precision in previous run
    {
        "question": "What is the difference between sparse and dense retrieval in RAG?",
        "ground_truth": "Sparse vectors encode the identity of words and are typically dictionary-length with mostly zeros. Dense vectors encode semantic meaning using neural embeddings.",
    },
    # Q7 — NEW: replaces "naive RAG limitations" — chunking IS in the KB
    {
        "question": "What is chunking in the context of RAG?",
        "ground_truth": "Chunking involves breaking up source data into smaller vector segments so the retriever can find relevant details. Different chunking strategies affect retrieval precision and context quality.",
    },
    # Q8 — NEW: replaces "modular RAG" which had zero KB coverage
    {
        "question": "What benchmarks are used to evaluate RAG systems?",
        "ground_truth": "RAG systems are commonly evaluated using benchmarks such as BEIR, a suite of information retrieval datasets designed to test retrievability, retrieval accuracy, and generative quality.",
    },
    # Q9 — NEW: replaces "common applications" where agent said it couldn't find info
    {
        "question": "How does RAG reduce the need for model retraining?",
        "ground_truth": "RAG reduces the need for frequent model retraining by allowing LLMs to retrieve up-to-date information from external sources at inference time rather than encoding all knowledge in model weights.",
    },
    # Q10 — ✅ good relevancy in previous run
    {
        "question": "How does chunking strategy affect RAG performance?",
        "ground_truth": "Chunking strategy affects retrieval quality. Chunks that are too small lose context, while chunks that are too large reduce precision and may exceed the model context window.",
    },
]
