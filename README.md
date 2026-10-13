cat > README.md << 'EOF'
# Agentic RAG Knowledge Assistant

A production-grade Retrieval-Augmented Generation system with an agent brain,
built from first principles and evaluated empirically.

> **Status:** Week 1 — Baseline RAG (vector search only)

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| LLM | Qwen3 8B (Ollama) | Best local tool-calling model; 128K context; Apache 2.0 |
| Embeddings | nomic-embed-text (Ollama) | 768 dims; local; strong retrieval benchmarks |
| Vector DB | pgvector (Postgres 18) | No new infra; hybrid search possible in one DB |
| Ingestion | LlamaIndex | 130+ formats; chunking; indexing |

**Total monthly cost: $0** — local models, free tiers.

## Quick Start

### Prerequisites

- Docker
- Python 3.11+
- [Ollama](https://ollama.com)

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
