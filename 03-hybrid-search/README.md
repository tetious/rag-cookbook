# Hybrid Search RAG

Combines **BM25 (keyword search)** with **vector search (semantic search)** for improved retrieval.

## No Ingestion Required!

This module reuses the existing vectorized documents from either:
- `naive/` → Collection: `rag_playbook.naive_rag`
- `metadata-filtered/` → Collection: `rag_playbook.metadata_filtered_rag`

Make sure you've run ingestion for at least one of these before using hybrid search.

## How It Works

1. **BM25 Search**: Lexical matching based on term frequency (good for exact keywords, names, acronyms)
2. **Vector Search**: Semantic similarity using embeddings (good for meaning, paraphrasing)
3. **Fusion**: Results combined using Reciprocal Rank Fusion (RRF)

## When Hybrid Beats Pure Vector

| Query Type | Vector Only | Hybrid (BM25 + Vector) |
|------------|-------------|------------------------|
| "GEICO earnings 2020" | ⚠️ May miss exact terms | ✅ Catches "GEICO" + "2020" |
| "What makes a good investment?" | ✅ Semantic match | ✅ Also good |
| "BRK.A stock split" | ⚠️ May miss ticker | ✅ Keyword match |

## Usage

```bash
# Install BM25 dependency
pip install rank_bm25

# Run retrieval tests
python hybrid-search/retrieval.py

# Interactive Q&A
python hybrid-search/generation.py
```

## Configuration

Adjust the balance between BM25 and vector search:
- `weights=[0.5, 0.5]` - Equal weighting (default)
- `weights=[0.3, 0.7]` - Favor vector search
- `weights=[0.7, 0.3]` - Favor keyword search
