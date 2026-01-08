# Agentic RAG

An AI agent that **dynamically decides** when to retrieve, what retrieval method to use, and whether it has enough information to answer.

## No Ingestion Required

Uses existing MongoDB vector collections:
- `naive_rag` (basic vectors)
- `metadata_filtered_rag` (vectors + metadata)

## How It Works

```
User Question
      │
      ▼
┌─────────────────┐
│  1. ANALYZE     │  "Do I need retrieval? Is this complex?"
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────┐
│SIMPLE │ │ COMPLEX  │
│       │ │decompose │
└───┬───┘ └────┬─────┘
    │          │
    ▼          ▼
┌─────────────────┐
│  2. DECIDE      │  "Retrieval needed? Which method?"
│  - no_retrieval │
│  - vector_search│
│  - filtered     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. EXECUTE     │  Run selected tool(s)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. EVALUATE    │  "Is this enough? Should I retry?"
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 ENOUGH    NOT ENOUGH
    │          │
    │          └──► Retry with different query/method
    ▼
┌─────────────────┐
│  5. SYNTHESIZE  │  Generate final answer
└─────────────────┘
```

## Agent Tools

| Tool | Description | When Used |
|------|-------------|-----------|
| `no_retrieval` | Skip retrieval, use model knowledge | Simple factual questions |
| `vector_search` | Semantic similarity search | General questions about content |
| `filtered_search` | Vector search with metadata filters | Questions about specific years, topics, companies |

## Usage

```bash
python agentic-rag/agent.py

# Or interactive mode
python agentic-rag/generation.py
```

## Example Agent Reasoning

**Question:** "How did Berkshire's insurance perform in 2020 vs 2008?"

```
🤔 ANALYZING: Complex comparison question requiring two time periods

📋 DECOMPOSING into sub-questions:
   1. "Berkshire insurance performance 2008"
   2. "Berkshire insurance performance 2020"

🔧 SUB-QUESTION 1:
   Tool: filtered_search (year=2008, topic=insurance)
   Retrieved: 3 documents
   ✅ Sufficient information

🔧 SUB-QUESTION 2:
   Tool: filtered_search (year=2020, topic=insurance)
   Retrieved: 3 documents
   ✅ Sufficient information

📝 SYNTHESIZING final answer from both retrievals...
```

## What Makes It "Agentic"

1. **Decides IF** retrieval is needed (vs using model knowledge)
2. **Chooses HOW** to retrieve (vector vs filtered)
3. **Decomposes** complex queries into sub-queries
4. **Evaluates** if retrieved info is sufficient
5. **Retries** with different approach if needed
6. **Synthesizes** from multiple retrieval steps
