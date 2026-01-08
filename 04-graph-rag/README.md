# GraphRAG - Knowledge Graph Enhanced Retrieval

Combines **knowledge graph traversal** with **vector search** for richer context retrieval.

## How It Works

1. **Extract** entities and relationships from document chunks using LLM
2. **Build** a knowledge graph in Neo4j
3. **Query** by finding relevant entities → traversing the graph → enriching vector results
4. **Generate** answers with graph-enhanced context

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  MongoDB Chunks │────▶│  LLM Extraction │
│  (existing)     │     │  (entities/rels)│
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │    Neo4j Graph  │
                        │  (nodes/edges)  │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Vector Search │       │Graph Traversal│       │   Combined    │
│   (MongoDB)   │       │   (Neo4j)     │       │   Context     │
└───────────────┘       └───────────────┘       └───────────────┘
```

## Setup

### 1. Start Neo4j with Docker

```bash
cd graph-rag
docker-compose up -d
```

Wait ~30 seconds for Neo4j to start, then verify:
- Browser: http://localhost:7474
- Login: neo4j / password123

### 2. Add Neo4j credentials to .env

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

### 3. Install dependencies

```bash
pip install neo4j langchain-neo4j
```

### 4. Build the knowledge graph

```bash
python graph-rag/graph_builder.py
```

This extracts entities and relationships from your existing MongoDB chunks.

### 5. Run retrieval/generation

```bash
python graph-rag/retrieval.py    # Test graph-enhanced retrieval
python graph-rag/generation.py   # Interactive Q&A
```

## Graph Schema

### Node Types
- `Entity` - Companies, people, concepts (e.g., "Apple", "Buffett", "Insurance")
- `Document` - Source documents (e.g., "2020ltr.pdf")
- `Topic` - High-level topics (e.g., "Investments", "Insurance")
- `Year` - Temporal nodes (e.g., "2020")

### Relationship Types
- `MENTIONS` - Document mentions an entity
- `RELATED_TO` - Entity is related to another entity
- `DISCUSSED_IN` - Topic discussed in a year
- `INVESTS_IN` - Investment relationship
- `OWNS` - Ownership relationship
- `MANAGES` - Management relationship

## Example Queries

**Find all entities related to insurance:**
```cypher
MATCH (e:Entity)-[:RELATED_TO*1..2]-(related)
WHERE e.name = 'insurance'
RETURN e, related
```

**Find documents where Apple and Buffett are both mentioned:**
```cypher
MATCH (d:Document)-[:MENTIONS]->(e1:Entity {name: 'apple'})
MATCH (d)-[:MENTIONS]->(e2:Entity {name: 'buffett'})
RETURN d.name
```

## Retrieval Strategy

1. **Extract query entities** - Identify entities mentioned in the user's question
2. **Graph traversal** - Find related entities within 2 hops
3. **Find relevant chunks** - Get document chunks that mention these entities
4. **Vector search** - Also run semantic search on the query
5. **Combine & rerank** - Merge results, boosting chunks with graph connections
6. **Generate** - Use enriched context for answer generation
