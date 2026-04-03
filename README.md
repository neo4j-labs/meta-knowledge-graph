# Meta Knowledge Graph: Self-Learning System for AI Agents

This repository contains the reference implementation for the **Meta Knowledge Graph** (MKG), an **Intelligence Layer for Enterprise AI Agents**.

The MKG is a self-improving, graph-structured metadata layer designed to enhance the reasoning capabilities of AI agents by providing rich enterprise context.

### Key Differentiator: Generating Intelligence

The MKG's core innovation is that it does not just *store* intelligence—it **generates** it through the **Graph Intelligence Engine**.

This engine uses algorithms from the Graph Data Science (GDS) library to:

  * **Discover Implicit Relationships:** Find relationships not explicitly created by humans or agents (using node similarity on bipartite graphs).
  * **Learn Agent Patterns:** Identify critical decision points (using betweenness centrality) and cluster agent behaviors into reusable patterns (using community detection).

Every agent run strengthens patterns, and GDS discovers new ones from the accumulated decision traces, creating a **compounding intelligence effect**.

### Four Metadata Categories Captured

The system harvests and captures decision traces and four categories of metadata from enterprise data platforms and agent interactions:

  * **Technical:** Schemas, column definitions, data types, and lineage.
  * **Operational:** Job execution stats, quality/trust scores, freshness, and pipeline success/failure.
  * **Business:** Glossary terms, KPI definitions, ontologies, and semantic mapping (e.g., table-to-domain).
  * **Agentic:** Decision traces with causal chains (`CAUSED`, `INFLUENCED`, `PRECEDENT_FOR`), policy application, user corrections with rationale, and accumulated patterns.

### Core Architecture

  * **Meta Knowledge Graph (Neo4j):** The central graph store, extending the Context Graph Demo model with nodes like `AgentRun`, `Correction`, `OrgKnowledge`, and `Schema`.
  * **GDS Intelligence Engine:** Runs analytical workflows to discover implicit relationships and learn patterns from accumulated decision traces.
  * **Hybrid Retrieval Agent:** Combines **semantic search** (text embeddings on decision reasoning) with **structural search** (enriched e.g. with graph embeddings) to retrieve precedents that are both semantically and structurally similar.
  * **External Metadata Connector:** Gather catalog metadata (schema, columns, types) from sources like Snowflake or Databricks.
  * **Approval Process:** Knowledge derived from user interactions (chat) requires explicit approval to become permanent rules, and patterns carry time-decaying confidence scores that trigger re-validation.

# Metagraph MCP

MCP server for Neo4j that provides tools for AI agents — importing, retrieval, and more — while building a **metagraph** alongside. The metagraph captures metadata about what was imported, queried, and how data relates, giving agents a persistent map of their own knowledge operations.

## Quick Start

### Run directly from the repo

```bash
npx @anthropic-ai/claude-code --mcp-server "metagraph-mcp: npx -y @anthropic-ai/sdk run -- pip install git+https://github.com/tomasonjo/metagraph-mcp.git && metagraph-mcp"
```

### Run locally during development

Add to your Claude Desktop `claude_desktop_config.json` or `.claude/settings.json`:

```json
{
  "mcpServers": {
    "metagraph-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/metagraph-mcp", "run", "metagraph-mcp"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "<your-password>",
        "NEO4J_DATABASE": "neo4j",
        "OPENAI_API_KEY": "<your-openai-api-key>"
      }
    }
  }
}
```

## Configuration

| Env Variable | CLI Flag | Default |
|---|---|---|
| `NEO4J_URI` | `--db-url` | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | `--username` | `neo4j` |
| `NEO4J_PASSWORD` | `--password` | `password` |
| `NEO4J_DATABASE` | `--database` | `neo4j` |
| `NEO4J_TRANSPORT` | `--transport` | `stdio` |
| `OPENAI_API_KEY` | — | — |
| `LLM_MODEL` | — | `gpt-5.4-mini` |

## Tools

### Knowledge Graph

| Tool | Description |
|---|---|
| `import_text_to_kg` | Extract entities and relationships from text using an LLM and import them as a knowledge graph into Neo4j |
| `neo4j_get_schema` | Get the schema of the Neo4j database (node labels, relationship types, properties) |
| `neo4j_read_cypher` | Run a read-only Cypher query against the Neo4j database |

### Memory

Persistent key-value memory stored in Neo4j, organized by category (`tools`, `user`, `general`).

| Tool | Description |
|---|---|
| `memory_write` | Write or update a markdown memory entry (learnings, user info, or facts) |
| `memory_read` | Read a memory entry by category and key |
| `memory_list` | List all stored memory entries, optionally filtered by category |
| `memory_delete` | Delete a memory entry by category and key |
