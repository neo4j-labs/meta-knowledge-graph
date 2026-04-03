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
