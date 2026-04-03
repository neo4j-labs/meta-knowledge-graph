"""Neo4j-backed markdown memory system for LLM agents.

Stores memories as :Memory nodes in Neo4j with markdown content.
The LLM sees plain markdown, but persistence is in the graph.

Categories:
- tools   — per-tool learnings, usage patterns, tips
- user    — user persona, preferences, context
- general — interesting facts, general knowledge
"""

import re
from typing import Optional

from neo4j import AsyncDriver

# Allowed top-level categories
CATEGORIES = {"tools", "user", "general"}


def _sanitize_key(key: str) -> str:
    """Normalize key to a consistent lowercase identifier."""
    key = key.strip().lower()
    key = re.sub(r"[^\w\-]", "_", key)
    key = key.strip("_")
    if not key:
        raise ValueError("Key must contain at least one alphanumeric character")
    return key


def _validate_category(category: str) -> str:
    category = category.strip().lower()
    if category not in CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {', '.join(sorted(CATEGORIES))}"
        )
    return category


async def memory_write(
    driver: AsyncDriver, database: str, category: str, key: str, content: str
) -> str:
    """Write or update a memory node. Uses MERGE to upsert by (category, key)."""
    category = _validate_category(category)
    key = _sanitize_key(key)

    records, _, _ = await driver.execute_query(
        "MERGE (m:Memory {category: $category, key: $key}) "
        "ON CREATE SET m.content = $content, m.created_at = datetime(), m.updated_at = datetime() "
        "ON MATCH SET m.content = $content, m.updated_at = datetime() "
        "RETURN m.created_at = m.updated_at AS is_new",
        parameters_={"category": category, "key": key, "content": content},
        database_=database,
    )
    is_new = records[0]["is_new"]
    action = "Created" if is_new else "Updated"
    return f"{action} memory: {category}/{key}"


async def memory_read(
    driver: AsyncDriver, database: str, category: str, key: str
) -> str:
    """Read a memory node. Returns the markdown content."""
    category = _validate_category(category)
    key = _sanitize_key(key)

    records, _, _ = await driver.execute_query(
        "MATCH (m:Memory {category: $category, key: $key}) RETURN m.content AS content",
        parameters_={"category": category, "key": key},
        database_=database,
    )
    if not records:
        return f"No memory found for {category}/{key}"
    return records[0]["content"]


async def memory_list(
    driver: AsyncDriver, database: str, category: Optional[str] = None
) -> dict[str, list[str]]:
    """List all memories, optionally filtered by category."""
    if category:
        category = _validate_category(category)
        records, _, _ = await driver.execute_query(
            "MATCH (m:Memory {category: $category}) "
            "RETURN m.category AS category, m.key AS key ORDER BY m.key",
            parameters_={"category": category},
            database_=database,
        )
    else:
        records, _, _ = await driver.execute_query(
            "MATCH (m:Memory) "
            "RETURN m.category AS category, m.key AS key ORDER BY m.category, m.key",
            database_=database,
        )

    result: dict[str, list[str]] = {}
    for record in records:
        cat = record["category"]
        result.setdefault(cat, []).append(record["key"])
    return result


async def memory_delete(
    driver: AsyncDriver, database: str, category: str, key: str
) -> str:
    """Delete a memory node."""
    category = _validate_category(category)
    key = _sanitize_key(key)

    records, _, _ = await driver.execute_query(
        "MATCH (m:Memory {category: $category, key: $key}) "
        "DELETE m RETURN count(*) AS deleted",
        parameters_={"category": category, "key": key},
        database_=database,
    )
    deleted = records[0]["deleted"]
    if deleted == 0:
        return f"No memory found for {category}/{key}"
    return f"Deleted memory: {category}/{key}"
