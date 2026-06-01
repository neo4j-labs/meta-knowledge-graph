import argparse
import asyncio
import os

from dotenv import load_dotenv

from meta_knowledge_graph import server

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Meta Knowledge Graph MCP Server")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URL",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", "password"),
        help="Neo4j password",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default=os.environ.get("NEO4J_TRANSPORT", "stdio"),
        help="MCP transport type",
    )
    args = parser.parse_args()
    asyncio.run(
        server.main(
            db_url=args.db_url,
            username=args.username,
            password=args.password,
            database=args.database,
            transport=args.transport,
        )
    )
