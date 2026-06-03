"""Build sales-assistant.excalidraw — architecture sketch for the sales assistant.

Run:
    uv run python scripts/build_sales_excalidraw.py

Output: sales-assistant.excalidraw at the repo root, importable directly into
the Excalidraw web app or the Excalidraw VS Code extension.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "sales-assistant.excalidraw"

BLUE   = "#a5d8ff"
YELLOW = "#ffec99"
GREEN  = "#b2f2bb"
PINK   = "#ffc9c9"
PURPLE = "#d0bfff"
INK    = "#1e1e1e"
GREY_ARROW = "#495057"

elements: list[dict] = []
rng = random.Random(20260601)


def _id() -> str:
    return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=12))


def _seed() -> int:
    return rng.randint(1, 2_000_000_000)


def _nonce() -> int:
    return rng.randint(1, 2_000_000_000)


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    bg: str = BLUE,
    label: str = "",
    font_size: int = 16,
) -> str:
    rid = _id()
    elements.append({
        "id": rid, "type": "rectangle",
        "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": INK, "backgroundColor": bg,
        "fillStyle": "solid", "strokeWidth": 1.5, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 3},
        "seed": _seed(), "version": 1, "versionNonce": _nonce(),
        "isDeleted": False, "boundElements": [], "updated": 0,
        "link": None, "locked": False,
    })
    if label:
        tid = _id()
        elements[-1]["boundElements"].append({"id": tid, "type": "text"})
        elements.append({
            "id": tid, "type": "text",
            "x": x, "y": y + (h - font_size * 1.25) / 2,
            "width": w, "height": font_size * 1.25,
            "angle": 0, "strokeColor": INK, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 1.5, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
            "roundness": None,
            "seed": _seed(), "version": 1, "versionNonce": _nonce(),
            "isDeleted": False, "boundElements": [], "updated": 0,
            "link": None, "locked": False,
            "text": label, "fontSize": font_size, "fontFamily": 5,
            "textAlign": "center", "verticalAlign": "middle",
            "baseline": font_size - 2, "containerId": rid,
            "originalText": label, "lineHeight": 1.25,
        })
    return rid


def label(x: float, y: float, t: str, *, size: int = 16, w: float = 600) -> str:
    tid = _id()
    elements.append({
        "id": tid, "type": "text",
        "x": x, "y": y, "width": w, "height": size * 1.5,
        "angle": 0, "strokeColor": INK, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1.5, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None,
        "seed": _seed(), "version": 1, "versionNonce": _nonce(),
        "isDeleted": False, "boundElements": [], "updated": 0,
        "link": None, "locked": False,
        "text": t, "fontSize": size, "fontFamily": 5,
        "textAlign": "left", "verticalAlign": "top",
        "baseline": size - 2, "containerId": None,
        "originalText": t, "lineHeight": 1.25,
    })
    return tid


def arrow(
    from_id: str,
    to_id: str,
    x1: float, y1: float,
    x2: float, y2: float,
    *,
    color: str = GREY_ARROW,
    edge_label: str | None = None,
) -> str:
    aid = _id()
    elements.append({
        "id": aid, "type": "arrow",
        "x": x1, "y": y1, "width": abs(x2 - x1), "height": abs(y2 - y1),
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1.5, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 2},
        "seed": _seed(), "version": 1, "versionNonce": _nonce(),
        "isDeleted": False, "boundElements": [], "updated": 0,
        "link": None, "locked": False,
        "startBinding": {"elementId": from_id, "focus": 0, "gap": 6},
        "endBinding": {"elementId": to_id, "focus": 0, "gap": 6},
        "lastCommittedPoint": None,
        "startArrowhead": None, "endArrowhead": "arrow",
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "elbowed": False,
    })
    if edge_label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        label(mx - 80, my - 10, edge_label, size=12, w=160)
    return aid


# -------- title --------
label(40, 30,
      "Sales Assistant · MCP for data access  ·  Hooks for behavior",
      size=24, w=1300)

# -------- Row 1: external sources + the sales rep --------
y1 = 110
bq      = rect(60,   y1, 230, 90, bg=YELLOW,
               label="BigQuery\nproducts · accounts · usage\n(read-only)")
enhance = rect(340,  y1, 230, 90, bg=PURPLE,
               label="Diffbot Enhance\norg + person enrichment")
news    = rect(620,  y1, 230, 90, bg=PURPLE,
               label="Diffbot News\nrecent articles")
rep     = rect(1100, y1, 230, 90, bg=GREEN,
               label="Sales Rep")

# -------- Row 2: MCP layer (data access) --------
y2 = 260
mcp = rect(60, y2, 1280, 110, bg=BLUE,
           label="meta-knowledge-graph MCP   ·   data access\n"
                 "bigquery_execute_query · enhance_entity · search_news\n"
                 "sales_brief_account · sales_sync_account_usage · "
                 "sales_track_news · sales_upsert_deal")

# -------- Row 3: agent (left) + hooks (right) --------
y3 = 430
h3 = 150
agent = rect(60, y3, 880, h3, bg=GREEN,
             label="Sales Assistant Agent (LLM)\n"
                   "plans tool calls · drafts outreach · ranks accounts")

hooks = rect(990, y3, 350, h3, bg="#fcc2d7",
             label="Hooks   ·   behavior\n"
                   "SessionStart → inject SystemPrompt\n"
                   "UserPromptSubmit → inject scoped\nAccount brief\n"
                   "Stop / SessionEnd → log events\n+ adjudicate :Learning · :Decision",
             font_size=13)

# -------- Row 4: Neo4j Customer Graph (long-term memory) --------
y4 = 630
h4 = 260
mem = rect(60, y4, 1280, h4, bg=PINK, label="")
label(80, y4 + 14, "Neo4j · Customer Graph  (long-term memory)", size=18, w=900)

inner_y = y4 + 70
account_b = rect(90,   inner_y, 180, 70, bg="#ffe3e3", label=":Account")
contact_b = rect(290,  inner_y, 180, 70, bg="#ffe3e3", label=":Contact")
deal_b    = rect(490,  inner_y, 180, 70, bg="#ffe3e3",
                 label=":Deal\n(assistant-owned)")
product_b = rect(690,  inner_y, 180, 70, bg="#ffe3e3",
                 label=":Product\n(mirrors BQ)")
news_b    = rect(890,  inner_y, 180, 70, bg="#ffe3e3", label=":NewsArticle")
snap_b    = rect(1090, inner_y, 230, 70, bg="#ffe3e3", label=":DiffbotSnapshot")

label(90, y4 + h4 - 56,
      "Diffbot enrichment + news are persisted here as durable memory.\n"
      "Deals also live here — the assistant writes stages, notes, owners.\n"
      "Hooks also read :SystemPrompt and write :Learning / :Decision into this graph.",
      size=13, w=1180)

# -------- Arrows --------
# sources → MCP
arrow(bq,      mcp, 175, y1 + 90, 175, y2,       edge_label="read-only")
arrow(enhance, mcp, 455, y1 + 90, 455, y2,       edge_label="enrich")
arrow(news,    mcp, 735, y1 + 90, 735, y2,       edge_label="news search")

# MCP → Agent (tool calls go to the agent)
arrow(mcp,   agent, 500, y2 + 110, 500, y3,      edge_label="tool calls")

# Hooks → Agent (inject behavior)
arrow(hooks, agent, 990, y3 + 70, 940, y3 + 70,  edge_label="inject prompt + context")

# Agent → Hooks (lifecycle events captured)
arrow(agent, hooks, 940, y3 + 110, 990, y3 + 110, edge_label="lifecycle events")

# Agent ↔ Memory (bidirectional, two arrows)
arrow(agent, mem,   360, y3 + h3, 360, y4,       edge_label="writes: deals · enrichment · news")
arrow(mem,   agent, 640, y4,      640, y3 + h3,  edge_label="reads: account graph")

# Hooks ↔ Memory (load SystemPrompt + write adjudicated learnings)
arrow(hooks, mem,   1165, y3 + h3, 1165, y4,     edge_label="write :Learning · :Decision")
arrow(mem,   hooks, 1050, y4,      1050, y3 + h3, edge_label="load :SystemPrompt")

# Sales Rep ↔ Agent (questions, drafts)
arrow(rep,   agent, 1100, y1 + 60, 940,  y3 + 30, edge_label="questions · deal updates")


out = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://github.com/tomasonjo/meta-knowledge-graph",
    "elements": elements,
    "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
OUT.write_text(json.dumps(out, indent=2))
print(f"wrote {OUT}  ({len(elements)} elements)")
