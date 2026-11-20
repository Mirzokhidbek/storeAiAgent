"""Connects the agent to the two MCP servers (Phase 6b) through langchain-mcp-adapters.

The async problem: MCP clients are async, but our graph, the terminal and Streamlit are
sync, and Streamlit re-runs the script in changing threads.

The simplest reliable fix used here: ONE background thread runs ONE asyncio event loop
for the whole process. Both MCP sessions are opened on that loop once and stay open.
Sync code hands a coroutine to that loop and waits for the answer
(asyncio.run_coroutine_threadsafe). So there is no asyncio.run() per request (that would
start both server processes again for every tool call) and no "event loop is already
running" / "attached to a different loop" errors.
"""

import asyncio
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

ROOT = Path(__file__).resolve().parent.parent
SERVERS = {"catalog": "mcp_servers.catalog_server", "orders": "mcp_servers.orders_server"}
START_TIMEOUT = 60   # seconds: the catalog server loads Chroma on start
CALL_TIMEOUT = 60


def server_connections(extra_env: dict | None = None) -> dict:
    """How to start each server: `python -m mcp_servers.<name>` over stdio.

    The MCP SDK passes only a few "safe" environment variables to a subprocess, so we
    pass ours explicitly (tests add SHOP_DB_PATH). cwd=ROOT lets the server import
    shop/ and find .env.
    """
    env = {**os.environ, **(extra_env or {})}
    return {name: {"transport": "stdio", "command": sys.executable,
                   "args": ["-m", module], "cwd": str(ROOT), "env": env}
            for name, module in SERVERS.items()}


def parse_result(message: ToolMessage) -> dict:
    """MCP returns the tool's dict as JSON text; turn it back into the same dict."""
    text = "".join(block["text"] for block in message.content
                   if isinstance(block, dict) and block.get("type") == "text")
    if message.status == "error":
        return {"error": "Xizmatda xatolik yuz berdi. Iltimos, keyinroq urinib ko'ring."}
    return json.loads(text)


class MCPToolbox:
    """Keeps both MCP sessions open and lets sync code call their tools."""

    def __init__(self, extra_env: dict | None = None):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop").start()
        self._ready = threading.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self._tools: dict[str, dict] = {}
        self._task = asyncio.run_coroutine_threadsafe(
            self._keep_sessions_open(server_connections(extra_env)), self._loop)
        if not self._ready.wait(START_TIMEOUT) or self._error:
            raise RuntimeError("MCP serverlarini ishga tushirib bo'lmadi") from self._error

    async def _keep_sessions_open(self, connections: dict) -> None:
        # Opening and closing must happen in the SAME task (the MCP SDK uses anyio task
        # groups), so this one task opens the sessions, then waits until close().
        try:
            client = MultiServerMCPClient(connections)
            async with AsyncExitStack() as stack:
                for name in connections:
                    session = await stack.enter_async_context(client.session(name))
                    self._tools[name] = {t.name: t for t in await load_mcp_tools(session)}
                self._ready.set()
                await self._stop.wait()
        except BaseException as err:   # noqa: BLE001 — reported to __init__
            self._error = err
            self._ready.set()

    def tool_names(self, server: str) -> set[str]:
        return set(self._tools[server])

    def call(self, server: str, tool: str, args: dict) -> dict:
        mcp_tool = self._tools[server][tool]
        # A full tool call (with id) makes the adapter return a ToolMessage with status.
        call = {"type": "tool_call", "id": f"{server}-{tool}", "name": tool, "args": args}
        future = asyncio.run_coroutine_threadsafe(mcp_tool.ainvoke(call), self._loop)
        return parse_result(future.result(CALL_TIMEOUT))

    def close(self) -> None:
        """Stop both server processes (they exit when their stdin closes)."""
        self._loop.call_soon_threadsafe(self._stop.set)
        self._task.result(CALL_TIMEOUT)
        self._loop.call_soon_threadsafe(self._loop.stop)
