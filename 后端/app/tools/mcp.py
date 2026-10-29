"""MCP 工具加载框架：读取 mcp.json 配置，加载 MCP 服务器工具。

mcp.json 格式（langchain-mcp-adapters >= 0.1 需要 transport 键）:
{
  "mcpServers": {
    "server-name": {
      "transport": "stdio",       // stdio | sse | websocket | http
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/path"],
      "env": {"KEY": "value"}
    },
    "remote-server": {
      "transport": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}

每个 Agent 运行创建一次 MCP 客户端会话，结束后关闭。
加载失败不影响主流程（返回空工具并记录原因）。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from ..config import get_effective

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    path = get_effective("mcp_config_path", "")
    if not path or not os.path.exists(path):
        return {"mcpServers": {}}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("MCP 配置解析失败: %s", e)
        return {"mcpServers": {}}


def _normalize_servers(servers: dict[str, Any]) -> dict[str, Any]:
    """旧格式（无 transport）自动补 stdio；新格式原样传递。"""
    out = {}
    for name, cfg in servers.items():
        if isinstance(cfg, dict) and "transport" not in cfg:
            cfg = {**cfg, "transport": "stdio"}
        out[name] = cfg
    return out


async def load_mcp_tools(workspace: str, emit) -> tuple[list[BaseTool], list[str]]:
    """加载 MCP 工具。返回 (tools, errors)。"""
    config = _load_config()
    servers = _normalize_servers(config.get("mcpServers", {}))
    if not servers:
        return [], []

    tools: list[BaseTool] = []
    errors: list[str] = []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        errors.append(f"MCP 依赖未安装: {e}")
        return [], errors

    client = MultiServerMCPClient(servers, tool_name_prefix=True)
    try:
        # 新版 langchain-mcp-adapters 已移除上下文管理器（工具每次调用自建 session），
        # 不能使用 async with，否则直接抛 NotImplementedError。
        tools = await client.get_tools()
    except Exception as e:
        errors.append(f"连接 MCP 服务器失败: {e}")
        return [], errors

    # 去重：不同服务器可能暴露同名工具；重名后加序号，避免 name_to_tool 静默覆盖
    seen: set[str] = set()
    unique = []
    for t in tools:
        name = t.name
        counter = 2
        while name in seen:
            name = f"{t.name}_{counter}"
            counter += 1
        if name != t.name:
            try:
                t = t.model_copy(update={"name": name})
            except Exception:
                continue
        seen.add(name)
        unique.append(t)
    tools = unique

    if tools:
        try:
            await emit({"type": "status", "status": "mcp_loaded", "detail": f"已加载 MCP 工具: {[t.name for t in tools]}"})
        except Exception:
            pass
    return tools, errors
