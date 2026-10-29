"""环境变量检查工具：只读查看项目相关环境变量。

帮助 Agent 快速排查配置问题（如 DATABASE_URL 是否设置、API Key 是否可用等），
不暴露完整密钥值，只显示是否存在和前几位字符。
"""

from __future__ import annotations

import os
import re

from langchain_core.tools import tool

from .context import ToolContext
from .registry import tool_spec

_SENSITIVE_KEYS = re.compile(r"(key|secret|token|password|passwd|pwd|auth)", re.IGNORECASE)
_CREDENTIAL_URL = re.compile(r"(\b[a-zA-Z][\w+.-]*://[^:/@\s]+:)([^@\s]+)(@)")


def _mask_value(key: str, value: str) -> str:
    if not value:
        return "(未设置)"
    # DATABASE_URL / REDIS_URL 等变量名不含敏感词，但值里可能带 user:pass@
    value = _CREDENTIAL_URL.sub(r"\1***\3", value)
    if _SENSITIVE_KEYS.search(key):
        if len(value) <= 8:
            return "***（已设置，已隐藏）"
        return f"{value[:4]}...{value[-2:]}（长度 {len(value)}）"
    return value


_COMMON_ENV_VARS = [
    "DATABASE_URL", "DB_URL", "DB_DSN",
    "REDIS_URL", "REDIS_HOST", "REDIS_PORT",
    "OPENAI_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY",
    "NODE_ENV", "PYTHON_ENV", "ENV",
    "PORT", "HOST",
    "PATH",
    "PYTHONPATH", "VIRTUAL_ENV",
    "npm_execpath", "npm_lifecycle_event",
]


def make_env_tools(ctx: ToolContext):

    @tool_spec(category="general", idempotent=True, timeout=10)
    @tool(parse_docstring=True)
    async def check_env(variables: str = "") -> str:
        """
        检查环境变量是否已配置（只读，敏感值自动脱敏）

        Args:
            variables: 要检查的环境变量名，逗号或空格分隔；为空时检查常见项目变量

        Returns:
            各变量的值（敏感值自动脱敏）
        """
        ctx.check_stopped()
        names = (
            re.split(r"[,\s]+", variables.strip()) if variables.strip() else _COMMON_ENV_VARS
        )

        lines: list[str] = []
        for name in names:
            name = name.strip()
            if not name:
                continue
            value = os.environ.get(name)
            masked = _mask_value(name, value or "")
            status = "✓" if value else "✗"
            lines.append(f"  {status} {name} = {masked}")

        if not lines:
            return "未找到指定的环境变量。"
        return f"环境变量检查结果（共 {len(lines)} 项）:\n" + "\n".join(lines)

    return check_env
