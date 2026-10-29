"""工作区级上下文：项目规则文件 + 长期记忆，注入系统提示。"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import get_effective
from .file_io import read_document
from .paths import is_within

logger = logging.getLogger(__name__)


def _read_limited(path: Path, limit: int) -> str:
    try:
        if not path.is_file():
            return ""
        doc = read_document(path)
        return doc.text.strip()[:limit]
    except Exception:
        return ""


def load_project_rules(workspace: str, max_chars: int = 8000) -> str:
    """读取 AGENTS.md / .agent/rules.md / CLAUDE.md 等规则文件。"""
    if not workspace:
        return ""
    raw = str(get_effective("project_rules_files", "") or "")
    names = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    parts: list[str] = []
    used = 0
    for name in names:
        p = Path(workspace) / name
        try:
            if not is_within(workspace, p):
                continue
        except Exception:
            continue
        text = _read_limited(p, max_chars - used)
        if not text:
            continue
        parts.append(f"### {name}\n{text}")
        used += len(text)
        if used >= max_chars:
            break
    if not parts:
        return ""
    return "【项目规则（必须遵守）】\n" + "\n\n".join(parts)


def load_memory_text(workspace: str, max_chars: int = 6000) -> str:
    """读取 .agent/memory.md 的长期记忆。"""
    if not workspace:
        return ""
    text = _read_limited(Path(workspace) / ".agent" / "memory.md", max_chars)
    return f"【项目长期记忆】\n{text}" if text else ""


def workspace_context(workspace: str) -> str:
    chunks = [load_project_rules(workspace), load_memory_text(workspace)]
    return "\n\n".join(x for x in chunks if x)
