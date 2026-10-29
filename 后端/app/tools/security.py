"""密钥扫描工具：发现硬编码的 API Key / 密码 / 私钥 / 带凭证的连接串。

只读检查，输出中的密钥一律脱敏（保留少量前缀便于定位，不泄露完整值）。
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from langchain_core.tools import tool

from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import tool_spec

_IGNORED = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".next", ".nuxt", "target", ".tox", ".mypy_cache", ".pytest_cache",
    ".agent_logs", ".agent_uploads", ".agent_backups", "coverage", ".turbo",
}
_MAX_FILE_BYTES = 1024 * 1024
_MAX_FILES = 5000

_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("私钥", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"), "high"),
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "high"),
    ("GitHub Token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "high"),
    ("Slack Token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "high"),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"), "high"),
    ("Anthropic Key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), "high"),
    ("OpenAI/DashScope Key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "high"),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "medium"),
    ("Bearer Token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._\-]{20,})"), "medium"),
    (
        "带凭证的连接串",
        re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|mssql)://[^\s:@/'\"]+:[^\s@/'\"]+@"),
        "high",
    ),
    (
        "疑似硬编码凭据",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|passwd|password|access[_-]?key)\b"
            r"\s*[:=]\s*(?:['\"]([^'\"]{8,})['\"]|([A-Za-z0-9_\-./+=]{12,}))"
        ),
        "medium",
    ),
]
_PLACEHOLDERS = (
    "your-", "your_", "xxx", "***", "placeholder", "changeme", "example",
    "dummy", "test", "process.env", "os.environ", "${", "<", "{{",
)


def _rel(workspace: str, p: Path) -> str:
    try:
        return os.path.relpath(str(p), workspace).replace("\\", "/")
    except ValueError:
        return str(p)


def _mask(value: str) -> str:
    value = value.strip()
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}***{value[-2:]}"


def _is_placeholder(token: str) -> bool:
    low = token.lower()
    return any(marker in low for marker in _PLACEHOLDERS)


def _match_value(m: re.Match) -> str:
    """取第一个非空捕获组作为命中值（无捕获组时用整段匹配）。"""
    if m.lastindex:
        for idx in range(1, m.lastindex + 1):
            value = m.group(idx)
            if value:
                return value
    return m.group(0)


def _candidates(workspace: str, base: Path, include: str) -> list[Path]:
    if base.is_file():
        return [base]
    files: list[Path] = []
    for root, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _IGNORED and not d.startswith(".agent_")]
        for name in names:
            p = Path(root) / name
            rel = _rel(workspace, p)
            if include and not (
                fnmatch.fnmatch(rel, include) or fnmatch.fnmatch(name, include)
            ):
                continue
            try:
                if p.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(p)
            if len(files) >= _MAX_FILES:
                return files
    return files


def make_security_tool(ctx: ToolContext):

    @tool_spec(category="search", idempotent=True, cacheable=False, timeout=90)
    @tool(parse_docstring=True)
    async def secret_scan(
        path: str = "",
        include: str = "",
        max_results: int = 100,
    ) -> str:
        """
        扫描工作区中的硬编码密钥、密码、私钥与带凭证的连接串（只读，结果脱敏）

        Args:
            path: 可选，扫描起始目录/文件，相对工作区，不传则整个工作区
            include: 可选，文件过滤 glob，例如 "*.py"、".env*"
            max_results: 可选，最多返回的命中条数（默认 100）

        Returns:
            按严重程度分组的命中清单（文件:行 + 脱敏片段 + 规则名）
        """
        ctx.check_stopped()
        try:
            base = resolve_workspace_path(ctx.workspace, path or ".")
        except (PathOutsideWorkspace, ValueError) as e:
            return f"错误：{e}"
        if not base.exists():
            return f"错误：路径不存在: {path}"

        limit = max(1, min(int(max_results), 500))
        candidates = _candidates(ctx.workspace, base, include)
        hits: list[tuple[str, str, int, str, str]] = []  # (severity, rule, line, rel, masked)
        scanned = 0
        for p in candidates:
            ctx.check_stopped()
            try:
                data = p.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                continue
            scanned += 1
            text = data.decode("utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                for rule, pattern, severity in _PATTERNS:
                    m = pattern.search(line)
                    if not m:
                        continue
                    token = _match_value(m)
                    if _is_placeholder(token):
                        continue
                    hits.append((severity, rule, lineno, _rel(ctx.workspace, p), _mask(token)))
                    break  # 同一行只记第一条规则，避免 sk- 同时命中通用规则重复上报
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break

        if not hits:
            return f"已扫描 {scanned} 个文件：未发现硬编码密钥（常见占位符已自动忽略）。"

        high = [h for h in hits if h[0] == "high"]
        medium = [h for h in hits if h[0] == "medium"]
        lines = [f"已扫描 {scanned} 个文件，发现 {len(hits)} 处疑似敏感信息："]
        for title, items in (("高危", high), ("中危", medium)):
            if not items:
                continue
            lines.append(f"\n{title}（{len(items)} 处）:")
            for _, rule, lineno, rel, masked in items:
                lines.append(f"  {rel}:{lineno}  [{rule}]  {masked}")
        if len(hits) >= limit:
            lines.append(f"\n已达返回上限 {limit} 条，可用 path/include 缩小范围后继续。")
        lines.append("\n建议：确认后把密钥移入 .env（并确保 .env 在 .gitignore 中），代码里改读环境变量。")
        return "\n".join(lines)

    return secret_scan
