"""代码检索工具：纯文件系统搜索（grep + glob），无需 embedding。

优先尝试 ripgrep（rg），不可用时回退到 Python 实现。
所有搜索限定在工作区内，并对 rg 子进程加超时，避免病态正则挂住 Agent。
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re

from langchain_core.tools import tool

from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import tool_spec

IGNORED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next",
    ".idea", ".vscode", "target", ".tox", ".mypy_cache", ".pytest_cache", ".agent_logs",
    ".agent_uploads", ".agent_backups",
}
MAX_RESULTS = 200


def _has_rg() -> bool:
    try:
        import shutil

        return shutil.which("rg") is not None
    except Exception:
        return False


def _resolve(ctx: ToolContext, path: str):
    return resolve_workspace_path(ctx.workspace, path or ".")


def make_search_tools(ctx: ToolContext):
    rg_available = _has_rg()

    @tool_spec(category="search", idempotent=True, cacheable=False, timeout=60)
    @tool(parse_docstring=True)
    async def grep_search(
        pattern: str,
        include: str = "",
        path: str = "",
        max_results: int = MAX_RESULTS,
    ) -> str:
        """
        在项目代码中搜索文本或正则表达式

        Args:
            pattern: 要搜索的正则表达式，例如函数名或错误信息关键词
            include: 可选，文件类型过滤 glob，例如 "*.py" 只搜索 Python 文件
            path: 可选，相对工作区的搜索目录/文件，不传则搜索整个工作区
            max_results: 可选，最多返回的匹配条数（默认 200，上限 1000）

        Returns:
            返回匹配的文件路径、行号和内容
        """
        ctx.check_stopped()
        try:
            target = _resolve(ctx, path)
        except (PathOutsideWorkspace, ValueError, OSError) as e:
            return str(e)
        if not target.exists():
            return f"错误：路径不存在: {target}"
        limit = max(1, min(int(max_results), 1000))

        if rg_available:
            cmd = ["rg", "--line-number", "--no-heading", "--color", "never", "--max-count", "50"]
            if include:
                cmd += ["--glob", include]
            for d in IGNORED_DIRS:
                cmd += ["--glob", f"!{d}/**"]
            cmd += ["-e", pattern, str(target)]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=45)
            except TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return "grep 超时（45s），请缩小 path 范围或简化正则。"
            except OSError as e:
                return f"ripgrep 启动失败: {e}"
            out = stdout.decode("utf-8", errors="replace")
            lines = [line for line in out.splitlines() if line.strip()]
            if not lines:
                return "没有找到匹配结果。"
            total = len(lines)
            body = "\n".join(lines[:limit])
            return f"找到 {total} 处匹配（显示前 {min(total, limit)} 处）:\n{body}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"正则错误: {e}"

        matches: list[str] = []
        if target.is_file():
            files = [str(target)]
        else:
            files = []
            for root, dirs, names in os.walk(target):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".agent_")]
                for name in names:
                    files.append(os.path.join(root, name))
                    if len(files) >= 20000:
                        break
                if len(files) >= 20000:
                    break
        for fpath in files:
            if include and not fnmatch.fnmatch(os.path.basename(fpath), include):
                continue
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = os.path.relpath(fpath, ctx.workspace)
                            matches.append(f"{rel}:{lineno}:{line.rstrip()[:200]}")
                            if len(matches) >= limit:
                                break
            except (OSError, PermissionError):
                continue
            if len(matches) >= limit:
                break

        if not matches:
            return "没有找到匹配结果。"
        return f"找到 {len(matches)} 处匹配（最多显示 {limit} 处）:\n" + "\n".join(matches)

    @tool_spec(category="search", idempotent=True, cacheable=False, timeout=60)
    @tool(parse_docstring=True)
    async def list_files(
        pattern: str = "**/*",
        path: str = "",
    ) -> str:
        """
        列出工作区内的文件

        Args:
            pattern: glob 匹配模式，例如 "**/*.py" 列出所有 Python 文件，"src/**" 列出 src 目录下的文件
            path: 可选，相对工作区的起始目录，不传则从工作区根目录开始

        Returns:
            返回匹配的文件相对路径列表
        """
        ctx.check_stopped()
        try:
            base = _resolve(ctx, path)
        except (PathOutsideWorkspace, ValueError, OSError) as e:
            return str(e)
        if base.is_file():
            return os.path.relpath(str(base), ctx.workspace)
        results: list[str] = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".agent_")]
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, ctx.workspace)
                if fnmatch.fnmatch(rel.replace(os.sep, "/"), pattern) or fnmatch.fnmatch(fname, pattern.split("/")[-1] if "/" in pattern else pattern):
                    results.append(rel)
                    if len(results) >= 500:
                        break
            if len(results) >= 500:
                break
        results.sort()
        if not results:
            return "没有匹配的文件。"
        return f"找到 {len(results)} 个文件（最多显示 500 个）:\n" + "\n".join(results)

    return grep_search, list_files
