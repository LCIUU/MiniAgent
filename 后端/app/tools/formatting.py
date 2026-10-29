"""代码格式化工具：按项目类型选择 ruff / black / prettier / gofmt / cargo fmt。"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

from langchain_core.tools import tool

from ..config import get_effective
from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


async def _run(cmd: list[str], cwd: str, timeout: int = 180) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        return 124, "格式化命令超时。"
    return proc.returncode or 0, _decode(out).strip()


def _has_python_tool(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


_PRETTIER_SUFFIXES = {
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
    ".json", ".jsonc", ".css", ".scss", ".less", ".html", ".vue", ".svelte",
    ".yaml", ".yml", ".md", ".mdx",
}
_LOCKFILES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb", "composer.lock"}


def make_format_tool(ctx: ToolContext):

    @tool_spec(category="build", risk_level=RISK_MEDIUM, timeout=240)
    @tool(parse_docstring=True)
    async def format_code(path: str = "") -> str:
        """
        用项目现有的格式化工具整理代码

        Args:
            path: 可选，只格式化工作区内的某个文件/目录；不传则格式化整个项目（可能较慢）

        Returns:
            格式化命令的输出与结果
        """
        ctx.check_stopped()
        try:
            target = resolve_workspace_path(ctx.workspace, path) if path else Path(ctx.workspace)
        except (PathOutsideWorkspace, ValueError) as e:
            return str(e)
        rel_target = path or "."
        suffix = target.suffix.lower()
        is_dir = target.is_dir()

        def project_has(*names: str) -> bool:
            return any(os.path.exists(os.path.join(ctx.workspace, n)) for n in names)

        # 按目标类型分派格式化器：先判断语言，再判断工具是否可用，
        # 避免在同时装了 prettier 的多语言仓库里用错工具（如对 .go 跑 prettier）。
        if suffix == ".py" or (is_dir and project_has("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg")):
            if _has_python_tool("ruff"):
                code, out = await _run([sys.executable, "-m", "ruff", "format", str(target)], ctx.workspace)
                return f"[ruff format {rel_target}] 退出码 {code}\n{out or '(无输出)'}"
            if _has_python_tool("black"):
                code, out = await _run([sys.executable, "-m", "black", str(target)], ctx.workspace)
                return f"[black {rel_target}] 退出码 {code}\n{out or '(无输出)'}"
            return "未找到 ruff/black，请先安装：uv add --dev ruff（或 pip install ruff）。"

        if suffix == ".go" or (is_dir and project_has("go.mod")):
            if not shutil.which("gofmt"):
                return "未找到 gofmt，请先安装 Go 工具链。"
            code, out = await _run(["gofmt", "-w", str(target)], ctx.workspace)
            return f"[gofmt {rel_target}] 退出码 {code}\n{out or '(无输出)'}"

        if suffix == ".rs" or (is_dir and project_has("Cargo.toml")):
            if project_has("Cargo.toml") and shutil.which("cargo"):
                code, out = await _run(["cargo", "fmt"], ctx.workspace, timeout=int(get_effective("terminal_timeout_seconds", 300)))
                return f"[cargo fmt] 退出码 {code}\n{out or '(无输出)'}"
            if shutil.which("rustfmt"):
                code, out = await _run(["rustfmt", str(target)], ctx.workspace)
                return f"[rustfmt {rel_target}] 退出码 {code}\n{out or '(无输出)'}"
            return "未找到 cargo 或 rustfmt，请先安装 Rust 工具链。"

        if is_dir and not project_has("package.json"):
            return "无法判断目录所属语言（未找到 pyproject.toml / go.mod / Cargo.toml / package.json）。"
        if not is_dir and suffix not in _PRETTIER_SUFFIXES:
            return f"不支持格式化的文件类型: {suffix or target.name}"

        if target.name.lower() in _LOCKFILES:
            return f"跳过锁文件（不格式化）: {rel_target}"

        # Node：优先本地 prettier，避免 npx 联网安装
        local_prettier = os.path.join(ctx.workspace, "node_modules", ".bin", "prettier")
        if os.name == "nt":
            local_prettier += ".cmd"
        if os.path.exists(local_prettier):
            code, out = await _run([local_prettier, "--write", str(target)], ctx.workspace)
            return f"[prettier {rel_target}] 退出码 {code}\n{out or '(无输出)'}"
        if shutil.which("npx"):
            code, out = await _run(["npx", "--no-install", "prettier", "--write", str(target)], ctx.workspace)
            return f"[npx prettier {rel_target}] 退出码 {code}\n{out or '(无输出)'}"

        return "未找到 prettier，请在项目中安装：npm i -D prettier。"

    return format_code
