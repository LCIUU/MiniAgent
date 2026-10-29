"""测试/构建执行工具：检测项目类型并运行对应命令。"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys

from langchain_core.tools import tool

from ..config import get_effective
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec
from .runtime_tools import _python_exe

MAX_OUTPUT = 8000


def _detect_project(workspace: str) -> str:
    """根据根目录文件判断项目类型（多语言项目按优先级）。"""
    try:
        files = set(os.listdir(workspace))
    except OSError:
        return "unknown"
    if "pyproject.toml" in files or "requirements.txt" in files or "setup.py" in files:
        return "python"
    if "package.json" in files:
        return "node"
    if "go.mod" in files:
        return "go"
    if "Cargo.toml" in files:
        return "rust"
    if "pom.xml" in files or "build.gradle" in files or "build.gradle.kts" in files:
        return "java"
    return "unknown"


def _node_pm(workspace: str) -> str:
    """检测 Node 包管理器（有锁文件优先）。"""
    if os.path.exists(os.path.join(workspace, "pnpm-lock.yaml")):
        return "pnpm"
    if os.path.exists(os.path.join(workspace, "yarn.lock")):
        return "yarn"
    if os.path.exists(os.path.join(workspace, "bun.lockb")):
        return "bun"
    return "npm"


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    if sys.platform == "win32":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/T", "/F", "/PID", str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=10)
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    """保留头尾：失败摘要通常在尾部，只截头会丢掉关键报错。"""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n... [输出已截断，共 {len(text)} 字符，以下为末尾] ...\n\n{tail}"


def _commands(workspace: str) -> dict[str, dict[str, str]]:
    # 用目标工作区自己的虚拟环境跑测试，而不是后端进程的 Python
    py = _python_exe(workspace)
    if " " in py:
        py = f'"{py}"'
    pm = _node_pm(workspace)
    node_test = "npm test" if pm == "npm" else f"{pm} test"
    node_build = f"{pm} run build"
    node_lint = f"{pm} run lint"
    return {
        "python": {
            "test": f"{py} -m pytest -x -q",
            "build": f"{py} -m compileall . -q",
            "lint": f"{py} -m ruff check .",
        },
        "node": {"test": node_test, "build": node_build, "lint": node_lint},
        "go": {"test": "go test ./...", "build": "go build ./...", "lint": "go vet ./..."},
        "rust": {"test": "cargo test", "build": "cargo build", "lint": "cargo clippy"},
        "java": {"test": "mvn test", "build": "mvn compile", "lint": "mvn checkstyle:check"},
        "unknown": {"test": "", "build": "", "lint": ""},
    }


def _shell() -> list[str]:
    if sys.platform == "win32":
        return ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command"]
    return ["bash", "-lc"]


def make_build_test_tools(ctx: ToolContext):

    @tool_spec(category="build", risk_level=RISK_MEDIUM, timeout=660)
    @tool(parse_docstring=True)
    async def run_checks(
        kind: str = "test",
    ) -> str:
        """
        运行项目的测试、构建或代码检查

        Args:
            kind: 检查类型，可选值 "test"（运行测试）、"build"（构建项目）、"lint"（代码风格检查）

        Returns:
            返回命令执行的退出码和输出结果
        """
        ctx.check_stopped()
        if kind not in ("test", "build", "lint"):
            return f"未知 kind: {kind}（支持 test / build / lint）"
        project = _detect_project(ctx.workspace)
        cmd = _commands(ctx.workspace).get(project, {}).get(kind, "")
        if not cmd:
            return f"无法识别项目类型（{project}），请使用 terminal 工具手动运行测试/构建命令。"
        # 命令是否存在，提前给出可读错误
        first = cmd.split()[0].strip('"')
        if not (os.path.isabs(first) or shutil.which(first)):
            return f"未找到命令 `{first}`，请确认依赖已安装。项目类型: {project}"

        await ctx.emit_event({"type": "status", "status": "executing", "detail": f"运行 {kind}: {cmd}"})
        try:
            timeout = min(int(get_effective("terminal_timeout_seconds", 300)) * 2, 600)
        except Exception:
            timeout = 600
        try:
            proc = await asyncio.create_subprocess_exec(
                *_shell(), cmd, cwd=ctx.workspace,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                await _kill_tree(proc)
                return f"[{kind}] 超时（{timeout}s）已终止。"
            except asyncio.CancelledError:
                # 用户停止/外层取消：回收子进程，避免测试进程变孤儿
                await _kill_tree(proc)
                raise
            out = _decode(stdout).strip()
            err = _decode(stderr).strip()
            combined = out + ("\n[stderr]\n" + err if err and err not in out else "")
            combined = _clip(combined)
            if proc.returncode != 0:
                # 让模型更容易注意到失败
                combined = f"【失败】\n{combined}"
            return f"[{kind}] 退出码 {proc.returncode}\n{combined}" if combined else f"[{kind}] 退出码 {proc.returncode}（无输出）"
        except Exception as e:
            return f"[{kind}] 执行失败: {e}"

    return run_checks
