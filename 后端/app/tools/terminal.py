"""终端命令执行工具：在工作区内异步执行 shell 命令。

- Windows 默认 PowerShell；POSIX 用 bash
- 高危命令识别（大小写不敏感）→ 触发审批；杀进程族命令硬拦截
- 前台命令支持流式输出（tool_output 事件）与停止/超时杀进程树
- 支持后台运行长驻命令（dev server 等），配合 process 工具查看日志/停止/列清单
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from typing import Any

from langchain_core.tools import tool

from ..config import get_effective
from ..services.paths import PathOutsideWorkspace, is_within, resolve_workspace_path
from .context import ToolContext
from .registry import RISK_HIGH, RISK_MEDIUM, tool_spec

# 高危命令模式（触发审批，无论模式；all 模式全部命令都审批）
# 注意：is_high_risk 会对命令做 lower()，这里的模式也统一按小写匹配；
# 为防未来误加大小写，匹配时统一 re.IGNORECASE。
HIGH_RISK_PATTERNS = [
    r"\brm\s+-[a-z]*r[a-z]*f",
    r"\bdel\s+/[a-z]*[sfq]",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdrop\s+database",
    r"\bdrop\s+table",
    r"\bgit\s+push\b",
    r"\bgit\s+reset\s+--hard",
    r"\bgit\s+clean\b",
    r"\bgit\s+restore\b",
    r"\bgit\s+checkout\s+--\b",
    r"\bkill\b.*-9",
    r"\bshutdown\b",
    r"\brestart\b",
    r"remove-item\b.*-recurse",
    r"\brmdir\s+/[a-z]*s",
    r"reg\s+delete",
    r"diskpart",
    r"\btaskkill\b",
    r"stop-process\b",
    r"\bchmod\s+-r\s+777\b",
    r"\bchown\s+-r\b",
    r"\bsudo\b",
    r"\bcurl\b.*\|\s*(?:ba)?sh",
    r"\bwget\b.*\|\s*(?:ba)?sh",
]

# 硬拦截模式：杀整个进程族（taskkill /F /IM node.exe 会杀掉所有 node 进程）
_PROCESS_FAMILY_KILL_PATTERNS = [
    re.compile(r"taskkill\s+(?:/\w+\s+)*?/IM\s+\S+", re.IGNORECASE),
    re.compile(r"Stop-Process\s+(?:-\w+\s+)*?-Name\s+\S+", re.IGNORECASE),
    re.compile(r"pkill\s+-\w*f", re.IGNORECASE),
    re.compile(r"killall\b", re.IGNORECASE),
]

LOG_DIR_NAME = ".agent_logs"
# taskkill 在中文 Windows 输出 GBK："错误: 没有找到进程 "xxx""
_NOT_FOUND_RE = re.compile(r"没有找到进程|找不到进程|not found", re.IGNORECASE)

# 本次后端进程内启动的后台进程对象：用于自然退出时回收，以及 stop 时避免
# 仅凭可能被复用的 PID 去杀无关进程（跨重启的旧 PID 仍需回退到 taskkill）
_BG_PROCS: dict[str, asyncio.subprocess.Process] = {}
_BG_REAPERS: set[asyncio.Task] = set()


def _decode_output(data: bytes) -> str:
    """按 UTF-8 → GBK → 系统编码的顺序解码命令输出。"""
    if not data:
        return ""
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    try:
        import locale

        return data.decode(locale.getpreferredencoding(False), errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


def _log_dir(ctx: ToolContext) -> str:
    path = os.path.join(ctx.workspace, LOG_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _index_path(ctx: ToolContext) -> str:
    return os.path.join(_log_dir(ctx), "index.json")


def _normalize_index(raw: Any) -> dict[str, dict[str, Any]]:
    """兼容旧版 pid -> 日志文件名 与新版 pid -> {log, command, ...}。"""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for pid, value in raw.items():
        if isinstance(value, str):
            out[str(pid)] = {"log": value}
        elif isinstance(value, dict):
            out[str(pid)] = value
    return out


def _index_load(ctx: ToolContext) -> dict[str, dict[str, Any]]:
    """pid -> {log, command, cwd, started_at} 的映射。"""
    try:
        with open(_index_path(ctx), encoding="utf-8") as f:
            return _normalize_index(json.load(f))
    except (OSError, ValueError):
        return {}


def _index_save(ctx: ToolContext, index: dict[str, dict[str, Any]]) -> None:
    try:
        with open(_index_path(ctx), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
    except OSError:
        pass


def _tail_text(path: str, max_lines: int = 40, max_bytes: int = 16000) -> str:
    """读取日志文件尾部若干行，用于后台启动预览和 process(log) 查看。"""
    try:
        with open(path, "rb") as f:
            data = f.read()[-max_bytes:]
    except OSError:
        return "(暂无输出)"
    text = _decode_output(data)
    return "\n".join(text.splitlines()[-max_lines:])


def _tail_text_lines(path: str, max_lines: int = 40, max_bytes: int = 16000) -> str:
    return _tail_text(path, max_lines=max_lines, max_bytes=max_bytes)


def is_high_risk(command: str) -> bool:
    """高危命令检测（大小写不敏感，修复原实现 PowerShell 规则永不命中的问题）。"""
    lowered = command.lower()
    return any(re.search(p, lowered, re.IGNORECASE) for p in HIGH_RISK_PATTERNS)


def _shell() -> tuple[str, list[str]]:
    if sys.platform == "win32":
        return "powershell", ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command"]
    return "bash", ["-lc"]


def _limit() -> int:
    try:
        return max(1000, int(get_effective("tool_output_max_chars", 12000)))
    except Exception:
        return 12000


def _truncate(text: str, limit: int | None = None) -> str:
    limit = limit or _limit()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n... [输出已截断，共 {len(text)} 字符] ...\n\n{tail}"


def _env(ctx: ToolContext) -> dict[str, str]:
    return {
        **os.environ,
        **ctx.env,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """强制终止进程及其全部子进程。"""
    if sys.platform == "win32":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/T",
                "/F",
                "/PID",
                str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=10)
        except Exception:
            proc.kill()
    else:
        proc.kill()


async def _wait_exit(proc: asyncio.subprocess.Process, timeout: float = 5.0) -> None:
    """带上限等待进程退出（规避 Windows Proactor 的 wait 挂起问题）。"""
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except TimeoutError:
        pass


async def _reap_background(ctx: ToolContext, proc: asyncio.subprocess.Process) -> None:
    """回收后台进程句柄，并在索引里标记已退出（日志仍可按 PID 查看）。"""
    pid = str(proc.pid)
    try:
        await proc.wait()
    except Exception:
        pass
    finally:
        _BG_PROCS.pop(pid, None)
    try:
        index = _index_load(ctx)
        meta = index.get(pid)
        if meta is not None and "exited_at" not in meta:
            meta["exited_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta["exit_code"] = proc.returncode
            _index_save(ctx, index)
    except Exception:
        pass


async def _run_background(
    ctx: ToolContext,
    shell: str,
    shell_args: list[str],
    command: str,
    resolved_cwd: str,
) -> str:
    """后台启动长驻命令：分离进程 + 日志重定向，立即返回 PID 与日志路径。"""
    log_dir = _log_dir(ctx)
    log_path = os.path.join(log_dir, f"bg-{int(time.time() * 1000)}.log")

    creation_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        creation_kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        creation_kwargs["start_new_session"] = True

    log_file = open(log_path, "ab")
    try:
        proc = await asyncio.create_subprocess_exec(
            shell,
            *shell_args,
            command,
            cwd=resolved_cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=_env(ctx),
            **creation_kwargs,
        )
    finally:
        log_file.close()

    try:
        await asyncio.sleep(2.5)
    except asyncio.CancelledError:
        # 启动等待期间被停止/取消：不能把刚拉起的进程留成孤儿
        await _kill_process_tree(proc)
        await _wait_exit(proc)
        raise
    if proc.returncode is not None:
        return (
            f"后台命令启动后立即退出（退出码 {proc.returncode}）。\n"
            f"日志: {log_path}\n\n"
            f"[启动日志]\n{_tail_text(log_path)}"
        )

    index = _index_load(ctx)
    index[str(proc.pid)] = {
        "log": os.path.basename(log_path),
        "command": command[:500],
        "cwd": resolved_cwd,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _index_save(ctx, index)
    _BG_PROCS[str(proc.pid)] = proc
    reaper = asyncio.create_task(_reap_background(ctx, proc))
    _BG_REAPERS.add(reaper)
    reaper.add_done_callback(_BG_REAPERS.discard)

    return (
        f"已在后台启动，PID: {proc.pid}\n"
        f"日志: {log_path}\n"
        f'查看日志: process(action="log", pid={proc.pid})；'
        f'停止: process(action="stop", pid={proc.pid})\n\n'
        f"[启动日志]\n{_tail_text(log_path)}"
    )


async def _run_foreground(
    ctx: ToolContext,
    shell: str,
    shell_args: list[str],
    command: str,
    resolved_cwd: str,
    timeout: int,
) -> str:
    """前台执行：流式读取输出，支持用户停止与超时杀进程树。"""
    started = time.time()
    proc = await asyncio.create_subprocess_exec(
        shell,
        *shell_args,
        command,
        cwd=resolved_cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=_env(ctx),
    )
    assert proc.stdout is not None

    buffer: deque[str] = deque()
    buffer_chars = 0
    buffer_dropped = False
    # 缓冲只保留末尾约 2MB：结果本身会截断到 tool_output_max_chars，
    # 无上限累积只会让长输出命令占满内存
    buffer_limit = 2_000_000
    pending: list[str] = []
    last_emit = 0.0

    async def reader() -> None:
        nonlocal last_emit, buffer_chars, buffer_dropped
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = _decode_output(chunk)
            buffer.append(text)
            buffer_chars += len(text)
            while buffer_chars > buffer_limit and len(buffer) > 1:
                buffer_chars -= len(buffer.popleft())
                buffer_dropped = True
            pending.append(text)
            now = time.monotonic()
            # 限流：约每 120ms 或累计 2KB 推一次，避免事件刷屏
            if now - last_emit >= 0.12 or sum(len(x) for x in pending) >= 2048:
                await ctx.emit_output("".join(pending))
                pending.clear()
                last_emit = now

    reader_task = asyncio.create_task(reader())
    stop_task = asyncio.create_task(ctx.stop_event.wait())
    proc_task = asyncio.create_task(proc.wait())
    stopped = False
    timed_out = False
    try:
        done, _ = await asyncio.wait(
            {proc_task, stop_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if proc_task not in done:
            if stop_task in done:
                stopped = True
            else:
                timed_out = True
            await _kill_process_tree(proc)
            await _wait_exit(proc)
    finally:
        stop_task.cancel()
        # 外层取消（停止/超时打断本协程）时也必须杀掉子进程，避免孤儿进程
        if proc.returncode is None:
            await _kill_process_tree(proc)
            await _wait_exit(proc)
        try:
            await asyncio.wait_for(reader_task, timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            reader_task.cancel()

    if pending:
        await ctx.emit_output("".join(pending))
    combined = "".join(buffer).strip()
    if buffer_dropped:
        combined = "…（输出过长，较早内容已省略）…\n" + combined
    duration_ms = int((time.time() - started) * 1000)

    if stopped:
        return f"命令已按用户请求停止（耗时 {duration_ms}ms）。\n{_truncate(combined) if combined else '(无输出)'}"
    if timed_out:
        return (
            f"命令超时（>{timeout}s）已被终止。\n"
            f"{_truncate(combined) if combined else '(无输出)'}"
        )
    header = f"退出码: {proc.returncode} | 耗时: {duration_ms}ms | 工作目录: {resolved_cwd}"
    body = _truncate(combined) if combined else "(无输出)"
    return f"{header}\n{body}"


def make_process_tool(ctx: ToolContext):
    """闭包创建 process 工具：查看/停止/列出 terminal 后台启动的进程。"""

    @tool_spec(category="terminal", risk_level=RISK_MEDIUM, timeout=30)
    @tool(parse_docstring=True)
    async def process(action: str, pid: int = 0, lines: int = 100) -> str:
        """
        查看、列出或停止由 terminal 工具后台启动的进程

        Args:
            action: "log" 查看后台进程日志；"stop" 停止后台进程（连同子进程）；"list" 列出本工作区后台进程
            pid: 后台进程的 PID（log/stop 模式需要）
            lines: log 模式下返回的最后行数（1-500，默认 100）

        Returns:
            log 模式返回日志内容；stop 模式返回停止结果；list 模式返回进程清单
        """
        ctx.check_stopped()
        pid = int(pid)
        lines = max(1, min(int(lines), 500))
        index = _index_load(ctx)

        if action == "list":
            if not index:
                return "当前工作区没有由 Agent 启动的后台进程。"
            rows = []
            for key, meta in index.items():
                state_note = (
                    f" | 已退出（退出码 {meta.get('exit_code')}" f"，{meta.get('exited_at')}）"
                    if meta.get("exited_at")
                    else ""
                )
                rows.append(
                    f"PID {key} | 启动 {meta.get('started_at', '?')}{state_note} | "
                    f"{meta.get('command', '?')[:160]} | 日志 {meta.get('log', '?')}"
                )
            return "后台进程:\n" + "\n".join(rows)

        if action == "log":
            if pid <= 0:
                return "错误：请提供 pid（terminal 后台模式返回的 PID）。"
            log_dir = _log_dir(ctx)
            meta = index.get(str(pid), {})
            filename = os.path.basename(str(meta.get("log") or f"pid-{pid}.log"))
            log_path = os.path.join(log_dir, filename)
            if not is_within(log_dir, log_path):
                return f"拒绝读取 PID {pid} 的日志：日志路径不在本工作区日志目录内。"
            if not os.path.isfile(log_path):
                return f"未找到 PID {pid} 的日志（{log_path}）。进程可能已被停止或 PID 不正确。"
            return f"[PID {pid} 日志最后 {lines} 行]\n{_tail_text_lines(log_path, max_lines=lines)}"

        if action == "stop":
            if pid <= 0:
                return "错误：请提供 pid。"
            if str(pid) not in index:
                return "错误：该 PID 不是由 Agent 在本工作区启动的后台进程，已拒绝停止。"
            live = _BG_PROCS.get(str(pid))
            if live is not None:
                if live.returncode is not None:
                    index.pop(str(pid), None)
                    _index_save(ctx, index)
                    return f"进程 {pid} 已退出（退出码 {live.returncode}），已从清单移除。"
                await _kill_process_tree(live)
                await _wait_exit(live)
                index.pop(str(pid), None)
                _index_save(ctx, index)
                return f"已停止进程树 PID {pid}。"
            if sys.platform == "win32":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/T",
                    "/F",
                    "/PID",
                    str(pid),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                _, err_raw = await asyncio.wait_for(killer.communicate(), timeout=15)
                if killer.returncode != 0:
                    err_text = _decode_output(err_raw).strip()
                    if _NOT_FOUND_RE.search(err_text):
                        index.pop(str(pid), None)
                        _index_save(ctx, index)
                        return f"进程 {pid} 不存在（可能已退出），视为已停止。"
                    return f"停止失败（退出码 {killer.returncode}）: {err_text}"
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    index.pop(str(pid), None)
                    _index_save(ctx, index)
                    return f"进程 {pid} 不存在（可能已退出），视为已停止。"
                except PermissionError as e:
                    return f"停止失败: {e}"
            index.pop(str(pid), None)
            _index_save(ctx, index)
            return f"已停止进程树 PID {pid}。"

        return f"未知 action: {action}（支持 log / stop / list）"

    return process


def make_terminal_tool(ctx: ToolContext):
    """闭包创建 terminal 工具（绑定当前运行的上下文）。"""

    @tool_spec(category="terminal", risk_level=RISK_HIGH, timeout=0)
    @tool(parse_docstring=True)
    async def terminal(
        command: str,
        cwd: str = "",
        timeout: int = 120,
        background: bool = False,
    ) -> str:
        """
        在工作区内执行 shell 命令

        Args:
            command: 要执行的完整命令字符串，例如 "python --version" 或 "git status"
            cwd: 可选，相对工作区的子目录路径，不传则在工作区根目录执行；禁止越出工作区
            timeout: 可选，前台模式下的命令超时秒数，超过会被强制终止
            background: 是否后台运行。dev server、watch 等长驻命令必须传 true，
                之后用 process 工具查看日志（action="log"）或停止（action="stop"）

        Returns:
            前台模式返回命令输出（含退出码、耗时和工作目录）；后台模式返回 PID、日志路径和启动日志预览
        """
        ctx.check_stopped()

        for pat in _PROCESS_FAMILY_KILL_PATTERNS:
            if pat.search(command):
                return (
                    f"【命令被拒绝】禁止执行 `{command.strip()}`。"
                    f"你试图杀掉整个进程族（/IM 或 -Name），这会终止机器上所有同名进程。"
                    f"如需停止由本 Agent 启动的特定后台服务，请用 process 工具的 action='stop'；"
                    f"如需释放特定端口，请先用 netstat 找到 PID，再用 taskkill /F /PID <pid> 只杀那个 PID。"
                )

        # 工作区边界校验：防止 cwd=".." 或绝对路径逃逸
        try:
            resolved_cwd = str(resolve_workspace_path(ctx.workspace, cwd))
        except (PathOutsideWorkspace, ValueError, OSError) as e:
            return f"错误：工作目录不合法（{e}）"
        if not os.path.isdir(resolved_cwd):
            return f"错误：目录不存在: {resolved_cwd}"

        risk = "high_risk" if is_high_risk(command) else "normal"
        approved = await ctx.approvals.request(
            ctx.emit_event,
            "terminal",
            risk,
            {"command": command, "cwd": resolved_cwd},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了该命令的执行。请换一种方式实现目标（例如用文件编辑工具修改代码，而不是运行该命令）。"

        shell, shell_args = _shell()
        if background:
            return await _run_background(ctx, shell, shell_args, command, resolved_cwd)

        try:
            max_timeout = int(get_effective("terminal_timeout_seconds", 300))
        except Exception:
            max_timeout = 300
        timeout = max(1, min(int(timeout), max_timeout))
        try:
            return await _run_foreground(ctx, shell, shell_args, command, resolved_cwd, timeout)
        except Exception as e:
            return f"命令执行失败: {e}"

    return terminal
