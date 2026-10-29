"""执行与诊断工具：跑代码片段、跑测试子集、等本地服务、看端口、查依赖漏洞。

定位是补上 terminal 之外的"短平快"能力：
- run_code 用临时文件执行代码，避免 PowerShell 引号转义地狱
- run_tests 直接调用测试框架，支持只跑某个文件/关键字
- wait_for_server 轮询本地端口，替代 sleep 后盲目 curl
- list_ports 查看端口占用，定位 EADDRINUSE
- audit_dependencies 只读检查已知漏洞，不改 lock 文件
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
import shutil
import sys
import time
import uuid
from urllib.parse import urlparse

from langchain_core.tools import tool

from ..config import get_effective
from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec


def _limit() -> int:
    try:
        return max(1000, int(get_effective("tool_output_max_chars", 12000)))
    except Exception:
        return 12000


def _truncate(text: str, limit: int | None = None) -> str:
    limit = limit or _limit()
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n...[已截断，共 {len(text)} 字符]...\n" + text[-limit // 2 :]


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _env(ctx: ToolContext) -> dict[str, str]:
    return {
        **os.environ,
        **ctx.env,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def _python_exe(workspace: str) -> str:
    for candidate in (
        os.path.join(workspace, ".venv", "Scripts", "python.exe"),
        os.path.join(workspace, ".venv", "bin", "python"),
        os.path.join(workspace, "venv", "Scripts", "python.exe"),
        os.path.join(workspace, "venv", "bin", "python"),
    ):
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("python") or shutil.which("python3")
    return found or sys.executable


def _node_pm(workspace: str) -> str:
    for lock, pm in (
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
        ("bun.lockb", "bun"),
        ("bun.lock", "bun"),
    ):
        if os.path.exists(os.path.join(workspace, lock)):
            return pm
    return "npm"


def _project_kind(workspace: str) -> str:
    try:
        files = set(os.listdir(workspace))
    except OSError:
        return "unknown"
    if "pyproject.toml" in files or "requirements.txt" in files or "setup.py" in files or "tests" in files:
        return "python"
    if "package.json" in files:
        return "node"
    if "go.mod" in files:
        return "go"
    if "Cargo.toml" in files:
        return "rust"
    return "unknown"


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


async def _exec(
    ctx: ToolContext,
    argv: list[str],
    *,
    timeout: int,
    cwd: str | None = None,
) -> tuple[int, str, bool, bool]:
    """执行命令，返回 (退出码, 合并输出, 是否超时, 是否被停止)。"""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd or ctx.workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=_env(ctx),
    )
    wait_task = asyncio.create_task(proc.communicate())
    stop_task: asyncio.Task | None = None
    reaped = False
    try:
        if ctx.stop_event is not None:
            stop_task = asyncio.create_task(ctx.stop_event.wait())
        pending = {wait_task} | ({stop_task} if stop_task else set())
        done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        timed_out = stopped = False
        if wait_task in done:
            raw, _ = wait_task.result()
            reaped = True
        else:
            stopped = stop_task is not None and stop_task in done
            timed_out = not stopped
            await _kill_tree(proc)
            try:
                raw, _ = await asyncio.wait_for(wait_task, timeout=5)
                reaped = True
            except Exception:
                raw = b""
    finally:
        if stop_task is not None:
            stop_task.cancel()
        # 外层工具超时/取消（asyncio.wait_for 取消本协程）时也要回收子进程，
        # 否则 pytest/构建进程会变成孤儿继续运行
        if not reaped:
            await _kill_tree(proc)
            try:
                await asyncio.wait_for(wait_task, timeout=5)
            except Exception:
                wait_task.cancel()

    code = proc.returncode if proc.returncode is not None else -1
    return code, _decode(raw).strip(), timed_out, stopped


def _missing(cmd: str, hint: str) -> str:
    return f"未找到 `{cmd}`，{hint}"


_RUNNERS = {"python": ".py", "py": ".py", "node": ".js", "javascript": ".js", "js": ".js", "bash": ".sh", "sh": ".sh", "shell": ".sh"}


def _port_of(local: str) -> int:
    try:
        return int(local.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 0


async def _collect_listeners(ctx: ToolContext) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """收集本机监听的 TCP 端口：返回 (pid -> 进程名, [(协议, 本地地址, pid)])。"""
    names: dict[str, str] = {}
    entries: list[tuple[str, str, str]] = []
    if sys.platform == "win32":
        if shutil.which("netstat") is None:
            return names, entries
        code, text, _, _ = await _exec(ctx, ["netstat", "-ano", "-p", "tcp"], timeout=20)
        if code != 0:
            return names, entries
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0].upper() != "TCP" or parts[3].upper() != "LISTENING":
                continue
            entries.append(("tcp", parts[1], parts[4]))
        # tasklist 在部分环境只返回少量进程，PowerShell 的 Get-Process 更完整
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if ps:
            code, text, _, _ = await _exec(
                ctx,
                [
                    ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                    "Get-Process | ForEach-Object { \"$($_.Id),$($_.ProcessName)\" }",
                ],
                timeout=25,
            )
            if code == 0:
                for line in text.splitlines():
                    pid, _, name = line.strip().partition(",")
                    if pid.isdigit():
                        names[pid] = name
        elif shutil.which("tasklist"):
            code, text, _, _ = await _exec(
                ctx, ["tasklist", "/FO", "CSV", "/NH"], timeout=20
            )
            if code == 0:
                for row in csv.reader(io.StringIO(text)):
                    if len(row) >= 2:
                        names[row[1].strip()] = row[0].strip()
    else:
        ss = shutil.which("ss")
        if ss:
            code, text, _, _ = await _exec(ctx, [ss, "-ltnp"], timeout=20)
            if code == 0:
                for line in text.splitlines()[1:]:
                    parts = line.split()
                    if len(parts) < 4 or parts[0] != "LISTEN":
                        continue
                    local = parts[3]
                    pid_m = re.search(r"pid=(\d+)", line)
                    name_m = re.search(r'users:\(\("([^"]+)"', line)
                    pid = pid_m.group(1) if pid_m else "?"
                    if name_m:
                        names[pid] = name_m.group(1)
                    entries.append(("tcp", local, pid))
        else:
            lsof = shutil.which("lsof")
            if not lsof:
                return names, entries
            code, text, _, _ = await _exec(
                ctx, [lsof, "-nP", "-iTCP", "-sTCP:LISTEN"], timeout=20
            )
            if code != 0 and not text:
                return names, entries
            for line in text.splitlines()[1:]:
                parts = line.split()
                if len(parts) < 9:
                    continue
                names[parts[1]] = parts[0]
                entries.append(("tcp", parts[8], parts[1]))
    return names, entries


async def _process_details(ctx: ToolContext, pid: str) -> dict[str, str]:
    """读取进程详情（名称/可执行文件/命令行/启动时间/父 PID），跨平台尽力而为。"""
    info: dict[str, str] = {}
    if not pid.isdigit():
        return info
    if sys.platform == "win32":
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if ps is None:
            return info
        script = (
            "$ErrorActionPreference='SilentlyContinue'; "
            f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
            "if ($p) { 'Name=' + $p.Name; 'Path=' + $p.ExecutablePath; "
            "'Cmd=' + $p.CommandLine; 'Parent=' + $p.ParentProcessId; "
            "if ($p.CreationDate) { 'Started=' + $p.CreationDate.ToString('yyyy-MM-dd HH:mm:ss') } }"
        )
        code, text, _, _ = await _exec(
            ctx, [ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            timeout=25,
        )
        if code == 0:
            for line in text.splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key in ("Name", "Path", "Cmd", "Parent", "Started") and value.strip():
                    info[key] = value.strip()
        return info
    ps = shutil.which("ps")
    if ps:
        code, text, _, _ = await _exec(
            ctx, [ps, "-p", pid, "-o", "ppid=,comm=,args="], timeout=15
        )
        if code == 0 and text.strip():
            parts = text.strip().split(None, 2)
            if len(parts) >= 1:
                info["Parent"] = parts[0]
            if len(parts) >= 2:
                info["Name"] = parts[1]
            if len(parts) >= 3:
                info["Cmd"] = parts[2]
    try:
        info["Path"] = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        pass
    return info


def make_runtime_tools(ctx: ToolContext):

    @tool_spec(category="terminal", risk_level=RISK_MEDIUM, timeout=130)
    @tool(parse_docstring=True)
    async def run_code(language: str, code: str, timeout: int = 30) -> str:
        """
        直接运行一小段代码并返回输出（自动用临时文件，避免 shell 引号转义问题）

        适合验证语法、跑几行数据处理、试验正则等；不要用它替代项目本身的测试。
        临时文件在运行结束后自动删除，代码若在运行时创建的文件会保留。

        Args:
            language: 语言，支持 python / node / bash
            code: 要执行的完整代码
            timeout: 可选，超时秒数（1-120，默认 30）

        Returns:
            退出码、耗时与合并后的标准输出/错误输出
        """
        ctx.check_stopped()
        if not code.strip():
            return "错误：code 不能为空。"
        lang = language.lower().strip()
        if lang not in _RUNNERS:
            return f"错误：不支持的语言 {language}（支持 python / node / bash）"
        timeout = max(1, min(int(timeout), 120))

        if lang in ("python", "py"):
            exe = _python_exe(ctx.workspace)
            if not exe:
                return _missing("python", "请先安装 Python 或创建 .venv。")
        elif lang in ("node", "javascript", "js"):
            exe = shutil.which("node")
            if not exe:
                return _missing("node", "请先安装 Node.js。")
        else:
            exe = shutil.which("bash") or (sys.platform != "win32" and shutil.which("sh"))
            if not exe:
                return _missing("bash", "Windows 请安装 Git Bash，或改用 python/node。")

        log_dir = os.path.join(ctx.workspace, ".agent_logs")
        os.makedirs(log_dir, exist_ok=True)
        suffix = _RUNNERS[lang]
        tmp = os.path.join(log_dir, f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}{suffix}")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(code if code.endswith("\n") else code + "\n")
        except OSError as e:
            return f"临时文件写入失败: {e}"

        argv = [exe, tmp]
        started = time.monotonic()
        try:
            code_rc, output, timed_out, stopped = await _exec(ctx, argv, timeout=timeout)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        elapsed = int((time.monotonic() - started) * 1000)

        if stopped:
            return f"已按用户请求停止（耗时 {elapsed}ms）。\n{_truncate(output) if output else '(无输出)'}"
        if timed_out:
            return f"执行超时（>{timeout}s）已终止。\n{_truncate(output) if output else '(无输出)'}"
        body = _truncate(output) if output else "(无输出)"
        return f"退出码: {code_rc} | 耗时: {elapsed}ms | 语言: {lang}\n{body}"

    @tool_spec(category="build", risk_level=RISK_MEDIUM, timeout=330)
    @tool(parse_docstring=True)
    async def run_tests(
        target: str = "",
        keyword: str = "",
        fail_fast: bool = False,
        timeout: int = 300,
    ) -> str:
        """
        运行项目测试，可只跑指定文件/目录或按关键字过滤

        比 run_checks(kind="test") 更精准：改完一个文件后可只跑相关测试，快速验证。

        Args:
            target: 可选，测试文件/目录（相对工作区），如 tests/test_api.py 或 src/foo.test.ts
            keyword: 可选，只跑名称匹配该关键字的用例（pytest -k / jest -t / go -run）
            fail_fast: 可选，第一个失败就停止（pytest -x），默认 False
            timeout: 可选，超时秒数（10-300，默认 300；需小于工具总超时 330s）

        Returns:
            退出码、耗时与测试输出
        """
        ctx.check_stopped()
        # 必须小于 @tool_spec(timeout=330)：否则外层先超时取消，子进程回收依赖 _exec 的兜底
        timeout = max(10, min(int(timeout), 300))
        kind = _project_kind(ctx.workspace)
        resolved_target = ""
        if target.strip():
            try:
                p = resolve_workspace_path(ctx.workspace, target.strip())
            except (PathOutsideWorkspace, ValueError) as e:
                return f"错误：{e}"
            if not p.exists():
                return f"错误：测试目标不存在: {target}"
            resolved_target = str(p)

        if kind == "python":
            exe = _python_exe(ctx.workspace)
            argv = [exe, "-m", "pytest", "-q"]
            if fail_fast:
                argv.append("-x")
            if keyword.strip():
                argv += ["-k", keyword.strip()]
            argv.append(resolved_target or ".")
        elif kind == "node":
            node_modules = os.path.join(ctx.workspace, "node_modules")
            if not os.path.isdir(node_modules):
                return "未找到 node_modules，请先安装依赖（可用 install_dependencies）。"
            deps: dict = {}
            try:
                with open(os.path.join(ctx.workspace, "package.json"), encoding="utf-8") as f:
                    pkg = json.load(f)
                deps = {**pkg.get("devDependencies", {}), **pkg.get("dependencies", {})}
            except (OSError, ValueError):
                pkg = {}
            if "vitest" in deps:
                npx = shutil.which("npx")
                if not npx:
                    return _missing("npx", "请确认 Node.js 安装完整。")
                argv = [npx, "vitest", "run"]
                if keyword.strip():
                    argv += ["-t", keyword.strip()]
                if resolved_target:
                    argv.append(resolved_target)
            elif "jest" in deps:
                npx = shutil.which("npx")
                if not npx:
                    return _missing("npx", "请确认 Node.js 安装完整。")
                argv = [npx, "jest"]
                if fail_fast:
                    argv.append("--bail")
                if keyword.strip():
                    argv += ["-t", keyword.strip()]
                if resolved_target:
                    argv.append(resolved_target)
            else:
                pm = _node_pm(ctx.workspace)
                scripts = pkg.get("scripts", {}) if isinstance(pkg, dict) else {}
                if "test" not in scripts:
                    return "package.json 没有 test 脚本，且未检测到 vitest/jest。请手动指定测试命令或用 terminal。"
                argv = [pm, "test"]
                if resolved_target or keyword.strip():
                    argv += ["--", resolved_target or "", *(["-k", keyword.strip()] if keyword.strip() else [])]
        elif kind == "go":
            argv = ["go", "test"]
            if resolved_target:
                argv.append(os.path.relpath(resolved_target, ctx.workspace).replace("\\", "/"))
            else:
                argv.append("./...")
            if fail_fast:
                argv.append("-failfast")
            if keyword.strip():
                argv += ["-run", keyword.strip()]
        elif kind == "rust":
            argv = ["cargo", "test"]
            if keyword.strip():
                argv.append(keyword.strip())
            if fail_fast:
                argv.append("--")
                argv.append("--fail-fast")
        else:
            return "无法识别项目类型，请用 terminal 手动运行测试命令。"

        if shutil.which(argv[0]) is None and not os.path.isfile(argv[0]):
            return _missing(argv[0], "请确认运行环境已安装。")

        await ctx.emit_event({
            "type": "status", "status": "executing",
            "detail": f"运行测试: {' '.join(argv[1:4])}",
        })
        started = time.monotonic()
        try:
            code, output, timed_out, stopped = await _exec(ctx, argv, timeout=timeout)
        except Exception as e:
            return f"测试执行失败: {e}"
        elapsed = int((time.monotonic() - started) * 1000)

        if stopped:
            return f"测试已按用户请求停止（耗时 {elapsed}ms）。\n{_truncate(output) if output else '(无输出)'}"
        if timed_out:
            return f"测试超时（>{timeout}s）已终止。\n{_truncate(output) if output else '(无输出)'}"
        head = f"测试退出码: {code} | 耗时: {elapsed}ms | 命令: {' '.join(argv)}"
        body = _truncate(output) if output else "(无输出)"
        if code != 0:
            return f"【测试失败】{head}\n{body}"
        return f"【测试通过】{head}\n{body}"

    @tool_spec(category="web", idempotent=True, timeout=75)
    @tool(parse_docstring=True)
    async def wait_for_server(url: str = "http://127.0.0.1:8000", timeout: int = 60) -> str:
        """
        轮询本地服务直到可访问或超时

        启动 dev server 后用它等待就绪，替代 sleep 后盲目访问；
        只允许 localhost / 127.0.0.1 / ::1，不会访问外部地址。

        Args:
            url: 本地服务地址，默认 http://127.0.0.1:8000
            timeout: 可选，最长等待秒数（1-180，默认 60）

        Returns:
            服务就绪时返回状态码与耗时；超时返回最后一次错误
        """
        ctx.check_stopped()
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            return f"错误：只支持 http/https，收到 {parsed.scheme or '(空)'}"
        host = (parsed.hostname or "").lower()
        if host not in ("localhost", "127.0.0.1", "::1"):
            return f"安全限制：wait_for_server 只允许本地地址，拒绝 {host or '(空)'}"
        try:
            import httpx
        except ImportError:
            return "httpx 未安装，无法检测服务。"

        timeout = max(1, min(int(timeout), 180))
        started = time.monotonic()
        last_error = ""
        async with httpx.AsyncClient(timeout=5.0, verify=False, follow_redirects=False) as client:
            while True:
                ctx.check_stopped()
                try:
                    resp = await client.get(url.strip())
                    elapsed = int((time.monotonic() - started) * 1000)
                    return (
                        f"服务已就绪：{url.strip()}\n"
                        f"状态码: {resp.status_code} | 等待耗时: {elapsed}ms"
                    )
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                if time.monotonic() - started >= timeout:
                    elapsed = int((time.monotonic() - started) * 1000)
                    return (
                        f"等待超时（{timeout}s，实际 {elapsed}ms）：{url.strip()} 仍不可访问。\n"
                        f"最后错误：{last_error}\n"
                        "请检查服务是否已启动、端口是否正确（可用 list_ports 查看占用）。"
                    )
                await asyncio.sleep(0.5)

    @tool_spec(category="terminal", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def list_ports(port: int = 0) -> str:
        """
        列出本机正在监听的 TCP 端口及对应进程

        排查端口占用（EADDRINUSE）、确认 dev server 是否真的起来时使用。

        Args:
            port: 可选，只看指定端口；0 表示列出全部监听端口

        Returns:
            端口、监听地址、PID、进程名列表
        """
        ctx.check_stopped()

        names, entries = await _collect_listeners(ctx)

        if port:
            entries = [e for e in entries if _port_of(e[1]) == int(port)]
        if not entries:
            if not names and port == 0:
                return "没有读取到监听端口（系统可能缺少 netstat / ss / lsof）。"
            return f"没有找到监听端口{(' ' + str(port)) if port else ''}。"

        rows = []
        seen: set[tuple[str, str]] = set()
        for proto, local, pid in sorted(entries, key=lambda e: _port_of(e[1])):
            key = (local, pid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(f"{local:<28} {proto:<5} PID {pid:<8} {names.get(pid, '?')}")
        if len(rows) > 200:
            rows = [*rows[:200], "...(仅显示前 200 条)"]
        return "监听端口（地址 / 协议 / PID 进程）:\n" + "\n".join(rows)

    @tool_spec(category="terminal", idempotent=True, timeout=60)
    @tool(parse_docstring=True)
    async def port_info(port: int) -> str:
        """
        查看指定端口的占用详情（只读，不会结束任何进程）

        返回监听地址、PID、进程名、可执行文件、命令行与来源；判断端口被谁占用、
        是否是 Agent 启动的后台服务时使用。非 Agent 启动的进程一律不得终止。

        Args:
            port: 要查看的端口号（1-65535）

        Returns:
            端口占用详情；未被监听时明确说明
        """
        ctx.check_stopped()
        try:
            port = int(port)
        except (TypeError, ValueError):
            return "错误：port 必须是整数。"
        if not 1 <= port <= 65535:
            return "错误：端口号必须在 1-65535 之间。"

        names, entries = await _collect_listeners(ctx)
        matched = [e for e in entries if _port_of(e[1]) == port]
        if not matched:
            return (
                f"端口 {port} 当前未被监听。"
                "如果服务刚启动，可用 wait_for_server 等待就绪后再确认。"
            )

        listeners: dict[str, list[str]] = {}
        for proto, local, pid in matched:
            listeners.setdefault(pid, []).append(f"{local} ({proto})")

        try:
            from .terminal import _index_load

            agent_index = _index_load(ctx)
        except Exception:
            agent_index = {}

        lines = [f"端口 {port} 占用详情（只读，不会结束任何进程）:"]
        for pid, addrs in listeners.items():
            detail = await _process_details(ctx, pid)
            name = detail.get("Name") or names.get(pid, "?")
            lines.append("")
            lines.append(f"监听地址: {', '.join(sorted(addrs))}")
            lines.append(f"PID: {pid} | 进程名: {name}")
            if detail.get("Path"):
                lines.append(f"可执行文件: {detail['Path']}")
            if detail.get("Cmd"):
                lines.append(f"命令行: {detail['Cmd'][:500]}")
            if detail.get("Started"):
                lines.append(f"启动时间: {detail['Started']}")
            if detail.get("Parent"):
                lines.append(f"父进程 PID: {detail['Parent']}")

            meta = agent_index.get(pid)
            owner_pid = pid
            if meta is None and detail.get("Parent", "").isdigit():
                # 后台 shell 的子进程（如 npm run dev 派生的 node）通过父 PID 归属
                parent = detail["Parent"]
                meta = agent_index.get(parent)
                owner_pid = parent
            if meta:
                lines.append(
                    f"来源: 由 Agent 后台启动（{meta.get('started_at', '时间未知')}，"
                    f"日志 {meta.get('log', '?')}）"
                )
                lines.append(f'停止方式: process(action="stop", pid={owner_pid})（仅限该 Agent 进程）')
            else:
                lines.append("来源: 非本工作区 Agent 启动的进程（禁止终止）")
        return "\n".join(lines)

    @tool_spec(category="build", idempotent=True, timeout=200)
    @tool(parse_docstring=True)
    async def audit_dependencies(timeout: int = 180) -> str:
        """
        只读检查依赖中的已知安全漏洞（不修改 lock 文件）

        Node 用 npm/pnpm audit，Python 用 pip-audit（需已安装），
        Go 用 govulncheck，Rust 用 cargo audit。

        Args:
            timeout: 可选，超时秒数（30-180，默认 180；需小于工具总超时 200s）

        Returns:
            漏洞数量、严重级别与受影响包列表；缺少检查工具时给出安装提示
        """
        ctx.check_stopped()
        timeout = max(30, min(int(timeout), 180))
        workspace = ctx.workspace

        if os.path.exists(os.path.join(workspace, "package-lock.json")) or os.path.exists(
            os.path.join(workspace, "pnpm-lock.yaml")
        ):
            pnpm = os.path.exists(os.path.join(workspace, "pnpm-lock.yaml"))
            cmd = "pnpm" if pnpm else "npm"
            if shutil.which(cmd) is None:
                return _missing(cmd, "请先安装 Node.js。")
            argv = [cmd, "audit", "--json"]
            code, text, timed_out, _ = await _exec(ctx, argv, timeout=timeout)
            if timed_out:
                return f"[{cmd} audit] 超时（{timeout}s）。"
            try:
                data = json.loads(text[text.index("{"):])
            except (ValueError, json.JSONDecodeError):
                return f"[{cmd} audit] 无法解析输出（退出码 {code}）:\n{_truncate(text, 4000)}"
            return _summarize_audit(cmd, data)

        if os.path.exists(os.path.join(workspace, "pyproject.toml")) or os.path.exists(
            os.path.join(workspace, "requirements.txt")
        ):
            exe = _python_exe(workspace)
            pid_audit = shutil.which("pip-audit")
            argv = [pid_audit, "-f", "json"] if pid_audit else [exe, "-m", "pip_audit", "-f", "json"]
            code, text, timed_out, _ = await _exec(ctx, argv, timeout=timeout)
            if timed_out:
                return "pip-audit 超时。"
            if code != 0 and "No module named" in text:
                return (
                    "未安装 pip-audit，无法检查 Python 依赖漏洞。\n"
                    f"可执行：{exe} -m pip install pip-audit 后重试（或直接告诉用户）。"
                )
            try:
                data = json.loads(text[text.index("["):] if "[" in text else text)
            except (ValueError, json.JSONDecodeError):
                return f"[pip-audit] 无法解析输出（退出码 {code}）:\n{_truncate(text, 4000)}"
            return _summarize_pip_audit(data)

        if os.path.exists(os.path.join(workspace, "go.mod")):
            if shutil.which("govulncheck") is None:
                return "未安装 govulncheck。可执行：go install golang.org/x/vuln/cmd/govulncheck@latest 后重试。"
            code, text, timed_out, _ = await _exec(ctx, ["govulncheck", "./..."], timeout=timeout)
            if timed_out:
                return "govulncheck 超时。"
            return _truncate(f"govulncheck 退出码 {code}\n{text}")

        if os.path.exists(os.path.join(workspace, "Cargo.toml")):
            if shutil.which("cargo") is None:
                return _missing("cargo", "请先安装 Rust 工具链。")
            code, text, timed_out, _ = await _exec(ctx, ["cargo", "audit"], timeout=timeout)
            if timed_out:
                return "cargo audit 超时。"
            if "no such command" in text.lower():
                return "cargo-audit 未安装。可执行：cargo install cargo-audit 后重试。"
            return _truncate(f"cargo audit 退出码 {code}\n{text}")

        return "未识别到可审计的依赖清单（package.json / pyproject.toml / go.mod / Cargo.toml）。"

    return run_code, run_tests, wait_for_server, list_ports, port_info, audit_dependencies


def _summarize_audit(cmd: str, data: dict) -> str:
    if not isinstance(data, dict):
        return f"[{cmd} audit] 输出格式异常。"
    meta = data.get("metadata", {}).get("vulnerabilities") or {}
    advisories = data.get("vulnerabilities") or data.get("advisories") or {}
    if isinstance(advisories, dict):
        items = list(advisories.values())
    elif isinstance(advisories, list):
        items = advisories
    else:
        items = []
    total = sum(int(v) for v in meta.values() if isinstance(v, (int, float))) if meta else len(items)
    if not items and not meta:
        return f"[{cmd} audit] 未发现已知漏洞。"
    lines = [f"[{cmd} audit] 共 {total} 项漏洞"]
    if meta:
        lines.append("严重级别: " + " | ".join(f"{k}={v}" for k, v in meta.items()))
    for item in items[:20]:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("module_name") or "?"
        severity = item.get("severity") or "?"
        via = item.get("via") or []
        title = ""
        if via and isinstance(via[0], dict):
            title = via[0].get("title", "")
        elif via and isinstance(via[0], str):
            title = via[0]
        fix = item.get("fixAvailable")
        fix_text = f" | 修复: {fix}" if fix not in (None, False) else ""
        lines.append(f"- {name} [{severity}] {title[:100]}{fix_text}")
    if len(items) > 20:
        lines.append(f"...(仅显示前 20 项，共 {len(items)} 项)")
    return "\n".join(lines)


def _summarize_pip_audit(data: object) -> str:
    if not isinstance(data, list):
        return "[pip-audit] 输出格式异常。"
    total = sum(len(item.get("vulns", [])) for item in data if isinstance(item, dict))
    if total == 0:
        return "[pip-audit] 未发现已知漏洞。"
    lines = [f"[pip-audit] 共 {total} 项漏洞，涉及 {len(data)} 个包"]
    for item in data[:25]:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "?")
        version = item.get("version", "?")
        for vuln in item.get("vulns", [])[:5]:
            vid = vuln.get("id", "?")
            fixes = vuln.get("fix_versions") or []
            fix_text = f" | 修复版本: {', '.join(fixes)}" if fixes else ""
            lines.append(f"- {name} {version}: {vid}{fix_text}")
    return "\n".join(lines)
