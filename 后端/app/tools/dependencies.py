"""依赖管理工具：自动识别包管理器，安装/卸载/查看依赖。

支持：
- Python: pip / uv / poetry / pdm（按项目文件自动识别）
- Node.js: npm / pnpm / yarn / bun（按 lock 文件自动识别）
- Rust: cargo
- Go: go mod
"""

from __future__ import annotations

import asyncio
import os
import shutil

from langchain_core.tools import tool

from ..config import get_effective
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec

_MAX = 12000


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _truncate(text: str) -> str:
    limit = int(get_effective("tool_output_max_chars", 12000))
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n...[已截断，共 {len(text)} 字符]...\n" + text[-limit // 2 :]


def _detect_python_pm(workspace: str) -> str:
    if os.path.exists(os.path.join(workspace, "uv.lock")):
        return "uv"
    if os.path.exists(os.path.join(workspace, "poetry.lock")):
        return "poetry"
    if os.path.exists(os.path.join(workspace, "pdm.lock")):
        return "pdm"
    return "pip"


def _detect_node_pm(workspace: str) -> str:
    if os.path.exists(os.path.join(workspace, "bun.lockb")) or os.path.exists(os.path.join(workspace, "bun.lock")):
        return "bun"
    if os.path.exists(os.path.join(workspace, "pnpm-lock.yaml")):
        return "pnpm"
    if os.path.exists(os.path.join(workspace, "yarn.lock")):
        return "yarn"
    return "npm"


def _detect_project_type(workspace: str) -> str:
    if os.path.exists(os.path.join(workspace, "Cargo.toml")):
        return "rust"
    if os.path.exists(os.path.join(workspace, "go.mod")):
        return "go"
    if any(os.path.exists(os.path.join(workspace, f)) for f in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg")):
        return "python"
    if os.path.exists(os.path.join(workspace, "package.json")):
        return "node"
    return "unknown"


async def _run_cmd(cmd: list[str], workspace: str, timeout: int = 120) -> tuple[int, str]:
    exe = shutil.which(cmd[0])
    if exe is None:
        return 127, f"未找到可执行文件: {cmd[0]}，请先安装。"
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        return 124, f"命令超时（{timeout}s）。"
    return proc.returncode or 0, _decode(out).strip()


def make_dependency_tools(ctx: ToolContext):

    @tool_spec(category="build", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def detect_dependencies() -> str:
        """
        检测当前项目的类型和包管理器（只读）

        适用：安装/卸载依赖前确认项目类型与应当使用的包管理器。按项目根目录的配置文件
        判断：Python（pyproject.toml/requirements.txt/setup.py/setup.cfg，包管理器按
        uv.lock/poetry.lock/pdm.lock 推断 pip/uv/poetry/pdm）、Node（package.json，按
        bun/pnpm/yarn lock 推断）、Rust（Cargo.toml → cargo）、Go（go.mod → go mod）。
        不会修改任何文件。

        Returns:
            项目类型、包管理器与配置文件清单；无法识别时返回提示。
        """
        ctx.check_stopped()
        ws = ctx.workspace
        ptype = _detect_project_type(ws)
        if ptype == "unknown":
            return "无法识别项目类型，未找到常见项目配置文件。"

        info = [f"项目类型: {ptype}"]

        if ptype == "python":
            pm = _detect_python_pm(ws)
            info.append(f"包管理器: {pm}")
            for f in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"):
                if os.path.exists(os.path.join(ws, f)):
                    info.append(f"配置文件: {f}")
        elif ptype == "node":
            pm = _detect_node_pm(ws)
            info.append(f"包管理器: {pm}")
        elif ptype == "rust":
            info.append("包管理器: cargo")
        elif ptype == "go":
            info.append("包管理器: go mod")

        return "\n".join(info)

    @tool_spec(category="build", idempotent=True, timeout=60)
    @tool(parse_docstring=True)
    async def list_dependencies() -> str:
        """
        列出当前项目已安装的依赖（只读）

        适用：确认某个包是否已安装、查看版本、排查依赖冲突。根据项目类型和包管理器
        执行对应命令：Python 用 pip list / uv pip list / poetry show / pdm list；
        Node 用 npm/pnpm/yarn/bun 的 list；Rust 用 cargo tree；Go 用 go list -m all。
        不会修改项目。

        输出可能较长，超过系统上限时保留头尾；命令失败时返回退出码和错误输出。

        Returns:
            依赖清单；无法识别项目类型或命令失败时返回原因。
        """
        ctx.check_stopped()
        ws = ctx.workspace
        ptype = _detect_project_type(ws)

        if ptype == "python":
            pm = _detect_python_pm(ws)
            if pm == "uv":
                code, out = await _run_cmd(["uv", "pip", "list"], ws)
            elif pm == "poetry":
                code, out = await _run_cmd(["poetry", "show"], ws)
            elif pm == "pdm":
                code, out = await _run_cmd(["pdm", "list"], ws)
            else:
                code, out = await _run_cmd(["pip", "list"], ws)
        elif ptype == "node":
            pm = _detect_node_pm(ws)
            if pm == "bun":
                code, out = await _run_cmd(["bun", "pm", "ls"], ws)
            elif pm == "pnpm":
                code, out = await _run_cmd(["pnpm", "list", "--depth=0"], ws)
            elif pm == "yarn":
                code, out = await _run_cmd(["yarn", "list", "--depth=0"], ws)
            else:
                code, out = await _run_cmd(["npm", "list", "--depth=0"], ws)
        elif ptype == "rust":
            code, out = await _run_cmd(["cargo", "tree", "--depth=1"], ws)
        elif ptype == "go":
            code, out = await _run_cmd(["go", "list", "-m", "all"], ws)
        else:
            return "无法识别项目类型。"

        if code != 0:
            return f"命令执行失败（退出码 {code}）:\n{out}"
        return _truncate(out) if out else "没有找到已安装的依赖。"

    @tool_spec(category="build", risk_level=RISK_MEDIUM, timeout=300)
    @tool(parse_docstring=True)
    async def install_dependency(packages: str, dev: bool = False) -> str:
        """
        安装一个或多个依赖包（自动使用项目现有的包管理器，需要用户审批）

        适用：项目缺少运行/测试所需依赖时。自动识别项目类型并调用对应包管理器：
        Python（uv add / poetry add / pdm add / pip install）、Node（bun add / pnpm add /
        yarn add / npm install）、Rust（cargo add）、Go（go get）。会修改依赖清单和
        lock 文件，属于中等风险操作。

        dev=true 时尽可能安装为开发依赖（uv --dev、poetry --group dev、pnpm -D、
        npm --save-dev 等）；pip/cargo/go 没有统一的 dev 语义，会忽略该参数。
        超时 300 秒，输出过长时保留头尾。

        Args:
            packages: 要安装的包名，空格分隔，例如 "requests flask" 或 "react lodash"。
            dev: 是否安装为开发依赖，默认 False。

        Returns:
            安装结果与包管理器输出；用户拒绝或失败时返回原因。
        """
        ctx.check_stopped()
        ws = ctx.workspace
        ptype = _detect_project_type(ws)
        if ptype == "unknown":
            return "无法识别项目类型，请先确认项目根目录。"

        pkgs = packages.strip().split()
        if not pkgs:
            return "错误：packages 不能为空。"

        ctx.check_stopped()
        approved = await ctx.approvals.request(
            ctx.emit_event,
            "install_dependency",
            "normal",
            {"command": f"安装依赖: {packages}", "cwd": ws},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了安装操作。"

        if ptype == "python":
            pm = _detect_python_pm(ws)
            if pm == "uv":
                cmd = ["uv", "add"] + pkgs + (["--dev"] if dev else [])
            elif pm == "poetry":
                cmd = ["poetry", "add"] + pkgs + (["--group", "dev"] if dev else [])
            elif pm == "pdm":
                cmd = ["pdm", "add"] + pkgs + (["--dev"] if dev else [])
            else:
                cmd = ["pip", "install", *pkgs]
        elif ptype == "node":
            pm = _detect_node_pm(ws)
            if pm == "bun":
                cmd = ["bun", "add"] + pkgs + (["--dev"] if dev else [])
            elif pm == "pnpm":
                cmd = ["pnpm", "add"] + pkgs + (["-D"] if dev else [])
            elif pm == "yarn":
                cmd = ["yarn", "add"] + pkgs + (["--dev"] if dev else [])
            else:
                cmd = ["npm", "install"] + pkgs + (["--save-dev"] if dev else [])
        elif ptype == "rust":
            cmd = ["cargo", "add", *pkgs]
        elif ptype == "go":
            cmd = ["go", "get", *pkgs]
        else:
            return "无法识别项目类型。"

        code, out = await _run_cmd(cmd, ws, timeout=300)
        if code != 0:
            return f"安装失败（退出码 {code}）:\n{_truncate(out)}"
        return f"依赖安装成功: {packages}\n{_truncate(out)}" if out else f"依赖安装成功: {packages}"

    @tool_spec(category="build", risk_level=RISK_MEDIUM, timeout=120)
    @tool(parse_docstring=True)
    async def uninstall_dependency(packages: str) -> str:
        """
        卸载一个或多个依赖包（自动使用项目现有的包管理器，需要用户审批）

        适用：清理不再使用的包、解决版本冲突。按项目类型调用 uv/poetry/pdm/pip、
        bun/pnpm/yarn/npm、cargo remove 或 go mod edit -droprequire。会修改依赖清单和
        lock 文件，且属于高风险操作，必须经用户确认；卸载前请确认代码中已无引用。

        超时 120 秒。若只是暂时不用、之后可能恢复，优先考虑保留依赖。

        Args:
            packages: 要卸载的包名，空格分隔。

        Returns:
            卸载结果；用户拒绝或失败时返回原因。
        """
        ctx.check_stopped()
        ws = ctx.workspace
        ptype = _detect_project_type(ws)
        if ptype == "unknown":
            return "无法识别项目类型。"

        pkgs = packages.strip().split()
        if not pkgs:
            return "错误：packages 不能为空。"

        approved = await ctx.approvals.request(
            ctx.emit_event,
            "uninstall_dependency",
            "high_risk",
            {"command": f"卸载依赖: {packages}", "cwd": ws},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了卸载操作。"

        if ptype == "python":
            pm = _detect_python_pm(ws)
            if pm == "uv":
                cmd = ["uv", "remove", *pkgs]
            elif pm == "poetry":
                cmd = ["poetry", "remove", *pkgs]
            elif pm == "pdm":
                cmd = ["pdm", "remove", *pkgs]
            else:
                cmd = ["pip", "uninstall", "-y", *pkgs]
        elif ptype == "node":
            pm = _detect_node_pm(ws)
            if pm == "bun":
                cmd = ["bun", "remove", *pkgs]
            elif pm == "pnpm":
                cmd = ["pnpm", "remove", *pkgs]
            elif pm == "yarn":
                cmd = ["yarn", "remove", *pkgs]
            else:
                cmd = ["npm", "uninstall", *pkgs]
        elif ptype == "rust":
            cmd = ["cargo", "remove", *pkgs]
        elif ptype == "go":
            cmd = ["go", "mod", "edit", "-droprequire", *pkgs]
        else:
            return "无法识别项目类型。"

        code, out = await _run_cmd(cmd, ws)
        if code != 0:
            return f"卸载失败（退出码 {code}）:\n{_truncate(out)}"
        return f"依赖已卸载: {packages}"

    return detect_dependencies, list_dependencies, install_dependency, uninstall_dependency
