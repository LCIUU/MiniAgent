"""Docker 感知工具：检测 Docker 环境，查看容器/镜像状态，在容器内执行命令。

支持：
- 检测 Docker 是否可用
- 列出运行中的容器和镜像
- 查看容器日志
- 在容器内执行命令（需审批）
"""

from __future__ import annotations

import asyncio
import os
import shutil

from langchain_core.tools import tool

from ..config import get_effective
from .context import ToolContext
from .registry import RISK_HIGH, RISK_MEDIUM, tool_spec

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


async def _run_docker(*args: str, workspace: str = "", timeout: int = 60) -> tuple[int, str]:
    exe = shutil.which("docker")
    if exe is None:
        return 127, "未找到 docker 可执行文件，请先安装 Docker。"
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        cwd=workspace or None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        return 124, "docker 命令超时。"
    return proc.returncode or 0, _decode(out).strip()


def _detect_docker_files(workspace: str) -> list[str]:
    found = []
    for name in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml",
                 "compose.yml", "compose.yaml", ".dockerignore"):
        if os.path.exists(os.path.join(workspace, name)):
            found.append(name)
    return found


def make_docker_tools(ctx: ToolContext):

    @tool_spec(category="docker", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def docker_status() -> str:
        """
        检查 Docker 环境状态：是否可用、当前容器、项目 Docker 配置

        Returns:
            Docker 可用性和项目 Docker 配置概览
        """
        ctx.check_stopped()
        exe = shutil.which("docker")
        if exe is None:
            return "Docker 未安装。"

        code, version = await _run_docker("version", "--format", "{{.Server.Version}}")
        if code != 0:
            return f"Docker 已安装但未运行: {version}"

        files = _detect_docker_files(ctx.workspace)
        parts = [f"Docker 版本: {version}"]
        if files:
            parts.append(f"项目 Docker 配置: {', '.join(files)}")
        else:
            parts.append("项目目录未发现 Docker 配置文件。")

        code, containers = await _run_docker("ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
        if code == 0 and containers:
            parts.append(f"\n运行中的容器:\n{_truncate(containers)}")
        else:
            parts.append("\n当前没有运行中的容器。")

        return "\n".join(parts)

    @tool_spec(category="docker", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def docker_logs(container: str, tail: int = 100) -> str:
        """
        查看 Docker 容器日志

        Args:
            container: 容器名称或 ID
            tail: 查看最后多少行日志，默认 100

        Returns:
            容器日志内容
        """
        ctx.check_stopped()
        if not container.strip():
            return "错误：container 不能为空。"
        tail = max(1, min(int(tail), 1000))
        code, text = await _run_docker("logs", "--tail", str(tail), container.strip())
        if code != 0:
            return f"获取日志失败: {text}"
        return _truncate(text) if text else "(无日志)"

    @tool_spec(category="docker", risk_level=RISK_MEDIUM, timeout=300)
    @tool(parse_docstring=True)
    async def docker_compose(action: str = "up") -> str:
        """
        管理 docker-compose 服务

        Args:
            action: "up" 启动；"down" 停止并删除；"restart" 重启；"ps" 查看状态；"build" 构建

        Returns:
            操作结果
        """
        ctx.check_stopped()
        files = _detect_docker_files(ctx.workspace)
        compose_files = [f for f in files if "compose" in f.lower()]
        if not compose_files:
            return "未找到 docker-compose 配置文件。"

        if action == "ps":
            code, text = await _run_docker("compose", "ps", workspace=ctx.workspace)
        elif action == "up":
            approved = await ctx.approvals.request(
                ctx.emit_event, "docker_compose", "normal",
                {"command": "docker compose up -d", "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了 docker compose up 操作。"
            code, text = await _run_docker("compose", "up", "-d", workspace=ctx.workspace, timeout=300)
        elif action == "down":
            approved = await ctx.approvals.request(
                ctx.emit_event, "docker_compose", "high_risk",
                {"command": "docker compose down", "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了 docker compose down 操作。"
            code, text = await _run_docker("compose", "down", workspace=ctx.workspace)
        elif action == "restart":
            approved = await ctx.approvals.request(
                ctx.emit_event, "docker_compose", "normal",
                {"command": "docker compose restart", "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了 docker compose restart 操作。"
            code, text = await _run_docker("compose", "restart", workspace=ctx.workspace, timeout=300)
        elif action == "build":
            approved = await ctx.approvals.request(
                ctx.emit_event, "docker_compose", "normal",
                {"command": "docker compose build", "cwd": ctx.workspace},
                stop_event=ctx.stop_event,
            )
            if not approved:
                return "用户拒绝了 docker compose build 操作。"
            code, text = await _run_docker("compose", "build", workspace=ctx.workspace, timeout=300)
        else:
            return f"未知 action: {action}（支持 up / down / restart / ps / build）"

        if code != 0:
            return f"docker compose {action} 失败:\n{_truncate(text)}"
        return _truncate(text) if text else f"docker compose {action} 完成。"

    @tool_spec(category="docker", risk_level=RISK_HIGH, timeout=120)
    @tool(parse_docstring=True)
    async def docker_exec(container: str, command: str) -> str:
        """
        在 Docker 容器内执行命令（需要用户审批）

        Args:
            container: 容器名称或 ID
            command: 要在容器内执行的命令

        Returns:
            命令输出
        """
        ctx.check_stopped()
        if not container.strip():
            return "错误：container 不能为空。"
        if not command.strip():
            return "错误：command 不能为空。"

        approved = await ctx.approvals.request(
            ctx.emit_event,
            "docker_exec",
            "high_risk",
            {"command": f"docker exec {container.strip()} {command.strip()}", "cwd": ctx.workspace},
            stop_event=ctx.stop_event,
        )
        if not approved:
            return "用户拒绝了容器内命令执行。"

        code, text = await _run_docker("exec", container.strip(), "sh", "-c", command.strip())
        if code != 0:
            return f"容器内命令执行失败（退出码 {code}）:\n{_truncate(text)}"
        return _truncate(text) if text else "(无输出)"

    return docker_status, docker_logs, docker_compose, docker_exec
