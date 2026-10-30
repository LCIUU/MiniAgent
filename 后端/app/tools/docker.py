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
        检查 Docker 环境状态：是否安装/运行、项目 Docker 配置、运行中的容器

        适用：项目涉及容器化部署时，先确认 Docker daemon 是否可用、当前有哪些容器在跑、
        工作区是否包含 Dockerfile / compose 文件。只读，不会启动或修改任何容器。

        Docker 未安装或未运行时返回明确提示；容器清单过长时按系统输出上限截断。
        本工具不会启动 Docker Desktop：未运行时请先让用户手动启动，之后可用
        docker_compose(action="up") 拉起项目服务；定位容器问题再用 docker_logs 查日志。

        Returns:
            Docker 版本、项目 Docker 配置文件与运行中容器清单。
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
        查看 Docker 容器日志（只读）

        适用：容器启动失败、服务行为异常时查看最后若干行日志。通过 docker logs --tail
        读取，只显示 stdout/stderr，不会进入容器或修改容器状态。

        container 可用容器名或 ID（可先用 docker_status 查看运行中的容器）；
        tail 范围 1-1000，默认 100；日志过长时保留头尾。
        容器已停止时仍可读取历史日志；需要实时跟踪新日志（类似 docker logs -f）时，
        改用 terminal 手动执行并注意该命令会持续阻塞，需配合超时或手动停止。

        Args:
            container: 容器名称或 ID，不能为空。
            tail: 读取最后多少行日志，默认 100，范围 1-1000。

        Returns:
            容器日志内容；容器不存在或读取失败时返回原因。
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
        管理 docker-compose 服务（启动/停止/重启/查看/构建）

        适用：工作区存在 docker-compose.yml / compose.yml 时管理整套服务。action=up 以
        后台方式启动（docker compose up -d）；down 停止并删除容器/网络（高风险，需审批）；
        restart 重启；ps 查看状态；build 重新构建镜像。除 ps 外均会触发审批。

        未找到 compose 配置文件时直接返回提示。命令在项目根目录执行；up/restart/build
        最多等待 300 秒，日志过长时保留头尾。

        Args:
            action: "up" 启动；"down" 停止并删除；"restart" 重启；"ps" 查看状态；"build" 构建。

        Returns:
            命令输出或完成提示；用户拒绝或执行失败时返回原因。
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
        在 Docker 容器内执行命令（高风险，需要用户审批）

        适用：检查容器内文件、环境变量、进程状态，或在容器内运行诊断命令。
        通过 docker exec <容器> sh -c <命令> 执行，需要容器内存在 sh；不支持交互式命令。
        会修改容器状态的命令（如删除文件、改配置）请先向用户确认影响。

        必须提供容器名/ID 和命令；执行前会触发高风险审批。命令输出过长时保留头尾。

        命令以非交互方式执行：不要使用 -it、vim、top 等需要终端的命令，也不要启动常驻进程；
        查看服务日志用 docker_logs，管理整套服务用 docker_compose。退出码非 0 时会附带退出码
        与输出，便于判断失败原因；容器没有 sh 或未运行时直接返回对应错误。

        Args:
            container: 容器名称或 ID，不能为空。
            command: 要在容器内执行的完整 shell 命令，不能为空。

        Returns:
            容器内命令的输出；用户拒绝或命令失败时返回原因。
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
