"""后台进程管理接口：停止 terminal 工具后台启动的进程树。"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/processes", tags=["processes"])

# taskkill 在中文 Windows 输出 GBK："错误: 没有找到进程 "xxx""
_NOT_FOUND_RE = re.compile(r"没有找到进程|找不到进程|not found", re.IGNORECASE)


def _decode_proc_output(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("gbk", errors="replace")


class StopRequest(BaseModel):
    pid: int


async def _kill_tree_windows(pid: int) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "taskkill",
        "/T",
        "/F",
        "/PID",
        str(pid),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    _, err_raw = await asyncio.wait_for(proc.communicate(), timeout=15)
    return proc.returncode or 0, _decode_proc_output(err_raw)


@router.get("/alive")
async def alive_processes(pids: str) -> dict:
    """批量检查 PID 是否存活，返回 {"alive": [存活的 pid]}。"""
    alive: list[int] = []
    pid_list = [int(t) for t in pids.split(",") if t.strip().isdigit()]
    if not pid_list:
        return {"alive": alive}

    if sys.platform == "win32":
        # Windows：一次 tasklist 查询所有 PID，避免每个 PID 生成一个子进程
        filters = " ".join(f'/FI "PID eq {pid}"' for pid in pid_list)
        probe = await asyncio.create_subprocess_exec(
            "tasklist",
            *filters.split(),
            "/FO", "CSV", "/NH",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        out_raw, _ = await asyncio.wait_for(probe.communicate(), timeout=15)
        output = _decode_proc_output(out_raw)
        for pid in pid_list:
            if f'"{pid}"' in output:
                alive.append(pid)
    else:
        # Unix：os.kill(pid, 0) 是轻量级检查
        for pid in pid_list:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except OSError:
                pass
    return {"alive": alive}


@router.post("/stop")
async def stop_process(req: StopRequest) -> dict:
    """按 PID 停止后台进程（Windows 下连同全部子进程）。

    进程本就不存在时视为停止成功（ok=true, already_gone=true）。
    """
    if req.pid <= 0:
        raise HTTPException(status_code=400, detail="pid 无效")

    if sys.platform == "win32":
        rc, err = await _kill_tree_windows(req.pid)
        if rc != 0:
            if _NOT_FOUND_RE.search(err):
                return {"ok": True, "already_gone": True}
            raise HTTPException(status_code=404, detail=f"停止失败: {err.strip()}")
        return {"ok": True}

    try:
        os.kill(req.pid, signal.SIGKILL)
        return {"ok": True}
    except ProcessLookupError:
        return {"ok": True, "already_gone": True}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=f"停止失败: {e}") from e
