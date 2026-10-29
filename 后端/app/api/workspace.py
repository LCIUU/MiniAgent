"""工作区 API：浏览本地目录、设置/读取当前工作区。

前端工作区选择器调用：列出目录树，选择后保存。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import get_effective, set_effective
from ..schemas import WorkspaceEntry, WorkspaceListOut

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


@router.get("", response_model=WorkspaceListOut)
async def list_workspace(path: str = Query("", description="相对或绝对路径，空表示默认工作区")):
    base = get_effective("workspace", os.path.expanduser("~"))
    if path:
        target = Path(path)
        if not target.is_absolute():
            target = Path(base) / target
    else:
        target = Path(base)
    target = target.resolve()

    if not target.exists():
        raise HTTPException(404, f"路径不存在: {target}")
    if not target.is_dir():
        raise HTTPException(400, f"不是目录: {target}")

    entries: list[WorkspaceEntry] = []
    try:
        for child in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if child.name.startswith(".") and child.name not in (".git",):
                continue
            if child.name in ("node_modules", ".venv", "__pycache__", "dist", "build", ".next"):
                continue
            try:
                entries.append(WorkspaceEntry(
                    name=child.name,
                    path=str(child),
                    is_dir=child.is_dir(),
                    size=child.stat().st_size if child.is_file() else 0,
                ))
            except OSError:
                continue
    except PermissionError as e:
        raise HTTPException(403, f"没有权限访问: {target}") from e

    parent = str(target.parent) if target != target.parent else None
    return WorkspaceListOut(path=str(target), parent=parent, entries=entries[:500])


@router.post("/select")
async def select_workspace(body: dict):
    path = body.get("path", "").strip()
    if not path:
        raise HTTPException(400, "路径不能为空")
    p = Path(path).resolve()
    if not p.is_dir():
        raise HTTPException(404, f"目录不存在: {p}")
    await set_effective("workspace", str(p))
    return {"ok": True, "workspace": str(p)}


@router.get("/current")
async def current_workspace():
    return {"workspace": get_effective("workspace", "")}
