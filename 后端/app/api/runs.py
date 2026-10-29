"""运行记录与用量 API：审计、token 成本、历史回看。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import run_store

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("")
async def list_runs(
    session_id: str = Query("", description="按会话过滤"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    items = await run_store.list_runs(session_id or None, limit=limit, offset=offset)
    return {"runs": items}


@router.get("/usage")
async def usage(session_id: str = Query("", description="按会话过滤")) -> dict:
    return await run_store.usage_summary(session_id or None)


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict:
    item = await run_store.get_run(run_id)
    if item is None:
        raise HTTPException(404, "运行记录不存在")
    return item
