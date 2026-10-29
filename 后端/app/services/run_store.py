"""Agent 运行记录（runs 表）的读写：审计、用量统计、重启后追溯。"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, update

from ..database import SessionLocal
from ..models import Run

logger = logging.getLogger(__name__)


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def _run_dict(run: Run) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "session_id": run.session_id,
        "status": run.status,
        "workspace": run.workspace,
        "model": run.model,
        "request_preview": run.request_preview,
        "error": run.error,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "llm_calls": run.llm_calls,
        "duration_ms": run.duration_ms,
        "started_at": _iso(run.started_at),
        "ended_at": _iso(run.ended_at),
    }


async def create_run(
    run_id: str,
    session_id: str,
    workspace: str,
    model: str,
    request_preview: str,
) -> None:
    """运行开始即落库；失败只记录日志，不影响主流程。"""
    try:
        async with SessionLocal() as db:
            db.add(
                Run(
                    id=run_id,
                    session_id=session_id,
                    status="running",
                    workspace=workspace,
                    model=model,
                    request_preview=request_preview[:500],
                    started_at=datetime.now(UTC),
                )
            )
            await db.commit()
    except Exception as e:
        logger.warning("run 记录创建失败（降级继续）: %s", e)


async def finish_run(
    run_id: str,
    status: str,
    *,
    error: str = "",
    token_usage: dict[str, Any] | None = None,
    duration_ms: int = 0,
) -> None:
    """运行结束更新状态与 token 用量；失败只记录日志。"""
    usage = token_usage or {}
    try:
        async with SessionLocal() as db:
            run = await db.get(Run, run_id)
            if run is None:
                return
            run.status = status
            run.error = (error or "")[:4000]
            run.input_tokens = int(usage.get("input_tokens") or 0)
            run.output_tokens = int(usage.get("output_tokens") or 0)
            run.llm_calls = int(usage.get("llm_calls") or 0)
            run.duration_ms = int(duration_ms)
            run.ended_at = datetime.now(UTC)
            await db.commit()
    except Exception as e:
        logger.warning("run 记录更新失败（降级继续）: %s", e)


async def mark_stale_runs_interrupted() -> int:
    """启动时收敛上次进程遗留的 running 记录（崩溃/强杀后状态永远停在 running）。"""
    try:
        async with SessionLocal() as db:
            result = await db.execute(
                update(Run)
                .where(Run.status == "running")
                .values(
                    status="interrupted",
                    ended_at=datetime.now(UTC),
                    error="服务重启，运行被中断",
                )
            )
            await db.commit()
            return int(result.rowcount or 0)
    except Exception as e:
        logger.warning("清理遗留运行记录失败: %s", e)
        return 0


async def get_run(run_id: str) -> dict[str, Any] | None:
    async with SessionLocal() as db:
        run = await db.get(Run, run_id)
        return _run_dict(run) if run else None


async def list_runs(
    session_id: str | None = None, *, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    async with SessionLocal() as db:
        stmt = select(Run).order_by(Run.started_at.desc()).limit(min(limit, 200)).offset(offset)
        if session_id:
            stmt = stmt.where(Run.session_id == session_id)
        rows = (await db.execute(stmt)).scalars().all()
        return [_run_dict(r) for r in rows]


async def usage_summary(session_id: str | None = None) -> dict[str, Any]:
    """按会话/全局聚合 token 与运行次数。"""
    async with SessionLocal() as db:
        stmt = select(
            func.count(Run.id),
            func.coalesce(func.sum(Run.input_tokens), 0),
            func.coalesce(func.sum(Run.output_tokens), 0),
            func.coalesce(func.sum(Run.llm_calls), 0),
            func.coalesce(func.sum(Run.duration_ms), 0),
        )
        if session_id:
            stmt = stmt.where(Run.session_id == session_id)
        row = (await db.execute(stmt)).one()
        return {
            "runs": int(row[0] or 0),
            "input_tokens": int(row[1] or 0),
            "output_tokens": int(row[2] or 0),
            "llm_calls": int(row[3] or 0),
            "total_duration_ms": int(row[4] or 0),
        }


async def delete_for_session(session_id: str) -> None:
    try:
        async with SessionLocal() as db:
            await db.execute(delete(Run).where(Run.session_id == session_id))
            await db.commit()
    except Exception as e:
        logger.debug("run 记录清理失败: %s", e)
