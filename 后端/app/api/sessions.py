"""会话 CRUD API。"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from ..database import get_db
from ..models import Message, Session, ToolCache
from ..schemas import MessageOut, SessionCreate, SessionOut, SessionRename
from ..services import run_store
from ..services.attachments import validate_stored_path
from ..services.runtime import manager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _dt(v: datetime) -> str:
    return v.isoformat() if v else ""


def _session_out(s: Session) -> SessionOut:
    return SessionOut(
        id=s.id,
        title=s.title,
        workspace=s.workspace,
        provider=s.provider,
        model=s.model,
        created_at=_dt(s.created_at),
        updated_at=_dt(s.updated_at),
    )


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    q: str = Query("", max_length=200, description="按标题/工作区搜索"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Session)
    if q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(Session.title.ilike(like) | Session.workspace.ilike(like))
    stmt = stmt.order_by(Session.updated_at.desc()).limit(limit).offset(offset)
    result = await db.execute(stmt)
    return [_session_out(s) for s in result.scalars().all()]


@router.post("", response_model=SessionOut)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    s = Session(title=body.title or "新会话", workspace=body.workspace)
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return _session_out(s)


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "会话不存在")
    return _session_out(s)


@router.patch("/{session_id}", response_model=SessionOut)
async def rename_session(session_id: str, body: SessionRename, db: AsyncSession = Depends(get_db)):
    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "会话不存在")
    s.title = body.title or s.title
    await db.commit()
    await db.refresh(s)
    return _session_out(s)


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "会话不存在")
    workspace = s.workspace or ""
    # 删除前先停止该会话的活动运行：否则后台 run 会继续执行工具、写工作区，
    # 并在会话已删除后产生孤儿消息/审计记录
    active = manager.find_active(session_id)
    if active is not None:
        manager.stop(active.run_id)
        if active.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(active.task), timeout=6)
            except (TimeoutError, asyncio.CancelledError):
                pass
    # 先收集附件文件路径（会话删除后 DB 记录就查不到了）
    rows = await db.execute(select(Message.attachments).where(Message.session_id == session_id))
    attachment_paths: list[str] = []
    for (atts,) in rows.fetchall():
        for att in atts or []:
            if isinstance(att, dict) and att.get("path"):
                attachment_paths.append(str(att["path"]))

    await db.delete(s)
    await db.commit()

    # 清理关联数据与磁盘文件，避免长期运行后磁盘无限增长
    try:
        await db.execute(delete(ToolCache).where(ToolCache.session_id == session_id))
        await db.commit()
    except Exception:
        await db.rollback()
    await run_store.delete_for_session(session_id)
    for raw in attachment_paths:
        try:
            p = Path(raw)
            if workspace and validate_stored_path(workspace, raw) and p.is_file():
                p.unlink()
        except OSError:
            pass
    return {"ok": True}


@router.get("/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=200, description="返回最近 N 条消息，默认 50"),
    offset: int = Query(0, ge=0, description="偏移量，用于分页加载更早的消息"),
    db: AsyncSession = Depends(get_db),
):
    # 按创建时间倒序取最近 limit 条，再正序返回（id 作为同一时刻的稳定 tie-breaker）
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
        .offset(offset)
        # 大字段不随列表下发/加载：完整消息流与附件提取正文只在下一轮构建上下文时使用
        .options(defer(Message.context_messages), defer(Message.request_content))
    )
    messages = result.scalars().all()
    messages.reverse()  # 恢复正序
    out = []
    for m in messages:
        out.append(MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            attachments=m.attachments or [],
            tool_calls=m.tool_calls or [],
            timeline=m.timeline or [],
            token_usage=m.token_usage,
            duration_ms=m.duration_ms or 0,
            created_at=_dt(m.created_at),
        ))
    return out
