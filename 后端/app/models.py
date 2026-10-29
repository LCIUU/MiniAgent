"""ORM 模型：会话、消息。"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_updated", "updated_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    workspace: Mapped[str] = mapped_column(Text, default="")
    provider: Mapped[str] = mapped_column(String(50), default="")
    model: Mapped[str] = mapped_column(String(100), default="")
    summary: Mapped[str] = mapped_column(Text, default="")  # 上下文摘要（中间件产出）
    # 摘要覆盖到的最后一条历史消息时间：跨轮跳过更早的原始消息，避免重复摘要
    summary_cutoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_session_created", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[str] = mapped_column(Text, default="")
    # 用户消息附件元信息（上传接口返回的 {name, path, size, kind} 列表），前端据此渲染图片缩略图
    attachments: Mapped[list | None] = mapped_column(JSON, default=list)
    # 本轮实际发给模型的内容（含附件提取正文/图片说明）；下一轮据此还原，
    # 否则追问附件内容时模型失忆。content 保持纯文本供前端展示
    request_content: Mapped[str | None] = mapped_column(Text, default=None)
    # assistant 消息的工具调用记录：[{name, args, result, status, started_at, finished_at}]
    tool_calls: Mapped[list | None] = mapped_column(JSON, default=list)
    # 思考过程时间轴（持久化）：[{kind: thinking|tool|plan, ...}]，供历史回看完整思考链
    timeline: Mapped[list | None] = mapped_column(JSON, default=list)
    # 本轮完整 LLM 消息流（human/ai/tool 序列，含 tool_calls 与工具结果），供下一轮原样重建上下文
    context_messages: Mapped[list | None] = mapped_column(JSON, default=None)
    # token 用量统计：{input_tokens, output_tokens, llm_calls}（模型返回为准，可能缺失）
    token_usage: Mapped[dict | None] = mapped_column(JSON, default=None)
    # 本轮运行总耗时（毫秒）：刷新/切换会话后仍可展示"总耗时"
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[Session] = relationship(back_populates="messages")


class ToolCache(Base):
    """幂等工具结果缓存：跨 run 复用（同会话、限 TTL），写操作后整会话失效。"""

    __tablename__ = "tool_cache"
    __table_args__ = (
        Index("ix_tool_cache_session_key_created", "session_id", "cache_key", "created_at"),
        Index("ix_tool_cache_created", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(32), index=True)
    # sha256(json.dumps({name, args, workspace, fingerprint}, sort_keys=True))
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    tool_name: Mapped[str] = mapped_column(String(100), default="")
    result: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Run(Base):
    """一次 Agent 运行的可查询审计记录（进程重启后仍可追溯）。"""

    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_session_started", "session_id", "started_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # run_id
    session_id: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running|done|error|cancelled
    workspace: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(100), default="")
    request_preview: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
