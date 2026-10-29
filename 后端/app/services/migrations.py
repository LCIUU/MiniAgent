"""轻量数据库迁移：幂等 DDL + 版本记录，替代“每次启动裸 ALTER + 吞异常”。

不引入 Alembic 的迁移图/自动生成能力，但保留可审计、可重复执行的迁移记录：
- 新库：``create_all`` 一次建全，迁移仍会记录版本
- 旧库：逐条补齐缺失列/索引，失败会抛出，避免“表结构坏了但服务照常启动”
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

Migration = tuple[int, str, Callable[[AsyncConnection], Awaitable[None]]]


async def _column_names(conn: AsyncConnection, table: str) -> set[str]:
    def _get(sync_conn):
        inspector = inspect(sync_conn)
        if table not in inspector.get_table_names():
            return set()
        return {c["name"] for c in inspector.get_columns(table)}

    return await conn.run_sync(_get)


async def ensure_column(conn: AsyncConnection, table: str, column: str, ddl_type: str) -> None:
    """缺列则补列；兼容 PostgreSQL / SQLite。"""
    if column in await _column_names(conn, table):
        return
    await conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl_type}'))
    logger.info("数据库迁移：%s.%s 已添加", table, column)


async def _migration_001_message_columns(conn: AsyncConnection) -> None:
    """messages 的思考时间轴 / token 用量 / 附件列。"""
    await ensure_column(conn, "messages", "timeline", "JSON")
    await ensure_column(conn, "messages", "token_usage", "JSON")
    await ensure_column(conn, "messages", "attachments", "JSON")


async def _migration_002_indexes(conn: AsyncConnection) -> None:
    """高频查询索引：消息列表、缓存、run 查询、会话排序。"""
    for ddl in (
        (
            'CREATE INDEX IF NOT EXISTS "ix_messages_session_created" '
            'ON "messages" ("session_id", "created_at")'
        ),
        (
            'CREATE INDEX IF NOT EXISTS "ix_tool_cache_session_key_created" '
            'ON "tool_cache" ("session_id", "cache_key", "created_at")'
        ),
        (
            'CREATE INDEX IF NOT EXISTS "ix_tool_cache_created" '
            'ON "tool_cache" ("created_at")'
        ),
        'CREATE INDEX IF NOT EXISTS "ix_sessions_updated" ON "sessions" ("updated_at")',
        (
            'CREATE INDEX IF NOT EXISTS "ix_runs_session_started" '
            'ON "runs" ("session_id", "started_at")'
        ),
    ):
        await conn.execute(text(ddl))


async def _migration_003_message_duration(conn: AsyncConnection) -> None:
    """assistant 消息补运行总耗时列（刷新后仍可展示总耗时）。"""
    await ensure_column(conn, "messages", "duration_ms", "INTEGER DEFAULT 0")


async def _migration_004_message_context(conn: AsyncConnection) -> None:
    """assistant 消息补完整 LLM 消息流列（下一轮恢复工具调用与工具结果）。"""
    await ensure_column(conn, "messages", "context_messages", "JSON")


async def _migration_005_session_summary_cutoff(conn: AsyncConnection) -> None:
    """会话补摘要覆盖截止时间：跨轮跳过已摘要的原始消息。"""
    await ensure_column(conn, "sessions", "summary_cutoff", "TIMESTAMP WITH TIME ZONE")


async def _migration_006_message_request_content(conn: AsyncConnection) -> None:
    """用户消息补“实际发送内容”列：跨轮保留附件提取正文，避免追问时失忆。"""
    await ensure_column(conn, "messages", "request_content", "TEXT")


MIGRATIONS: list[Migration] = [
    (1, "messages: timeline/token_usage/attachments", _migration_001_message_columns),
    (2, "indexes: messages/tool_cache/sessions/runs", _migration_002_indexes),
    (3, "messages: duration_ms", _migration_003_message_duration),
    (4, "messages: context_messages", _migration_004_message_context),
    (5, "sessions: summary_cutoff", _migration_005_session_summary_cutoff),
    (6, "messages: request_content", _migration_006_message_request_content),
]


async def run_migrations(conn: AsyncConnection) -> int:
    """执行未应用的迁移，返回本次应用数量。"""
    await conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, "
            "description VARCHAR(200), "
            "applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    result = await conn.execute(text("SELECT version FROM schema_migrations"))
    applied = {row[0] for row in result.fetchall()}
    count = 0
    for version, description, fn in MIGRATIONS:
        if version in applied:
            continue
        try:
            await fn(conn)
            await conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, description) "
                    "VALUES (:v, :d) ON CONFLICT (version) DO NOTHING"
                ),
                {"v": version, "d": description},
            )
            count += 1
            logger.info("数据库迁移 %03d 完成: %s", version, description)
        except Exception as e:
            logger.error("数据库迁移 %03d 失败: %s", version, e)
            raise
    return count
