"""异步数据库会话（SQLAlchemy 2.0 + asyncpg）。"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_effective

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _url() -> str:
    return get_effective("database_url", "postgresql+asyncpg://postgres@localhost:5432/coding_agent")


engine = create_async_engine(
    _url(),
    echo=False,
    pool_pre_ping=True
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncSession:
    """FastAPI 依赖：每个请求一个会话。"""
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """建表 + 幂等迁移（失败抛出，由启动器决定是否继续）。"""
    from . import models  # noqa: F401  确保模型注册
    from .services.migrations import run_migrations

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await run_migrations(conn)


async def ping_db(timeout: float = 3.0) -> bool:
    """健康检查用：带超时的 DB 连通性探测。"""
    import asyncio

    try:
        async def _ping() -> None:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

        await asyncio.wait_for(_ping(), timeout=timeout)
        return True
    except Exception as e:
        logger.warning("数据库健康检查失败: %s", e)
        return False
