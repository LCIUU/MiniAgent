"""幂等工具结果缓存（PostgreSQL 持久化）：跨 run 复用，写操作后整会话失效。

设计约束：
- 仅幂等工具（registry ToolSpec.idempotent=True）的结果允许入库
- 缓存键包含工作区指纹 + 被读取文件的 mtime/size，避免用户在 IDE 改完文件后
  仍命中旧缓存（原实现只按工具名+参数，存在脏读窗口）
- TTL 限制（默认 tool_cache_ttl_seconds=600s）
- 所有操作静默降级：DB 不可用时不影响 Agent 主流程
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_effective
from ..database import SessionLocal
from ..models import ToolCache
from .paths import is_within

logger = logging.getLogger(__name__)

# 参数里可能代表文件/目录的字段名（read_files 用 paths 一次读多个文件）
_PATH_ARG_KEYS = ("path", "file_path", "paths", "directory", "dir", "target_path")
# 目录递归指纹的条目上限：超过后不再深入，避免在工作区根目录调用时拖慢缓存键计算
_DIR_FINGERPRINT_MAX_ENTRIES = 2000


def _path_signature(p: Path) -> str:
    """文件取 mtime/size；目录递归取最新 mtime、条目数与总大小。"""
    try:
        st = p.stat()
    except OSError:
        return ""
    norm = os.path.normcase(str(p))
    if not p.is_dir():
        return f"{norm}:{st.st_mtime_ns}:{st.st_size}"
    newest = 0
    count = 0
    total = 0
    stack = [p]
    try:
        while stack and count < _DIR_FINGERPRINT_MAX_ENTRIES:
            cur = stack.pop()
            with os.scandir(cur) as it:
                for entry in it:
                    if count >= _DIR_FINGERPRINT_MAX_ENTRIES:
                        break
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    else:
                        try:
                            est = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        newest = max(newest, est.st_mtime_ns)
                        total += est.st_size
                    count += 1
    except OSError:
        pass
    # 目录自身 mtime 已覆盖"增删改名"，newest 覆盖"子文件被外部修改"
    return f"{norm}:{st.st_mtime_ns}:{newest}:{count}:{total}"


def _workspace_fingerprint(workspace: str, args: dict[str, Any]) -> str:
    """工作区绝对路径 + 被访问文件/目录状态的稳定指纹。"""
    if not workspace:
        return ""
    ws = os.path.normcase(os.path.realpath(workspace))
    parts: list[str] = [ws]
    for key in _PATH_ARG_KEYS:
        raw = args.get(key)
        if isinstance(raw, str):
            values: list[Any] = (
                [x for x in re.split(r"[\n,;]+", raw)] if key == "paths" else [raw]
            )
        elif isinstance(raw, (list, tuple)):
            values = list(raw)
        else:
            continue
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            try:
                p = Path(value)
                if not p.is_absolute():
                    p = Path(workspace) / p
                p = p.resolve()
                if is_within(workspace, p):
                    sig = _path_signature(p)
                    if sig:
                        parts.append(sig)
            except OSError:
                continue
    return "|".join(parts)


def make_cache_key(tool_name: str, args: dict[str, Any], workspace: str = "") -> str:
    """工具名 + 参数 + 工作区/文件指纹 → 稳定散列键。"""
    payload = {
        "name": tool_name,
        "args": args,
        "fingerprint": _workspace_fingerprint(workspace, args),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ttl_seconds() -> int:
    try:
        return int(get_effective("tool_cache_ttl_seconds", 600))
    except Exception:
        return 600


async def get_cached(
    session_id: str, tool_name: str, args: dict[str, Any], workspace: str = "",
    *, db: AsyncSession | None = None,
) -> str | None:
    """查询会话内未过期的缓存结果；未命中/出错返回 None。"""
    if not session_id:
        return None
    try:
        cutoff = datetime.now(UTC) - timedelta(seconds=_ttl_seconds())
        _do = db if db is not None else SessionLocal()
        try:
            result = await _do.execute(
                select(ToolCache.result)
                .where(
                    ToolCache.session_id == session_id,
                    ToolCache.cache_key == make_cache_key(tool_name, args, workspace),
                    ToolCache.created_at >= cutoff,
                )
                .order_by(ToolCache.created_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()
        finally:
            if db is None:
                await _do.close()
    except Exception as e:
        logger.debug("工具缓存读取失败（降级跳过）: %s", e)
        return None


async def put_cached(
    session_id: str,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    workspace: str = "",
    *, db: AsyncSession | None = None,
) -> None:
    """写入缓存（同键先删后插，保持单条最新）。"""
    if not session_id:
        return
    try:
        key = make_cache_key(tool_name, args, workspace)
        _do = db if db is not None else SessionLocal()
        try:
            await _do.execute(
                delete(ToolCache).where(
                    ToolCache.session_id == session_id,
                    ToolCache.cache_key == key,
                )
            )
            _do.add(ToolCache(session_id=session_id, cache_key=key, tool_name=tool_name, result=result))
            await _do.commit()
        finally:
            if db is None:
                await _do.close()
    except Exception as e:
        logger.debug("工具缓存写入失败（降级跳过）: %s", e)


async def invalidate_session(session_id: str, *, db: AsyncSession | None = None) -> None:
    """写操作后调用：清空该会话全部缓存，避免脏读。"""
    if not session_id:
        return
    try:
        _do = db if db is not None else SessionLocal()
        try:
            await _do.execute(delete(ToolCache).where(ToolCache.session_id == session_id))
            await _do.commit()
        finally:
            if db is None:
                await _do.close()
    except Exception as e:
        logger.debug("工具缓存失效清理失败（降级跳过）: %s", e)


async def cleanup_expired() -> int:
    """后台清理过期缓存，返回删除行数；DB 不可用时静默返回 0。"""
    try:
        cutoff = datetime.now(UTC) - timedelta(seconds=_ttl_seconds())
        async with SessionLocal() as db:
            result = await db.execute(delete(ToolCache).where(ToolCache.created_at < cutoff))
            await db.commit()
            return int(result.rowcount or 0)
    except Exception as e:
        logger.debug("工具缓存清理失败（降级跳过）: %s", e)
        return 0
