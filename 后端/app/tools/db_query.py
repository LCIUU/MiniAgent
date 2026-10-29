"""数据库查询工具：安全地执行只读 SQL 查询、查看表结构。

支持通过环境变量 DATABASE_URL 或 .env 文件连接 PostgreSQL / SQLite / MySQL。
仅允许只读操作（SELECT / EXPLAIN / SHOW），写操作被拦截。
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil

from langchain_core.tools import tool

from ..config import get_effective
from .context import ToolContext
from .registry import tool_spec

_MAX = 12000
_READ_ONLY_RE = re.compile(
    r"^\s*(SELECT|EXPLAIN|SHOW|DESCRIBE|DESC|WITH)\b",
    re.IGNORECASE,
)
_WRITE_RE = re.compile(
    # REPLACE 只拦截 REPLACE INTO 语句；SELECT REPLACE(col, ...) 是常用只读函数
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|GRANT|REVOKE)\b"
    r"|\bREPLACE\s+INTO\b",
    re.IGNORECASE,
)


def _truncate(text: str) -> str:
    limit = int(get_effective("tool_output_max_chars", 12000))
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n...[已截断，共 {len(text)} 字符]...\n" + text[-limit // 2 :]


def _detect_db_type(dsn: str) -> str:
    dsn_lower = dsn.lower()
    if "postgresql" in dsn_lower or "postgres" in dsn_lower:
        return "postgresql"
    if "mysql" in dsn_lower or "mariadb" in dsn_lower:
        return "mysql"
    if "sqlite" in dsn_lower:
        return "sqlite"
    return "unknown"


def _get_dsn(workspace: str) -> str:
    for name in ("DATABASE_URL", "DB_URL", "DB_DSN"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    for env_file in (".env",):
        path = os.path.join(workspace, env_file)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(("DATABASE_URL=", "DB_URL=", "DB_DSN=")):
                            _, _, val = line.partition("=")
                            val = val.strip().strip("'\"")
                            if val:
                                return val
            except Exception:
                pass
    return ""


async def _run_psql(sql: str, dsn: str, workspace: str) -> str:
    if shutil.which("psql") is None:
        return "未找到 psql 客户端，请安装 PostgreSQL 客户端工具。"
    proc = await asyncio.create_subprocess_exec(
        "psql", dsn, "-c", sql, "--no-psqlrc", "-t",
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        return "查询超时（30s）。"
    text = out.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return f"查询失败:\n{text}"
    return _truncate(text) if text else "(无结果)"


async def _run_sqlite(sql: str, db_path: str, workspace: str) -> str:
    if shutil.which("sqlite3") is None:
        return "未找到 sqlite3 客户端。"
    proc = await asyncio.create_subprocess_exec(
        "sqlite3", db_path, sql,
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        return "查询超时（30s）。"
    text = out.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return f"查询失败:\n{text}"
    return _truncate(text) if text else "(无结果)"


async def _run_mysql(sql: str, dsn: str, workspace: str) -> str:
    if shutil.which("mysql") is None:
        return "未找到 mysql 客户端。"
    proc = await asyncio.create_subprocess_exec(
        "mysql", "-e", sql, "--batch", dsn,
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        return "查询超时（30s）。"
    text = out.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        return f"查询失败:\n{text}"
    return _truncate(text) if text else "(无结果)"


def make_db_tools(ctx: ToolContext):

    @tool_spec(category="db", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def db_schema() -> str:
        """
        查看当前项目数据库的表结构概览

        Returns:
            数据库类型与表清单
        """
        ctx.check_stopped()
        dsn = _get_dsn(ctx.workspace)
        if not dsn:
            return "未找到数据库连接配置（DATABASE_URL 环境变量或 .env 文件）。"
        db_type = _detect_db_type(dsn)

        if db_type == "postgresql":
            sql = "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"
            result = await _run_psql(sql, dsn, ctx.workspace)
        elif db_type == "sqlite":
            db_path = dsn.replace("sqlite:///", "").replace("sqlite:", "")
            result = await _run_sqlite(".tables", db_path, ctx.workspace)
        elif db_type == "mysql":
            result = await _run_mysql("SHOW TABLES;", dsn, ctx.workspace)
        else:
            return f"不支持的数据库类型: {db_type}"

        return f"数据库类型: {db_type}\n\n{result}"

    @tool_spec(category="db", idempotent=True, timeout=30)
    @tool(parse_docstring=True)
    async def db_query(sql: str) -> str:
        """
        执行只读 SQL 查询（仅允许 SELECT / EXPLAIN / SHOW / DESCRIBE）

        Args:
            sql: SQL 查询语句，只允许只读操作

        Returns:
            查询结果
        """
        ctx.check_stopped()
        if not sql.strip():
            return "错误：sql 不能为空。"
        if _WRITE_RE.search(sql):
            return "安全限制：只允许只读查询（SELECT / EXPLAIN / SHOW / DESCRIBE），禁止写操作。"
        if not _READ_ONLY_RE.match(sql):
            return "安全限制：SQL 必须以 SELECT / EXPLAIN / SHOW / DESCRIBE 开头。"

        dsn = _get_dsn(ctx.workspace)
        if not dsn:
            return "未找到数据库连接配置（DATABASE_URL 环境变量或 .env 文件）。"
        db_type = _detect_db_type(dsn)

        if db_type == "postgresql":
            return await _run_psql(sql, dsn, ctx.workspace)
        elif db_type == "sqlite":
            db_path = dsn.replace("sqlite:///", "").replace("sqlite:", "")
            return await _run_sqlite(sql, db_path, ctx.workspace)
        elif db_type == "mysql":
            return await _run_mysql(sql, dsn, ctx.workspace)
        else:
            return f"不支持的数据库类型: {db_type}"

    return db_schema, db_query
