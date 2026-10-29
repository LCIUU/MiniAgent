"""AI Coding Agent 后端启动入口（直接运行本文件即可）。

用法：
    python main.py              # 或 uv run main.py
    python main.py --cors "*"   # 允许所有来源跨域（开发用）
    start.bat                   # Windows 双击启动

跨域配置（优先级：命令行 --cors > 环境变量 CORS_ORIGINS > .env > 默认值）：
    python main.py --cors "http://localhost:5173,http://192.168.1.10:5173"
    set CORS_ORIGINS=*   （或在 .env 中配置 CORS_ORIGINS=...）

启动流程：
    1. 读取 .env 配置
    2. 检查 PostgreSQL 数据库是否存在，不存在则自动创建
    3. 自动创建数据表（sessions / messages）
    4. 启动 FastAPI 服务（http://127.0.0.1:8000，文档在 /docs）
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Windows 控制台强制 UTF-8 输出（避免 GBK 编码错误与乱码）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 确保可以从项目根目录导入 app 包（双击/任意目录运行都有效）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncpg  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.config import BACKEND_DIR, get_effective, settings  # noqa: E402

BANNER = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║         AI Coding Agent · 后端服务启动器                     ║
  ║         Nocturne Glassmorphism Edition                    ║
  ╚═══════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数（主要是跨域配置）。"""
    parser = argparse.ArgumentParser(description="AI Coding Agent 后端启动器")
    parser.add_argument(
        "--cors",
        metavar="ORIGINS",
        help="CORS 允许来源，逗号分隔；* 表示允许所有（例如 --cors 'http://localhost:5173,http://x.com'）",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="监听地址（默认读 .env 的 HOST，未配置为 127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="监听端口（默认读 .env 的 PORT，未配置为 8000）",
    )
    return parser.parse_args()


def apply_cors_config(cors_arg: str | None) -> list[str]:
    """应用跨域配置，返回生效的 origins 列表。

    优先级：命令行 --cors > 环境变量 CORS_ORIGINS > .env（Settings 已加载）> 默认。
    解析规则：逗号分隔；* 允许所有；空值回退默认。
    """
    raw: str | None = cors_arg
    if raw is None:
        raw = os.environ.get("CORS_ORIGINS")
    if raw is None:
        raw = getattr(settings, "cors_origins", None)
        if isinstance(raw, list):
            return raw

    if raw is None:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    raw = str(raw).strip()
    if not raw:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    # 环境变量优先写回，供 FastAPI 应用层读取
    os.environ["CORS_ORIGINS"] = raw
    origins = [x.strip() for x in raw.split(",") if x.strip()]
    return origins


def print_cors_status(origins: list[str]) -> None:
    print("\n  ── 跨域配置 (CORS) ─────────────────────────")
    if "*" in origins:
        print("  允许来源 : *（允许所有来源，不携带凭证）")
    else:
        print("  允许来源 : " + ", ".join(origins))
    print("  修改方式 : python main.py --cors \"来源1,来源2\" 或 .env 中 CORS_ORIGINS=...")



def _resolve_database() -> tuple[str, str, str]:
    """从 DATABASE_URL 解析出 (数据库名, 管理连接 DSN, 业务 DSN)。

    管理连接连到 postgres 库（用于创建目标库），业务连接连到目标库。
    """
    url = get_effective("database_url", settings.database_url)
    u = make_url(url)
    dbname = u.database or "coding_agent"
    port = u.port or 5432

    def dsn(db: str) -> str:
        auth = ""
        if u.username:
            auth = u.username
            if u.password:
                auth += f":{u.password}"
            auth += "@"
        return f"postgresql://{auth}{u.host or 'localhost'}:{port}/{db}"

    return dbname, dsn("postgres"), dsn(dbname)


async def ensure_database() -> str:
    """确保目标数据库存在，不存在则自动创建。返回业务 DSN。"""
    dbname, admin_dsn, business_dsn = _resolve_database()
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", dbname)
        if not exists:
            # 数据库名需要加引号（防止特殊字符注入）
            quoted = f'"{dbname}"'
            await conn.execute(f"CREATE DATABASE {quoted}")
            print(f"  ✔ 已自动创建数据库: {dbname}")
        else:
            print(f"  ✔ 数据库已存在: {dbname}")
    finally:
        await conn.close()
    return business_dsn


async def init_tables() -> None:
    """建表（幂等）。"""
    from app.database import init_db

    await init_db()
    print("  ✔ 数据表与迁移就绪 (sessions / messages / tool_cache / runs)")


async def check_settings() -> None:
    """启动前提示关键配置状态（不阻塞启动）。"""
    api_key = get_effective("openai_api_key", "")
    tavily = get_effective("tavily_api_key", "")
    base_url = get_effective("base_url", "")
    model = get_effective("model", "")

    print("\n  ── 当前模型配置 ──────────────────────────")
    print(f"  Base URL : {base_url or '(未设置，默认百炼)'}")
    print(f"  模型     : {model or '(未设置，默认 qwen-max)'}")
    if api_key:
        print("  API Key  : 已配置（已加密存储，不在终端显示）")
    else:
        print("  API Key  : ⚠ 未配置 —— 启动后在浏览器设置页填写，或写入 .env 的 OPENAI_API_KEY")
    print(f"  Tavily   : {'已配置' if tavily else '未配置（网络搜索工具不可用，可选）'}")
    mode = get_effective("command_confirm_mode", "auto")
    mode_cn = {"auto": "自动执行", "high_risk": "高危命令需确认", "all": "全部命令需确认"}.get(mode, mode)
    print(f"  命令确认 : {mode_cn} ({mode})")


def print_frontend_help() -> None:
    print("\n  ── 前端访问 ─────────────────────────────")
    print("  前端地址   : http://localhost:5173   （需先运行: cd 前端 && pnpm dev）")
    print("  API 文档   : http://127.0.0.1:8000/docs")
    print("  健康检查   : http://127.0.0.1:8000/api/health")


async def _bootstrap() -> None:
    """初始化（同一事件循环内完成，避免 SQLAlchemy 引擎跨 loop 使用）。"""
    try:
        await ensure_database()
        await init_tables()
        await check_settings()
    except Exception as e:  # noqa: BLE001
        print(f"\n  ✘ 数据库初始化失败: {e}")
        print("    请确认 PostgreSQL 已启动，且 .env 中 DATABASE_URL 正确。")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    print(BANNER)
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        print("  ✔ 已加载配置: .env")
    else:
        print("  ⚠ 未找到 .env，使用默认配置（可复制 .env.example 后修改）")

    # 跨域配置（命令行 > 环境变量 > .env > 默认）
    cors_origins = apply_cors_config(args.cors)

    # 监听地址（命令行 > .env > 默认）
    host = args.host or get_effective("host", settings.host)
    port = args.port or int(get_effective("port", settings.port))

    print_frontend_help()
    print_cors_status(cors_origins)

    import uvicorn

    async def _run() -> None:
        await _bootstrap()
        print(f"\n  ▶ 正在启动服务: http://{host}:{port}  (CORS: {', '.join(cors_origins) if '*' not in cors_origins else '*'})\n")
        config = uvicorn.Config("app.main:app", host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
