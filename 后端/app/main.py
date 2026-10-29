"""FastAPI 应用入口。"""

from __future__ import annotations

import asyncio
import hmac
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import agent, processes, runs, sessions, settings, tools, workspace
from .config import migrate_settings_secrets
from .config import settings as cfg
from .database import engine, init_db, ping_db
from .services import run_store, tool_cache
from .services.runtime import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# 第三方库降噪
for _noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "sqlalchemy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|tvly-[A-Za-z0-9_\-]{6,})")


class _SecretFilter(logging.Filter):
    """日志脱敏：避免 API Key 被写进日志/终端/IDE 缓存。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = _SECRET_RE.sub("***", record.msg)
            if record.args:
                record.args = tuple(
                    _SECRET_RE.sub("***", a) if isinstance(a, str) else a for a in record.args
                )
        except Exception:
            pass
        return True


for _handler in logging.getLogger().handlers:
    _handler.addFilter(_SecretFilter())

logger = logging.getLogger(__name__)

_cleanup_task: asyncio.Task | None = None


async def _cleanup_loop() -> None:
    """后台清理过期工具缓存（每 10 分钟一次）。"""
    while True:
        try:
            removed = await tool_cache.cleanup_expired()
            if removed:
                logger.info("已清理 %d 条过期工具缓存", removed)
        except Exception as e:
            logger.debug("工具缓存清理异常: %s", e)
        await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    app.state.db_ready = False
    try:
        await init_db()
        app.state.db_ready = True
        logger.info("数据库初始化完成")
        stale = await run_store.mark_stale_runs_interrupted()
        if stale:
            logger.info("已将 %d 条上次进程遗留的 running 运行记录标记为中断", stale)
    except Exception as e:
        logger.error("数据库初始化失败（服务继续启动，/api/health 会报告 not ready）: %s", e)
    try:
        if migrate_settings_secrets():
            logger.info("已将历史明文 API Key 迁移为加密存储")
    except Exception as e:
        logger.warning("设置密钥迁移失败: %s", e)
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        if _cleanup_task:
            _cleanup_task.cancel()
            try:
                await _cleanup_task
            except asyncio.CancelledError:
                pass
        await manager.shutdown()
        try:
            await engine.dispose()
        except Exception:
            pass
        logger.info("服务已关闭，后台运行已取消")


app = FastAPI(title=cfg.app_name, version="0.2.0", lifespan=lifespan)

# CORS 跨域配置
_origins = cfg.cors_origins
_allow_all = "*" in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 可选本地认证：设置 API_TOKEN 后，所有 /api 请求需带 X-API-Token / Bearer
_OPEN_PATHS = {"/api/health", "/api/health/ready", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def _auth_and_rate_limit(request: Request, call_next):
    path = request.url.path
    token = (cfg.api_token or "").strip()
    if token and path.startswith("/api") and path not in _OPEN_PATHS:
        provided = request.headers.get("x-api-token", "").strip()
        auth = request.headers.get("authorization", "").strip()
        if not provided and auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        if not hmac.compare_digest(provided, token):
            return JSONResponse({"detail": "未授权：缺少或错误的 API Token"}, status_code=401)

    # 简单内存限流（仅写请求）；默认阈值较高，避免影响正常使用
    limit = int(cfg.rate_limit_per_mutation or 0)
    if limit > 0 and path.startswith("/api") and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        bucket = app.state.__dict__.setdefault("_rate_bucket", {})
        now = asyncio.get_running_loop().time()
        ip = request.client.host if request.client else "unknown"
        window = bucket.setdefault(ip, [])
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= limit:
            return JSONResponse({"detail": "请求过于频繁，请稍后再试"}, status_code=429)
        window.append(now)
        # 清理过期 bucket（超过 5 分钟无活动的 IP）
        if len(bucket) > 100:
            stale_ips = [k for k, v in bucket.items() if not v or now - v[-1] > 300]
            for k in stale_ips:
                bucket.pop(k, None)
    return await call_next(request)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("未处理异常: %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "服务器内部错误"}, status_code=500)


app.include_router(sessions.router)
app.include_router(settings.router)
app.include_router(workspace.router)
app.include_router(agent.router)
app.include_router(processes.router)
app.include_router(tools.router)
app.include_router(runs.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "app": cfg.app_name,
        "version": "0.2.0",
        "db_ready": bool(getattr(app.state, "db_ready", False)),
    }


@app.get("/api/health/ready")
async def readiness():
    ok = await ping_db()
    return JSONResponse(
        {"status": "ready" if ok else "not_ready", "db": ok},
        status_code=200 if ok else 503,
    )
