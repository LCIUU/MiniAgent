"""应用配置：从环境变量 / .env 读取，敏感项（API Key）可被前端设置覆盖并持久化。"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .services.secrets import (
    decrypt_settings,
    encrypt_settings,
    has_plaintext_secrets,
    merge_preserving_unreadable,
    restrict_file_permissions,
)

# 后端根目录
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 服务
    app_name: str = "AI Coding Agent"
    host: str = "127.0.0.1"
    port: int = 8000

    # 数据库
    database_url: str = "postgresql+asyncpg://postgres@localhost:5432/coding_agent"

    # 模型 Provider（OpenAI 兼容端点）
    # 统一走 OpenAI 兼容协议：base_url + api_key + model
    # 例：阿里百炼 https://dashscope.aliyuncs.com/compatible-mode/v1
    #      DeepSeek  https://api.deepseek.com/v1
    # .env 中使用 BASE_URL / MODEL；default_* 为旧字段名，保留作为兜底
    default_provider: str = "openai"
    base_url: str = ""
    model: str = ""
    default_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model: str = "qwen3.7-flash"
    openai_api_key: str = ""
    # 各模型服务商凭证（.env 预留空值，按需填写；设置页按模型保存的 Key 优先于此处）
    dashscope_api_key: str = ""
    dashscope_base_url: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = ""

    # Tavily 网络搜索
    tavily_api_key: str = ""

    # 工具行为
    # 命令确认模式: auto | high_risk | all
    command_confirm_mode: str = "auto"
    tool_output_max_chars: int = 12000
    terminal_timeout_seconds: int = 300
    tool_cache_ttl_seconds: int = 600
    tool_cache_max_chars: int = 20000
    # 文件修改前自动备份到 .agent_backups（非 Git 项目的兜底撤销）
    enable_file_backups: bool = True

    # 上下文管理
    # 摘要阈值：>0 为显式值（默认 5 万）；0 = 按模型窗口 × summary_threshold_ratio 自动计算
    summary_threshold_tokens: int = 50000
    summary_threshold_ratio: float = 0.6
    summary_keep_messages: int = 30

    # 模型调用
    llm_max_retries: int = 2
    # 流式空闲超时（秒）：超过该时长未收到新 chunk 视为连接卡死；0 = 关闭该护栏
    # 上游（百炼）在高思考档位/限流排队时会短暂停推 SSE，180s 内被判死会整轮重跑，
    # 白丢已经流出的内容；放宽到 240s 给排队留余量。
    llm_stream_chunk_timeout: float = 240
    # 主模型不可用时的降级模型（留空则不降级）
    fallback_model: str = ""

    # 流式事件
    sse_heartbeat_seconds: int = 15
    sse_replay_max_events: int = 20000

    # 项目规则文件（逗号分隔，相对工作区；启动时注入系统提示）
    project_rules_files: str = "AGENTS.md,.agent/rules.md,CLAUDE.md,.cursorrules"

    # 本地安全（默认关闭，开启后所有 /api 请求需带 X-API-Token）
    api_token: str = ""
    # 每 IP 每分钟允许的写请求数（0=不限）
    rate_limit_per_mutation: int = 120

    # MCP 配置
    mcp_config_path: str = str(BACKEND_DIR / "mcp.json")

    # 设置持久化文件（前端可写）
    settings_file: str = str(BACKEND_DIR / "agent_settings.json")

    # CORS
    # CORS 跨域配置（.env 中用逗号分隔，如 CORS_ORIGINS=http://localhost:5173,http://192.168.1.10:5173）
    # 填 * 表示允许所有来源（不携带凭证）
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: Any) -> Any:
        """解析 CORS 配置：支持 JSON 数组 或 逗号分隔字符串。"""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


settings = Settings()


_SETTINGS_CACHE: dict[str, Any] = {"path": "", "mtime_ns": -1, "data": {}}
_SETTINGS_LOCK = asyncio.Lock()


def _read_settings_raw(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def load_settings_dict() -> dict[str, Any]:
    """读取持久化的用户设置（带 mtime 缓存，避免每次 get_effective 都读盘+解密）。

    敏感项（API Key）在磁盘上为 Fernet 密文，这里统一解密后返回；
    返回深拷贝，调用方可安全修改后再 save_settings_dict。
    """
    path = Path(settings.settings_file)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        _SETTINGS_CACHE.update({"path": str(path), "mtime_ns": -1, "data": {}})
        return {}
    if _SETTINGS_CACHE["path"] == str(path) and _SETTINGS_CACHE["mtime_ns"] == mtime_ns:
        return copy.deepcopy(_SETTINGS_CACHE["data"])
    data = decrypt_settings(_read_settings_raw(path))
    _SETTINGS_CACHE.update({"path": str(path), "mtime_ns": mtime_ns, "data": data})
    return copy.deepcopy(data)


def save_settings_dict(data: dict[str, Any]) -> None:
    """原子写入用户设置（敏感项加密后落盘），并刷新内存缓存。"""
    path = Path(settings.settings_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = copy.deepcopy(data)
    # 解密失败的 Key（密钥文件丢失/更换）不能被空串覆盖掉
    merge_preserving_unreadable(to_write, _read_settings_raw(path) if path.exists() else {})
    payload = json.dumps(encrypt_settings(to_write), ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(prefix=".agent-settings-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
        os.replace(tmp, path)
        restrict_file_permissions(path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        _SETTINGS_CACHE.update({
            "path": str(path),
            "mtime_ns": path.stat().st_mtime_ns,
            "data": copy.deepcopy(data),
        })
    except OSError:
        _SETTINGS_CACHE.update({"path": "", "mtime_ns": -1, "data": {}})


def migrate_settings_secrets() -> bool:
    """启动时把历史明文 API Key 一次性加密；返回是否发生了迁移。"""
    path = Path(settings.settings_file)
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, dict) or not has_plaintext_secrets(raw):
        return False
    save_settings_dict(raw)
    return True


def get_effective(key: str, default: Any = None) -> Any:
    """优先返回持久化设置，其次返回环境变量/.env 默认值。"""
    user = load_settings_dict()
    if key in user:
        return user[key]
    env = getattr(settings, key, None)
    if env in (None, "") and key in ("model", "base_url"):
        # 兼容旧字段名 DEFAULT_MODEL / DEFAULT_BASE_URL
        env = getattr(settings, f"default_{key}", None)
    return default if env in (None, "") else env


async def set_effective(key: str, value: Any) -> None:
    async with _SETTINGS_LOCK:
        data = load_settings_dict()
        data[key] = value
        save_settings_dict(data)


# 模型目录：模型名 → 端点/凭证字段/思考能力适配
# thinking 适配方式：
#   "qwen"     通过 extra_body.enable_thinking 控制思考（阿里百炼 Qwen3 系）
#   "reasoner" 推理模型，自动返回 reasoning_content，无需额外参数（DeepSeek 推理系）
#   "none"     无思考能力，不传任何思考参数
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "qwen3.7-flash": {
        "label": "Qwen3.7 Flash（阿里百炼）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_field": "dashscope_api_key",
        "url_field": "dashscope_base_url",
        "key_env": "DASHSCOPE_API_KEY",
        "thinking": "qwen",
        "context_window": 131072,
        "vision": True,  # Qwen3 系支持多模态图片输入
    },
    "qwen3.8-flash": {
        "label": "Qwen3.8 Flash（阿里百炼）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_field": "dashscope_api_key",
        "url_field": "dashscope_base_url",
        "key_env": "DASHSCOPE_API_KEY",
        "thinking": "qwen",
        "context_window": 131072,
        "vision": True,
    },
    "qwen3.7-plus": {
        "label": "Qwen3.7 Plus（阿里百炼）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_field": "dashscope_api_key",
        "url_field": "dashscope_base_url",
        "key_env": "DASHSCOPE_API_KEY",
        "thinking": "qwen",
        "context_window": 131072,
        "vision": True,
    },
    "deepseek-v4-flash": {
        "label": "DeepSeek V4 Flash",
        "base_url": "https://api.deepseek.com/v1",
        "key_field": "deepseek_api_key",
        "url_field": "deepseek_base_url",
        "key_env": "DEEPSEEK_API_KEY",
        "thinking": "none",
        "context_window": 131072,
        "vision": False,  # DeepSeek 文本模型，不支持图像输入
    },
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "base_url": "https://api.deepseek.com/v1",
        "key_field": "deepseek_api_key",
        "url_field": "deepseek_base_url",
        "key_env": "DEEPSEEK_API_KEY",
        "thinking": "reasoner",
        "context_window": 131072,
        "vision": False,
    },
}


def resolve_model_endpoint(model: str) -> dict[str, Any]:
    """解析模型的最终请求端点与凭证（每次调用实时读取，切换即时生效）。

    凭证优先级：设置页按模型保存的 Key > 服务商环境变量(.env) > 全局 openai_api_key；
    端点优先级：设置页按模型保存的 base_url > 服务商环境变量 > 目录默认值。
    目录外的自定义模型回退到全局 base_url / openai_api_key（兼容旧行为）。
    """
    entry = MODEL_CATALOG.get(model)
    user = load_settings_dict()
    per_keys = user.get("model_api_keys") or {}
    per_urls = user.get("model_base_urls") or {}
    per_key = str(per_keys.get(model) or "").strip()
    per_url = str(per_urls.get(model) or "").strip()
    if entry:
        api_key = per_key or get_effective(entry["key_field"], "") or get_effective("openai_api_key", "")
        base_url = per_url or get_effective(entry["url_field"], "") or entry["base_url"]
        thinking = entry["thinking"]
        key_env = entry["key_env"]
        label = entry["label"]
        vision = bool(entry.get("vision", False))
    else:
        api_key = per_key or get_effective("openai_api_key", "")
        base_url = per_url or get_effective("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        thinking = "qwen" if model.lower().startswith("qwen3") else "none"
        key_env = "OPENAI_API_KEY"
        label = model
        vision = model.lower().startswith("qwen3")  # 目录外 qwen3 自定义模型按多模态处理
    return {
        "model": model,
        "base_url": str(base_url or "").rstrip("/"),
        "api_key": str(api_key or "").strip(),
        "thinking": thinking,
        "vision": vision,
        "context_window": int(entry.get("context_window", 0)) if entry else 0,
        "in_catalog": entry is not None,
        "key_env": key_env,
        "label": label,
    }
