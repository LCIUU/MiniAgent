"""设置 API：读取/更新持久化设置（前端设置页调用）。

API Key 只写不回传（响应中仅返回是否已配置）。
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException

from ..agent.llm import init_chat_models
from ..config import (
    MODEL_CATALOG,
    get_effective,
    load_settings_dict,
    resolve_model_endpoint,
    save_settings_dict,
    set_effective,
)
from ..schemas import ModelCatalogItem, SettingsOut, SettingsUpdate

router = APIRouter(prefix="/api/settings", tags=["settings"])

# 设置项白名单
_KEYS = {
    "base_url",
    "model",
    "api_key",
    "tavily_api_key",
    "command_confirm_mode",
    "workspace",
}

# 推理等级合法值（其余一律按 Default 处理）
_EFFORTS = {"off", "low", "medium", "high", "xhigh"}


def _norm_effort(v: object) -> str | None:
    return v if v in _EFFORTS else None


def _catalog_status() -> list[ModelCatalogItem]:
    """模型目录各项的配置状态（凭证是否就绪，供前端提示与禁用）。"""
    items: list[ModelCatalogItem] = []
    for name, entry in MODEL_CATALOG.items():
        ep = resolve_model_endpoint(name)
        items.append(
            ModelCatalogItem(
                name=name,
                label=entry["label"],
                base_url=ep["base_url"],
                configured=bool(ep["api_key"]),
                supports_thinking=entry["thinking"] != "none",
                supports_vision=bool(entry.get("vision", False)),
                key_env=entry["key_env"],
            )
        )
    return items


@router.get("", response_model=SettingsOut)
async def get_settings():
    current_model = get_effective("model", "")
    ep = resolve_model_endpoint(current_model)
    return SettingsOut(
        base_url=get_effective("base_url", ""),
        model=current_model,
        has_api_key=bool(ep["api_key"]),
        current_model_configured=bool(ep["api_key"]),
        has_tavily_key=bool(get_effective("tavily_api_key", "")),
        command_confirm_mode=get_effective("command_confirm_mode", "auto"),
        workspace=get_effective("workspace", ""),
        model_options=list(MODEL_CATALOG.keys()),
        model_catalog=_catalog_status(),
        reasoning_effort=_norm_effort(get_effective("reasoning_effort", None)),
    )


@router.post("", response_model=SettingsOut)
async def update_settings(body: SettingsUpdate):
    from ..config import _SETTINGS_LOCK

    data = body.model_dump(exclude_none=True)
    # 推理等级：允许显式传 null 恢复 Default（exclude_none 会丢弃 null，需单独处理）
    if "reasoning_effort" in body.model_fields_set:
        await set_effective("reasoning_effort", _norm_effort(body.reasoning_effort))
    # 统一键名：前端传 api_key -> 存 openai_api_key
    for k, v in data.items():
        if k not in _KEYS:
            continue
        if k == "api_key":
            await set_effective("openai_api_key", v.strip() if isinstance(v, str) else v)
        elif k == "base_url":
            await set_effective("base_url", v.strip().rstrip("/") if isinstance(v, str) else v)
        else:
            await set_effective(k, v)

    target = (body.model or get_effective("model", "")).strip()

    # 模型专属 Key / Base URL / 清除：在同一个锁内完成读-改-写，避免并发写丢数据
    async with _SETTINGS_LOCK:
        user = load_settings_dict()
        changed = False

        if body.clear_model_api_key:
            keys = user.get("model_api_keys") or {}
            if target in keys:
                keys.pop(target, None)
                user["model_api_keys"] = keys
                changed = True

        if isinstance(body.model_api_key, str) and body.model_api_key.strip() and target:
            keys = user.setdefault("model_api_keys", {})
            keys[target] = body.model_api_key.strip()
            changed = True

        if isinstance(body.model_base_url, str) and target:
            urls = user.setdefault("model_base_urls", {})
            value = body.model_base_url.strip().rstrip("/")
            if value:
                urls[target] = value
            elif target in urls:
                urls.pop(target, None)
            changed = True

        if changed:
            save_settings_dict(user)

    return await get_settings()


@router.post("/test")
async def test_model(model: str = ""):
    """用指定模型（默认当前模型）发一次极短请求，验证端点/Key/模型是否可用。

    前端选择模型后可直接测试，无需先保存；测试不走降级模型，避免掩盖真实故障。
    """
    target = (model or "").strip() or str(get_effective("model", ""))
    started = time.time()
    try:
        llm = init_chat_models(reasoning_effort="off", model=target, use_fallback=False)
        from langchain_core.messages import HumanMessage

        resp = await asyncio.wait_for(llm.ainvoke([HumanMessage(content="ping")]), timeout=30)
        text = getattr(resp, "content", "")
        if isinstance(text, list):
            text = " ".join(str(x) for x in text if isinstance(x, str))
        return {
            "ok": True,
            "latency_ms": int((time.time() - started) * 1000),
            "model": target,
            "reply": str(text)[:200],
        }
    except TimeoutError:
        raise HTTPException(504, "模型连接超时（>30s）") from None
    except Exception as e:
        raise HTTPException(400, f"模型连接失败: {e}") from e
