"""LLM 工厂：按持久化设置创建 LangChain ChatModel。

统一入口：model = init_chat_models()
统一走 OpenAI 兼容协议：
- base_url 可指向阿里百炼 / DeepSeek / OpenAI / 任意兼容端点
- api_key 从持久化设置读取，其次环境变量 OPENAI_API_KEY / .env
- 使用 ChatDeepSeek：ChatOpenAI 子类，额外把流式 delta 中的 reasoning_content
  解析到 additional_kwargs，任意兼容端点的思考过程均可透出
- 原生 max_retries 重试；可选 FALLBACK_MODEL 降级（主模型失败且尚未输出时切换）
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

from ..config import get_effective, resolve_model_endpoint

logger = logging.getLogger(__name__)


def _create_model(
    model: str,
    *,
    temperature: float,
    reasoning_effort: str | None,
    extra: dict[str, Any] | None = None,
) -> BaseChatModel:
    ep = resolve_model_endpoint(model)
    if not ep["api_key"]:
        raise ValueError(
            f"模型 {model} 未配置 API Key，请任选其一：\n"
            f"1. 设置页选择「{ep['label']}」后填写该模型的 API Key；\n"
            f"2. 在后端 .env 中设置 {ep['key_env']}=你的密钥；\n"
            f"3. 配置全局 OPENAI_API_KEY 作为回退。"
        )

    # 思考参数适配：仅 qwen 系通过 extra_body 控制；reasoner 系自动思考；其余不传避免端点报错
    extra_body: dict | None = None
    if ep["thinking"] == "qwen":
        if reasoning_effort == "off":
            extra_body = {"enable_thinking": False}
        elif reasoning_effort in {"low", "medium", "high", "xhigh"}:
            extra_body = {"enable_thinking": True, "reasoning_effort": reasoning_effort}
        else:
            extra_body = {"enable_thinking": True}

    # 流式空闲超时：Qwen 长上下文/大文件生成时，服务端可能长时间不推流；
    # langchain_openai 默认 120s 会直接抛 StreamChunkTimeoutError 中断任务。
    # 放宽到可配置值（0 = 关闭该护栏），仍保留最终的重试与错误上报。
    try:
        chunk_timeout = float(get_effective("llm_stream_chunk_timeout", 180) or 0)
    except (TypeError, ValueError):
        chunk_timeout = 180.0
    return ChatDeepSeek(
        model=ep["model"],
        base_url=ep["base_url"],
        api_key=ep["api_key"],
        temperature=temperature,
        extra_body=extra_body,
        max_retries=int(get_effective("llm_max_retries", 2)),
        request_timeout=max(120, int(chunk_timeout) if chunk_timeout > 0 else 120),
        stream_chunk_timeout=chunk_timeout if chunk_timeout > 0 else None,
        **(extra or {}),
    )


class _FallbackChatModel:
    """简单降级包装：主模型失败且尚未产出任何 chunk 时切到备用模型。"""

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback
        self._llm_type = getattr(primary, "_llm_type", "fallback")

    @property
    def _default_params(self) -> dict:
        return getattr(self.primary, "_default_params", {})

    def bind_tools(self, tools: Any, **kwargs: Any) -> _FallbackChatModel:
        return _FallbackChatModel(
            self.primary.bind_tools(tools, **kwargs),
            self.fallback.bind_tools(tools, **kwargs),
        )

    def with_retry(self, **kwargs: Any) -> _FallbackChatModel:
        return self

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yielded = False
        try:
            async for chunk in self.primary.astream(messages, **kwargs):
                yielded = True
                yield chunk
            return
        except Exception as e:
            if yielded:
                raise
            logger.warning("主模型流式调用失败，切换降级模型: %s", e)
        async for chunk in self.fallback.astream(messages, **kwargs):
            yield chunk

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        try:
            return await self.primary.ainvoke(messages, **kwargs)
        except Exception as e:
            logger.warning("主模型调用失败，切换降级模型: %s", e)
            return await self.fallback.ainvoke(messages, **kwargs)


def init_chat_models(
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    model: str = "",
    use_fallback: bool = True,
    **kwargs,
) -> Any:
    """按当前有效设置创建 ChatModel（每次调用重新解析端点与凭证 → 切换模型实时生效）。

    reasoning_effort: 思考强度档位（前端发送栏可调），按模型思考能力适配。
    model: 显式指定模型（设置页“测试连接”用）；为空则用当前生效模型。
    use_fallback: 是否启用 FALLBACK_MODEL 降级（测试连接必须关闭，否则会掩盖主模型故障）。
    """
    model = (model or get_effective("model", "qwen3.7-flash")).strip()
    primary = _create_model(
        model, temperature=temperature, reasoning_effort=reasoning_effort, extra=kwargs
    )
    fallback = str(get_effective("fallback_model", "") or "").strip()
    if use_fallback and fallback and fallback != model:
        try:
            return _FallbackChatModel(
                primary,
                _create_model(
                    fallback, temperature=temperature, reasoning_effort=reasoning_effort, extra=kwargs
                ),
            )
        except Exception as e:
            logger.warning("降级模型 %s 创建失败，仅使用主模型: %s", fallback, e)
    return primary
