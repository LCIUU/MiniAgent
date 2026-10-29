"""思考强度（reasoning_effort）参数链路单测。

覆盖四个环节（不发真实网络请求）：
1. extra_body 按档位正确构造（llm.py）
2. extra_body 真正进入模型请求参数（ChatOpenAI._default_params → 请求载荷）
3. graph._make_llm 把 state["reasoning_effort"] 透传给模型工厂
4. ChatRequest API 模型能接收该参数
"""

import asyncio

import pytest

from app.agent import graph
from app.agent import llm as llm_mod
from app.agent.llm import init_chat_models
from app.schemas import ChatRequest


@pytest.fixture(autouse=True)
def _fake_settings(monkeypatch):
    """屏蔽真实配置/API Key 依赖，固定模型与密钥。"""

    def fake_get(key, default=None):
        return {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.8-max",
            "openai_api_key": "test-key",
        }.get(key, default)

    monkeypatch.setattr(llm_mod, "get_effective", fake_get)
    # 目录路由替身：固定端点与凭证，thinking 行为与目录外回退分支一致
    monkeypatch.setattr(
        llm_mod,
        "resolve_model_endpoint",
        lambda model: {
            "model": model,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "test-key",
            "thinking": "qwen" if model.lower().startswith("qwen3") else "none",
            "in_catalog": False,
            "key_env": "OPENAI_API_KEY",
            "label": model,
        },
    )


def _default_params(effort: str | None) -> dict:
    """按指定档位构建模型，返回会合并进请求载荷的默认参数。"""
    model = init_chat_models(reasoning_effort=effort)
    params = model._default_params  # noqa: SLF001 - 测试需要断言真实请求参数
    return params


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("off", {"enable_thinking": False}),
        ("low", {"enable_thinking": True, "reasoning_effort": "low"}),
        ("medium", {"enable_thinking": True, "reasoning_effort": "medium"}),
        ("high", {"enable_thinking": True, "reasoning_effort": "high"}),
        ("xhigh", {"enable_thinking": True, "reasoning_effort": "xhigh"}),
        # 未指定：qwen3 系列默认开启思考，但不设档位
        (None, {"enable_thinking": True}),
        # 非法档位：按未指定处理，绝不把脏值发给端点
        ("banana", {"enable_thinking": True}),
    ],
)
def test_extra_body_levels(effort, expected):
    params = _default_params(effort)
    assert params.get("extra_body") == expected


def test_off_has_no_effort_key():
    params = _default_params("off")
    assert "reasoning_effort" not in (params.get("extra_body") or {})


def test_none_for_non_qwen3_model(monkeypatch):
    """非 qwen3 模型且未指定档位时不注入任何 extra_body（兼容任意端点）。"""

    def fake_get(key, default=None):
        return {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "openai_api_key": "k"}.get(key, default)

    monkeypatch.setattr(llm_mod, "get_effective", fake_get)
    assert _default_params(None).get("extra_body") is None


def test_make_llm_passthrough(monkeypatch):
    """_make_llm 必须把 state 中的 reasoning_effort / model 原样传给模型工厂。"""
    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return "FAKE_LLM"

    monkeypatch.setattr(graph, "init_chat_models", fake_factory)

    out = asyncio.run(graph._make_llm({"reasoning_effort": "xhigh", "model": "qwen3.7-plus"}))
    assert out == "FAKE_LLM"
    assert captured == {"reasoning_effort": "xhigh", "model": "qwen3.7-plus"}

    captured.clear()
    out = asyncio.run(graph._make_llm({}))
    assert captured == {"reasoning_effort": None, "model": ""}


def test_make_llm_prefers_injected_override(monkeypatch):
    """注入 state['_llm'] 替身时不走工厂（reasoning_effort 不应触发真实构建）。"""
    calls = []

    def fake_factory(**kwargs):
        calls.append(kwargs)
        return "FAKE_LLM"

    monkeypatch.setattr(graph, "init_chat_models", fake_factory)
    out = asyncio.run(graph._make_llm({"_llm": "OVERRIDE", "reasoning_effort": "low"}))
    assert out == "OVERRIDE"
    assert calls == []


def test_chat_request_accepts_reasoning_effort():
    req = ChatRequest(message="hi", reasoning_effort="xhigh")
    assert req.reasoning_effort == "xhigh"
    assert ChatRequest(message="hi").reasoning_effort is None
