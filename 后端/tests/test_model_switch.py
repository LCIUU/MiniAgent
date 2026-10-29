"""模型切换与目录路由单测（不发真实网络请求）。

覆盖：
1. 目录模型按各自端点路由（qwen → 百炼，deepseek → DeepSeek），同进程切换即时生效
2. 凭证优先级：模型专属 Key > 服务商环境变量 > 全局 Key
3. 未配置时的明确错误提示（含环境变量名指引）
4. 思考参数按模型类型适配（qwen 传 enable_thinking，reasoner/none 不传）
5. 目录外自定义模型回退全局端点（兼容旧行为）
6. 设置 API 返回目录配置状态；模型专属 Key 保存后即时就绪
"""

import json

import pytest
from fastapi.testclient import TestClient

from app import config as config_mod
from app.agent.llm import init_chat_models
from app.main import app


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """隔离持久化设置文件与 env 凭证字段，返回设置文件路径。"""
    f = tmp_path / "agent_settings.json"
    f.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config_mod.settings, "settings_file", str(f))
    for field in (
        "openai_api_key",
        "dashscope_api_key",
        "dashscope_base_url",
        "deepseek_api_key",
        "deepseek_base_url",
    ):
        monkeypatch.setattr(config_mod.settings, field, "")
    return f


def _write(path, data: dict):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_catalog_routing_and_hot_switch(isolated):
    """同进程切换模型：请求路由到各自服务商端点，无需重启。"""
    _write(isolated, {"model": "qwen3.7-flash", "openai_api_key": "sk-g"})
    llm = init_chat_models()
    assert llm.model_name == "qwen3.7-flash"
    assert "dashscope.aliyuncs.com" in llm.openai_api_base

    _write(isolated, {"model": "deepseek-v4-pro", "openai_api_key": "sk-g"})
    llm = init_chat_models()
    assert llm.model_name == "deepseek-v4-pro"
    assert "api.deepseek.com" in llm.openai_api_base


def test_per_model_key_priority(isolated):
    """模型专属 Key 优先于全局 Key。"""
    _write(
        isolated,
        {
            "model": "deepseek-v4-flash",
            "openai_api_key": "sk-global",
            "model_api_keys": {"deepseek-v4-flash": "sk-per"},
        },
    )
    llm = init_chat_models()
    assert llm.api_key.get_secret_value() == "sk-per"


def test_env_key_fallback(isolated, monkeypatch):
    """无专属/全局 Key 时回退到服务商环境变量凭证。"""
    monkeypatch.setattr(config_mod.settings, "deepseek_api_key", "sk-env")
    _write(isolated, {"model": "deepseek-v4-flash"})
    llm = init_chat_models()
    assert llm.api_key.get_secret_value() == "sk-env"


def test_global_key_fallback(isolated):
    _write(isolated, {"model": "qwen-3.8flash", "openai_api_key": "sk-global"})
    assert init_chat_models().api_key.get_secret_value() == "sk-global"


def test_missing_key_message(isolated):
    """未配置任何凭证时给出明确指引（模型名 + 环境变量名 + 操作路径）。"""
    _write(isolated, {"model": "deepseek-v4-flash"})
    with pytest.raises(ValueError, match="未配置 API Key") as exc:
        init_chat_models()
    assert "DEEPSEEK_API_KEY" in str(exc.value)
    assert "设置页" in str(exc.value)


def test_thinking_adaptation(isolated):
    """qwen 系注入 enable_thinking；reasoner / none 不注入任何思考参数。"""
    _write(isolated, {"model": "qwen3.7-flash", "openai_api_key": "k"})
    assert init_chat_models(reasoning_effort="high")._default_params["extra_body"] == {  # noqa: SLF001
        "enable_thinking": True,
        "reasoning_effort": "high",
    }
    assert init_chat_models(reasoning_effort="off")._default_params["extra_body"] == {  # noqa: SLF001
        "enable_thinking": False
    }

    _write(isolated, {"model": "deepseek-v4-pro", "openai_api_key": "k"})
    assert init_chat_models(reasoning_effort="high")._default_params.get("extra_body") is None  # noqa: SLF001

    _write(isolated, {"model": "deepseek-v4-flash", "openai_api_key": "k"})
    assert init_chat_models()._default_params.get("extra_body") is None  # noqa: SLF001


def test_custom_model_fallback(isolated):
    """目录外自定义模型回退全局端点与 Key（兼容旧行为）。"""
    _write(
        isolated,
        {
            "model": "my-fine-tune",
            "base_url": "https://my-endpoint.example.com/v1",
            "openai_api_key": "sk-g",
        },
    )
    llm = init_chat_models()
    assert "my-endpoint.example.com" in llm.openai_api_base
    assert llm._default_params.get("extra_body") is None  # noqa: SLF001


def test_settings_api_catalog_and_per_model_key(isolated):
    """GET 返回 5 项目录与配置状态；POST 专属 Key 后对应模型即时就绪。"""
    client = TestClient(app)
    r = client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    assert data["model_options"] == [
        "qwen3.7-flash",
        "qwen3.8-flash",
        "qwen3.7-plus",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    by_name = {m["name"]: m for m in data["model_catalog"]}
    assert by_name["deepseek-v4-pro"]["configured"] is False
    assert by_name["deepseek-v4-pro"]["supports_thinking"] is True
    assert by_name["deepseek-v4-flash"]["supports_thinking"] is False
    assert by_name["qwen3.7-flash"]["key_env"] == "DASHSCOPE_API_KEY"

    r = client.post("/api/settings", json={"model": "deepseek-v4-pro", "model_api_key": "sk-new"})
    assert r.status_code == 200
    data = r.json()
    by_name = {m["name"]: m for m in data["model_catalog"]}
    assert by_name["deepseek-v4-pro"]["configured"] is True
    assert data["model"] == "deepseek-v4-pro"
