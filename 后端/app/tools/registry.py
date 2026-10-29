"""工具注册中心：工具元数据（ToolSpec）的注册、发现与查询。

用途：
- 执行层（graph.execute_node）按 spec 决策并行分组（idempotent）、超时、缓存策略
- 发现 API（GET /api/tools）输出 manifest 供前端展示
- MCP 等运行时加载的工具未显式注册，由 build_tools 统一补默认 spec

注册方式：在工具工厂内紧贴 @tool 下方加 @tool_spec(...) 装饰器；
工厂每次被调用时重复注册是幂等的（同名覆盖）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

# 风险等级：low（只读）| medium（写工作区）| high（系统级/外发）
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"


@dataclass
class ToolSpec:
    """工具元数据：治理与调度的决策依据。"""

    category: str = "general"  # fs | search | terminal | web | build | mcp | general
    risk_level: str = RISK_LOW
    timeout: float = 120.0  # 单次执行超时（秒）
    idempotent: bool = False  # 只读/幂等：允许并行执行
    cacheable: bool = True  # 是否允许结果跨 run 复用（全库搜索等易过期工具关闭）
    description: str = ""


# 默认 spec：未注册工具（含 MCP）按最保守策略——非幂等（串行）、不缓存
_DEFAULT_SPEC = ToolSpec()

_REGISTRY: dict[str, ToolSpec] = {}
_REGISTERED_AT: dict[str, float] = {}


def register_spec(name: str, spec: ToolSpec) -> None:
    """注册（或覆盖）工具元数据。"""
    _REGISTRY[name] = spec
    _REGISTERED_AT[name] = time.time()


def tool_spec(
    *,
    category: str = "general",
    risk_level: str = RISK_LOW,
    timeout: float = 120.0,
    idempotent: bool = False,
    cacheable: bool = True,
) -> Callable[[Any], Any]:
    """装饰器：把元数据绑定到 @tool 生成的 StructuredTool 上。

    用法（注意装饰顺序，@tool_spec 在 @tool 之外层）：

        @tool_spec(category="fs", idempotent=True, timeout=30)
        @tool(parse_docstring=True)
        async def read_file(...): ...
    """

    def deco(t: Any) -> Any:
        desc = (getattr(t, "description", None) or "").strip().splitlines()
        register_spec(
            t.name,
            ToolSpec(
                category=category,
                risk_level=risk_level,
                timeout=timeout,
                idempotent=idempotent,
                cacheable=cacheable,
                description=(desc[0] if desc else "")[:200],
            ),
        )
        return t

    return deco


def ensure_spec(name: str, *, category: str = "mcp") -> ToolSpec:
    """查询工具 spec；未注册则按保守默认补注册（MCP 等动态工具）。"""
    spec = _REGISTRY.get(name)
    if spec is None:
        spec = ToolSpec(category=category)
        register_spec(name, spec)
    return spec


def spec_of(name: str) -> ToolSpec:
    """查询工具 spec（只读，不触发注册）；未注册返回保守默认。"""
    return _REGISTRY.get(name, _DEFAULT_SPEC)


def manifest() -> list[dict[str, Any]]:
    """全部已注册工具的清单（供发现 API 输出）。"""
    return [
        {"name": name, **asdict(spec)}
        for name, spec in sorted(_REGISTRY.items(), key=lambda kv: kv[0])
    ]
