"""Agent 图状态：单循环 ReAct（规划由模型自主决定）的 LangGraph 状态定义。

注意：messages 使用覆盖语义（不加 reducer），因为每个节点都返回完整列表，
避免与旧状态重复合并。
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    # 用户请求
    request: str
    workspace: str

    # 消息历史（含工具消息），供 LLM 上下文
    messages: list[BaseMessage]

    # 计划（TodoListMiddleware 状态 schema：todos=[{content, status}]，由模型按需调用 write_todos 写入）
    # plan 为 todos 的纯文本投影（steps），供前端渲染与持久化
    todos: list[dict[str, str]]
    plan: list[str]

    # 工具执行
    tool_rounds: int

    # 上下文管理
    summary: str
    summary_count: int

    # 思考强度档位（off | low | medium | high | xhigh | None=跟随模型默认）
    reasoning_effort: str | None

    # 本次 run 使用的模型（运行期间固定，避免用户中途切换设置导致模型漂移）
    model: str

    # token 用量累计：{input_tokens, output_tokens, llm_calls}（模型返回为准）
    token_usage: dict[str, int]

    # 运行控制
    run_id: str
    session_id: str
    stopped: bool
    error: str | None

    # 事件回调：async callable (event: dict)
    emit: Any

    # 审批管理器（跨节点共享）
    approvals: Any

    # 停止事件（用户中断）
    stop_event: Any

    # 运行时注入的工具列表（不进 LLM 上下文）
    _tools: Any

    # 运行时注入的模型替身（仅测试用，不进 LLM 上下文）
    _llm: Any


def new_run_id() -> str:
    return uuid.uuid4().hex
