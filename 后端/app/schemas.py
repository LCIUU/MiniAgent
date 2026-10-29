"""Pydantic Schema（API 请求/响应）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------- 会话 ----------
class SessionOut(BaseModel):
    id: str
    title: str
    workspace: str
    provider: str
    model: str
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class SessionCreate(BaseModel):
    title: str = "新会话"
    workspace: str = ""


class SessionRename(BaseModel):
    title: str


# ---------- 消息 ----------
class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] | None = None
    duration_ms: int = 0
    created_at: str

    model_config = {"from_attributes": True}


class ToolCallOut(BaseModel):
    name: str
    args: dict[str, Any]
    result: str = ""
    status: str = "pending"  # running | success | error | rejected | cancelled
    duration_ms: int = 0


# ---------- 聊天请求 ----------
class AttachmentIn(BaseModel):
    """已上传附件的引用（上传接口返回的元信息原样回传）。"""

    name: str
    path: str
    size: int = 0
    kind: str = ""


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    workspace: str = ""
    attachments: list[AttachmentIn] = Field(default_factory=list)
    # 思考强度档位：off | low | medium | high | xhigh，None=跟随模型默认
    reasoning_effort: str | None = None


# ---------- 设置 ----------
class ModelCatalogItem(BaseModel):
    name: str
    label: str
    base_url: str
    configured: bool = False        # 是否已有可用 API Key（专属 Key / env / 全局回退）
    supports_thinking: bool = False  # 是否具备思考能力（决定前端推理档位可用性）
    supports_vision: bool = False    # 是否支持图片输入（多模态）
    key_env: str = ""               # 未配置时提示用的环境变量名


class SettingsOut(BaseModel):
    base_url: str = ""
    model: str = ""
    has_api_key: bool = False
    current_model_configured: bool = False
    has_tavily_key: bool = False
    command_confirm_mode: Literal["auto", "high_risk", "all"] = "auto"
    workspace: str = ""
    model_options: list[str] = Field(default_factory=list)
    model_catalog: list[ModelCatalogItem] = Field(default_factory=list)
    reasoning_effort: str | None = None  # 推理等级（null=跟随模型默认）


class SettingsUpdate(BaseModel):
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    model_api_key: str | None = None  # 当前所选模型的专属 API Key（与 model 搭配保存）
    model_base_url: str | None = None  # 当前所选模型的专属端点
    clear_model_api_key: bool | None = None  # 是否清除当前模型的专属 Key
    tavily_api_key: str | None = None
    command_confirm_mode: Literal["auto", "high_risk", "all"] | None = None
    workspace: str | None = None
    reasoning_effort: str | None = None  # 显式传 null 表示恢复 Default


# ---------- 工作区 ----------
class WorkspaceEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int = 0


class WorkspaceListOut(BaseModel):
    path: str
    parent: str | None
    entries: list[WorkspaceEntry]


# ---------- Agent 运行 ----------
class RunOut(BaseModel):
    run_id: str
    session_id: str
    status: str  # running | done | error | cancelled


class ApprovalDecision(BaseModel):
    approval_id: str
    approve: bool


class QuestionAnswer(BaseModel):
    question_id: str
    answers: list[str] = Field(default_factory=list)
