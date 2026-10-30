"""Agent API：发起聊天（返回 run_id）、SSE 事件流、WebSocket 停止/审批。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..agent.context import extract_summary, extract_summary_cutoff
from ..config import get_effective, resolve_model_endpoint
from ..database import get_db
from ..models import Message, Session
from ..schemas import ApprovalDecision, ChatRequest, QuestionAnswer, RunOut
from ..services.attachments import (
    AttachmentError,
    build_attachment_block,
    image_data_url,
    kind_of,
    save_upload,
    validate_stored_path,
)
from ..services.paths import ensure_workspace
from ..services.runtime import _DONE, RunBusyError, manager
from ..tools.mcp import load_mcp_tools

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])


_WRITE_TOOLS = {"write_file", "edit_file", "apply_patch", "multi_edit", "create_file"}
_READ_TOOLS = {"read_file", "list_dir", "list_files", "glob", "grep", "search_files"}
_SHELL_TOOLS = {"terminal", "run_command", "bash", "shell"}


def _tool_digest(calls: list) -> str:
    """把上一轮工具调用压成分组摘要，供下一轮快速了解改过/读过/跑过什么。"""
    groups: dict[str, list[str]] = {"改动": [], "读取": [], "执行": [], "失败": [], "其他": []}
    for tc in calls[:24]:
        if not isinstance(tc, dict):
            continue
        name = str(tc.get("name") or "?")
        args = tc.get("args") or {}
        hint = ""
        if isinstance(args, dict):
            hint = (
                args.get("path")
                or args.get("directory")
                or args.get("dir")
                or args.get("command")
                or args.get("source")
                or ""
            )
        label = f"{name}({str(hint)[:80]})" if hint else name
        if str(tc.get("status") or "success") != "success":
            reason = str(tc.get("result") or "").replace("\n", " ").strip()
            groups["失败"].append(label + (f": {reason[:80]}" if reason else ""))
        elif name in _WRITE_TOOLS:
            groups["改动"].append(label)
        elif name in _READ_TOOLS:
            groups["读取"].append(label)
        elif name in _SHELL_TOOLS:
            groups["执行"].append(label)
        else:
            groups["其他"].append(label)
    parts = [f"{k}: {', '.join(v)}" for k, v in groups.items() if v]
    return ("\n[上轮操作] " + "；".join(parts)[:900]) if parts else ""


def _json_content(value: Any) -> Any:
    """把消息 content 转成可 JSON 序列化的结构（文本或多模态 block 列表）。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_content(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_content(v) for k, v in value.items()}
    return str(value)


def _ensure_tool_pairs(items: list[dict]) -> list[dict]:
    """给缺少结果的 AI tool_calls 补占位 ToolMessage，避免历史里 tool_call_id 悬空。"""
    out: list[dict] = []
    pending: list[str] = []

    def _flush() -> None:
        for call_id in pending:
            out.append({
                "role": "tool",
                "content": "（该工具调用未返回结果：本轮被中断）",
                "tool_call_id": call_id,
            })
        pending.clear()

    for item in items:
        role = item.get("role")
        if pending and role != "tool":
            _flush()
        out.append(item)
        if role == "ai":
            for tc in item.get("tool_calls") or []:
                call_id = str(tc.get("id") or "") if isinstance(tc, dict) else ""
                if call_id:
                    pending.append(call_id)
        elif role == "tool":
            call_id = str(item.get("tool_call_id") or "")
            if call_id in pending:
                pending.remove(call_id)
    _flush()
    return out


def _serialize_run_messages(messages: list, initial_ids: set[int]) -> list[dict]:
    """提取本轮新产生的消息（排除系统提示、摘要、内部提醒、初始历史）。

    摘要之前的消息已由摘要代表，只持久化摘要之后的尾部；否则下一轮会把
    同一行里的完整工具流重新载入，再次触发压缩。
    """
    last_summary = -1
    for i, m in enumerate(messages):
        if (
            id(m) not in initial_ids
            and isinstance(m, HumanMessage)
            and getattr(m, "additional_kwargs", {}).get("is_summary")
        ):
            last_summary = i
    items: list[dict] = []
    for i, m in enumerate(messages):
        if i <= last_summary or id(m) in initial_ids:
            continue
        if isinstance(m, SystemMessage):
            # 头部系统提示每轮重建、不入库；内部提醒（is_nudge）必须原样入库：
            # 它们当时就在模型上下文里，丢掉会让下一轮重建的历史与上一轮实际发送的
            # 前缀不一致，提醒之后的所有内容都无法命中隐式缓存。
            if getattr(m, "additional_kwargs", {}).get("is_nudge"):
                items.append({"role": "system", "content": _json_content(m.content)})
            continue
        if isinstance(m, HumanMessage):
            extra = getattr(m, "additional_kwargs", {})
            if extra.get("is_summary") or extra.get("is_nudge"):
                continue
            items.append({"role": "human", "content": _json_content(m.content)})
        elif isinstance(m, AIMessage):
            items.append({
                "role": "ai",
                "content": _json_content(m.content),
                "tool_calls": _json_content(list(m.tool_calls or [])),
            })
        elif isinstance(m, ToolMessage):
            item = {
                "role": "tool",
                "content": _json_content(m.content),
                "tool_call_id": m.tool_call_id or "",
            }
            if getattr(m, "name", None):
                item["name"] = m.name
            items.append(item)
    return _ensure_tool_pairs(items)


def _messages_from_run(raw: Any, db_ts: float | None = None) -> list:
    """把持久化的完整消息流还原为 LangChain 消息（AI tool_calls + ToolMessage 成对）。

    db_ts 标记这些消息所属 DB 行的创建时间，供摘要计算覆盖截止点。
    """
    out = []
    if not isinstance(raw, list):
        return out
    extra: dict[str, Any] = {"db_ts": db_ts} if db_ts is not None else {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if content is None:
            content = ""
        if role == "human":
            out.append(HumanMessage(content=content, additional_kwargs=dict(extra)))
        elif role == "system":
            # 历史里的内部提醒：原样放回原位，保证前缀与上一轮实际发送的一致
            out.append(SystemMessage(content=content, additional_kwargs={**extra, "is_nudge": True}))
        elif role == "ai":
            out.append(AIMessage(
                content=content,
                tool_calls=item.get("tool_calls") or [],
                additional_kwargs=dict(extra),
            ))
        elif role == "tool":
            kwargs: dict[str, Any] = {
                "content": content,
                "tool_call_id": str(item.get("tool_call_id") or ""),
                "additional_kwargs": dict(extra),
            }
            if item.get("name"):
                kwargs["name"] = str(item["name"])
            out.append(ToolMessage(**kwargs))
    return out


def _drop_orphan_tool_messages(messages: list) -> list:
    """丢弃没有对应 AI tool_calls 的 ToolMessage。

    旧数据可能因摘要截断问题以孤立工具结果开头；直接发给模型会违反
    OpenAI 工具消息协议（tool 消息必须紧跟在带 tool_calls 的 assistant 之后）。
    """
    out: list = []
    pending: set[str] = set()
    for m in messages:
        if isinstance(m, AIMessage):
            pending = {str(tc.get("id") or "") for tc in (m.tool_calls or [])}
            out.append(m)
        elif isinstance(m, ToolMessage):
            call_id = str(m.tool_call_id or "")
            if call_id in pending:
                pending.discard(call_id)
                out.append(m)
        else:
            pending.clear()
            out.append(m)
    return out


def _final_summary(state: dict[str, Any]) -> str:
    """取本轮最终摘要：中途压缩只更新消息流，state["summary"] 可能仍是旧值。"""
    return extract_summary(state.get("messages") or []) or str(state.get("summary") or "")


def _db_ts_of(m: Any) -> float | None:
    """DB 行创建时间（epoch 秒），作为摘要覆盖范围的边界标记。"""
    dt = getattr(m, "created_at", None)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _history_from_db(messages: list[Message]) -> list:
    """把 DB 消息还原为 LangChain 消息列表（用于多轮上下文）。

    新数据带有 context_messages（本轮完整 LLM 消息流：AI tool_calls、工具结果、
    中间文本），直接原样重建，保证跨轮不丢工具信息；旧数据没有该列时回退到
    "最终回答 + 工具操作摘要" 的压缩形式。每条消息带 db_ts，供摘要记录覆盖截止点。
    """
    out = []
    for m in messages:
        atts = m.attachments or []
        att_note = ""
        if atts:
            names = "、".join(str(a.get("name", "")) for a in atts if isinstance(a, dict))
            if names:
                att_note = f"\n[本轮附件: {names}]"
        ts = _db_ts_of(m)
        extra: dict[str, Any] = {"db_ts": ts} if ts is not None else {}
        if m.role == "user":
            # 新数据带 request_content（含附件提取正文）；旧数据回退到纯文本 + 附件名。
            # 原样使用、不做 strip：本轮实际发送的就是这个字符串，任何首尾空白差异
            # 都会让重建的历史与上一轮的前缀对不上，其后的上下文全部无法命中缓存。
            stored_request = getattr(m, "request_content", None)
            if isinstance(stored_request, str) and stored_request.strip():
                content = stored_request
            else:
                content = (m.content or "") + att_note
            out.append(HumanMessage(content=content, additional_kwargs=dict(extra)))
        elif m.role == "assistant":
            stored = getattr(m, "context_messages", None)
            if stored:
                out.extend(_messages_from_run(stored, ts))
            else:
                tool_note = _tool_digest(m.tool_calls or [])
                out.append(
                    AIMessage(content=(m.content or "") + tool_note, additional_kwargs=dict(extra))
                )
    return _drop_orphan_tool_messages(out)


def _timeline_from_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从事件历史重建执行过程流时间轴（与前端 useAgent 分段规则一致）。

    分段规则（思考与阶段正文各自独立的 forceNew 标志）：
    - thinking delta：追加到最后一个 thinking 条目；plan/tool_*/token 之后强制新段
    - token delta：追加到最后一个 text 条目（阶段正文）；plan/tool_* 之后强制新段
    - tool_start → 新建 tool 条目；tool_end → 回填状态与结果
    - plan → 覆盖最后一个 plan 条目（无则追加）
    - summary_start → 新建 compress 条目；summary_done → 回填状态、耗时与 token
    最终回答即时间轴中最后一个 text 条目。
    """
    timeline: list[dict[str, Any]] = []
    force_think = False
    force_text = False
    # 用 dict 维护 pending 工具，避免每次 tool_end/diff 都线性扫描 timeline
    pending_tools: dict[str, dict[str, Any]] = {}

    def _close_thinking(end_ts: float) -> None:
        """给最后一段已结束的思考补时长。"""
        if timeline and timeline[-1].get("kind") == "thinking" and timeline[-1].get("duration_ms") is None:
            timeline[-1]["duration_ms"] = round((end_ts - timeline[-1]["start"]) * 1000)

    for ev in history:
        etype = ev.get("type")
        ts = ev.get("_ts", 0.0)
        if etype == "thinking":
            delta = ev.get("delta", "")
            if not delta:
                continue
            last = timeline[-1] if timeline else None
            if last and last.get("kind") == "thinking" and not force_think:
                last["text"] += delta
                if ev.get("duration_ms") is not None:
                    last["duration_ms"] = ev["duration_ms"]
            else:
                timeline.append({
                    "kind": "thinking",
                    "text": delta,
                    "start": ts,
                    "duration_ms": ev.get("duration_ms"),
                })
            force_think = False
        elif etype == "thinking_end":
            # 思考段真实结束时刻由后端在生成结束时下发（含被闸门暂存的短思考），
            # 用它收口，避免历史重建按下一个事件时间算出包含工具参数生成的假耗时
            if timeline and timeline[-1].get("kind") == "thinking" and timeline[-1].get("duration_ms") is None:
                timeline[-1]["duration_ms"] = ev.get("duration_ms") or round(
                    (ts - timeline[-1].get("start", ts)) * 1000
                )
        elif etype == "token":
            delta = ev.get("delta", "")
            if not delta:
                continue
            force_think = True  # token 之后到来的思考属于新阶段
            _close_thinking(ts)
            last = timeline[-1] if timeline else None
            if last and last.get("kind") == "text" and not force_text:
                last["text"] += delta
            else:
                timeline.append({"kind": "text", "text": delta})
            force_text = False
        elif etype == "tool_start":
            force_think = True
            force_text = True
            _close_thinking(ts)
            record = {
                "kind": "tool",
                "call_id": ev.get("call_id", ""),
                "name": ev.get("tool", ""),
                "args": ev.get("args", {}),
                "status": "running",
                "result": "",
                "duration_ms": 0,
                "started_at": ev.get("started_at", ""),
            }
            timeline.append(record)
            pending_tools[ev.get("call_id", "")] = record
        elif etype == "tool_end":
            force_think = True
            force_text = True
            call_id = ev.get("call_id", "")
            record = pending_tools.pop(call_id, None)
            if record is not None:
                # 读文件类工具在历史卡片中保留全文（用户要求不截断），其余工具仍限制体积
                result_text = str(ev.get("result", ""))
                record["result"] = (
                    result_text if record.get("name") in ("read_file", "read_files")
                    else result_text[:4000]
                )
                record["status"] = ev.get("status", "success")
                record["duration_ms"] = ev.get("duration_ms", 0)
        elif etype == "diff":
            force_think = True
            force_text = True
            call_id = ev.get("call_id", "")
            if call_id:
                record = pending_tools.get(call_id)
                if record is not None:
                    record["diff"] = str(ev.get("diff", ""))[:12000]
            else:
                # 无 call_id 的旧事件：回退到最近一个仍在运行的工具
                for it in reversed(timeline):
                    if it.get("kind") == "tool" and it.get("status") == "running":
                        it["diff"] = str(ev.get("diff", ""))[:12000]
                        break
        elif etype == "plan":
            force_think = True
            force_text = True
            _close_thinking(ts)
            steps = ev.get("steps") or []
            if not steps:
                continue
            todos = ev.get("todos") or []
            for i in range(len(timeline) - 1, -1, -1):
                if timeline[i].get("kind") == "plan":
                    timeline[i]["steps"] = steps
                    if todos:
                        timeline[i]["todos"] = todos
                    break
            else:
                plan_item: dict[str, Any] = {"kind": "plan", "steps": steps}
                if todos:
                    plan_item["todos"] = todos
                timeline.append(plan_item)
        elif etype == "summary_start":
            force_think = True
            force_text = True
            _close_thinking(ts)
            timeline.append({
                "kind": "compress",
                "status": "running",
                "start": ts,
                "before_tokens": ev.get("before_tokens"),
            })
        elif etype == "summary_done":
            force_think = True
            force_text = True
            _close_thinking(ts)
            for i in range(len(timeline) - 1, -1, -1):
                item = timeline[i]
                if item.get("kind") == "compress" and item.get("status") == "running":
                    item["status"] = "done"
                    item["failed"] = not bool(ev.get("ok", True))
                    item["duration_ms"] = ev.get("duration_ms") or round(
                        (ts - float(item.get("start") or ts)) * 1000
                    )
                    item["after_tokens"] = ev.get("after_tokens")
                    item["summary"] = str(ev.get("summary") or "")[:12000]
                    break

    # 兜底：最后一段思考以运行结束为终点
    if history:
        _close_thinking(history[-1].get("_ts", 0.0))

    # 未收到 tool_end 的工具（运行被中断/进程崩溃）标记为 cancelled，
    # 否则刷新后卡片会永远停在"运行中"转圈
    for record in pending_tools.values():
        record["status"] = "cancelled"
    # 被中断的压缩过程同样收尾，避免历史里显示"进行中"和异常巨大的耗时
    for item in timeline:
        if item.get("kind") == "compress" and item.get("status") == "running":
            item["status"] = "done"
            item["failed"] = True
            item["duration_ms"] = item.get("duration_ms") or 0

    return timeline


async def _persist_result(handle, session_id: str) -> None:
    """运行结束后：把最终 assistant 消息 + 工具调用写入 DB。"""
    from ..database import SessionLocal

    state = handle.final_state or {}
    messages = state.get("messages", [])
    # 最终回答 = 本轮（最后一条用户消息之后）最后一条不含工具调用的非空 assistant 消息。
    # 必须限定本轮：state.messages 含完整历史，中断/出错时直接取最后一条会把上一轮的
    # 回答重复写成这一轮的结果。
    last_user_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_user_idx = i
    final_text = ""
    for m in messages[last_user_idx + 1 :]:
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            content = m.content if isinstance(m.content, str) else ""
            if content.strip():
                final_text = content
    final_text = final_text.strip()

    timeline = _timeline_from_history(handle.history)
    if not final_text:
        # final_state 缺失（取消/出错）时，从事件历史重建已产生的最后一段正文，
        # 保住中断前的输出，刷新页面仍可见
        for it in timeline:
            if it.get("kind") == "text" and (it.get("text") or "").strip():
                final_text = it["text"].strip()

    # 从事件历史重建工具调用记录（按 call_id 关联，兼容并行工具的乱序结束）
    tool_records: list[dict[str, Any]] = []
    pending_tools: dict[str, dict[str, Any]] = {}
    for ev in handle.history:
        etype = ev.get("type")
        if etype == "tool_start":
            record = {
                "name": ev.get("tool", ""),
                "args": ev.get("args", {}),
                "result": "",
                "status": "running",
                "duration_ms": 0,
                "call_id": ev.get("call_id", ""),
            }
            tool_records.append(record)
            pending_tools[str(ev.get("call_id", ""))] = record
        elif etype == "diff":
            record = pending_tools.get(str(ev.get("call_id", "")))
            if record is not None:
                record["diff"] = str(ev.get("diff", ""))[:12000]
        elif etype == "tool_end":
            call_id = str(ev.get("call_id", ""))
            record = pending_tools.pop(call_id, None)
            if record is None:
                # 没有 call_id 的旧事件：回退到最近一个仍在运行的工具
                for candidate in reversed(tool_records):
                    if candidate.get("status") == "running":
                        record = candidate
                        break
            if record is not None:
                # 读文件类工具在历史卡片中保留全文（用户要求不截断），其余工具仍限制体积
                result_text = str(ev.get("result", ""))
                record["result"] = (
                    result_text if record.get("name") in ("read_file", "read_files")
                    else result_text[:4000]
                )
                record["status"] = ev.get("status", "success")
                record["duration_ms"] = ev.get("duration_ms", 0)

    # 未收到 tool_end 的调用（中断/崩溃）标记为 cancelled，避免历史卡片永远转圈
    for record in pending_tools.values():
        record["status"] = "cancelled"

    # 运行失败必须落进消息正文：否则刷新页面后错误事件消失，只剩一条空回复，
    # 用户只能看到“跑着跑着停了”。已有部分回答时把中断原因追加在末尾。
    error = str(state.get("error") or "").strip()
    if error:
        notice = f"【运行中断】{error}"
        final_text = f"{final_text}\n\n{notice}".strip() if final_text else notice
    elif not final_text and not tool_records:
        final_text = "（运行被中断或无输出）"

    # 本轮完整消息流：按对象身份排除启动时已有的历史/摘要/系统提示，只留本轮新产生的
    context_messages: list[dict] = []
    initial_refs = getattr(handle, "initial_message_refs", None) or []
    if initial_refs and messages:
        context_messages = _serialize_run_messages(messages, {id(m) for m in initial_refs})
        if final_text:
            last_ai = next(
                (it for it in reversed(context_messages) if it.get("role") == "ai"), None
            )
            last_content = last_ai.get("content") if last_ai else None
            if (
                last_ai is None
                or last_ai.get("tool_calls")
                or not isinstance(last_content, str)
                or last_content.strip() != final_text
            ):
                context_messages.append({"role": "ai", "content": final_text, "tool_calls": []})
        context_messages = _ensure_tool_pairs(context_messages)
        if context_messages:
            logger.info(
                "持久化完整消息流: run=%s %d 条（含 AI tool_calls 与工具结果）",
                getattr(handle, "run_id", ""), len(context_messages),
            )

    async with SessionLocal() as db:
        session = await db.get(Session, session_id)
        if session is None:
            return
        # 摘要可能在本轮中途（plan/execute）生成：state["summary"] 只在
        # context_node 更新，必须从最终消息流提取，否则跨轮摘要丢失、
        # 下一轮会重新把全部原始历史当上下文并再次压缩。
        summary = _final_summary(state)
        if summary:
            session.summary = summary
            cutoff = extract_summary_cutoff(messages)
            if cutoff is not None:
                session.summary_cutoff = datetime.fromtimestamp(cutoff, tz=UTC)
            logger.info(
                "持久化上下文摘要: run=%s %d 字，覆盖截止=%s",
                getattr(handle, "run_id", ""), len(summary), session.summary_cutoff,
            )
        started_at = getattr(handle, "started_at", 0.0)
        ended_at = getattr(handle, "ended_at", 0.0) or time.time()
        duration_ms = int((ended_at - started_at) * 1000) if started_at else 0
        msg = Message(
            session_id=session_id,
            role="assistant",
            content=final_text,
            tool_calls=tool_records,
            timeline=timeline,
            context_messages=context_messages or None,
            token_usage=state.get("token_usage"),
            duration_ms=max(0, duration_ms),
        )
        db.add(msg)
        await db.commit()


@router.post("/upload")
async def upload_attachments(
    workspace: str = Form("", description="目标工作区（必须属于当前全局工作区或已有会话）"),
    files: list[UploadFile] = File(default_factory=list),
    db: AsyncSession = Depends(get_db),
):
    """上传附件（PDF/Word/图片/文本），保存到工作区 .agent_uploads/ 并返回元信息。"""
    from ..services.attachments import MAX_FILE_SIZE

    requested = (workspace or get_effective("workspace", "")).strip()
    if not requested:
        raise HTTPException(400, "未设置工作区，无法上传附件")
    try:
        ws = ensure_workspace(requested)
    except (ValueError, OSError) as e:
        raise HTTPException(400, f"工作区无效: {e}") from e

    # 只允许写入当前全局工作区或已有会话绑定的工作区，避免任意目录写入
    global_ws = (get_effective("workspace", "") or "").strip()
    allowed = {global_ws} if global_ws else set()
    rows = await db.execute(select(Session.workspace).distinct())
    allowed.update(str(x).strip() for (x,) in rows.fetchall() if x and str(x).strip())

    if os.path.normcase(os.path.realpath(ws)) not in {
        os.path.normcase(os.path.realpath(p)) for p in allowed if p
    }:
        raise HTTPException(403, "不允许向该工作区上传附件，请先在界面中选择工作区")

    if not files:
        raise HTTPException(400, "未选择文件")
    saved, errors = [], []
    for f in files:
        try:
            data = await f.read(MAX_FILE_SIZE + 1)
            if len(data) > MAX_FILE_SIZE:
                raise AttachmentError(f"文件过大（>20MB）: {f.filename}")
            saved.append(save_upload(ws, f.filename or "", data))
        except AttachmentError as e:
            errors.append(str(e))
        except OSError as e:
            errors.append(f"{f.filename}: 保存失败（{e}）")
    if not saved:
        raise HTTPException(400, "；".join(errors) or "附件保存失败")
    return {"files": saved, "errors": errors}


@router.post("/chat", response_model=RunOut)
async def start_chat(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    """发起一次 Agent 运行，立即返回 run_id；事件通过 SSE 消费。"""
    if not body.message.strip():
        raise HTTPException(400, "消息不能为空")

    workspace = (body.workspace or get_effective("workspace", "")).strip()
    if not workspace:
        raise HTTPException(400, "未设置工作区，请先在工作区选择器中指定目录")
    try:
        workspace = ensure_workspace(workspace)
    except (ValueError, OSError) as e:
        raise HTTPException(400, f"工作区无效: {e}") from e

    # 会话：新建或复用
    session_id = body.session_id
    if session_id:
        session = await db.get(Session, session_id)
        if session is None:
            raise HTTPException(404, "会话不存在")
        if not session.workspace:
            session.workspace = workspace
        else:
            workspace = session.workspace
    else:
        title = re.sub(r"\s+", " ", body.message).strip()[:30] or "新会话"
        session = Session(title=title, workspace=workspace)
        db.add(session)
        await db.commit()
        await db.refresh(session)
        session_id = session.id

    # 记录本会话实际使用的模型/服务商（会话列表与审计用）
    model = str(get_effective("model", "") or "")
    session.provider = "openai-compatible"
    session.model = model
    await db.commit()

    # 历史消息：限制最近 200 条；已有摘要时跳过摘要覆盖过的旧消息，
    # 否则每轮都会把已摘要的全量历史重新计入 token，导致反复触发压缩。
    summary_active = bool(session.summary.strip())
    history_stmt = select(Message).where(Message.session_id == session_id)
    if summary_active and session.summary_cutoff is not None:
        history_stmt = history_stmt.where(Message.created_at > session.summary_cutoff)
    result = await db.execute(
        history_stmt.order_by(Message.created_at.desc()).limit(200)
    )
    history = _history_from_db(list(reversed(result.scalars().all())))

    # 用户消息入库：attachments 独立存 JSON 列（前端渲染图片缩略图用），
    # content 保持纯文本，不再拼 "[附件] xxx" 行（避免历史消息里图片退化为文本）
    atts = [a.model_dump() for a in body.attachments]
    user_msg = Message(
        session_id=session_id,
        role="user",
        content=body.message,
        attachments=atts or [],
    )
    db.add(user_msg)
    await db.commit()

    # 附件分类注入：
    # - 文本类（PDF/Word/文本）→ 提取正文拼入请求文本（不变）
    # - 图片类 → 视觉模型走多模态 image_url 注入（base64 data URL）；
    #            非视觉模型降级为文本说明（明确告知模型"用户上传了图片但当前模型不支持图像识别"）
    text_atts = [a for a in atts if (a.get("kind") or kind_of(a.get("name", ""))) != "image"]
    image_atts = [a for a in atts if (a.get("kind") or kind_of(a.get("name", ""))) == "image"]

    request_message = body.message + build_attachment_block(text_atts, workspace)

    image_blocks: list[dict[str, Any]] = []
    if image_atts:
        ep = resolve_model_endpoint(get_effective("model", ""))
        if ep["vision"]:
            failed_names: list[str] = []
            for att in image_atts:
                try:
                    url = image_data_url(workspace, att["path"])
                    image_blocks.append({"type": "image_url", "image_url": {"url": url}})
                except AttachmentError:
                    failed_names.append(att.get("name", "未知文件"))
            if failed_names:
                # 只告知文件名，不暴露内部技术错误（防止模型向用户复述报错文案）
                request_message += (
                    f"\n\n【图片读取失败】以下图片未能成功读取：{'、'.join(failed_names)}。"
                    "请告知用户这些图片无法识别，建议重新上传。"
                )
            if image_blocks:
                request_message += f"\n\n【用户上传图片】共 {len(image_blocks)} 张，已作为图片输入随本条消息一并提供，请直接识别图片内容。"
        else:
            names = "、".join(a.get("name", "") for a in image_atts)
            request_message += (
                f"\n\n【用户上传图片】{names}。当前模型不支持图像识别，无法读取图片内容；"
                "如用户的问题依赖图片内容，请如实告知并建议切换到支持视觉的模型（如 Qwen 系列）。"
            )

    # 保存本轮实际发送内容（含附件提取正文与图片说明），供下一轮还原上下文；
    # 追问附件内容时不再只看到文件名
    user_msg.request_content = request_message
    await db.commit()

    # 加载 MCP 工具（失败不影响主流程）
    mcp_tools, mcp_errors = [], []
    try:
        mcp_tools, mcp_errors = await load_mcp_tools(workspace, lambda ev: asyncio.sleep(0))
    except Exception as e:
        mcp_errors = [str(e)]

    try:
        handle = await manager.start(
            session_id=session_id,
            request=request_message,
            workspace=workspace,
            history=history,
            summary=session.summary,
            summary_cutoff=(
                session.summary_cutoff.timestamp()
                if summary_active and session.summary_cutoff is not None
                else None
            ),
            persist=lambda h: _persist_result(h, session_id),
            mcp_tools=mcp_tools,
            reasoning_effort=body.reasoning_effort,
            image_blocks=image_blocks,
            model=model,
        )
    except RunBusyError as e:
        # 用户消息已入库：run 未启动时回滚，避免重试产生重复提问污染上下文
        await db.delete(user_msg)
        await db.commit()
        raise HTTPException(409, str(e)) from e
    if mcp_errors:
        # MCP 加载失败不影响主流程：标记非致命，前端只提示、不中断事件流
        await handle.emit({
            "type": "error",
            "message": "MCP: " + "; ".join(mcp_errors),
            "fatal": False,
        })

    return RunOut(run_id=handle.run_id, session_id=session_id, status="running")


@router.get("/attachment_file")
async def get_attachment_file(
    path: str = Query(..., description="附件存储路径"),
    db: AsyncSession = Depends(get_db),
):
    """下载附件原始文件（图片缩略图渲染、文件下载共用）。

    安全：校验路径属于数据库中任一会话的工作区 .agent_uploads 内，
    支持跨工作区访问（用户切换全局工作区后，旧会话附件仍可正常渲染）。
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "附件文件不存在")

    # 收集所有已知工作区（含当前全局设置），逐一校验 path 是否在其 .agent_uploads 内
    workspaces = {get_effective("workspace", "").strip()}
    rows = await db.execute(select(Session.workspace).distinct())
    for (ws,) in rows.fetchall():
        if ws and ws.strip():
            workspaces.add(ws.strip())

    for ws in workspaces:
        if validate_stored_path(ws, path):
            return FileResponse(str(p), filename=p.name)

    raise HTTPException(403, "无权访问该附件路径")


@router.get("/sessions/{session_id}/active_run")
async def get_active_run(session_id: str):
    """查询该会话正在进行中的运行（前端切换会话/刷新后重连接管用）。"""
    h = manager.find_active(session_id)
    if h is None:
        return {"run_id": None}
    return {"run_id": h.run_id, "started_at": h.started_at}


@router.get("/runs/{run_id}/stream")
async def stream_run(
    run_id: str,
    request: Request,
    since: int = Query(0, ge=0, description="只回放 seq 大于该值的事件（断线重连用）"),
):
    """SSE 事件流：回放历史事件 → 实时推送；带心跳与事件序号。"""
    handle = manager.get(run_id)
    if handle is None:
        raise HTTPException(404, "运行不存在或已结束")

    # 兼容标准 EventSource 的 Last-Event-ID
    since_seq = since
    header_last = request.headers.get("last-event-id", "").strip()
    if header_last.isdigit():
        since_seq = max(since_seq, int(header_last))

    async def event_gen():
        q = handle.subscribe(since_seq)
        heartbeat = max(5, int(get_effective("sse_heartbeat_seconds", 15)))
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=heartbeat)
                except TimeoutError:
                    # SSE 注释行：保持连接活跃，避免代理/浏览器空闲断开
                    yield ": ping\n\n"
                    continue
                if item is _DONE:
                    break
                seq = item.get("seq")
                prefix = f"id: {seq}\n" if isinstance(seq, int) and seq > 0 else ""
                yield f"{prefix}data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            handle.unsubscribe(q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str):
    ok = manager.stop(run_id)
    if not ok:
        raise HTTPException(404, "运行不存在或已结束")
    return {"ok": True}


@router.post("/runs/{run_id}/approval", response_model=dict)
async def decide_approval(run_id: str, body: ApprovalDecision):
    ok = manager.decide(run_id, body.approval_id, body.approve)
    if not ok:
        raise HTTPException(404, "审批请求不存在或已处理")
    return {"ok": True}


@router.post("/runs/{run_id}/answer", response_model=dict)
async def answer_question(run_id: str, body: QuestionAnswer):
    """回答 ask_user / 删除确认等结构化提问。"""
    ok = manager.answer(run_id, body.question_id, body.answers)
    if not ok:
        raise HTTPException(404, "提问不存在或已回答")
    return {"ok": True}


@router.websocket("/ws/{run_id}")
async def ws_control(ws: WebSocket, run_id: str):
    """WebSocket 控制通道：客户端 → 服务端。

    消息格式: {"type": "stop"}
            | {"type": "approval", "approval_id": "...", "approve": true}
            | {"type": "answer", "question_id": "...", "answers": ["..."]}
    """
    token = str(get_effective("api_token", "") or "").strip()
    if token and ws.query_params.get("token", "") != token:
        await ws.close(code=4401)
        return
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = data.get("type")
            if mtype == "stop":
                manager.stop(run_id)
                await ws.send_json({"ok": True})
            elif mtype == "approval":
                ok = manager.decide(run_id, data.get("approval_id", ""), bool(data.get("approve")))
                await ws.send_json({"ok": ok})
            elif mtype == "answer":
                answers = data.get("answers")
                if not isinstance(answers, list):
                    answers = [str(answers)] if answers else []
                ok = manager.answer(run_id, str(data.get("question_id", "")), answers)
                await ws.send_json({"ok": ok})
            else:
                await ws.send_json({"error": "unknown type"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WS 异常: %s", e)
