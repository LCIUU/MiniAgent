"""流式事件协议：后端 → 前端 (SSE) / 前端 → 后端 (WebSocket)。

SSE 事件（data 为 JSON；均带 seq 自增序号，可作为 Last-Event-ID）：
  {"type": "run_start", "run_id": "...", "started_at": 1234567890.123}
  {"type": "status", "status": "executing|stopped|loop_guard|reasoning_repeat_nudge|reasoning_repeat_blocked|duplicate_read_blocked|file_write_nudge|replay_truncated", "detail": "..."}
  {"type": "thinking", "delta": "..."}                     # 模型推理增量
  {"type": "token", "delta": "..."}                         # 流式文本增量
  {"type": "plan", "steps": [...], "todos": [{"content", "status"}]}
  {"type": "tool_start", "tool": "terminal", "args": {...}, "call_id": "...", "started_at": "HH:MM:SS"}
  {"type": "tool_output", "call_id": "...", "stream": "stdout", "delta": "..."}   # 工具实时输出
  {"type": "diff", "call_id": "...", "path": "a.py", "diff": "--- a/...", "added": 1, "removed": 2}
  {"type": "tool_end", "tool": "...", "result": "...", "call_id": "...", "status": "success|error", "started_at": "HH:MM:SS", "duration_ms": 123}
  {"type": "approval", "approval_id": "...", "tool": "terminal", "command": "...", "reason": "high_risk|normal|all"}
  {"type": "summary_start", "before_tokens": 80790, "messages_before": 268}   # 上下文压缩开始
  {"type": "summary_done", "ok": true, "before_tokens": 80790, "after_tokens": 10500, "messages_before": 268, "messages_after": 31, "duration_ms": 12345, "summary": "...", "db_cutoff": 1730000000.123}  # summary 供前端展开；db_cutoff 供跨轮跳过已摘要历史
  {"type": "summary", "summary": "..."}                     # 上下文摘要完成
  {"type": "usage", "input_tokens": 1, "output_tokens": 2, "llm_calls": 3}
  {"type": "message_done", "message_id": "..."}             # 预留给后续按消息流式渲染
  {"type": "run_done", "session_id": "...", "ended_at": 1234567890.456, "duration_ms": 12345, "token_usage": {...}}
  {"type": "error", "message": "..."}

WS 消息（前端 → 后端）：
  {"type": "stop"}
  {"type": "approval", "approval_id": "...", "approve": true|false}
"""

from __future__ import annotations

from typing import Any


def sse_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": event_type, **data}
