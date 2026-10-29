"""工具共享上下文：工作区、事件发射、审批、停止控制。

每个 Agent 运行创建一个 ToolContext，通过闭包注入各个工具。
``current_call_id`` 用 ContextVar 保存当前工具调用 id，使并行工具也能把
``tool_output`` 事件关联到正确的调用卡片。
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]

# 当前正在执行的工具调用 id（并行任务各自持有独立 ContextVar 上下文）
current_call_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_call_id", default="")


@dataclass
class ApprovalManager:
    """工具命令审批：工具调用 wait() 阻塞，等待外部决策。"""

    mode: str = "auto"  # auto | high_risk | all
    _pending: dict[str, asyncio.Future] = field(default_factory=dict)
    _counter: int = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"ap_{int(time.time() * 1000)}_{self._counter}"

    async def request(
        self,
        emit: EmitFn,
        tool: str,
        reason: str,
        payload: dict[str, Any],
        stop_event: asyncio.Event | None = None,
    ) -> bool:
        """发起审批请求，返回是否批准。mode=auto 时直接通过。

        stop_event 被设置（用户点停止）时立即返回 False，避免卡 600 秒。
        """
        if self.mode == "auto":
            return True
        if self.mode == "high_risk" and reason != "high_risk":
            return True
        if stop_event is not None and stop_event.is_set():
            return False
        approval_id = self._new_id()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = fut
        await emit({"type": "approval", "approval_id": approval_id, "tool": tool, **payload, "reason": reason})

        watcher: asyncio.Task | None = None
        if stop_event is not None:
            async def _watch_stop() -> None:
                await stop_event.wait()
                if not fut.done():
                    fut.set_result(False)

            watcher = asyncio.create_task(_watch_stop())
        approved = False
        try:
            approved = bool(await asyncio.wait_for(fut, timeout=600))
            return approved
        except TimeoutError:
            return False
        finally:
            if watcher is not None:
                watcher.cancel()
            self._pending.pop(approval_id, None)
            # 回执事件让前端收起审批卡片；刷新后重放历史时也能还原"已批准/已拒绝"终态
            try:
                await emit({"type": "approval_done", "approval_id": approval_id, "approved": approved})
            except Exception:
                pass

    def decide(self, approval_id: str, approve: bool) -> bool:
        """外部（WS/HTTP）提交决策。"""
        fut = self._pending.get(approval_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approve)
        return True

    def cancel_all(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result(False)
        self._pending.clear()


@dataclass
class QuestionManager:
    """结构化提问：工具 ask() 阻塞，等待用户点选/填写后继续。"""

    _pending: dict[str, asyncio.Future] = field(default_factory=dict)
    _counter: int = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"q_{int(time.time() * 1000)}_{self._counter}"

    async def ask(
        self,
        emit: EmitFn,
        tool: str,
        question: str,
        options: list[str] | None = None,
        *,
        allow_multiple: bool = False,
        stop_event: asyncio.Event | None = None,
        timeout: float = 600,
    ) -> list[str] | None:
        """发起提问并等待回答；用户停止或超时未答时返回 None。"""
        if stop_event is not None and stop_event.is_set():
            return None
        question_id = self._new_id()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[question_id] = fut
        await emit({
            "type": "question",
            "question_id": question_id,
            "tool": tool,
            "question": question,
            "options": list(options or []),
            "allow_multiple": allow_multiple,
        })

        watcher: asyncio.Task | None = None
        if stop_event is not None:
            async def _watch_stop() -> None:
                await stop_event.wait()
                if not fut.done():
                    fut.set_result(None)

            watcher = asyncio.create_task(_watch_stop())
        try:
            answers = await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            answers = None
        finally:
            if watcher is not None:
                watcher.cancel()
            self._pending.pop(question_id, None)
        # 回执事件让前端收起卡片；断线重放历史时也能还原"已答/已取消"的终态
        try:
            await emit({"type": "question_done", "question_id": question_id, "answers": answers or []})
        except Exception:
            pass
        return answers if answers else None

    def answer(self, question_id: str, answers: list[str]) -> bool:
        """外部（WS/HTTP）提交回答。"""
        fut = self._pending.get(question_id)
        if fut is None or fut.done():
            return False
        picked = [str(a).strip() for a in answers if str(a).strip()]
        fut.set_result(picked)
        return True

    def cancel_all(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result(None)
        self._pending.clear()


@dataclass
class ToolContext:
    workspace: str
    emit: EmitFn
    approvals: ApprovalManager
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    env: dict[str, str] = field(default_factory=dict)
    questions: QuestionManager = field(default_factory=QuestionManager)

    @property
    def call_id(self) -> str:
        return current_call_id.get()

    def check_stopped(self) -> None:
        """工具应在每步前调用；已停止则抛 CancelledError。"""
        if self.stop_event.is_set():
            raise asyncio.CancelledError("运行已被用户停止")

    async def emit_event(self, event: dict[str, Any]) -> None:
        try:
            await self.emit(event)
        except Exception:
            pass

    async def emit_output(self, text: str, *, stream: str = "stdout", limit: int = 8000) -> None:
        """推送工具实时输出片段（终端流式输出用）。"""
        if not text:
            return
        await self.emit_event({
            "type": "tool_output",
            "call_id": self.call_id,
            "stream": stream,
            "delta": text[:limit],
        })
