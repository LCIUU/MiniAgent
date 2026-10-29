"""Agent 运行管理：后台任务、SSE 事件队列、停止、审批决策、运行持久化。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from ..agent.context import SUMMARY_MARKER
from ..agent.graph import graph
from ..agent.state import AgentState, new_run_id
from ..config import get_effective
from ..services import run_store
from ..tools.context import ApprovalManager, QuestionManager

logger = logging.getLogger(__name__)

_DONE = object()


class RunBusyError(RuntimeError):
    """同一会话已有运行中的任务。"""


@dataclass
class RunHandle:
    run_id: str
    session_id: str
    workspace: str = ""
    model: str = ""
    task: asyncio.Task | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    approvals: ApprovalManager = field(default_factory=ApprovalManager)
    questions: QuestionManager = field(default_factory=QuestionManager)
    status: str = "running"  # running | done | error | cancelled
    error: str | None = None
    final_state: dict[str, Any] | None = None
    # 本轮启动时的初始消息对象（按 id 区分本轮新生成的消息，持久化完整消息流用）
    initial_message_refs: list[Any] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)  # 重放缓冲
    _history_max: int = 50000  # 防止长任务内存无限增长
    started_at: float = 0.0  # 运行开始时刻（epoch 秒）
    ended_at: float = 0.0
    # 消息落库只做一次：取消发生在 _finish 内部时会被 shield 重入，避免同一轮写两条消息
    persisted: bool = False
    # 多订阅者：每个 SSE 连接一个独立队列
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    _seq: int = 0

    async def emit(self, event: dict[str, Any]) -> None:
        """广播并缓存事件；缓存项带单调时间戳与自增序号。"""
        self._seq += 1
        cached = {**event, "seq": self._seq, "_ts": time.monotonic()}
        self.history.append(cached)
        if len(self.history) > self._history_max:
            self.history = self.history[-self._history_max :]
        broadcast = {**event, "seq": self._seq}
        for q in list(self.subscribers):
            q.put_nowait(broadcast)

    def subscribe(self, since_seq: int = 0) -> asyncio.Queue:
        """新订阅者：回放 since_seq 之后的历史事件，再实时接收。

        回放条数受 sse_replay_max_events 限制；超过时只回放最近一段并明确告知前端。
        """
        q: asyncio.Queue = asyncio.Queue()
        max_events = int(get_effective("sse_replay_max_events", 20000) or 20000)
        replay = [e for e in self.history if int(e.get("seq", 0)) > since_seq]
        if len(replay) > max_events:
            dropped = len(replay) - max_events
            replay = replay[-max_events:]
            q.put_nowait({
                "type": "status",
                "status": "replay_truncated",
                "detail": f"历史事件过多，已省略最早 {dropped} 条（可刷新页面查看最终结果）",
                "seq": 0,
            })
        for ev in replay:
            q.put_nowait({k: v for k, v in ev.items() if k != "_ts"})
        if self.status != "running":
            q.put_nowait(_DONE)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def finish(self, status: str, error: str | None = None, final_state: dict | None = None) -> None:
        self.status = status
        self.error = error
        self.final_state = final_state
        self.ended_at = time.time()
        for q in list(self.subscribers):
            q.put_nowait(_DONE)


class RunManager:
    """管理所有进行中的 Agent 运行。"""

    def __init__(self) -> None:
        self._runs: dict[str, RunHandle] = {}
        self._start_lock = asyncio.Lock()
        self._force_tasks: set[asyncio.Task] = set()

    def get(self, run_id: str) -> RunHandle | None:
        return self._runs.get(run_id)

    def find_active(self, session_id: str) -> RunHandle | None:
        for h in self._runs.values():
            if h.session_id == session_id and h.status == "running":
                return h
        return None

    async def start(
        self,
        session_id: str,
        request: str,
        workspace: str,
        history: list,
        summary: str = "",
        summary_cutoff: float | None = None,
        persist: Any = None,
        mcp_tools: list | None = None,
        reasoning_effort: str | None = None,
        image_blocks: list[dict] | None = None,
        model: str = "",
    ) -> RunHandle:
        """启动一个 Agent 运行；同一会话已有运行中的任务时抛 RunBusyError。"""
        async with self._start_lock:
            active = self.find_active(session_id)
            if active is not None:
                raise RunBusyError(f"该会话已有运行中的任务（run_id={active.run_id}）")
            run_id = new_run_id()
            handle = RunHandle(run_id=run_id, session_id=session_id, workspace=workspace, model=model)
            self._runs[run_id] = handle

        async def _fail_open(status: str, error: BaseException | None) -> None:
            """兜底收尾：初始化阶段异常（create_run/build_tools 等）也必须
            补发 run_done 并落运行状态，否则 handle 永远 running、该会话被 409 卡死。"""
            if handle.status != "running":
                return
            message = str(error) if error else "运行被取消"
            duration_ms = int((time.time() - (handle.started_at or time.time())) * 1000)
            try:
                if status == "error":
                    await handle.emit({"type": "error", "message": f"Agent 运行失败: {message}"})
                else:
                    await handle.emit({"type": "status", "status": "stopped"})
            except Exception:
                pass
            try:
                await run_store.finish_run(
                    run_id, status, error=str(error or ""), token_usage={}, duration_ms=duration_ms
                )
            except Exception:
                logger.exception("记录运行失败状态时出错")
            try:
                await handle.emit({
                    "type": "run_done",
                    "session_id": session_id,
                    "ended_at": time.time(),
                    "duration_ms": duration_ms,
                    "token_usage": {},
                })
            except Exception:
                pass
            handle.finish(status, error=str(error or ""))
            logger.info("run %s 兜底收尾: status=%s error=%s", run_id, status, message)
            loop = asyncio.get_running_loop()
            loop.call_later(600, self._runs.pop, run_id, None)

        async def _run() -> None:
            from ..tools import build_tools

            run_started_at = time.time()
            handle.started_at = run_started_at
            logger.info("run %s 开始 (session=%s, workspace=%s)", run_id, session_id, workspace)
            await run_store.create_run(run_id, session_id, workspace, model, request)
            try:
                await handle.emit({"type": "run_start", "run_id": run_id, "started_at": run_started_at})
            except Exception:
                pass

            tools = build_tools(
                workspace,
                handle.emit,
                handle.approvals,
                handle.stop_event,
                mcp_tools,
                questions=handle.questions,
            )
            handle.approvals.mode = get_effective("command_confirm_mode", "auto")

            # 当前用户消息：多模态场景 content 为 list[blocks]，纯文本场景为 str
            current_content: Any = request
            if image_blocks:
                current_content = [{"type": "text", "text": request}, *image_blocks]

            # 摘要恢复：把上一次 run 持久化的摘要注回上下文，否则长会话第二轮就失忆
            initial_messages: list[Any] = []
            if summary and summary.strip():
                initial_messages.append(
                    HumanMessage(
                        content=f"{SUMMARY_MARKER}\n{summary.strip()}",
                        additional_kwargs={"is_summary": True, "db_cutoff": summary_cutoff},
                    )
                )
            initial_messages.extend(history)
            initial_messages.append(HumanMessage(content=current_content))
            handle.initial_message_refs = list(initial_messages)

            initial: AgentState = {
                "request": request,
                "workspace": workspace,
                "messages": initial_messages,
                "plan": [],
                "todos": [],
                "tool_rounds": 0,
                "summary": summary,
                "summary_count": 0,
                "reasoning_effort": reasoning_effort,
                "model": model,
                "run_id": run_id,
                "session_id": session_id,
                "stopped": False,
                "error": None,
                "emit": handle.emit,
                "approvals": handle.approvals,
                "stop_event": handle.stop_event,
                "_tools": tools,
            }

            async def _finish(status: str, final_state: dict | None, error: str | None = None) -> None:
                if final_state is not None:
                    handle.final_state = final_state
                ended_at = time.time()
                duration_ms = int((ended_at - run_started_at) * 1000)
                handle.ended_at = ended_at  # 持久化时读取，保证消息耗时与本轮 run 一致
                if not handle.persisted:
                    handle.persisted = True
                    try:
                        if persist is not None:
                            await persist(handle)
                    except Exception:
                        logger.exception("持久化运行结果失败")
                usage = (final_state or handle.final_state or {}).get("token_usage") or {}
                await run_store.finish_run(
                    run_id, status, error=error or "", token_usage=usage, duration_ms=duration_ms
                )
                try:
                    await handle.emit({
                        "type": "run_done",
                        "session_id": session_id,
                        "ended_at": ended_at,
                        "duration_ms": duration_ms,
                        "token_usage": usage,
                    })
                except Exception:
                    pass
                handle.finish(status, error=error, final_state=final_state)
                logger.info("run %s 结束: status=%s 耗时=%.1fs", run_id, status, ended_at - run_started_at)
                loop = asyncio.get_running_loop()
                loop.call_later(600, self._runs.pop, run_id, None)

            try:
                result = await graph.ainvoke(initial)
                # 用户主动停止：图会优雅返回 stopped=True，状态记为 cancelled
                stopped = isinstance(result, dict) and bool(result.get("stopped"))
                error = result.get("error") if isinstance(result, dict) else None
                if error:
                    await _finish("error", result, error=str(error))
                else:
                    await _finish("cancelled" if stopped else "done", result)
            except asyncio.CancelledError:
                handle.approvals.cancel_all()
                handle.questions.cancel_all()
                try:
                    await asyncio.shield(_finish("cancelled", handle.final_state))
                except asyncio.CancelledError:
                    handle.finish("cancelled")
            except Exception as e:
                logger.exception("Agent 运行失败")
                handle.approvals.cancel_all()
                handle.questions.cancel_all()
                try:
                    await handle.emit({"type": "error", "message": f"Agent 运行失败: {e}"})
                except Exception:
                    pass
                await _finish("error", handle.final_state, error=str(e))

        async def _run_wrapper() -> None:
            try:
                await _run()
            except asyncio.CancelledError:
                handle.approvals.cancel_all()
                handle.questions.cancel_all()
                try:
                    await asyncio.shield(_fail_open("cancelled", None))
                except asyncio.CancelledError:
                    handle.finish("cancelled")
                raise
            except Exception as e:
                logger.exception("Agent 运行未捕获异常，执行兜底收尾")
                try:
                    await asyncio.shield(_fail_open("error", e))
                except asyncio.CancelledError:
                    handle.finish("error", error=str(e))

        handle.task = asyncio.create_task(_run_wrapper())
        return handle

    def stop(self, run_id: str) -> bool:
        handle = self._runs.get(run_id)
        if handle is None or handle.status != "running":
            return False
        handle.stop_event.set()
        handle.approvals.cancel_all()
        handle.questions.cancel_all()
        # 给工具/模型 5 秒优雅退出时间；仍未结束则强制取消任务
        task = handle.task
        if task is not None and not task.done():
            async def _force_cancel() -> None:
                await asyncio.sleep(5)
                if handle.status == "running" and not task.done():
                    logger.warning("run %s 优雅停止超时，强制取消任务", run_id)
                    task.cancel()

            force = asyncio.create_task(_force_cancel())
            self._force_tasks.add(force)
            force.add_done_callback(self._force_tasks.discard)
        return True

    def decide(self, run_id: str, approval_id: str, approve: bool) -> bool:
        handle = self._runs.get(run_id)
        if handle is None:
            return False
        return handle.approvals.decide(approval_id, approve)

    def answer(self, run_id: str, question_id: str, answers: list[str]) -> bool:
        handle = self._runs.get(run_id)
        if handle is None:
            return False
        return handle.questions.answer(question_id, answers)

    async def shutdown(self) -> None:
        """应用关闭：取消所有运行中的任务，避免后台进程/协程泄漏。"""
        tasks = [h.task for h in self._runs.values() if h.task and not h.task.done()]
        for h in self._runs.values():
            h.stop_event.set()
            h.approvals.cancel_all()
            h.questions.cancel_all()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runs.clear()
        for task in list(self._force_tasks):
            task.cancel()
        self._force_tasks.clear()


manager = RunManager()
