"""深度优化回归测试：安全/编码/patch/缓存/摘要/契约。"""

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent import context as ctx_mod
from app.agent.graph import _path_key
from app.schemas import MessageOut
from app.services import secrets
from app.services.paths import PathOutsideWorkspace, resolve_workspace_path
from app.services.tool_cache import make_cache_key
from app.tools.patch import _apply_hunks, parse_patch
from app.tools.terminal import is_high_risk


def test_high_risk_windows_patterns_case_insensitive():
    assert is_high_risk("Remove-Item -Recurse -Force .\\dist")
    assert is_high_risk("stop-process -Name node")
    assert is_high_risk("git push origin main")
    assert not is_high_risk("git status")


def test_workspace_path_escape_blocked():
    ws = tempfile.mkdtemp()
    with pytest.raises(PathOutsideWorkspace):
        resolve_workspace_path(ws, "..")
    inside = resolve_workspace_path(ws, "a/b.txt")
    assert inside == Path(ws).resolve() / "a" / "b.txt"


def test_path_key_normalizes_relative_and_absolute():
    ws = tempfile.mkdtemp()
    a = _path_key(ws, "a.txt")
    b = _path_key(ws, os.path.join(ws, "a.txt"))
    assert a == b


def test_file_edit_preserves_crlf_and_bom():
    from app.tools.context import ApprovalManager, ToolContext
    from app.tools.files import make_file_tools

    async def run():
        ws = Path(tempfile.mkdtemp())
        crlf = ws / "crlf.txt"
        crlf.write_bytes(b"one\r\ntwo\r\n")
        bom = ws / "bom.txt"
        bom.write_bytes(b"\xef\xbb\xbfhello\n")
        ctx = ToolContext(
            workspace=str(ws),
            emit=lambda e: asyncio.sleep(0),
            approvals=ApprovalManager(mode="auto"),
        )
        _, _, edit_file, *_ = make_file_tools(ctx)
        await edit_file.ainvoke({"path": "crlf.txt", "old_string": "two", "new_string": "TWO"})
        await edit_file.ainvoke({"path": "bom.txt", "old_string": "hello", "new_string": "hi"})
        return crlf.read_bytes(), bom.read_bytes()

    crlf, bom = asyncio.run(run())
    assert crlf == b"one\r\nTWO\r\n"
    assert bom == b"\xef\xbb\xbfhi\n"


def test_cache_key_changes_when_file_changes():
    ws = tempfile.mkdtemp()
    p = Path(ws) / "a.txt"
    p.write_text("v1", encoding="utf-8")
    args = {"path": "a.txt"}
    k1 = make_cache_key("read_file", args, ws)
    time.sleep(0.01)
    p.write_text("v2-longer", encoding="utf-8")
    os.utime(p, None)
    k2 = make_cache_key("read_file", args, ws)
    assert k1 != k2


def test_patch_parser_and_apply():
    original = "one\ntwo\nthree\n"
    patch = parse_patch(
        "--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
    )[0]
    assert _apply_hunks(original, patch) == "one\nTWO\nthree\n"
    with pytest.raises(ValueError):
        _apply_hunks(original, parse_patch(
            "--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-nope\n+x\n"
        )[0])


def test_graph_is_single_react_loop():
    """图不再有独立规划节点：模型在执行循环里自主决定是否调用 write_todos。"""
    import app.agent.graph as graph_mod

    nodes = set(graph_mod.graph.get_graph().nodes)
    assert {"execute", "context"} <= nodes
    assert "plan" not in nodes


def test_system_prompt_leaves_planning_to_model():
    import app.agent.graph as graph_mod

    text = graph_mod._system_message(
        {"workspace": tempfile.mkdtemp(), "_tools": []}
    ).content
    assert "write_todos" in text
    assert "自主判断" in text


def test_extract_raw_write_todos_text():
    from app.agent.graph import _extract_raw_write_todos

    raw = (
        "write_todos(todos=["
        "{'content': 'Create a synthesized sound effect utility', 'status': 'in_progress'}, "
        "{'content': 'Add global volume control to the Navbar', 'status': 'pending'}])"
    )
    todos = _extract_raw_write_todos(raw)
    assert [t["content"] for t in todos] == [
        "Create a synthesized sound effect utility",
        "Add global volume control to the Navbar",
    ]
    assert todos[0]["status"] == "in_progress"
    assert _extract_raw_write_todos(f"```python\n{raw}\n```") == todos
    assert _extract_raw_write_todos("先说明一下：write_todos(todos=[])") is None
    assert _extract_raw_write_todos("write_todos(todos=[{bad}])") is None
    assert _extract_raw_write_todos("") is None


def test_execute_parses_raw_write_todos_text():
    """模型把 write_todos 写成正文时，执行循环要解析为计划更新并继续执行。"""
    import app.agent.graph as graph_mod
    from app.tools.context import ApprovalManager

    raw = (
        "write_todos(todos=["
        "{'content': '加音效', 'status': 'in_progress'}, "
        "{'content': '加全局音量控制', 'status': 'completed'}])"
    )

    class RawPlanLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def astream(self, messages, **kwargs):
            self.calls += 1
            chunk = type("Chunk", (), {})()
            chunk.content = raw if self.calls == 1 else "完成"
            chunk.tool_calls = []
            chunk.usage_metadata = None
            chunk.additional_kwargs = {}
            yield chunk

    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    state = {
        "request": "继续完善游戏体验", "workspace": tempfile.mkdtemp(),
        "messages": [HumanMessage(content="继续完善游戏体验")],
        "plan": [], "todos": [],
        "tool_rounds": 0, "summary": "", "summary_count": 0,
        "run_id": "raw_plan_exec", "session_id": "raw_plan_session",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": [], "_llm": RawPlanLLM(), "model": "test-model",
    }

    out = asyncio.run(graph_mod.execute_node(state))

    assert out["plan"] == ["加音效", "加全局音量控制"]
    assert [t["status"] for t in out["todos"]] == ["in_progress", "completed"]
    plan_events = [e for e in events if e["type"] == "plan"]
    assert plan_events and plan_events[-1]["todos"] == out["todos"]
    ai_idx = next(
        i for i, m in enumerate(out["messages"])
        if isinstance(m, AIMessage) and m.tool_calls
    )
    ai_msg = out["messages"][ai_idx]
    tool_msg = out["messages"][ai_idx + 1]
    assert ai_msg.tool_calls[0]["name"] == "write_todos"
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == ai_msg.tool_calls[0]["id"]
    assert out["messages"][-1].content == "完成"


def test_execute_always_binds_write_todos():
    """没有预设计划时也要把 write_todos 与真实工具一起绑定：规划由模型自主决定。"""
    import app.agent.graph as graph_mod
    from app.tools.context import ApprovalManager

    class RecordingLLM:
        _llm_type = "mock"

        def __init__(self):
            self.bound: list[str] = []
            self.seen: list[list] = []

        def bind_tools(self, tools):
            self.bound = [t.name for t in tools]
            return self

        async def astream(self, messages, **kwargs):
            self.seen.append(messages)
            chunk = type("Chunk", (), {})()
            chunk.content = "好的"
            chunk.tool_calls = []
            chunk.usage_metadata = None
            chunk.additional_kwargs = {}
            yield chunk

    llm = RecordingLLM()
    state = {
        "request": "你好", "workspace": tempfile.mkdtemp(),
        "messages": [HumanMessage(content="你好")],
        "plan": [], "todos": [],
        "tool_rounds": 0, "summary": "", "summary_count": 0,
        "run_id": "bind_todos_test", "session_id": "bind_todos_session",
        "stopped": False, "error": None, "emit": None,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": [], "_llm": llm, "model": "test-model",
    }

    out = asyncio.run(graph_mod.execute_node(state))

    assert llm.bound == ["write_todos"]
    assert isinstance(llm.seen[0][0], SystemMessage)
    assert "write_todos" in str(llm.seen[0][0].content)
    assert out["messages"][-1].content == "好的"


def test_rebuild_llm_for_retry_switches_to_fallback(monkeypatch):
    """连接类失败重试时优先切换备用模型，且不再叠加降级包装。"""
    import app.agent.graph as graph_mod

    calls: list[dict] = []

    class Dummy:
        pass

    def fake_init(**kw):
        calls.append(kw)
        return Dummy()

    settings = {"model": "primary-model", "fallback_model": "fallback-model"}
    monkeypatch.setattr(graph_mod, "init_chat_models", fake_init)
    monkeypatch.setattr(
        graph_mod, "get_effective", lambda key, default=None: settings.get(key, default)
    )
    _, name = asyncio.run(
        graph_mod._rebuild_llm_for_retry({"model": "primary-model", "reasoning_effort": "high"})
    )

    assert name == "fallback-model"
    assert calls == [
        {"model": "fallback-model", "reasoning_effort": "high", "use_fallback": False}
    ]


def test_rebuild_llm_for_retry_falls_back_to_primary(monkeypatch):
    """备用模型不可用时重建主模型客户端，不能因此丢掉重试机会。"""
    import app.agent.graph as graph_mod

    calls: list[dict] = []

    class Dummy:
        pass

    def fake_init(**kw):
        if kw.get("model") == "fallback-model":
            raise ValueError("备用模型未配置 API Key")
        calls.append(kw)
        return Dummy()

    settings = {"model": "primary-model", "fallback_model": "fallback-model"}
    monkeypatch.setattr(graph_mod, "init_chat_models", fake_init)
    monkeypatch.setattr(
        graph_mod, "get_effective", lambda key, default=None: settings.get(key, default)
    )
    _, name = asyncio.run(graph_mod._rebuild_llm_for_retry({"model": "primary-model"}))

    assert name == "primary-model"
    assert calls == [{"model": "primary-model", "reasoning_effort": None}]


def test_rebuild_llm_for_retry_keeps_injected_llm(monkeypatch):
    """测试注入的 _llm 替身不应被重建逻辑替换。"""
    import app.agent.graph as graph_mod

    override = object()

    def boom(**kw):
        raise AssertionError("注入替身时不应创建新客户端")

    monkeypatch.setattr(graph_mod, "init_chat_models", boom)
    llm, name = asyncio.run(
        graph_mod._rebuild_llm_for_retry({"_llm": override, "model": "m1"})
    )

    assert llm is override
    assert name == "m1"


def test_summary_injected_and_extracted(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, messages, **kwargs):
            class R:
                content = "## 用户目标与约束\n测试摘要"

            return R()

    def fake_get(key, default=None):
        return {"summary_threshold_tokens": 10, "summary_keep_messages": 2}.get(key, default)

    monkeypatch.setattr(ctx_mod, "get_effective", fake_get)
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="这是一段很长的历史消息" * 10),
        HumanMessage(content="中间消息" * 10),
        HumanMessage(content="最近一条"),
    ]
    out = asyncio.run(ctx_mod.maybe_summarize_messages(messages, FakeLLM()))
    assert isinstance(out[0], SystemMessage)
    assert out[1].additional_kwargs.get("is_summary") is True
    assert "测试摘要" in ctx_mod.extract_summary(out)


def test_summary_emits_start_and_done_events(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, messages, **kwargs):
            class R:
                content = "新摘要"

            return R()

    monkeypatch.setattr(
        ctx_mod,
        "get_effective",
        lambda key, default=None: {
            "summary_threshold_tokens": 10,
            "summary_keep_messages": 2,
        }.get(key, default),
    )
    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="这是一段很长的历史消息" * 10),
        HumanMessage(content="中间消息" * 10),
        HumanMessage(content="最近一条"),
    ]
    out = asyncio.run(ctx_mod.maybe_summarize_messages(messages, FakeLLM(), emit=emit))

    assert [e["type"] for e in events] == ["summary_start", "summary_done"]
    start, done = events
    assert start["before_tokens"] == ctx_mod.count_tokens_zh_aware(messages)
    assert start["messages_before"] == len(messages)
    assert done["ok"] is True
    assert done["after_tokens"] < done["before_tokens"]
    assert done["messages_after"] == len(out)
    assert "新摘要" in done["summary"]


def test_eligible_start_never_splits_tool_pairs():
    from app.agent.context import _eligible_start

    msgs = [
        HumanMessage(content="任务"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "read_file", "args": {}}]),
        ToolMessage(content="r1", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "read_file", "args": {}}]),
        ToolMessage(content="r2", tool_call_id="c2"),
        HumanMessage(content="继续"),
        AIMessage(content="完成"),
    ]

    # 截点落在 ToolMessage 上：回退到产生它的 AIMessage，而不是再退到上一条 ToolMessage
    assert _eligible_start(msgs, 5) == 1
    # 截点落在 AIMessage(tool_calls) 上：它已是合法起点，保持不动
    assert _eligible_start(msgs, 4) == 3
    # 保留段绝不能以 ToolMessage 开头
    for keep in range(1, len(msgs) + 1):
        start = _eligible_start(msgs, keep)
        assert not (start < len(msgs) and isinstance(msgs[start], ToolMessage)), keep


def test_summary_cutoff_covers_only_summarized_db_rows():
    msg = HumanMessage(content="很长的历史" * 20, additional_kwargs={"db_ts": 100.0})
    older_ai = AIMessage(content="旧回答", additional_kwargs={"db_ts": 200.0})
    kept_human = HumanMessage(content="保留的问题", additional_kwargs={"db_ts": 300.0})
    kept_ai = AIMessage(content="保留的回答", additional_kwargs={"db_ts": 400.0})

    cutoff = ctx_mod._summary_cutoff([msg, older_ai], [kept_human, kept_ai])

    assert cutoff == 200.0


def test_summary_cutoff_keeps_partial_row_intact():
    # 截点落在同一 DB 行内：该行两侧消息时间相同，整体保留，截止点取更早的行
    old = HumanMessage(content="旧", additional_kwargs={"db_ts": 100.0})
    partial = AIMessage(content="同一行前半", additional_kwargs={"db_ts": 200.0})
    kept = ToolMessage(
        content="同一行后半", tool_call_id="c1", additional_kwargs={"db_ts": 200.0}
    )

    assert ctx_mod._summary_cutoff([old, partial], [kept]) == 100.0


def test_summary_window_shrinks_by_token_budget(monkeypatch):
    """压缩后必须明显低于阈值，否则下一轮立刻再压缩（每轮换摘要 = 缓存全灭）。"""

    class FakeLLM:
        async def ainvoke(self, messages, **kwargs):
            class R:
                content = "新摘要"

            return R()

    monkeypatch.setattr(
        ctx_mod,
        "get_effective",
        lambda key, default=None: {
            "summary_threshold_tokens": 2000,
            "summary_keep_messages": 30,
            "summary_keep_ratio": 0.45,
            "summary_min_keep_messages": 2,
        }.get(key, default),
    )
    msgs = [SystemMessage(content="system")]
    msgs += [HumanMessage(content="这是一段较长的历史消息内容" * 10) for _ in range(40)]
    before = ctx_mod.count_tokens_zh_aware(msgs)
    assert before > 2000

    out = asyncio.run(ctx_mod.maybe_summarize_messages(msgs, FakeLLM()))

    assert len(out) < len(msgs)
    # 保留窗口按 token 预算收缩：压缩结果要留出余量，不能还贴着阈值
    assert ctx_mod.count_tokens_zh_aware(out) < 2000 * 0.75
    # 最后一条（当前任务）必须留在窗口里
    assert out[-1] is msgs[-1]
    # 保留段不能以孤立 ToolMessage 开头，也不能从中间拆开工具对
    assert not isinstance(out[2], ToolMessage)


def test_summary_compacts_few_huge_messages(monkeypatch):
    """条数不足 keep 时也要按 token 预算压缩，否则少量超长消息会撑爆模型窗口。"""

    class FakeLLM:
        async def ainvoke(self, messages, **kwargs):
            class R:
                content = "新摘要"

            return R()

    monkeypatch.setattr(
        ctx_mod,
        "get_effective",
        lambda key, default=None: {
            "summary_threshold_tokens": 2000,
            "summary_keep_messages": 30,
            "summary_keep_ratio": 0.45,
            "summary_min_keep_messages": 4,
        }.get(key, default),
    )
    msgs = [SystemMessage(content="system")]
    msgs += [HumanMessage(content="超长工具结果内容" * 300) for _ in range(10)]
    assert len(msgs) < 30  # 条数远小于 keep
    assert ctx_mod.count_tokens_zh_aware(msgs) > 4000

    out = asyncio.run(ctx_mod.maybe_summarize_messages(msgs, FakeLLM()))

    assert len(out) < len(msgs)
    assert out[-1] is msgs[-1]
    assert ctx_mod.count_tokens_zh_aware(out) < ctx_mod.count_tokens_zh_aware(msgs)


def test_accumulate_usage_collects_cache_read():
    from app.agent import graph as graph_mod

    class Chunk:
        usage_metadata = {
            "input_tokens": 1000,
            "output_tokens": 20,
            "input_token_details": {"cache_read": 900},
        }

    class ChunkNoCache:
        usage_metadata = {"input_tokens": 500, "output_tokens": 10}

    state: dict = {}
    graph_mod._accumulate_usage(state, Chunk())
    graph_mod._accumulate_usage(state, ChunkNoCache())

    usage = state["token_usage"]
    assert usage["input_tokens"] == 1500
    assert usage["output_tokens"] == 30
    assert usage["cache_read_tokens"] == 900
    assert usage["llm_calls"] == 2


def test_serialize_run_messages_keeps_internal_nudges():
    """内部提醒当时就在模型上下文里，必须原样入库，否则下一轮前缀对不上。"""
    from app.api.agent import _messages_from_run, _serialize_run_messages

    nudge = SystemMessage(content="【系统自动提醒】先读后写", additional_kwargs={"is_nudge": True})
    messages = [AIMessage(content="完成"), nudge, SystemMessage(content="头部系统提示")]

    items = _serialize_run_messages(messages, set())

    assert [i["role"] for i in items] == ["ai", "system"]
    assert items[1]["content"] == "【系统自动提醒】先读后写"

    restored = _messages_from_run(items)
    assert isinstance(restored[1], SystemMessage)
    assert restored[1].content == nudge.content
    assert restored[1].additional_kwargs.get("is_nudge") is True


def test_history_preserves_request_whitespace():
    """request_content 原样还原：任何首尾空白差异都会打断缓存前缀。"""
    from app.api.agent import _history_from_db
    from app.models import Message

    m = Message(session_id="s", role="user", content="看下这个文件")
    m.request_content = "看下这个文件\n"

    history = _history_from_db([m])

    assert history[0].content == "看下这个文件\n"


def test_summary_message_carries_cutoff(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, messages, **kwargs):
            class R:
                content = "摘要"

            return R()

    monkeypatch.setattr(
        ctx_mod,
        "get_effective",
        lambda key, default=None: {
            "summary_threshold_tokens": 10,
            "summary_keep_messages": 2,
        }.get(key, default),
    )
    messages = [
        HumanMessage(content="旧消息" * 20, additional_kwargs={"db_ts": 100.0}),
        AIMessage(content="旧回答", additional_kwargs={"db_ts": 200.0}),
        HumanMessage(content="保留一", additional_kwargs={"db_ts": 300.0}),
        HumanMessage(content="保留二", additional_kwargs={"db_ts": 400.0}),
    ]
    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    out = asyncio.run(ctx_mod.maybe_summarize_messages(messages, FakeLLM(), emit=emit))

    assert ctx_mod.extract_summary_cutoff(out) == 200.0
    assert events[-1]["db_cutoff"] == 200.0


def test_final_summary_prefers_messages_over_stale_state():
    from app.api.agent import _final_summary

    summary_msg = HumanMessage(
        content="【历史上下文摘要】\n中途生成的摘要",
        additional_kwargs={"is_summary": True},
    )
    state = {"messages": [HumanMessage(content="新问题"), summary_msg], "summary": "旧摘要"}

    assert "中途生成的摘要" in _final_summary(state)
    assert _final_summary({"messages": [], "summary": "旧摘要"}) == "旧摘要"


def test_history_from_db_tags_created_at_for_cutoff():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from app.api.agent import _history_from_db

    created = datetime(2026, 10, 24, 2, 17, tzinfo=UTC)
    user_row = SimpleNamespace(
        role="user",
        content="问题",
        attachments=None,
        tool_calls=[],
        context_messages=None,
        created_at=created,
    )
    ai_row = SimpleNamespace(
        role="assistant",
        content="回答",
        attachments=None,
        tool_calls=[],
        context_messages=[{"role": "ai", "content": "回答", "tool_calls": []}],
        created_at=created,
    )

    out = _history_from_db([user_row, ai_row])

    assert out[0].content == "问题"
    assert all(m.additional_kwargs.get("db_ts") == created.timestamp() for m in out)


def test_history_drops_orphan_tool_messages():
    from types import SimpleNamespace

    from app.api.agent import _history_from_db

    stored = [
        {"role": "tool", "content": "孤立结果", "tool_call_id": "orphan"},
        {
            "role": "ai",
            "content": "",
            "tool_calls": [{"id": "call_ok", "name": "read_file", "args": {"path": "a.js"}}],
        },
        {"role": "tool", "content": "正常结果", "tool_call_id": "call_ok"},
    ]
    row = SimpleNamespace(
        role="assistant",
        content="完成",
        attachments=None,
        tool_calls=[],
        context_messages=stored,
    )

    out = _history_from_db([row])

    assert not any(isinstance(m, ToolMessage) and m.tool_call_id == "orphan" for m in out)
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "call_ok" for m in out)


def test_timeline_thinking_uses_backend_duration():
    """思考耗时以后端给出的真实时长为准：
    - thinking_end 收口流式思考，不按后续工具开始时间计算（否则会算进参数生成时间）
    - 被闸门暂存到轮末的短思考，事件自带 duration_ms，直接采用
    """
    from app.api.agent import _timeline_from_history

    history = [
        {"type": "thinking", "delta": "第一段", "_ts": 1.0},
        {"type": "thinking", "delta": "思考", "_ts": 1.2},
        {"type": "thinking_end", "duration_ms": 1500, "_ts": 2.5},
        {"type": "tool_start", "tool": "read_file", "call_id": "c1", "args": {}, "_ts": 60.0},
        {"type": "tool_end", "call_id": "c1", "status": "success", "result": "ok", "_ts": 60.1},
        {"type": "thinking", "delta": "第二段短思考", "_ts": 61.0, "duration_ms": 900},
        {"type": "tool_start", "tool": "read_file", "call_id": "c2", "args": {}, "_ts": 62.0},
    ]
    timeline = _timeline_from_history(history)

    thinks = [it for it in timeline if it["kind"] == "thinking"]
    assert [it["duration_ms"] for it in thinks] == [1500, 900]


def test_timeline_reconstructs_compress_item():
    from app.api.agent import _timeline_from_history

    history = [
        {"type": "thinking", "delta": "思考", "_ts": 1.0},
        {"type": "summary_start", "before_tokens": 80000, "_ts": 2.0},
        {"type": "summary_done", "ok": True, "after_tokens": 9000, "duration_ms": 1200, "summary": "## 摘要\n- 要点", "_ts": 3.2},
        {"type": "token", "delta": "最终回答", "_ts": 4.0},
    ]
    timeline = _timeline_from_history(history)

    assert [it["kind"] for it in timeline] == ["thinking", "compress", "text"]
    compress = timeline[1]
    assert compress["status"] == "done"
    assert compress["after_tokens"] == 9000
    assert compress["summary"] == "## 摘要\n- 要点"
    assert compress["duration_ms"] == 1200


def test_summary_replaces_previous_summary(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, messages, **kwargs):
            class R:
                content = "新摘要"

            return R()

    monkeypatch.setattr(
        ctx_mod,
        "get_effective",
        lambda key, default=None: {
            "summary_threshold_tokens": 10,
            "summary_keep_messages": 2,
        }.get(key, default),
    )
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="【历史上下文摘要】旧摘要", additional_kwargs={"is_summary": True}),
        HumanMessage(content="很久以前" * 20),
        HumanMessage(content="中间消息" * 20),
        HumanMessage(content="最近一条"),
    ]

    out = asyncio.run(ctx_mod.maybe_summarize_messages(messages, FakeLLM()))

    summaries = [m for m in out if ctx_mod._is_summary_message(m)]
    assert len(summaries) == 1
    assert "新摘要" in ctx_mod.extract_summary(out)
    assert "旧摘要" not in ctx_mod.extract_summary(out)


def test_format_for_summary_keeps_head_and_tail():
    messages = [HumanMessage(content=f"消息{i}-" + "x" * 4000) for i in range(10)]

    text = ctx_mod._format_for_summary(messages, max_chars=10000)

    assert "消息0" in text
    assert "消息9" in text
    assert "中间省略" in text
    assert len(text) <= 12000


def test_format_for_summary_keeps_oversized_latest():
    """最新的待摘要消息再长也要进入摘要输入，不能被整条丢弃。"""
    messages = [
        HumanMessage(content="消息0-" + "x" * 4000),
        HumanMessage(content="最新-" + "y" * 9000),
    ]

    text = ctx_mod._format_for_summary(messages, max_chars=5000)

    assert "最新" in text
    assert "中间省略" in text
    assert len(text) <= 7000


def test_recent_context_keeps_latest_user_goal():
    """最近一次用户要求即使不在尾部窗口内，也必须带入摘要输入。"""
    preserved = [HumanMessage(content="现在把游戏数量开发到70个")] + [
        ToolMessage(content=f"结果{i}-" + "x" * 500, tool_call_id=f"c{i}")
        for i in range(20)
    ]

    text = ctx_mod._format_recent_context(preserved)

    assert "现在把游戏数量开发到70个" in text
    assert "最近一次用户消息" in text


def test_summary_prompt_includes_recent_messages(monkeypatch):
    """摘要提示同时包含旧历史与最近保留上下文，避免最新目标停留在旧版本。"""
    class FakeLLM:
        def __init__(self):
            self.prompts: list[str] = []

        async def ainvoke(self, messages, **kwargs):
            self.prompts.append(messages[0].content)

            class R:
                content = "## 用户目标与约束\n扩展到70个"

            return R()

    monkeypatch.setattr(
        ctx_mod,
        "get_effective",
        lambda key, default=None: {
            "summary_threshold_tokens": 10,
            "summary_keep_messages": 3,
            "summary_input_max_chars": 8000,
        }.get(key, default),
    )
    llm = FakeLLM()
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="旧任务：把游戏扩展到60个" + "历史" * 100),
        AIMessage(content="已完成60个"),
        HumanMessage(content="现在把游戏数量开发到70个"),
        AIMessage(content="正在创建新的游戏组件"),
    ]

    asyncio.run(ctx_mod.maybe_summarize_messages(messages, llm))

    assert llm.prompts
    prompt = llm.prompts[0]
    assert "旧任务：把游戏扩展到60个" in prompt
    assert "现在把游戏数量开发到70个" in prompt
    assert "最近一次用户消息" in prompt
    assert "## 最近进展" in prompt
    assert "<recent_messages>" in prompt


def test_summary_threshold_auto_from_model_window(monkeypatch):
    monkeypatch.setattr(
        ctx_mod,
        "get_effective",
        lambda key, default=None: {
            "summary_threshold_tokens": 0,
            "summary_threshold_ratio": 0.6,
        }.get(key, default),
    )
    monkeypatch.setattr(ctx_mod, "resolve_model_endpoint", lambda model: {"context_window": 100000})
    assert ctx_mod._resolve_threshold("qwen3.7-plus") == 60000


def test_summary_threshold_explicit_override(monkeypatch):
    monkeypatch.setattr(
        ctx_mod,
        "get_effective",
        lambda key, default=None: {"summary_threshold_tokens": 12345}.get(key, default),
    )
    assert ctx_mod._resolve_threshold("qwen3.7-plus") == 12345


def test_tool_digest_groups_actions():
    from app.api.agent import _tool_digest

    text = _tool_digest([
        {"name": "write_file", "args": {"path": "a.jsx"}, "status": "success"},
        {"name": "read_file", "args": {"path": "b.jsx"}, "status": "success"},
        {"name": "terminal", "args": {"command": "npm run build"}, "status": "error", "result": "exit 1"},
    ])
    assert "改动: write_file(a.jsx)" in text
    assert "读取: read_file(b.jsx)" in text
    assert "失败: terminal(npm run build): exit 1" in text


def test_settings_secret_roundtrip(monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("AGENT_SETTINGS_KEY", Fernet.generate_key().decode())
    secrets._fernet_ready = False
    secrets._fernet_cache = None
    try:
        token = secrets.encrypt_value("sk-test-123456")
        assert token.startswith(secrets.ENCRYPTED_PREFIX)
        assert secrets.decrypt_value(token) == "sk-test-123456"
    finally:
        secrets._fernet_ready = False
        secrets._fernet_cache = None


def test_message_out_keeps_attachments():
    m = MessageOut(
        id="1",
        role="user",
        content="hi",
        attachments=[{"name": "a.png", "path": "x", "size": 1, "kind": "image"}],
        created_at="",
    )
    assert m.attachments and m.attachments[0]["name"] == "a.png"


def test_web_fetch_rejects_private_hosts():
    from app.tools.web import validate_public_url

    for url in ("http://127.0.0.1/", "http://localhost:8000/", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            asyncio.run(validate_public_url(url))


def test_tool_call_signature_ignores_generated_ids():
    from app.agent.graph import _tool_calls_signature

    first = [{"id": "call_a", "name": "read_file", "args": {"path": "a.py"}}]
    second = [{"id": "call_b", "name": "read_file", "args": {"path": "a.py"}}]

    assert _tool_calls_signature(first) == _tool_calls_signature(second)
    changed = [{"id": "call_c", "name": "read_file", "args": {"path": "b.py"}}]
    assert _tool_calls_signature(first) != _tool_calls_signature(changed)


def test_exception_brief_never_empty():
    from app.agent.graph import _exception_brief

    assert _exception_brief(Exception()) == "Exception（无错误信息）"
    assert _exception_brief(ValueError("boom")).startswith("ValueError: boom")

    class EmptyMessageError(Exception):
        def __str__(self) -> str:
            return ""

    brief = _exception_brief(EmptyMessageError())
    assert brief
    assert "EmptyMessageError" in brief


def test_exception_brief_includes_cause_chain():
    """openai 的 APIConnectionError 只说 "Connection error."，底层原因必须一并带出。"""
    from app.agent.graph import _exception_brief

    try:
        try:
            raise OSError(11001, "getaddrinfo failed")
        except OSError as inner:
            raise RuntimeError("Connection error.") from inner
    except RuntimeError as e:
        brief = _exception_brief(e)
    assert "getaddrinfo failed" in brief
    assert "caused by" in brief


def test_connection_error_classification_and_budget():
    from app.agent.graph import _is_connection_error, _retry_budget, _retry_delay

    class ReadError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class StreamChunkTimeoutError(Exception):
        pass

    wrapped = APIConnectionError("Connection error.")
    wrapped.__cause__ = ReadError("")
    assert _is_connection_error(wrapped)
    assert not _is_connection_error(ValueError("bad request"))

    # 空闲超时已经白等一整个窗口，只允许再试一次；连接类给满退避表
    assert _retry_budget(StreamChunkTimeoutError("x"), 5) == 2
    assert _retry_budget(wrapped, 5) == 5
    assert _retry_budget(ValueError("x"), 5) == 3
    assert _retry_budget(ValueError("x"), 1) == 1
    assert 1.6 <= _retry_delay(1) <= 2.4
    assert 16.0 <= _retry_delay(4) <= 24.0


def test_reasoning_acc_emits_short_tail_immediately():
    """短思考的最后一段必须立即下发，不能扣留到流结束（否则前端看起来被截断）。"""
    from app.agent.graph import ReasoningAcc

    async def run():
        acc = ReasoningAcc(None)
        chunks: list[str] = []

        async def send(delta: str) -> None:
            chunks.append(delta)

        await acc.push(
            "Let me continue creating the remaining 8 games. I'll create them in",
            send,
        )
        return "".join(chunks)

    assert "I'll create them in" in asyncio.run(run())


def test_reasoning_acc_holds_referral_prefix():
    """行首是「用户问我…」引导句时仍要暂存，等整行到齐后再统一剥离。"""
    from app.agent.graph import ReasoningAcc

    async def run():
        acc = ReasoningAcc(None)
        chunks: list[str] = []

        async def send(delta: str) -> None:
            chunks.append(delta)

        await acc.push("用户问我这个游戏怎么改，", send)
        held = "".join(chunks)
        await acc.push("我先看一下代码。\n直接修改。", send)
        return held, "".join(chunks)

    held, emitted = asyncio.run(run())

    assert held == ""
    assert "用户问我" not in emitted
    assert "直接修改。" in emitted


def test_reasoning_acc_holds_short_referral_first_chunk():
    """线上首块可能只有「用户」两个字：必须暂存，不能把复述句半截漏给前端。

    回归：旧判定只看「当前缓冲是否已整句匹配复述正则」，首块太短匹配不上，
    于是半句被即时下发并清空缓冲，后半句再没有可识别的行首，整句原样泄漏。
    """
    from app.agent.graph import ReasoningAcc

    async def run():
        acc = ReasoningAcc("只回复 ok，不要调用任何工具")
        chunks: list[str] = []

        async def send(delta: str) -> None:
            chunks.append(delta)

        for piece in [
            "用户", "要求", "我只", '回复"', 'ok"，不调', "用任何工具。",
            "这是一个简单的指令，", "我应该", "直接遵守。\n",
        ]:
            await acc.push(piece, send)
        await acc.flush(send)
        return "".join(chunks)

    assert asyncio.run(run()) == ""


def test_reasoning_acc_releases_non_referral_line_after_newline():
    """以「用」开头但不是复述句的正常思考，换行后要正常下发。"""
    from app.agent.graph import ReasoningAcc

    async def run():
        acc = ReasoningAcc(None)
        chunks: list[str] = []

        async def send(delta: str) -> None:
            chunks.append(delta)

        await acc.push("用", send)
        await acc.push(" read_file 看一下代码。\n", send)
        await acc.flush(send)
        return "".join(chunks)

    assert "用 read_file 看一下代码。" in asyncio.run(run())


def _run_execute(llm, run_id: str, tools: list | None = None) -> tuple[dict, list[dict]]:
    import app.agent.graph as graph_mod
    from app.tools.context import ApprovalManager

    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    state = {
        "request": "你好", "workspace": tempfile.mkdtemp(),
        "messages": [HumanMessage(content="你好")],
        "plan": [], "todos": [],
        "tool_rounds": 0, "summary": "", "summary_count": 0,
        "run_id": run_id, "session_id": "",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": tools or [], "_llm": llm, "model": "test-model",
    }
    result = asyncio.run(graph_mod.execute_node(state))
    return result, events


def test_reasoning_similarity_flags_verbatim_restatement():
    from app.agent.graph import _reasoning_shingles, _reasoning_similarity

    base = (
        "The file is valid UTF-8 but contains mojibake sequences, so the strings were "
        "merged and the JS syntax broke. I need another script to repair it before build."
    )
    near = base.replace("so the strings were merged", "because the strings got merged")
    unrelated = "接下来创建游戏组件，先写音效工具，再补全局音量控制。"

    assert _reasoning_similarity(_reasoning_shingles(base), _reasoning_shingles(base)) == 1.0
    assert _reasoning_similarity(_reasoning_shingles(base), _reasoning_shingles(near)) >= 0.8
    assert _reasoning_similarity(
        _reasoning_shingles(base), _reasoning_shingles(unrelated)
    ) < 0.2
    assert _reasoning_shingles("") == frozenset()


def test_clean_reasoning_strips_leaked_think_tags():
    from app.agent.graph import _clean_reasoning

    out = _clean_reasoning("先说结论。</think>\n</anttThinking>下一步验证。", None)

    assert "</think>" not in out
    assert "anttThinking" not in out
    assert "先说结论" in out
    assert "下一步验证" in out


def test_execute_escalates_repeated_reasoning():
    """连续多轮复述同一段思考（工具调用却略有变化）→ 先提醒换思路，持续重复后切总结模式。"""
    import json

    from langchain_core.messages import AIMessageChunk
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    reasoning = (
        "The file is valid UTF-8 but contains mojibake sequences, which means the "
        "encoding conversion merged string literals and broke the JS syntax. I need to "
        "inspect the file again and write another repair script that normalizes the "
        "bytes before the bundler parses them, otherwise the build will keep failing."
    )

    class RepeatingThinkLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls >= 5:
                yield AIMessageChunk(content="已给出结论：剩余问题需要手动处理。")
                return
            yield AIMessageChunk(
                content="", additional_kwargs={"reasoning_content": reasoning},
            )
            yield AIMessageChunk(content="", tool_call_chunks=[{
                "name": "fake_read",
                "args": json.dumps({"path": f"f{self.calls}.txt"}),
                "id": f"call_{self.calls}",
                "index": 0,
            }])

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    result, events = _run_execute(RepeatingThinkLLM(), "repeated_reasoning", [fake_read])

    statuses = [e.get("status") for e in events if e["type"] == "status"]
    assert "reasoning_repeat_nudge" in statuses
    assert "reasoning_repeat_blocked" in statuses
    assert result.get("_loop_summary_mode") is True
    assert result["messages"][-1].content.startswith("已给出结论")
    # 被取消的工具调用必须补配对 ToolMessage，否则下一轮请求会被端点拒绝
    blocked_idx = [
        i for i, m in enumerate(result["messages"])
        if isinstance(m, ToolMessage) and m.content.startswith("【思考重复检测】")
    ]
    assert blocked_idx
    # 内部提醒必须是系统消息，模型才不会把它当作用户发言
    assert isinstance(result["messages"][blocked_idx[0] + 1], SystemMessage)
    assert "不是用户留言" in result["messages"][blocked_idx[0] + 1].content


def test_reasoning_bridge_keeps_edges_and_caps_length():
    from app.agent.graph import _REASONING_BRIDGE_MAX_CHARS, _reasoning_bridge

    assert _reasoning_bridge("太短") == ""
    text = "开头结论：CardBattleIcon 定义在 gameIconMap 之后。" + ("中间分析。" * 400) + "结尾行动：立即移动 map。"
    bridge = _reasoning_bridge(text)

    assert bridge.startswith("开头结论")
    assert bridge.endswith("结尾行动：立即移动 map。")
    assert "中间省略" in bridge
    assert len(bridge) <= _REASONING_BRIDGE_MAX_CHARS + 20


def test_execute_bridges_previous_reasoning_into_next_round():
    """上一轮思考要以临时系统消息回填给下一轮，且不能进入最终消息历史。"""
    import json

    from langchain_core.messages import AIMessageChunk
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    reasoning = (
        "The TDZ error is caused by gameIconMap being defined before CardBattleIcon. "
        "The fix is known: move the map below all icon components, then rebuild."
    )

    class RecordingBridgeLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0
            self.seen: list[list] = []

        async def astream(self, messages, **kwargs):
            self.seen.append(list(messages))
            self.calls += 1
            if self.calls == 1:
                yield AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": reasoning},
                )
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "fake_read",
                    "args": json.dumps({"path": "a.txt"}),
                    "id": "call_1",
                    "index": 0,
                }])
                return
            yield AIMessageChunk(content="已修复。")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    llm = RecordingBridgeLLM()
    result, _events = _run_execute(llm, "bridge_reasoning", [fake_read])

    assert len(llm.seen) == 2
    assert not any("上一轮思考" in str(m.content) for m in llm.seen[0])
    bridged = [m for m in llm.seen[1] if "上一轮思考" in str(m.content)]
    assert bridged and isinstance(bridged[0], SystemMessage)
    assert "CardBattleIcon" in bridged[0].content
    assert bridged[0].additional_kwargs.get("is_nudge") is True
    # 回填只存在于当次调用，不能写进最终消息历史（否则会被持久化、参与摘要）
    assert not any("上一轮思考" in str(m.content) for m in result["messages"])
    assert result["messages"][-1].content == "已修复。"


def test_execute_cuts_self_repeating_reasoning():
    """单轮思考反复复写同一段时立即中断，并把已有思考回填给下一轮。"""
    from langchain_core.messages import AIMessageChunk

    paragraph = (
        "The problem is the same TDZ error: gameIconMap references CardBattleIcon before "
        "its declaration, so the module throws at import time and the page goes blank. "
        "I need to move the map below all icon definitions and then rebuild. "
    )

    class LoopingReasoningLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0
            self.seen: list[list] = []

        async def astream(self, messages, **kwargs):
            self.seen.append(list(messages))
            self.calls += 1
            if self.calls == 1:
                yield AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning_content": paragraph * 16},
                )
                return
            yield AIMessageChunk(content="已直接落盘修复。")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    llm = LoopingReasoningLLM()
    result, events = _run_execute(llm, "self_repeat_reasoning", [])

    statuses = [e.get("status") for e in events if e["type"] == "status"]
    assert "reasoning_self_repeat" in statuses
    assert llm.calls == 2
    assert any("上一轮思考" in str(m.content) for m in llm.seen[1])
    nudges = [
        m for m in llm.seen[1]
        if isinstance(m, SystemMessage)
        and m.additional_kwargs.get("is_nudge")
        and "系统自动提醒" in m.content
    ]
    assert nudges and "不要重复之前的推理" in nudges[0].content
    assert result["messages"][-1].content == "已直接落盘修复。"


def test_execute_does_not_cut_long_non_repetitive_reasoning():
    """正常的长推理（超过旧 4000 字上限）不能再被硬截断。"""
    import json

    from langchain_core.messages import AIMessageChunk
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    long_reasoning = "".join(
        f"检查第{i}项：变量v{i}与函数f{i * 3}之间没有冲突，继续核对下一个模块。\n"
        for i in range(200)
    )
    assert len(long_reasoning) > 5000

    class LongThinkLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": long_reasoning},
                )
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "fake_read",
                    "args": json.dumps({"path": "a.txt"}),
                    "id": "call_1",
                    "index": 0,
                }])
                return
            yield AIMessageChunk(content="完成。")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    _result, events = _run_execute(LongThinkLLM(), "long_reasoning", [fake_read])

    statuses = [e.get("status") for e in events if e["type"] == "status"]
    assert "reasoning_self_repeat" not in statuses
    assert "reasoning_too_long" not in statuses


def test_trailing_reasoning_after_final_answer_dropped():
    """模型在正文之后才下发的收尾 reasoning 不应在最终回答下方留下一条 Think。"""

    from langchain_core.messages import AIMessageChunk

    class TrailingLLM:
        _llm_type = "mock"

        async def astream(self, messages, **kwargs):
            yield AIMessageChunk(
                content="", additional_kwargs={"reasoning_content": "先想一下。"},
            )
            yield AIMessageChunk(content="最终回答正文")
            yield AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "所有步骤已完成。让我给出最终回答。"},
            )

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    result, events = _run_execute(TrailingLLM(), "trailing_reasoning")

    thinking = "".join(e.get("delta", "") for e in events if e["type"] == "thinking")
    assert "先想一下" in thinking
    assert "所有步骤已完成" not in thinking
    assert [e["type"] for e in events if e["type"] in ("token", "thinking")][-1] == "token"
    assert result["messages"][-1].content == "最终回答正文"


def test_trailing_reasoning_kept_when_tool_calls_follow():
    """正文后的思考若后面还有工具调用，则仍应展示，不能误丢过程信息。"""

    from langchain_core.messages import AIMessageChunk

    class TrailingWithToolLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": "先想一下。"},
                )
                yield AIMessageChunk(content="我先试一下。", tool_call_chunks=[{
                    "name": "no_such_tool", "args": "{}", "id": "c1", "index": 0,
                }])
                yield AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning_content": "补充思考：还要调用工具。"},
                )
            else:
                yield AIMessageChunk(content="完成")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    result, events = _run_execute(TrailingWithToolLLM(), "trailing_tool")

    thinking = "".join(e.get("delta", "") for e in events if e["type"] == "thinking")
    assert "补充思考" in thinking
    assert result["messages"][-1].content == "完成"


def test_execute_hides_near_duplicate_thinking_across_rounds():
    """连续两轮几乎相同的思考只展示一次：模型每轮从同样的证据重新推导时不重复刷屏。"""
    import json

    from langchain_core.messages import AIMessageChunk
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    same = "让我先看看现有的CSS文件和页面结构，确认卡片与导航栏的样式定义位置。\n"

    class DupThinkLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                yield AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": same},
                )
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "fake_read",
                    "args": json.dumps({"path": f"f{self.calls}.txt"}),
                    "id": f"call_{self.calls}",
                    "index": 0,
                }])
                return
            yield AIMessageChunk(content="完成。")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    _result, events = _run_execute(DupThinkLLM(), "dup_think", [fake_read])

    thinking = "".join(e.get("delta", "") for e in events if e["type"] == "thinking")
    assert thinking.count("让我先看看现有的CSS文件和页面结构") == 1


def test_execute_keeps_distinct_thinking_across_rounds():
    """不同轮次的新分析必须正常展示，去重不能误伤正常思考。"""
    import json

    from langchain_core.messages import AIMessageChunk
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    first = "先确认首页卡片区域的样式定义，再决定卡片阴影与圆角的改法。\n"
    second = "卡片样式已经确认，接下来检查导航栏的毛玻璃模糊参数是否过重。\n"

    class DiffThinkLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                text = first if self.calls == 1 else second
                yield AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": text},
                )
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "fake_read",
                    "args": json.dumps({"path": f"f{self.calls}.txt"}),
                    "id": f"call_{self.calls}",
                    "index": 0,
                }])
                return
            yield AIMessageChunk(content="完成。")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    _result, events = _run_execute(DiffThinkLLM(), "diff_think", [fake_read])

    thinking = "".join(e.get("delta", "") for e in events if e["type"] == "thinking")
    assert "先确认首页卡片区域" in thinking
    assert "接下来检查导航栏" in thinking


def test_execute_hides_duplicate_intermediate_narration():
    """中间轮次逐字重复上一轮已经说过的旁白，不再重复展示。"""
    import json

    from langchain_core.messages import AIMessageChunk
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    narration = "好的，我来全面美化前端UI。先改造首页游戏卡片和搜索区域，然后优化导航栏和页脚。"

    class EchoLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                for i in range(0, len(narration), 6):
                    yield AIMessageChunk(content=narration[i:i + 6])
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "fake_read",
                    "args": json.dumps({"path": f"f{self.calls}.txt"}),
                    "id": f"call_{self.calls}",
                    "index": 0,
                }])
                return
            yield AIMessageChunk(content="完成。")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    _result, events = _run_execute(EchoLLM(), "echo_dup", [fake_read])

    tokens = "".join(e.get("delta", "") for e in events if e["type"] == "token")
    assert tokens.count(narration) == 1
    assert "完成。" in tokens


def test_execute_keeps_distinct_intermediate_narration():
    """不同轮次的旁白内容不同时必须正常展示，回显抑制不能误伤。"""
    import json

    from langchain_core.messages import AIMessageChunk
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    first = "先读取首页样式，确认卡片区域的现状。"
    second = "首页已确认，接下来检查导航栏的毛玻璃参数。"

    class DiffEchoLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                text = first if self.calls == 1 else second
                yield AIMessageChunk(content=text)
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "fake_read",
                    "args": json.dumps({"path": f"f{self.calls}.txt"}),
                    "id": f"call_{self.calls}",
                    "index": 0,
                }])
                return
            yield AIMessageChunk(content="完成。")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    _result, events = _run_execute(DiffEchoLLM(), "echo_diff", [fake_read])

    tokens = "".join(e.get("delta", "") for e in events if e["type"] == "token")
    assert first in tokens
    assert second in tokens


def test_execute_never_drops_final_answer_even_if_repeated():
    """最终回答即使与上一轮旁白相同也必须展示，回显抑制不能吞掉回答。"""
    import json

    from langchain_core.messages import AIMessageChunk
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    narration = "好的，我现在开始修改首页卡片样式。"

    class FinalEchoLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            yield AIMessageChunk(content=narration)
            if self.calls == 1:
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "fake_read", "args": json.dumps({"path": "a.txt"}),
                    "id": "call_1", "index": 0,
                }])

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    result, events = _run_execute(FinalEchoLLM(), "echo_final", [fake_read])

    tokens = "".join(e.get("delta", "") for e in events if e["type"] == "token")
    assert tokens.count(narration) == 2
    assert result["messages"][-1].content == narration


def test_execute_node_retries_empty_stream_error():
    """模型流式响应中途断开且未产生可见输出时，应自动重试而不是静默停止。"""
    import app.agent.graph as graph_mod
    from app.tools.context import ApprovalManager

    class FlakyLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield type("M", (), {
                    "content": "",
                    "tool_calls": [],
                    "additional_kwargs": {"reasoning_content": "先分析一下"},
                })()
                raise RuntimeError("")
            yield type("M", (), {"content": "重试成功", "tool_calls": []})()

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    llm = FlakyLLM()
    state = {
        "request": "你好", "workspace": tempfile.mkdtemp(),
        "messages": [HumanMessage(content="你好")],
        "plan": [], "todos": [],
        "tool_rounds": 0, "summary": "", "summary_count": 0,
        "run_id": "retry_test", "session_id": "retry_session",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": [], "_llm": llm, "model": "test-model",
    }

    result = asyncio.run(graph_mod.execute_node(state))

    assert result.get("error") is None, result.get("error")
    assert llm.calls == 2, f"应重试一次，实际调用 {llm.calls} 次"
    assert any("自动重试" in str(e.get("delta", "")) for e in events)
    assert result["messages"][-1].content == "重试成功"


def test_execute_node_retries_timeout_with_partial_tool_call():
    """流在工具参数生成中途超时：半截 tool_call 未派发，应重试而不是直接失败。"""
    import app.agent.graph as graph_mod
    from app.tools.context import ApprovalManager

    class StallingLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield type("M", (), {
                    "content": "",
                    "tool_calls": [{"id": "partial", "name": "write_file", "args": {}}],
                    "additional_kwargs": {"reasoning_content": "准备写文件"},
                })()
                raise TimeoutError("No streaming chunk received for 180.0s")
            yield type("M", (), {"content": "重试后完成", "tool_calls": []})()

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    llm = StallingLLM()
    state = {
        "request": "你好", "workspace": tempfile.mkdtemp(),
        "messages": [HumanMessage(content="你好")],
        "plan": [], "todos": [],
        "tool_rounds": 0, "summary": "", "summary_count": 0,
        "run_id": "retry_timeout", "session_id": "retry_timeout_session",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": [], "_llm": llm, "model": "test-model",
    }

    result = asyncio.run(graph_mod.execute_node(state))

    assert result.get("error") is None, result.get("error")
    assert llm.calls == 2
    assert any("自动重试" in str(e.get("delta", "")) for e in events)
    assert result["messages"][-1].content == "重试后完成"


def test_serialize_run_messages_drops_pre_summary_and_nudges():
    from app.api.agent import _serialize_run_messages

    messages = [
        HumanMessage(content="任务"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "read_file", "args": {}}]),
        ToolMessage(content="结果", tool_call_id="c1"),
        HumanMessage(
            content="【历史上下文摘要】旧内容", additional_kwargs={"is_summary": True}
        ),
        HumanMessage(content="内部提醒", additional_kwargs={"is_nudge": True}),
        AIMessage(content="完成"),
    ]

    items = _serialize_run_messages(messages, set())

    assert [i["role"] for i in items] == ["ai"]
    assert items[0]["content"] == "完成"


def test_env_masks_credentials_inside_database_url():
    from app.tools.env_check import _mask_value

    masked = _mask_value("DATABASE_URL", "postgresql://user:supersecret@localhost/db")

    assert "supersecret" not in masked
    assert masked.startswith("postgresql://user:***@")


def test_stream_accumulator_keeps_streamed_args_over_empty_complete_call():
    """DashScope 首个工具分片带 name+id、参数为空，会被 langchain 合成为 args={}
    的“完整”调用；累计器必须让后到的参数分片生效（否则 read_file args={} 全部校验失败）。"""
    from langchain_core.messages import AIMessageChunk

    from app.agent.graph import _StreamAccumulator

    acc = _StreamAccumulator()
    acc.add(AIMessageChunk(content="", tool_call_chunks=[
        {"name": "read_file", "args": "", "id": "call_1", "index": 0},
    ]))
    acc.add(AIMessageChunk(content="", tool_call_chunks=[
        {"name": None, "args": '{"path": "src/', "id": None, "index": 0},
    ]))
    acc.add(AIMessageChunk(content="", tool_call_chunks=[
        {"name": None, "args": 'App.tsx", "limit": 80}', "id": None, "index": 0},
    ]))
    calls = acc.tool_calls()
    assert calls == [{
        "id": "call_1", "name": "read_file",
        "args": {"path": "src/App.tsx", "limit": 80},
    }]


def test_stream_accumulator_rejects_truncated_args():
    """流中途断开时参数是半截 JSON：绝不部分修复后执行，标记非法让模型重发。"""
    from langchain_core.messages import AIMessageChunk

    from app.agent.graph import _StreamAccumulator

    acc = _StreamAccumulator()
    acc.add(AIMessageChunk(content="", tool_call_chunks=[
        {"name": "read_file", "args": '{"path": "a.py', "id": "call_2", "index": 0},
    ]))
    calls = acc.tool_calls()
    assert calls and calls[0]["name"] == "read_file"
    assert calls[0]["args"] == {}
    assert calls[0]["_invalid_args"] == '{"path": "a.py'


def test_stream_accumulator_supports_complete_only_chunks():
    """测试替身/非流式端点一次下发完整 tool_calls（无参数分片）也要能取到参数。"""
    import json

    from app.agent.graph import _StreamAccumulator

    class Complete:
        tool_calls = [{"id": "call_3", "name": "write_file", "args": {"path": "b.txt", "content": "hi"}}]
        tool_call_chunks = []
        usage_metadata = None

    acc = _StreamAccumulator()
    acc.add(Complete())
    calls = acc.tool_calls()
    assert json.dumps(calls[0]["args"], sort_keys=True) == '{"content": "hi", "path": "b.txt"}'


def test_validation_error_tells_model_missing_params():
    from app.agent.graph import _format_validation_error
    from langchain_core.tools import tool
    from pydantic import ValidationError

    @tool(parse_docstring=True)
    def read_file(path: str, limit: int = 10) -> str:
        """读取文件

        Args:
            path: 文件路径，相对于工作区的路径
            limit: 最多读取的行数
        """
        return ""

    try:
        read_file.invoke({})
    except ValidationError as e:
        text = _format_validation_error(read_file, e)
    else:  # pragma: no cover
        raise AssertionError("未触发参数校验错误")

    assert "参数缺失" in text
    assert "path" in text
    assert "文件路径" in text


def test_loop_guard_switches_to_summary_instead_of_hard_abort():
    """重复工具调用先拦截并提醒换做法继续，仍然重复才切总结模式，
    最后才硬性中止；提醒必须以系统消息下发，不能像用户要求。"""
    from langchain_core.tools import tool

    from app.agent import graph as graph_mod
    from app.tools.context import ApprovalManager

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        return "ok"

    class Chunk:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []
            self.tool_call_chunks = []
            self.usage_metadata = None

    class LoopingLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            if self.calls <= 7:
                yield Chunk(tool_calls=[{
                    "id": f"call_{self.calls}", "name": "fake_read", "args": {"path": "a.txt"},
                }])
            else:
                yield Chunk(content="重复调用已停止：本次没有取得新信息，建议更换任务描述后重试。")

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    state = {
        "request": "看看这个项目", "workspace": tempfile.mkdtemp(),
        "messages": [HumanMessage(content="看看这个项目")],
        "plan": [], "todos": [],
        "tool_rounds": 0, "summary": "", "summary_count": 0,
        "run_id": "loop_guard_test", "session_id": "loop_guard_test_session",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": [fake_read], "_llm": LoopingLLM(), "model": "test-model",
    }

    result = asyncio.run(graph_mod.execute_node(state))

    assert result.get("error") is None
    assert result["messages"][-1].content.startswith("重复调用已停止")
    assert "已中止：连续多轮检测到完全相同的工具调用" not in result["messages"][-1].content
    statuses = [(e.get("status"), e.get("detail", "")) for e in events if e.get("type") == "status"]
    assert any(s == "loop_guard" and "换一种做法" in d for s, d in statuses)
    assert any(s == "loop_guard" and "总结模式" in d for s, d in statuses)
    # 第一阶段提醒：系统消息，且明确说明不是用户要求，任务继续
    nudges = [
        m for m in result["messages"]
        if isinstance(m, SystemMessage) and m.additional_kwargs.get("is_nudge")
    ]
    assert any("不是用户留言" in m.content and "换一种做法" in m.content for m in nudges)
    assert any(
        isinstance(m, ToolMessage) and m.content.startswith("【系统自动保护】")
        for m in result["messages"]
    )


def test_long_final_answer_accepted_without_length_nudge():
    """长回答不触发任何“分析过长”提醒，也不被丢弃，直接按最终回答收尾。"""

    class Chunk:
        def __init__(self, content=""):
            self.content = content
            self.tool_calls = []
            self.tool_call_chunks = []
            self.usage_metadata = None

    class LongAnswerLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            yield Chunk(content="分析" * 1500)

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    llm = LongAnswerLLM()
    result, events = _run_execute(llm, "long_answer")

    answer = "分析" * 1500
    assert result["messages"][-1].content == answer
    assert llm.calls == 1
    blocked = [e for e in events if e.get("type") == "status" and e.get("status") == "reasoning_loop_blocked"]
    assert blocked == []
    nudges = [
        m for m in result["messages"]
        if isinstance(m, SystemMessage) and m.additional_kwargs.get("is_nudge")
    ]
    assert nudges == []


def test_summary_mode_never_executes_tool_calls():
    """总结模式下模型仍返回工具调用：一律不执行，连续两轮则系统自动终止。"""
    from langchain_core.tools import tool

    from app.agent import graph as graph_mod
    from app.tools.context import ApprovalManager

    executed = {"n": 0}

    @tool(parse_docstring=True)
    def fake_read(path: str) -> str:
        """读取文件

        Args:
            path: 文件路径
        """
        executed["n"] += 1
        return "ok"

    class Chunk:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []
            self.tool_call_chunks = []
            self.usage_metadata = None

    class StubbornLLM:
        _llm_type = "mock"

        def __init__(self):
            self.calls = 0

        async def astream(self, messages, **kwargs):
            self.calls += 1
            yield Chunk(tool_calls=[{
                "id": f"c{self.calls}", "name": "fake_read", "args": {"path": "a.txt"},
            }])

        def bind_tools(self, tools):
            return self

        def with_retry(self, **kw):
            return self

    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    state = {
        "request": "你好", "workspace": tempfile.mkdtemp(),
        "messages": [HumanMessage(content="你好")],
        "plan": [], "todos": [],
        "tool_rounds": 0, "summary": "", "summary_count": 0,
        "run_id": "summary_tools", "session_id": "",
        "stopped": False, "error": None, "emit": emit,
        "approvals": ApprovalManager(mode="auto"), "stop_event": asyncio.Event(),
        "_tools": [fake_read], "_llm": StubbornLLM(), "model": "test-model",
        "_loop_summary_mode": True,
    }

    result = asyncio.run(graph_mod.execute_node(state))

    assert executed["n"] == 0, "总结模式下禁止执行任何真实工具"
    assert result["messages"][-1].content.startswith("系统自动终止：总结模式下仍返回工具调用")
    statuses = [e.get("status") for e in events if e.get("type") == "status"]
    assert "summary_tool_blocked" in statuses


def test_summary_input_skips_internal_nudges():
    """内部提醒是系统自动产生的，不得作为对话内容进入摘要输入。"""
    from app.agent.context import _format_for_summary

    text = _format_for_summary([
        HumanMessage(content="任务内容"),
        SystemMessage(content="【系统自动保护】内部提醒", additional_kwargs={"is_nudge": True}),
        AIMessage(content="继续执行"),
    ])

    assert "任务内容" in text
    assert "内部提醒" not in text
