"""跨轮上下文：完整消息流（AI tool_calls + ToolMessage）序列化/恢复。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.api.agent import (
    _ensure_tool_pairs,
    _history_from_db,
    _messages_from_run,
    _serialize_run_messages,
)
from app.models import Message


def test_serialize_run_messages_keeps_tool_stream():
    history = HumanMessage(content="旧问题")
    init_ai = AIMessage(content="旧回答")
    new_ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "read_file", "args": {"path": "a.py"}}],
    )
    tool = ToolMessage(content="print(1)", tool_call_id="call_1", name="read_file")
    final = AIMessage(content="完成了")
    items = _serialize_run_messages(
        [history, init_ai, SystemMessage(content="sys"), new_ai, tool, final],
        {id(history), id(init_ai)},
    )

    assert [i["role"] for i in items] == ["ai", "tool", "ai"]
    assert items[0]["tool_calls"][0]["id"] == "call_1"
    assert items[0]["tool_calls"][0]["args"] == {"path": "a.py"}
    assert items[1]["tool_call_id"] == "call_1"
    assert items[1]["content"] == "print(1)"
    assert items[2]["content"] == "完成了"


def test_serialize_run_messages_skips_summary_and_system():
    summary = HumanMessage(
        content="【历史上下文摘要】旧摘要", additional_kwargs={"is_summary": True}
    )
    items = _serialize_run_messages(
        [SystemMessage(content="sys"), summary, HumanMessage(content="提醒"), AIMessage(content="好")],
        set(),
    )
    assert [i["role"] for i in items] == ["human", "ai"]
    assert items[0]["content"] == "提醒"


def test_history_from_db_rebuilds_full_stream():
    m = Message(session_id="s", role="assistant", content="完成了")
    m.context_messages = [
        {
            "role": "ai",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "read_file", "args": {"path": "a.py"}}],
        },
        {"role": "tool", "content": "文件内容", "tool_call_id": "c1", "name": "read_file"},
        {"role": "ai", "content": "完成了", "tool_calls": []},
    ]

    out = _history_from_db([m])

    assert isinstance(out[0], AIMessage)
    assert out[0].tool_calls[0]["id"] == "c1"
    assert isinstance(out[1], ToolMessage)
    assert out[1].tool_call_id == "c1"
    assert out[2].content == "完成了"


def test_history_from_db_uses_enriched_request_content():
    """追问附件内容时，历史里应还原实际发送正文，而不是只看到文件名。"""
    m = Message(session_id="s", role="user", content="总结这份文档")
    m.attachments = [{"name": "报告.pdf", "path": ".agent_uploads/报告.pdf", "kind": "pdf"}]
    m.request_content = "总结这份文档\n\n【附件正文: 报告.pdf】\n核心结论：……"

    out = _history_from_db([m])

    assert isinstance(out[0], HumanMessage)
    assert "核心结论" in out[0].content
    assert "[本轮附件:" not in out[0].content


def test_history_from_db_falls_back_to_attachment_names_for_old_rows():
    m = Message(session_id="s", role="user", content="看看这个")
    m.attachments = [{"name": "a.png", "path": "x.png", "kind": "image"}]

    out = _history_from_db([m])

    assert "看看这个" in out[0].content
    assert "a.png" in out[0].content


def test_history_from_db_falls_back_to_digest_for_old_rows():
    m = Message(
        session_id="s",
        role="assistant",
        content="好了",
        tool_calls=[{"name": "write_file", "args": {"path": "a.py"}, "status": "success"}],
    )

    out = _history_from_db([m])

    assert isinstance(out[0], AIMessage)
    assert "上轮操作" in out[0].content
    assert "write_file(a.py)" in out[0].content


def test_ensure_tool_pairs_fills_interrupted_calls():
    items = [
        {"role": "ai", "content": "", "tool_calls": [{"id": "x", "name": "t", "args": {}}]},
    ]

    out = _ensure_tool_pairs(items)

    assert out[-1]["role"] == "tool"
    assert out[-1]["tool_call_id"] == "x"
    assert "中断" in out[-1]["content"]


def test_messages_from_run_roundtrip_tool_linkage():
    raw = [
        {
            "role": "ai",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "write_file", "args": {"path": "a.py", "content": "x"}}],
        },
        {"role": "tool", "content": "written", "tool_call_id": "c1", "name": "write_file"},
        {"role": "ai", "content": "写好了", "tool_calls": []},
    ]

    msgs = _messages_from_run(raw)

    assert len(msgs) == 3
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].tool_calls[0]["args"]["content"] == "x"
    assert isinstance(msgs[1], ToolMessage)
    assert msgs[1].tool_call_id == "c1"
    assert msgs[2].content == "写好了"
