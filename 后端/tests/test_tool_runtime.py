"""工具执行层测试：分组调度 / 并行提速 / 超时治理 / 校验自纠 / 缓存写失效 / 用量统计。

不依赖真实 LLM 与数据库（session_id="" 时 DB 缓存自动短路，内存缓存生效）。
pytest 与直接 `python tests/test_tool_runtime.py` 均可运行。
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool

import app.agent.graph as g
from app.tools.registry import ToolSpec, register_spec

# ---------------------------------------------------------------------------
# 测试专用工具（模块加载时注册 spec）
# ---------------------------------------------------------------------------

SLOW_DELAY = 0.3


@tool
async def slow_read(name: str) -> str:
    """慢速只读工具（模拟 IO）"""
    await asyncio.sleep(SLOW_DELAY)
    return f"read:{name}"


@tool
async def typed_tool(count: int) -> str:
    """严格类型工具（用于参数校验测试）"""
    return f"count:{count}"


@tool
async def very_slow(x: int = 1) -> str:
    """超慢工具（用于外层超时治理测试）"""
    await asyncio.sleep(5)
    return "done"


COUNTER = {"n": 0}


@tool
async def counted_read(k: str) -> str:
    """带执行计数的只读工具（用于缓存测试）"""
    COUNTER["n"] += 1
    return f"v{COUNTER['n']}"


@tool
async def plain_write(k: str) -> str:
    """写工具（用于缓存失效测试）"""
    return "w"


WRITE_COUNTER = {"n": 0}


@tool
async def counted_write(path: str, content: str) -> str:
    """带执行计数的写工具（用于同轮重复调用去重测试）"""
    WRITE_COUNTER["n"] += 1
    return f"written:{path}:{len(content)}"


register_spec("slow_read", ToolSpec(category="test", idempotent=True, timeout=5))
register_spec("typed_tool", ToolSpec(category="test", idempotent=True, timeout=5))
register_spec("very_slow", ToolSpec(category="test", idempotent=False, timeout=0.1))
register_spec("counted_read", ToolSpec(category="test", idempotent=True, timeout=5))
register_spec("plain_write", ToolSpec(category="test", idempotent=False, timeout=5))
register_spec("counted_write", ToolSpec(category="test", idempotent=False, timeout=5))

TOOLS_BY_NAME = {
    t.name: t
    for t in [slow_read, typed_tool, very_slow, counted_read, plain_write, counted_write]
}


def _state(events: list) -> dict:
    async def _emit(ev):
        events.append(ev)

    return {"run_id": "t", "session_id": "", "emit": _emit}


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

def test_group_real_calls():
    """分组：连续幂等并为一段，写操作单独成段，顺序语义保持。"""
    calls = [
        (0, {"id": "a", "name": "slow_read", "args": {}}),
        (1, {"id": "b", "name": "slow_read", "args": {}}),
        (2, {"id": "c", "name": "plain_write", "args": {}}),
        (3, {"id": "d", "name": "slow_read", "args": {}}),
        (4, {"id": "e", "name": "plain_write", "args": {}}),
        (5, {"id": "f", "name": "plain_write", "args": {}}),
    ]
    groups = g._group_real_calls(calls)
    sizes = [len(x) for x in groups]
    assert sizes == [2, 1, 1, 1, 1], f"分组结果不符: {sizes}"
    # 顺序保持
    flat = [i for grp in groups for i, _ in grp]
    assert flat == [0, 1, 2, 3, 4, 5]
    print("PASS test_group_real_calls")


def test_parallel_speedup():
    """并行段：2 个 0.3s 幂等工具并发执行，总耗时应显著低于串行。"""

    async def run():
        events = []
        state = _state(events)
        cache = g.ToolResultCache("")
        calls = [
            {"id": "c1", "name": "slow_read", "args": {"name": "a"}},
            {"id": "c2", "name": "slow_read", "args": {"name": "b"}},
        ]
        t0 = time.monotonic()
        msgs = await asyncio.gather(*(
            g._run_one_tool(state, tc, TOOLS_BY_NAME[tc["name"]], cache) for tc in calls
        ))
        elapsed = time.monotonic() - t0
        return events, msgs, elapsed

    events, msgs, elapsed = asyncio.run(run())
    assert elapsed < SLOW_DELAY * 1.8, f"未体现并行提速: {elapsed:.2f}s"
    assert all(m.content.startswith("read:") for m in msgs)
    starts = [e for e in events if e["type"] == "tool_start"]
    assert len(starts) == 2
    print(f"PASS test_parallel_speedup (elapsed={elapsed:.2f}s)")


def test_timeout_governance():
    """超时治理：spec.timeout=0.1 的工具被外层强制截断，返回结构化超时反馈。"""

    async def run():
        events = []
        state = _state(events)
        cache = g.ToolResultCache("")
        return await g._run_one_tool(
            state, {"id": "t1", "name": "very_slow", "args": {}}, very_slow, cache,
        ), events

    msg, events = asyncio.run(run())
    assert "超时" in msg.content, f"应返回超时提示: {msg.content}"
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["status"] == "error"
    print("PASS test_timeout_governance")


def test_validation_feedback():
    """校验自纠：非法参数 → 字段级结构化错误反馈，引导模型修正重调。"""

    async def run():
        events = []
        state = _state(events)
        cache = g.ToolResultCache("")
        return await g._run_one_tool(
            state, {"id": "v1", "name": "typed_tool", "args": {"count": "abc"}}, typed_tool, cache,
        )

    msg = asyncio.run(run())
    assert "【参数校验失败】" in msg.content, f"应为校验失败反馈: {msg.content}"
    assert "count" in msg.content, "应定位到出错字段"
    assert "typed_tool" in msg.content
    print("PASS test_validation_feedback")


def test_cache_invalidate_on_write():
    """缓存闭环：幂等结果命中缓存；写操作后缓存失效重新执行。"""

    async def run():
        events = []
        state = _state(events)
        cache = g.ToolResultCache("")
        tc = lambda i, name, args: {"id": i, "name": name, "args": args}  # noqa: E731
        COUNTER["n"] = 0
        m1 = await g._run_one_tool(state, tc("1", "counted_read", {"k": "x"}), counted_read, cache)
        m2 = await g._run_one_tool(state, tc("2", "counted_read", {"k": "x"}), counted_read, cache)
        assert COUNTER["n"] == 1, "第二次调用应命中缓存"
        assert "重复调用拦截" in m2.content
        await g._run_one_tool(state, tc("3", "plain_write", {"k": "x"}), plain_write, cache)
        m3 = await g._run_one_tool(state, tc("4", "counted_read", {"k": "x"}), counted_read, cache)
        assert COUNTER["n"] == 2, "写操作后缓存应失效并重新执行"
        assert m3.content == "v2"
        return m1

    asyncio.run(run())
    print("PASS test_cache_invalidate_on_write")


class _Chunk:
    def __init__(self, content="", tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage or {"input_tokens": 10, "output_tokens": 5}


class _ParallelMockLLM:
    """第 1 轮返回 2 个幂等调用（验证 execute_node 并行调度），第 2 轮输出文本。"""

    _llm_type = "mock"

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def with_retry(self, **kw):
        return self

    async def astream(self, messages, **kw):
        self.calls += 1
        if self.calls == 1:
            yield _Chunk(tool_calls=[
                {"id": "c1", "name": "slow_read", "args": {"name": "a"}},
                {"id": "c2", "name": "slow_read", "args": {"name": "b"}},
            ])
        else:
            yield _Chunk(content="完成")

    async def ainvoke(self, messages, **kw):
        return _Chunk(content="summary")


def test_execute_node_parallel_and_usage():
    """execute_node 集成：一轮双幂等调用并行执行 + token 用量累计。"""

    async def run():
        events = []

        async def _emit(ev):
            events.append(ev)

        state = {
            "request": "t", "workspace": ".", "messages": [],
            "plan": ["步骤"], "todos": [{"content": "步骤", "status": "in_progress"}],
            "tool_rounds": 0,
            "run_id": "t", "session_id": "", "stopped": False, "error": None,
            "emit": _emit, "stop_event": asyncio.Event(),
            "_tools": [slow_read], "_llm": _ParallelMockLLM(),
        }
        t0 = time.monotonic()
        result = await g.execute_node(state)
        return result, events, time.monotonic() - t0

    result, events, elapsed = asyncio.run(run())
    assert elapsed < SLOW_DELAY * 1.8, f"execute_node 未并行: {elapsed:.2f}s"
    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(ends) == 2 and all(e["status"] == "success" for e in ends)
    # ToolMessage 顺序与 tool_calls 顺序对应
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]
    # 用量累计：2 次 LLM 调用 × (10+5)
    usage = result.get("token_usage")
    assert usage and usage["llm_calls"] == 2 and usage["input_tokens"] == 20, f"用量累计异常: {usage}"
    print(f"PASS test_execute_node_parallel_and_usage (elapsed={elapsed:.2f}s)")


class _DuplicateWriteMockLLM:
    """第 1 轮输出两个完全相同的写调用，第 2 轮收尾。"""

    _llm_type = "mock"

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, messages, **kw):
        self.calls += 1
        if self.calls == 1:
            yield _Chunk(tool_calls=[
                {"id": "w1", "name": "counted_write",
                 "args": {"path": "a.css", "content": "x"}},
                {"id": "w2", "name": "counted_write",
                 "args": {"path": "a.css", "content": "x"}},
            ])
        else:
            yield _Chunk(content="完成")

    async def ainvoke(self, messages, **kw):
        return _Chunk(content="summary")


def test_same_round_duplicate_write_deduped():
    """同轮完全相同的写调用只执行一次，且每个 call_id 都有对应 ToolMessage。"""

    async def run():
        events = []

        async def _emit(ev):
            events.append(ev)

        state = {
            "request": "t", "workspace": ".", "messages": [],
            "plan": ["步骤"], "todos": [{"content": "步骤", "status": "in_progress"}],
            "tool_rounds": 0,
            "run_id": "t", "session_id": "", "stopped": False, "error": None,
            "emit": _emit, "stop_event": asyncio.Event(),
            "_tools": [counted_write], "_llm": _DuplicateWriteMockLLM(),
        }
        return await g.execute_node(state), events

    WRITE_COUNTER["n"] = 0
    result, events = asyncio.run(run())

    assert WRITE_COUNTER["n"] == 1, "重复写调用不应被执行两次"
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["w1", "w2"]
    assert "重复调用拦截" in tool_msgs[1].content
    assert "written:a.css:1" in tool_msgs[1].content
    # 重复调用不再产生第二条 tool_start/tool_end
    assert len([e for e in events if e["type"] == "tool_start"]) == 1
    assert len([e for e in events if e["type"] == "tool_end"]) == 1
    print("PASS test_same_round_duplicate_write_deduped")


if __name__ == "__main__":
    test_group_real_calls()
    test_parallel_speedup()
    test_timeout_governance()
    test_validation_feedback()
    test_cache_invalidate_on_write()
    test_execute_node_parallel_and_usage()
    test_same_round_duplicate_write_deduped()
    print("\nALL TOOL RUNTIME TESTS PASSED")
