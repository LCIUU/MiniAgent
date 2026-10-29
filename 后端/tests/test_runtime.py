"""RunHandle 订阅制广播与 find_active 的单元测试。"""

import asyncio

from app.services.runtime import _DONE, RunHandle, RunManager


def _without_seq(ev):
    return {k: v for k, v in ev.items() if k != "seq"}


def test_subscribe_replays_history_then_live():
    """新订阅者先收到全部历史事件（无 _ts），再收到实时事件。"""
    async def run():
        h = RunHandle(run_id="r1", session_id="s1")
        await h.emit({"type": "thinking", "delta": "前半"})
        q = h.subscribe()
        await h.emit({"type": "thinking", "delta": "后半"})
        return q

    q = asyncio.run(run())
    first = q.get_nowait()
    assert _without_seq(first) == {"type": "thinking", "delta": "前半"}  # 历史回放且剥离 _ts
    assert _without_seq(q.get_nowait()) == {"type": "thinking", "delta": "后半"}  # 实时事件紧随其后


def test_multiple_subscribers_independent():
    """多个订阅者各自收到完整事件流（切换会话/多标签页互不抢占）。"""
    async def run():
        h = RunHandle(run_id="r1", session_id="s1")
        q1 = h.subscribe()
        await h.emit({"type": "token", "delta": "x"})
        q2 = h.subscribe()  # 中途加入：拿到历史回放
        await h.emit({"type": "token", "delta": "y"})
        return q1, q2

    q1, q2 = asyncio.run(run())
    assert [_without_seq(q1.get_nowait()), _without_seq(q1.get_nowait())] == [
        {"type": "token", "delta": "x"},
        {"type": "token", "delta": "y"},
    ]
    assert [_without_seq(q2.get_nowait()), _without_seq(q2.get_nowait())] == [
        {"type": "token", "delta": "x"},  # 回放补齐
        {"type": "token", "delta": "y"},
    ]


def test_finished_run_subscribe_gets_done():
    """已结束的 run：订阅者收到历史回放 + 结束信号；run_done 在 _DONE 之前。"""
    async def run():
        h = RunHandle(run_id="r1", session_id="s1")
        await h.emit({"type": "token", "delta": "回答"})
        await h.emit({"type": "run_done", "duration_ms": 123})
        h.finish("done")
        return h.subscribe()

    q = asyncio.run(run())
    items = [q.get_nowait(), q.get_nowait(), q.get_nowait()]
    assert items[0]["type"] == "token"
    assert items[1]["type"] == "run_done"  # 结束信号前必有耗时事件
    assert items[2] is _DONE


def test_find_active():
    mgr = RunManager()
    h = RunHandle(run_id="r1", session_id="s1")
    mgr._runs["r1"] = h
    assert mgr.find_active("s1") is h
    assert mgr.find_active("s2") is None
    h.status = "done"
    assert mgr.find_active("s1") is None  # 已结束不算活动
