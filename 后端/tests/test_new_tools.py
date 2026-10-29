"""新增工具回归测试：快照 / 批量替换 / 编码 / 密钥扫描 / JSON 编辑 / patch 预览 / 执行诊断。"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def _emit(_event: dict) -> None:
    return None


def _make_tools(ws: str) -> dict:
    from app.tools import build_tools
    from app.tools.context import ApprovalManager

    tools = build_tools(ws, _emit, ApprovalManager(), asyncio.Event())
    return {t.name: t for t in tools}


def _workspace() -> str:
    return tempfile.mkdtemp(prefix="agent-tools-")


def _invoke(tools: dict, tool_name: str, **kwargs) -> str:
    return asyncio.run(tools[tool_name].ainvoke(kwargs))


def test_json_edit_preserves_format_and_chinese():
    ws = _workspace()
    try:
        p = Path(ws, "conf.json")
        p.write_text(
            json.dumps({"name": "示例", "scripts": {"build": "tsc"}, "list": [1, 2]}, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tools = _make_tools(ws)

        assert '"tsc"' in _invoke(tools, "json_edit", path="conf.json", pointer="/scripts/build", action="get")
        _invoke(tools, "json_edit", path="conf.json", pointer="/scripts/test", value='"pytest"', action="set")
        _invoke(tools, "json_edit", path="conf.json", pointer="/list/0", action="delete")

        text = p.read_text(encoding="utf-8")
        data = json.loads(text)
        assert data["scripts"]["test"] == "pytest"
        assert data["list"] == [2]
        assert data["name"] == "示例"
        assert "\n    \"scripts\"" in text  # 4 空格缩进保持
        assert "示例" in text and "\\u793a" not in text  # 中文未被转义
        assert list(json.loads(text).keys()) == ["name", "scripts", "list"]  # 键顺序保持

        missing = _invoke(tools, "json_edit", path="conf.json", pointer="/nope/x", value="1", action="set")
        assert "父节点不存在" in missing
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_json_edit_rejects_jsonc():
    ws = _workspace()
    try:
        Path(ws, "tsconfig.json").write_text('{\n  // 注释\n  "compilerOptions": {}\n}\n', encoding="utf-8")
        tools = _make_tools(ws)
        out = _invoke(tools, "json_edit", path="tsconfig.json", pointer="/compilerOptions", action="get")
        assert "JSONC" in out and "edit_file" in out
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_apply_patch_dry_run_does_not_write():
    ws = _workspace()
    try:
        p = Path(ws, "app.py")
        p.write_text("value = 1\n", encoding="utf-8")
        tools = _make_tools(ws)
        patch = """--- a/app.py
+++ b/app.py
@@ -1,1 +1,1 @@
-value = 1
+value = 2
"""
        preview = _invoke(tools, "apply_patch", patch=patch, dry_run=True)
        assert "预览" in preview and "+value = 2" in preview
        assert p.read_text(encoding="utf-8") == "value = 1\n"

        result = _invoke(tools, "apply_patch", patch=patch, dry_run=False)
        assert "apply_patch 完成" in result
        assert p.read_text(encoding="utf-8") == "value = 2\n"
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_replace_in_files_dry_run_then_apply():
    ws = _workspace()
    try:
        a = Path(ws, "a.txt")
        b = Path(ws, "nested", "b.txt")
        b.parent.mkdir()
        a.write_text("one two one\n", encoding="utf-8")
        b.write_text("one\n", encoding="utf-8")
        tools = _make_tools(ws)

        preview = _invoke(tools, "replace_in_files", pattern="one", replacement="three")
        assert "【预览】" in preview
        assert a.read_text(encoding="utf-8") == "one two one\n"

        applied = _invoke(tools, "replace_in_files", pattern="one", replacement="three", dry_run=False)
        assert "已替换 2 个文件，共 3 处" in applied
        assert a.read_text(encoding="utf-8") == "three two three\n"
        assert b.read_text(encoding="utf-8") == "three\n"
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_check_encoding_detects_and_fixes_gbk():
    ws = _workspace()
    try:
        gbk = Path(ws, "gbk.txt")
        gbk.write_bytes("中文内容\n".encode("gbk"))
        utf8 = Path(ws, "utf8.txt")
        utf8.write_text("正常内容\n", encoding="utf-8")
        tools = _make_tools(ws)

        report = _invoke(tools, "check_encoding")
        assert "GBK" in report and "gbk.txt" in report
        assert gbk.read_bytes() == "中文内容\n".encode("gbk")

        fixed = _invoke(tools, "check_encoding", fix=True)
        assert "转换" in fixed
        assert gbk.read_bytes().decode("utf-8") == "中文内容\n"
        assert utf8.read_text(encoding="utf-8") == "正常内容\n"
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_secret_scan_masks_and_skips_placeholders():
    ws = _workspace()
    try:
        Path(ws, "a.env").write_text(
            "TOKEN=abc123realtokenvalue\nPASSWORD=changeme\nAPI_KEY=${API_KEY}\nmax_tokens=4096\n",
            encoding="utf-8",
        )
        Path(ws, "b.py").write_text('key = "sk-abcdefghijklmnopqrstuvwx"\n', encoding="utf-8")
        tools = _make_tools(ws)

        out = _invoke(tools, "secret_scan")
        assert "sk-a***wx" in out
        assert "abc1***ue" in out
        assert "sk-abcdefghijklmnopqrstuvwx" not in out
        assert "changeme" not in out
        assert "max_tokens" not in out
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_checkpoint_restore_roundtrip():
    ws = _workspace()
    try:
        f = Path(ws, "data.txt")
        f.write_text("original\n", encoding="utf-8")
        tools = _make_tools(ws)

        created = _invoke(tools, "workspace_checkpoint", action="create", label="before-edit")
        assert "已创建快照" in created
        checkpoint_id = created.split("快照 ")[1].split("：")[0]

        f.write_text("broken\n", encoding="utf-8")
        Path(ws, "extra.txt").write_text("extra\n", encoding="utf-8")

        restored = _invoke(tools, "workspace_checkpoint", action="restore", checkpoint_id=checkpoint_id)
        assert "已恢复快照" in restored
        assert f.read_text(encoding="utf-8") == "original\n"
        assert Path(ws, "extra.txt").exists()  # 恢复只覆盖，不删除恢复后新增文件

        listed = _invoke(tools, "workspace_checkpoint", action="list")
        assert checkpoint_id in listed
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_read_files_and_reference_tools():
    ws = _workspace()
    try:
        Path(ws, "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        Path(ws, "b.py").write_text("from a import foo\nprint(foo())\n", encoding="utf-8")
        tools = _make_tools(ws)

        out = _invoke(tools, "read_files", paths="a.py, b.py")
        assert "=== a.py" in out and "=== b.py" in out
        assert "1 | def foo():" in out

        refs = _invoke(tools, "find_references", name="foo")
        assert "a.py:1" in refs and "b.py:2" in refs

        deps = _invoke(tools, "find_dependents", path="a.py")
        assert "b.py:1" in deps
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_read_file_returns_full_content_for_tool_card():
    """read_file 不再二次截断：完整返回所选行，tool_end 也回传完整内容供卡片展示。"""
    from app.agent import graph as g

    ws = _workspace()
    try:
        lines = [f"line_{i:04d} = '{'x' * 40}'" for i in range(2500)]
        Path(ws, "big.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
        tools = _make_tools(ws)

        out = _invoke(tools, "read_file", path="big.py")
        assert "已截断" not in out
        assert "总行数: 2500" in out
        assert "line_2499" in out
        assert len(out) > 150000

        events: list[dict] = []

        async def _emit(ev: dict) -> None:
            events.append(ev)

        state = {"run_id": "t", "session_id": "", "emit": _emit}
        cache = g.ToolResultCache("")
        call = {"name": "read_file", "args": {"path": "big.py"}}
        msg = asyncio.run(g._run_one_tool(
            state, {"id": "r1", **call}, tools["read_file"], cache,
        ))
        end = next(e for e in events if e["type"] == "tool_end")
        assert "line_2499" in end["result"], "工具卡片应展示完整读取内容"
        assert "line_2499" in msg.content, "模型也应看到完整内容"
        assert len(msg.content) > 150000, "读取类结果不再做字符上限截断"
        assert "工具输出已截断" not in msg.content

        # 重复读取走缓存，模型仍应拿到完整内容
        repeat = asyncio.run(g._run_one_tool(
            state, {"id": "r2", **call}, tools["read_file"], cache,
        ))
        assert "line_2499" in repeat.content
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_run_code_python_and_reject_remote_wait():
    ws = _workspace()
    try:
        tools = _make_tools(ws)
        out = _invoke(tools, "run_code", language="python", code="print('hello', 1 + 1)")
        assert "退出码: 0" in out and "hello 2" in out
        assert not list(Path(ws, ".agent_logs").glob("run-*"))  # 临时脚本已清理

        blocked = _invoke(tools, "wait_for_server", url="http://example.com", timeout=2)
        assert "安全限制" in blocked
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_run_tests_and_audit_unknown_project():
    ws = _workspace()
    try:
        tools = _make_tools(ws)
        assert "无法识别项目类型" in _invoke(tools, "run_tests")
        assert "未识别到可审计的依赖清单" in _invoke(tools, "audit_dependencies")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_git_stash_requires_repo():
    if shutil.which("git") is None:
        return
    ws = _workspace()
    try:
        tools = _make_tools(ws)
        assert "不是 Git 仓库" in _invoke(tools, "git_stash", action="list")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_list_ports_returns_readable_output():
    ws = _workspace()
    try:
        tools = _make_tools(ws)
        out = _invoke(tools, "list_ports")
        assert "监听端口" in out or "没有找到监听端口" in out
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def _make_tools_with_questions(ws: str, questions, emit=None) -> dict:
    from app.tools import build_tools
    from app.tools.context import ApprovalManager

    async def _noop(_event: dict) -> None:
        return None

    tools = build_tools(
        ws, emit or _noop, ApprovalManager(), asyncio.Event(), questions=questions,
    )
    return {t.name: t for t in tools}


async def _wait_question(events: list) -> dict:
    for _ in range(300):
        found = next((e for e in events if e.get("type") == "question"), None)
        if found is not None:
            return found
        await asyncio.sleep(0.01)
    raise AssertionError("未在超时时间内收到 question 事件")


def test_ask_user_roundtrip():
    from app.tools.context import QuestionManager

    ws = _workspace()
    try:
        async def run():
            events: list = []

            async def emit(ev):
                events.append(ev)

            qm = QuestionManager()
            tools = _make_tools_with_questions(ws, qm, emit)
            task = asyncio.create_task(
                tools["ask_user"].ainvoke({"question": "选哪个方案？", "options": "A 方案，B 方案"})
            )
            ev = await _wait_question(events)
            assert ev["options"] == ["A 方案", "B 方案"]
            assert ev["allow_multiple"] is False
            assert qm.answer(ev["question_id"], ["B 方案"]) is True
            assert "B 方案" in await asyncio.wait_for(task, timeout=5)
            assert any(
                e.get("type") == "question_done" and e["question_id"] == ev["question_id"]
                for e in events
            )

        asyncio.run(run())
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_delete_file_requires_tool_level_confirmation():
    from app.tools.context import QuestionManager

    ws = _workspace()
    try:
        async def run():
            events: list = []

            async def emit(ev):
                events.append(ev)

            qm = QuestionManager()
            tools = _make_tools_with_questions(ws, qm, emit)
            target = Path(ws, "obsolete.txt")
            target.write_text("x", encoding="utf-8")

            task = asyncio.create_task(tools["delete_file"].ainvoke({"path": "obsolete.txt"}))
            ev = await _wait_question(events)
            assert "确认删除" in ev["options"]
            qm.answer(ev["question_id"], ["取消"])
            assert "取消" in await asyncio.wait_for(task, timeout=5)
            assert target.exists()

            events.clear()
            task = asyncio.create_task(tools["delete_file"].ainvoke({"path": "obsolete.txt"}))
            ev = await _wait_question(events)
            qm.answer(ev["question_id"], ["确认删除"])
            assert "已删除" in await asyncio.wait_for(task, timeout=5)
            assert not target.exists()

        asyncio.run(run())
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_port_info_is_read_only():
    ws = _workspace()
    try:
        tools = _make_tools(ws)
        out = _invoke(tools, "port_info", port=65001)
        assert "65001" in out
        assert "只读" in out or "未被监听" in out
        assert "1-65535" in _invoke(tools, "port_info", port=0)
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_read_file_offset_limit_and_out_of_range():
    ws = _workspace()
    try:
        Path(ws, "a.txt").write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
        tools = _make_tools(ws)

        out = _invoke(tools, "read_file", path="a.txt", offset=3, limit=2)
        assert "总行数: 10" in out and "显示 3-4" in out
        assert "L3" in out and "L4" in out and "L5" not in out

        far = _invoke(tools, "read_file", path="a.txt", offset=99)
        assert "超出文件末尾" in far and "总行数: 10" in far
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_read_file_large_file_range_reads_without_full_load():
    """超过 8MB 的文件用 offset/limit 可分段读取，不再整文件载入；整文件读取仍明确拒绝。"""
    ws = _workspace()
    try:
        total = 85000
        line = "x" * 100
        with open(Path(ws, "huge.log"), "w", encoding="utf-8") as f:
            for _ in range(total):
                f.write(line + "\n")
        tools = _make_tools(ws)

        out = _invoke(tools, "read_file", path="huge.log", offset=total, limit=2)
        assert f"总行数: {total}" in out
        assert f"{total} |" in out
        assert "超出文件末尾" not in out

        full = _invoke(tools, "read_file", path="huge.log")
        assert "文件过大" in full and "offset/limit" in full
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def test_read_text_range_handles_gbk_and_utf16():
    from app.services.file_io import read_text_range

    ws = _workspace()
    try:
        gbk = Path(ws, "gbk.txt")
        gbk.write_bytes("第一行\n第二行\n第三行\n".encode("gbk"))
        rng = read_text_range(gbk, 2, 1)
        assert rng.lines == ["第二行"]
        assert rng.total_lines == 3
        assert rng.encoding.lower() == "gbk"

        u16 = Path(ws, "u16.txt")
        u16.write_bytes("甲\n乙\n".encode("utf-16"))
        rng16 = read_text_range(u16, 1, 0)
        assert rng16.lines == ["甲", "乙"]
        assert rng16.total_lines == 2

        out = _invoke(_make_tools(ws), "read_file", path="u16.txt")
        assert "甲" in out and "乙" in out and "编码: utf-16-le" in out
    finally:
        shutil.rmtree(ws, ignore_errors=True)
