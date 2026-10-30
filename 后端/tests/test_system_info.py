"""system_info 工具测试：分节解析、输出格式、脱敏、软件过滤与注册。"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools import system_info as si


async def _emit(_event: dict) -> None:
    return None


def _make_tools(ws: str) -> dict:
    from app.tools import build_tools
    from app.tools.context import ApprovalManager

    tools = build_tools(ws, _emit, ApprovalManager(), asyncio.Event())
    return {t.name: t for t in tools}


def _invoke(tools: dict, tool_name: str, **kwargs) -> str:
    return asyncio.run(tools[tool_name].ainvoke(kwargs))


def test_system_info_registered():
    tools = _make_tools(str(Path.cwd()))
    assert "system_info" in tools


def test_parse_sections_aliases_and_all():
    assert si._parse_sections("") == (["overview"], [])
    names, unknown = si._parse_sections("net, hw, cpu")
    assert unknown == []
    assert names[0] == "network"
    assert "cpu" in names and "memory" in names

    names, unknown = si._parse_sections("all")
    assert names == si._SECTION_ORDER and unknown == []

    names, unknown = si._parse_sections("overview, overview, bogus")
    assert names == ["overview"] and unknown == ["bogus"]


def test_overview_has_sections_and_key_fields():
    tools = _make_tools(str(Path.cwd()))
    text = _invoke(tools, "system_info", sections="overview,memory")
    assert "【概览】" in text
    assert "【内存】" in text
    assert "计算机" in text and "Python" in text
    assert len(text) < si._MAX_TOTAL_CHARS


def test_unknown_section_still_returns_overview():
    tools = _make_tools(str(Path.cwd()))
    text = _invoke(tools, "system_info", sections="bogus")
    assert "【概览】" in text
    assert "已忽略未知节" in text


def test_env_section_masks_sensitive_values(monkeypatch):
    monkeypatch.setenv("AAA_SYSTEM_INFO_TOKEN", "super-secret-value-123456")
    monkeypatch.setenv("PWD", "/tmp/project")
    text = asyncio.run(si._section_env("", 30))
    assert "AAA_SYSTEM_INFO_TOKEN" in text
    assert "super-secret-value-123456" not in text
    # PWD 只是当前目录，不应被当作敏感变量列入脱敏清单
    assert "PWD:" not in text


def test_software_keyword_filter_and_dedupe(monkeypatch):
    async def fake_items():
        return [
            ("Python 3.13.4", "3.13.4", "PSF"),
            ("Node.js", "24.1.0", "OpenJS"),
            ("Notepad++", "8.6", "Don Ho"),
        ]

    monkeypatch.setattr(si, "_software_windows", fake_items)
    monkeypatch.setattr(si, "_software_posix", fake_items)
    monkeypatch.setattr(si, "_is_windows", lambda: True)

    text = asyncio.run(si._section_software("python", 10))
    assert "共 1 个匹配" in text
    assert "Python 3.13.4（PSF）" in text
    assert "Node.js" not in text

    missing = asyncio.run(si._section_software("zzz-not-installed", 10))
    assert "未找到名称包含" in missing

    limited = asyncio.run(si._section_software("", 2))
    assert "其余 1 个未显示" in limited


def test_packages_section_reads_manifests(monkeypatch):
    root = Path(tempfile.mkdtemp(prefix="agent-sysinfo-"))

    def write(rel: str, content: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    write(
        "web/package.json",
        json.dumps({"dependencies": {"react": "^19.2.0"}, "devDependencies": {"vite": "^8.2.2"}}),
    )
    write("web/node_modules/vite/package.json", json.dumps({"name": "vite", "version": "8.2.2"}))
    write("web/pnpm-lock.yaml", "")
    write(
        "api/pyproject.toml",
        '[project]\nname = "demo"\ndependencies = ["fastapi>=0.115", "uvicorn>=0.30"]\n',
    )
    write("api/uv.lock", "")

    async def fake_global():
        return "C:/npm-global", ["typescript: 5.9.2", "vite: 8.2.2"]

    monkeypatch.setattr(si, "_global_npm_packages", fake_global)

    text = asyncio.run(si._section_packages("", 10, str(root)))
    assert "【项目与全局依赖】" in text
    assert "全局 npm 包（C:/npm-global）" in text and "typescript: 5.9.2" in text
    assert "web/package.json（Node 项目，pnpm；共 2 个依赖，已安装 1 个）" in text
    assert "vite: 声明 ^8.2.2，已安装 8.2.2" in text
    assert "react: 声明 ^19.2.0（未安装）" in text
    assert "api/pyproject.toml（Python 项目，uv；共 2 项）" in text
    assert "fastapi>=0.115" in text
    assert "node_modules/vite/package.json" not in text

    filtered = asyncio.run(si._section_packages("vite", 10, str(root)))
    assert "vite: 声明 ^8.2.2" in filtered
    assert "react:" not in filtered and "fastapi" not in filtered
    assert "typescript" not in filtered


def test_packages_section_without_manifests(monkeypatch):
    root = Path(tempfile.mkdtemp(prefix="agent-empty-"))

    async def fake_global():
        return "", []

    monkeypatch.setattr(si, "_global_npm_packages", fake_global)
    text = asyncio.run(si._section_packages("", 10, str(root)))
    assert "未发现 package.json" in text


def test_join_sections_truncates_oversized_output():
    big = "x" * (si._MAX_SECTION_CHARS + 100)
    joined = si._join_sections([big] * 4)
    assert "本节过长，已截断" in joined
    assert "总输出上限未显示" in joined
    assert len(joined) < si._MAX_TOTAL_CHARS + 500


def test_runtime_output_filters_error_text(monkeypatch):
    async def fake_run(_cmd, timeout=8.0):
        return "No module named pip"

    monkeypatch.setattr(si.shutil, "which", lambda _name: "C:/fake/python.exe")
    monkeypatch.setattr(si, "_run", fake_run)
    assert asyncio.run(si._runtime_versions()) == []
