"""轻量代码索引：项目大纲 + 符号查找（无外部依赖）。

- Python 用标准库 ``ast`` 精确解析函数/类/方法
- JS/TS/Go/Rust/Java 用保守正则提取函数/类/接口/结构体
- 结果按工作区 + 路径做短 TTL 缓存，避免重复扫描

这不是 tree-sitter/LSP 级别的索引，但足以让 Agent 在大项目里先定位符号，
再决定读哪个文件，显著减少盲目的 grep/全文读取。
"""

from __future__ import annotations

import ast
import os
import re
import time
from dataclasses import dataclass
from fnmatch import fnmatch as _fnmatch
from pathlib import Path

from langchain_core.tools import tool

from ..services.paths import PathOutsideWorkspace, resolve_workspace_path
from .context import ToolContext
from .registry import tool_spec

_IGNORED = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next",
    ".idea", ".vscode", "target", ".tox", ".mypy_cache", ".pytest_cache", ".agent_logs",
    ".agent_uploads", "coverage", ".turbo", ".agent_backups",
}
_SOURCE_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts", ".mts": "ts", ".cts": "ts",
    ".go": "go", ".rs": "rust",
    ".java": "java", ".kt": "kotlin",
}
_MAX_FILE_BYTES = 1_500_000
_CACHE_TTL = 60.0
_CACHE_MAX_SIZE = 1000  # 最大缓存条目数，防止内存无限增长
_cache: dict[str, tuple[float, list[Symbol]]] = {}


@dataclass(frozen=True)
class Symbol:
    path: str
    line: int
    kind: str
    name: str
    detail: str = ""


def _safe_read(path: Path) -> str:
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return ""
        return path.read_text("utf-8", errors="replace")
    except OSError:
        return ""


def _py_symbols(rel: str, text: str) -> list[Symbol]:
    out: list[Symbol] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                args = ", ".join(a.arg for a in child.args.args[:6])
                if child.args.vararg:
                    args += (", " if args else "") + "*" + child.args.vararg.arg
                if child.args.kwarg:
                    args += (", " if args else "") + "**" + child.args.kwarg.arg
                kind = "method" if prefix else ("async function" if isinstance(child, ast.AsyncFunctionDef) else "function")
                out.append(Symbol(rel, child.lineno, kind, name, f"({args})"))
                visit(child, prefix)
            elif isinstance(child, ast.ClassDef):
                bases = ", ".join(getattr(b, "id", getattr(b, "attr", "")) for b in child.bases[:4])
                out.append(Symbol(rel, child.lineno, "class", f"{prefix}{child.name}", f"({bases})" if bases else ""))
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree)
    return out


_JS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
    (re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)"), "type"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function"), "function"),
    (re.compile(r"^\s*(?:public|private|protected|static|async|readonly|\s)*\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{"), "method"),
]

_GO_PATTERNS = [
    (re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)"), "function"),
    (re.compile(r"^\s*type\s+([A-Za-z_][\w]*)\s+(?:struct|interface)"), "type"),
]

_RUST_PATTERNS = [
    (re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][\w]*)"), "function"),
    (re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|union)\s+([A-Za-z_][\w]*)"), "type"),
    (re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:[A-Za-z_][\w:<>]*\s+for\s+)?([A-Za-z_][\w]*)"), "impl"),
]

_JAVA_PATTERNS = [
    (re.compile(r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:class|interface|enum|record)\s+([A-Za-z_][\w]*)"), "type"),
    (re.compile(r"^\s*(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?[A-Za-z_<>\[\], ?]+\s+([A-Za-z_][\w]*)\s*\([^)]*\)\s*\{"), "method"),
]

_PATTERNS = {"js": _JS_PATTERNS, "ts": _JS_PATTERNS, "go": _GO_PATTERNS, "rust": _RUST_PATTERNS, "java": _JAVA_PATTERNS, "kotlin": _JAVA_PATTERNS}
_RESERVED = {
    "if", "for", "while", "switch", "catch", "function", "return", "typeof", "new",
    "do", "else", "try", "finally", "with", "await", "yield", "import", "export",
}


def _regex_symbols(rel: str, text: str, lang: str) -> list[Symbol]:
    out: list[Symbol] = []
    patterns = _PATTERNS.get(lang, [])
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#", "*", "/*")):
            continue
        for pattern, kind in patterns:
            m = pattern.match(line)
            if m and m.group(1) not in _RESERVED:
                out.append(Symbol(rel, lineno, kind, m.group(1), stripped[:160]))
                break
    return out


def _scan(workspace: str, scope: str = "") -> list[Symbol]:
    key = f"{os.path.normcase(os.path.realpath(workspace))}|{scope}"
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    try:
        base = resolve_workspace_path(workspace, scope) if scope else Path(workspace).resolve()
    except (PathOutsideWorkspace, ValueError):
        return []
    symbols: list[Symbol] = []
    if base.is_file():
        files = [base]
    else:
        files = []
        for root, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d not in _IGNORED and not d.startswith(".agent_")]
            for name in names:
                ext = Path(name).suffix.lower()
                if ext in _SOURCE_EXT:
                    files.append(Path(root) / name)
                if len(files) >= 5000:
                    break
            if len(files) >= 5000:
                break

    for file in files:
        lang = _SOURCE_EXT.get(file.suffix.lower())
        if not lang:
            continue
        try:
            rel = os.path.relpath(file, workspace).replace("\\", "/")
        except ValueError:
            rel = str(file)
        text = _safe_read(file)
        if not text:
            continue
        if lang == "python":
            symbols.extend(_py_symbols(rel, text))
        else:
            symbols.extend(_regex_symbols(rel, text, lang))
    _cache[key] = (now, symbols)
    # 缓存超过最大大小时，按时间戳清理最旧的条目
    if len(_cache) > _CACHE_MAX_SIZE:
        oldest_keys = sorted(_cache.keys(), key=lambda k: _cache[k][0])[: len(_cache) - _CACHE_MAX_SIZE]
        for k in oldest_keys:
            _cache.pop(k, None)
    return symbols


def make_code_intel_tools(ctx: ToolContext):

    @tool_spec(category="search", idempotent=True, cacheable=False, timeout=60)
    @tool(parse_docstring=True)
    async def project_outline(path: str = "", max_items: int = 300) -> str:
        """
        生成项目结构大纲：列出各源文件中的类/函数/方法定义

        Args:
            path: 可选，只分析工作区内的某个子目录或文件
            max_items: 最多返回的符号数（默认 300）

        Returns:
            按文件分组的符号清单，格式 file:line kind name(detail)
        """
        ctx.check_stopped()
        symbols = _scan(ctx.workspace, path)
        if not symbols:
            return "没有找到可分析的源代码文件。"
        limit = max(20, min(int(max_items), 1000))
        grouped: dict[str, list[Symbol]] = {}
        for sym in symbols[:limit]:
            grouped.setdefault(sym.path, []).append(sym)
        lines = [f"共 {len(symbols)} 个符号，显示 {min(len(symbols), limit)} 个："]
        for rel, items in grouped.items():
            lines.append(f"\n{rel}")
            for sym in items:
                lines.append(f"  {sym.line:>5}  {sym.kind:<14} {sym.name}{sym.detail}")
        return "\n".join(lines)

    @tool_spec(category="search", idempotent=True, cacheable=False, timeout=60)
    @tool(parse_docstring=True)
    async def find_symbol(name: str, kind: str = "", path: str = "") -> str:
        """
        按名称查找类/函数/方法/接口定义

        Args:
            name: 符号名（支持不区分大小写的子串匹配）
            kind: 可选，过滤符号类型，例如 function / class / method / type / impl
            path: 可选，只查工作区内的某个子目录或文件

        Returns:
            匹配的符号列表，格式 file:line kind name(detail)
        """
        ctx.check_stopped()
        needle = name.strip().lower()
        if not needle:
            return "错误：name 不能为空。"
        symbols = _scan(ctx.workspace, path)
        matches = [
            s for s in symbols
            if needle in s.name.lower() and (not kind or kind.lower() in s.kind.lower())
        ]
        if not matches:
            return f"没有找到符号: {name}"
        lines = [f"找到 {len(matches)} 个匹配（最多显示 100 个）："]
        for sym in matches[:100]:
            lines.append(f"{sym.path}:{sym.line}  {sym.kind}  {sym.name}{sym.detail}")
        return "\n".join(lines)

    @tool_spec(category="search", idempotent=True, cacheable=False, timeout=60)
    @tool(parse_docstring=True)
    async def find_references(
        name: str,
        path: str = "",
        include: str = "",
        max_results: int = 120,
    ) -> str:
        """
        查找某个符号/标识符在源码中的全部引用位置（定义、调用、导入都会列出）

        Args:
            name: 标识符名称（整词匹配，例如 SettingsPanel 或 resolve_workspace_path）
            path: 可选，只查工作区内的某个子目录或文件
            include: 可选，文件类型过滤 glob，例如 "*.tsx"
            max_results: 可选，最多返回的引用条数（默认 120，上限 500）

        Returns:
            引用列表，格式 file:line: 行内容
        """
        ctx.check_stopped()
        needle = name.strip()
        if not needle:
            return "错误：name 不能为空。"
        if not re.fullmatch(r"[A-Za-z_$][\w$.:\-]*", needle):
            return "错误：name 需为标识符（不支持正则、空格等特殊字符）。"
        try:
            base = resolve_workspace_path(ctx.workspace, path) if path else Path(ctx.workspace)
        except (PathOutsideWorkspace, ValueError) as e:
            return f"错误：{e}"
        if not base.exists():
            return f"错误：路径不存在: {path}"

        word = re.compile(rf"(?<![\w$]){re.escape(needle)}(?![\w$])")
        limit = max(1, min(int(max_results), 500))
        files = _source_files(base)
        if include:
            files = [f for f in files if _fnmatch(f.name, include) or _fnmatch(f.as_posix(), include)]

        hits: list[str] = []
        for file in files:
            ctx.check_stopped()
            text = _safe_read(file)
            if not text or needle not in text:
                continue
            rel = _rel_path(ctx.workspace, file)
            for lineno, line in enumerate(text.splitlines(), 1):
                if word.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        if not hits:
            return f"没有找到 `{needle}` 的引用。"
        head = f"找到 {len(hits)}{'+' if len(hits) >= limit else ''} 处引用（{len(files)} 个源文件）："
        return head + "\n" + "\n".join(hits)

    @tool_spec(category="search", idempotent=True, cacheable=False, timeout=60)
    @tool(parse_docstring=True)
    async def find_dependents(
        path: str,
        include: str = "",
        max_results: int = 120,
    ) -> str:
        """
        找出哪些文件通过 import/require 依赖了指定文件（改动前评估影响面）

        Args:
            path: 目标文件（或目录）路径，相对工作区，例如 "src/store.ts"
            include: 可选，只在这些文件中查找，glob 形式，例如 "*.py"
            max_results: 可选，最多返回条数（默认 120，上限 500）

        Returns:
            依赖该文件的源码文件与对应 import 行
        """
        ctx.check_stopped()
        try:
            target = resolve_workspace_path(ctx.workspace, path)
        except (PathOutsideWorkspace, ValueError) as e:
            return f"错误：{e}"
        if not target.exists():
            return f"错误：路径不存在: {path}"
        stem = target.stem
        if stem.lower() in ("index", "main", "__init__"):
            stem = target.parent.name or stem
        if not stem:
            return f"错误：无法从路径推断模块名: {path}"

        module_re = re.compile(rf"(?<![\w-]){re.escape(stem)}(?:[./\\][\w.-]+)?(?![\w-])")
        import_re = re.compile(
            r"""(?x)
            (?: from \s+ ['"]([^'"]+)['"]          # JS: from './x'
              | require \s* \( \s* ['"]([^'"]+)['"] # JS: require('./x')
              | import \s* \( \s* ['"]([^'"]+)['"]  # JS: import('./x')
              | ^\s*(?:from\s+([\w.]+)\s+import    # Python: from x import y
              |       import\s+([\w.]+))            # Python: import x
            )
            """
        )
        limit = max(1, min(int(max_results), 500))
        files = _source_files(Path(ctx.workspace))
        if include:
            files = [f for f in files if _fnmatch(f.name, include) or _fnmatch(f.as_posix(), include)]

        hits: list[str] = []
        for file in files:
            ctx.check_stopped()
            if file == target:
                continue
            text = _safe_read(file)
            if not text or stem not in text:
                continue
            rel = _rel_path(ctx.workspace, file)
            for lineno, line in enumerate(text.splitlines(), 1):
                m = import_re.search(line)
                if not m:
                    continue
                module = next((g for g in m.groups() if g), "")
                if module_re.search(module):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        if not hits:
            return f"没有找到依赖 `{path}` 的文件。"
        return f"找到 {len(hits)} 处依赖（{stem}）：\n" + "\n".join(hits)

    return project_outline, find_symbol, find_references, find_dependents


def _rel_path(workspace: str, p: Path) -> str:
    try:
        return os.path.relpath(str(p), workspace).replace("\\", "/")
    except ValueError:
        return str(p)


def _source_files(base: Path) -> list[Path]:
    """列出 base 下可分析的源码文件（忽略依赖/构建目录）。"""
    if base.is_file():
        return [base] if base.suffix.lower() in _SOURCE_EXT else []
    files: list[Path] = []
    for root, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _IGNORED and not d.startswith(".agent_")]
        for name in names:
            if Path(name).suffix.lower() in _SOURCE_EXT:
                files.append(Path(root) / name)
            if len(files) >= 5000:
                return files
    return files
