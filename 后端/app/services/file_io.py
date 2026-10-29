"""文本文件读写：编码/换行/BOM 保持，避免 Windows 上把 LF 悄悄改成 CRLF。

- 读取时识别 UTF-8 / UTF-8 BOM / UTF-16 / GBK，统一内部为 ``\\n``
- 写入时按原文件的编码、换行符、BOM 还原
- 通过临时文件 + ``os.replace`` 原子写入，降低中断导致文件损坏的风险
"""

from __future__ import annotations

import codecs
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_TEXT_BYTES = 8 * 1024 * 1024  # 单文件读取上限，避免一次读爆内存


class BinaryFileError(ValueError):
    """看起来是二进制文件，拒绝按文本读取。"""


@dataclass(frozen=True)
class TextDocument:
    text: str
    encoding: str
    newline: str
    bom: bytes


@dataclass(frozen=True)
class TextRange:
    lines: list[str]
    total_lines: int
    encoding: str


def _detect_bom(data: bytes) -> tuple[str, bytes]:
    for bom, enc in (
        (codecs.BOM_UTF8, "utf-8"),
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if data.startswith(bom):
            return enc, bom
    return "", b""


def _decode(data: bytes) -> tuple[str, str, bytes]:
    enc, bom = _detect_bom(data)
    if enc:
        try:
            return data[len(bom) :].decode(enc), enc, bom
        except UnicodeDecodeError:
            pass
    for candidate in ("utf-8", "gbk", "cp1252"):
        try:
            return data.decode(candidate), candidate, b""
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8", b""


def _detect_newline(raw: str) -> str:
    crlf = raw.count("\r\n")
    lf = raw.count("\n") - crlf
    cr = raw.count("\r") - crlf
    if crlf == 0 and lf == 0 and cr == 0:
        # 空文件/单行无换行：默认 LF，避免在 Windows 上把新建内容写成 CRLF
        return "\n"
    if crlf >= lf and crlf >= cr:
        return "\r\n"
    if cr > lf:
        return "\r"
    return "\n"


def read_document(path: Path, *, max_bytes: int = MAX_TEXT_BYTES) -> TextDocument:
    """读取文本文件并保留编码/换行信息。"""
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"文件过大（{size} 字节 > {max_bytes}），无法整体读取；"
            "请用 offset/limit 分段读取（大文件分段不会整文件载入），或先用 grep 定位"
        )
    data = path.read_bytes()
    # 带 BOM 的 UTF-16/32 文本每两字节就有一个 NUL，不能按二进制处理
    if not _detect_bom(data)[1] and b"\x00" in data[:4096]:
        raise BinaryFileError(f"二进制文件，无法按文本读取: {path.name}")
    raw, encoding, bom = _decode(data)
    newline = _detect_newline(raw)
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return TextDocument(text=text, encoding=encoding, newline=newline, bom=bom)


def _detect_encoding(prefix: bytes) -> str:
    enc, _bom = _detect_bom(prefix)
    if enc:
        return "utf-8-sig" if enc == "utf-8" else enc
    # 前缀可能在多字节字符中间被截断：最多回退 3 字节再判断
    for candidate in ("utf-8", "gbk"):
        for cut in range(4):
            sample = prefix[: len(prefix) - cut] if cut else prefix
            try:
                sample.decode(candidate)
                return candidate
            except UnicodeDecodeError:
                continue
    return "cp1252"


def read_text_range(path: Path, start: int, limit: int) -> TextRange:
    """按行读取指定范围，避免大文件用 offset/limit 时整文件载入内存。"""
    start = max(1, int(start))
    limit = max(0, int(limit))
    with open(path, "rb") as f:
        prefix = f.read(65536)
    _enc, bom = _detect_bom(prefix)
    if not bom and b"\x00" in prefix[:4096]:
        raise BinaryFileError(f"二进制文件，无法按文本读取: {path.name}")
    encoding = _detect_encoding(prefix)
    lines: list[str] = []
    total = 0
    with open(path, encoding=encoding, errors="replace", newline=None) as f:
        for total, raw in enumerate(f, 1):
            if total < start:
                continue
            if limit and len(lines) >= limit:
                continue  # 仍需继续计数总行数
            text_line = raw.rstrip("\n")
            if total == 1:
                text_line = text_line.lstrip("\ufeff")
            lines.append(text_line)
    return TextRange(lines=lines, total_lines=total, encoding=encoding)


def write_document(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str = "\n",
    bom: bytes = b"",
) -> None:
    """原子写入文本，按指定编码/换行/BOM 还原。"""
    if newline and newline != "\n":
        text = text.replace("\n", newline)
    data = text.encode(encoding)
    if bom:
        data = bom + data
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".agent-write-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            mode = path.stat().st_mode
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_new_text(path: Path, text: str) -> None:
    """新建文件默认 UTF-8 + LF（跨平台仓库友好）。"""
    write_document(path, text, encoding="utf-8", newline="\n")


async def async_read_document(path: Path, *, max_bytes: int = MAX_TEXT_BYTES) -> TextDocument:
    """异步版本的 read_document，避免阻塞事件循环。"""
    import asyncio
    return await asyncio.to_thread(read_document, path, max_bytes=max_bytes)


async def async_read_text_range(path: Path, start: int, limit: int) -> TextRange:
    """异步版本的范围读取。"""
    import asyncio
    return await asyncio.to_thread(read_text_range, path, start, limit)


async def async_write_document(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str = "\n",
    bom: bytes = b"",
) -> None:
    """异步版本的 write_document。"""
    import asyncio
    await asyncio.to_thread(write_document, path, text, encoding=encoding, newline=newline, bom=bom)


async def async_write_new_text(path: Path, text: str) -> None:
    """异步版本的 write_new_text。"""
    import asyncio
    await asyncio.to_thread(write_new_text, path, text)
