"""附件服务：上传保存 + 文本提取（PDF/Word/纯文本），供 Agent 注入上下文。

设计要点：
- 统一保存到工作区下 .agent_uploads/ 目录，保证 Agent 在既有安全边界
  （所有文件操作限定工作区内）内可以通过工具访问原始文件
- 上传时校验扩展名白名单与大小上限
- 发起聊天时按类型提取文本（PDF 用 pypdf；.docx 用标准库解析 OOXML；
  纯文本直接读取；图片不做文本提取），并截断防止撑爆上下文
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

# 允许的扩展名 → 类别
_KIND_BY_EXT = {
    ".pdf": "pdf",
    ".doc": "word",
    ".docx": "word",
    ".txt": "text",
    ".md": "text",
    ".py": "text",
    ".js": "text",
    ".ts": "text",
    ".json": "text",
    ".csv": "text",
    ".log": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 单文件 20MB
UPLOAD_DIRNAME = ".agent_uploads"
# 注入消息的单文件提取文本上限（字符）
MAX_EXTRACT_CHARS = 6000


class AttachmentError(ValueError):
    """附件校验/保存失败。"""


def kind_of(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return _KIND_BY_EXT.get(ext, "")


def uploads_dir(workspace: str) -> Path:
    """工作区附件目录（不存在则创建）。"""
    root = Path(workspace) / UPLOAD_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_upload(workspace: str, filename: str, data: bytes) -> dict:
    """保存一个上传文件，返回元信息 {name, path, size, kind}。

    - 文件名做安全化处理，加时间戳前缀防覆盖
    - 拒绝超限与不在白名单内的文件
    """
    if not filename or not Path(filename).name:
        raise AttachmentError("文件名无效")
    name = Path(filename).name
    kind = kind_of(name)
    if not kind:
        raise AttachmentError(f"不支持的文件类型: {name}")
    if len(data) > MAX_FILE_SIZE:
        raise AttachmentError(f"文件过大（>20MB）: {name}")
    if len(data) == 0:
        raise AttachmentError(f"文件为空: {name}")

    safe_stem = re.sub(r'[\\/:*?"<>|\s]+', "_", Path(name).stem)[:60] or "file"
    stored_name = f"{int(time.time()*1000):x}_{safe_stem}{Path(name).suffix.lower()}"
    dest = uploads_dir(workspace) / stored_name
    dest.write_bytes(data)
    return {"name": name, "path": str(dest), "size": len(data), "kind": kind}


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "（未安装 pypdf，无法提取 PDF 文本）"
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages[:50]:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        return f"（PDF 解析失败: {e}）"


def _docx_text(path: Path) -> str:
    """用标准库解析 .docx（OOXML）：取 word/document.xml，段落转行后去标签。"""
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception as e:
        return f"（Word 解析失败: {e}）"
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return xml


def _plain_text(path: Path) -> str:
    try:
        return path.read_text("utf-8", errors="ignore")
    except Exception as e:
        return f"（文本读取失败: {e}）"


def extract_text(path: str, kind: str) -> str:
    """按类别提取文本；图片与不支持的格式返回占位说明。"""
    p = Path(path)
    if kind == "pdf":
        text = _pdf_text(p)
    elif kind == "word":
        text = _docx_text(p) if p.suffix.lower() == ".docx" else "（旧版 .doc 格式暂不支持文本提取）"
    elif kind == "text":
        text = _plain_text(p)
    else:  # image 等
        text = "（二进制文件，未做文本提取；如需处理请使用工具）"
    text = text.strip()
    if len(text) > MAX_EXTRACT_CHARS:
        text = text[:MAX_EXTRACT_CHARS] + f"\n…（内容过长，已截断，完整文件见 {p}）"
    return text or "（未提取到文本内容）"


def validate_stored_path(workspace: str, path: str) -> bool:
    """校验附件路径确实位于工作区 .agent_uploads 内（防目录穿越）。

    使用 os.path.normcase 做平台感知比较：Windows 上忽略盘符/路径大小写差异
    （realpath 不统一盘符大小写，字符串 startswith 会误判），POSIX 上为 no-op。
    """
    try:
        root = os.path.normcase(os.path.realpath(Path(workspace) / UPLOAD_DIRNAME))
        real = os.path.normcase(os.path.realpath(path))
        return real.startswith(root + os.sep)
    except Exception:
        return False


# 多模态注入：单张图片原始字节上限（base64 后约 ×1.33，防止撑爆上下文/请求体）
MAX_IMAGE_INJECT_BYTES = 8 * 1024 * 1024  # 8 MB

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def image_data_url(_workspace: str, path: str) -> str:
    """读取已保存的图片附件并转为 data URL（OpenAI 多模态 image_url 格式）。

    安全约束：仅允许工作区 .agent_uploads 内、图片扩展名白名单内的文件。

    workspace 参数已弃用：校验时自动从 path 反推实际工作区（.agent_uploads 的父目录），
    解决"粘贴时前端全局工作区与会话工作区不一致"导致的路径校验失败。
    """
    import base64

    p = Path(path)
    # 从 path 反推实际工作区：找到 .agent_uploads 的父目录
    actual_ws = None
    for i, part in enumerate(p.parts):
        if part == UPLOAD_DIRNAME:
            actual_ws = str(Path(*p.parts[:i]))
            break
    if not actual_ws or not validate_stored_path(actual_ws, path):
        raise AttachmentError("图片路径无效或不在工作区附件目录内")

    ext = p.suffix.lower()
    mime = _MIME_BY_EXT.get(ext)
    if not mime:
        raise AttachmentError(f"不支持的图片格式: {p.name}")
    if not p.exists():
        raise AttachmentError(f"图片文件不存在: {p.name}")
    data = p.read_bytes()
    if len(data) > MAX_IMAGE_INJECT_BYTES:
        raise AttachmentError(f"图片超过注入上限（{MAX_IMAGE_INJECT_BYTES // 1024 // 1024}MB）: {p.name}")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def build_attachment_block(attachments: list[dict], workspace: str) -> str:
    """把附件提取结果组装为注入 Agent 请求的文本块。"""
    if not attachments:
        return ""
    lines = ["", "", "【用户上传附件】"]
    for i, att in enumerate(attachments, 1):
        path = att.get("path", "")
        name = att.get("name", Path(path).name)
        kind = att.get("kind") or kind_of(name)
        # 与 image_data_url 一致：从 path 反推实际工作区，避免切换全局工作区后附件失效
        actual_ws = workspace
        p = Path(path)
        for j, part in enumerate(p.parts):
            if part == UPLOAD_DIRNAME:
                actual_ws = str(Path(*p.parts[:j]))
                break
        if not validate_stored_path(actual_ws, path) or not Path(path).exists():
            lines.append(f"{i}. {name}（文件缺失或路径无效，已忽略）")
            continue
        lines.append(f"{i}. {name}（{kind}，已保存至 {path}）")
        lines.append("   提取内容：")
        lines.append("   " + extract_text(path, kind).replace("\n", "\n   "))
    lines.append("【附件结束】请在回答中结合上述附件内容。")
    return "\n".join(lines)
