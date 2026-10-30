"""网络工具：Tavily 搜索 + 安全的网页正文抓取。

web_fetch 的安全约束：
- 只允许 http/https
- 拒绝 localhost、私有网段、链路本地、保留地址（防 SSRF）
- 手动跟随重定向，每一跳都重新校验
- 限制响应大小（2MB）与超时，避免大文件/慢站点拖垮 Agent
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from typing import ClassVar

import httpx
from langchain_core.tools import tool

from ..config import get_effective
from .context import ToolContext
from .registry import tool_spec

TAVILY_ENDPOINT = "https://api.tavily.com/search"
_MAX_FETCH_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 4


def tavily_configured() -> bool:
    return bool(get_effective("tavily_api_key", ""))


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


async def validate_public_url(url: str) -> None:
    """校验 URL 可安全抓取；不合法抛 ValueError。"""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL 不能为空")
    from urllib.parse import urlparse

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("只支持 http/https URL")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("URL 缺少主机名")
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        raise ValueError("拒绝访问本地地址")
    # 直接是 IP 时先校验
    try:
        ipaddress.ip_address(host)
        if not _is_public_ip(host):
            raise ValueError("拒绝访问私有/保留 IP")
        return
    except ValueError as e:
        if "拒绝访问" in str(e):
            raise
    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM),
            timeout=8,
        )
    except Exception as e:
        raise ValueError(f"域名解析失败: {e}") from e
    for info in infos:
        if not _is_public_ip(info[4][0]):
            raise ValueError(f"域名解析到非公网地址，已拒绝: {info[4][0]}")


class _HTMLTextExtractor(HTMLParser):
    """轻量 HTML 正文提取：跳过 script/style/noscript/svg，保留段落换行。"""

    SKIP: ClassVar[set[str]] = {"script", "style", "noscript", "svg", "template", "head"}
    BLOCK: ClassVar[set[str]] = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._skip_depth == 0 and data:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> tuple[str, str]:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return "", re.sub(r"<[^>]+>", "", html)
    return parser.title.strip(), parser.text()


def _limit() -> int:
    try:
        return max(1000, int(get_effective("tool_output_max_chars", 12000)))
    except Exception:
        return 12000


async def _fetch(url: str) -> tuple[str, str, str, str]:
    """返回 (final_url, content_type, title, text)。"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 AgentFetcher/1.0"
        ),
        "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8",
    }
    current = url
    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=False) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            await validate_public_url(current)
            async with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location", "")
                    if not location:
                        raise ValueError(f"重定向缺少 Location（HTTP {resp.status_code}）")
                    current = str(httpx.URL(current).join(location))
                    continue
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                data = bytearray()
                async for chunk in resp.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > _MAX_FETCH_BYTES:
                        raise ValueError(f"响应超过 {_MAX_FETCH_BYTES // 1024 // 1024}MB 限制")
                encoding = resp.encoding or "utf-8"
                text = bytes(data).decode(encoding, errors="replace")
                if "html" in content_type or text.lstrip().startswith("<"):
                    title, body = html_to_text(text)
                    return str(resp.url), content_type or "text/html", title, body
                if "json" in content_type:
                    try:
                        return str(resp.url), content_type, "", json.dumps(json.loads(text), ensure_ascii=False, indent=2)
                    except ValueError:
                        pass
                return str(resp.url), content_type or "text/plain", "", text
        raise ValueError("重定向次数过多")


def make_web_search_tool(ctx: ToolContext):

    @tool_spec(category="web", idempotent=True, timeout=45)
    @tool(parse_docstring=True)
    async def web_search(
        query: str,
        max_results: int = 5,
    ) -> str:
        """
        搜索互联网获取最新信息（Tavily API）

        适用：查询最新的技术文档、库版本、API 用法、报错解决方案、新闻等模型知识之外的
        实时信息。搜索完成后通常用 web_fetch 抓取具体页面的完整正文。

        需要配置 Tavily API Key（设置页或 TAVILY_API_KEY）；未配置时返回明确提示。
        返回结果包含标题、URL 和内容摘要；max_results 最多 10 条。不要用它搜索本地代码，
        本地内容用 grep_search / find_symbol。

        Args:
            query: 搜索关键词或问题描述，例如 "FastAPI SSE 实现"。
            max_results: 返回结果条数，默认 5，最多 10。

        Returns:
            编号的搜索结果列表（标题、URL、摘要）；无结果或未配置 Key 时返回提示。
        """
        ctx.check_stopped()
        api_key = get_effective("tavily_api_key", "")
        if not api_key:
            return "网络搜索未启用：未配置 Tavily API Key。请在设置页填写 TAVILY_API_KEY。"

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(
                    TAVILY_ENDPOINT,
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": min(max_results, 10),
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                return f"Tavily 请求失败 (HTTP {e.response.status_code}): {e.response.text[:300]}"
            except Exception as e:
                return f"网络搜索失败: {e}"

        results = data.get("results", [])
        if not results:
            return f"「{query}」没有搜索结果。"
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            lines.append(f"[{i}] {title}\n    URL: {url}\n    {content[:400]}")
        return "搜索结果:\n" + "\n\n".join(lines)

    @tool_spec(category="web", idempotent=True, timeout=45)
    @tool(parse_docstring=True)
    async def web_fetch(url: str, max_chars: int = 0) -> str:
        """
        抓取公网网页并提取正文（用于阅读文档、issue、报错页面）

        适用：web_search 找到链接后读取完整内容，或直接打开已知的文档/博客链接。
        会自动把 HTML 转为纯文本并保留段落；JSON 响应会格式化输出。

        安全限制：只允许 http/https 公网地址；拒绝 localhost、私有网段、链路本地和保留地址
        （防 SSRF），每一跳重定向都会重新校验；响应上限 2MB、最多 4 次重定向、单次 30 秒超时。
        max_chars 控制返回正文长度，默认使用系统输出上限。不能访问本地服务，本地 API 调试
        请用 http_request。

        Args:
            url: 要抓取的 http/https URL，必须解析到公网地址。
            max_chars: 返回正文的最大字符数，默认 0 表示使用系统默认上限。

        Returns:
            网页标题、最终 URL、内容类型与正文；失败时返回具体原因。
        """
        ctx.check_stopped()
        try:
            final_url, content_type, title, body = await _fetch(url)
        except ValueError as e:
            return f"抓取失败: {e}"
        except httpx.HTTPStatusError as e:
            return f"抓取失败 (HTTP {e.response.status_code}): {url}"
        except Exception as e:
            return f"抓取失败: {e}"
        limit = max(500, int(max_chars)) if max_chars else _limit()
        body = body[:limit]
        head = f"标题: {title or '(无)'}\nURL: {final_url}\n类型: {content_type}\n"
        return head + "\n" + (body or "(未提取到正文)")

    return web_search, web_fetch
