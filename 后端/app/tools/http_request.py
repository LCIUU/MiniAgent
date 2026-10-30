"""HTTP 请求测试工具：发送 HTTP 请求验证 API 端点。

支持 GET / POST / PUT / DELETE / PATCH，可设置 headers、body、query params。
用于本地开发时快速测试 API，无需切换到 Postman 等外部工具。
"""

from __future__ import annotations

import json
import time
from urllib.parse import urlparse

from langchain_core.tools import tool

from ..config import get_effective
from .context import ToolContext
from .registry import RISK_MEDIUM, tool_spec

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2MB 响应上限，防止大响应撑爆内存
_MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)

# 必须拦截的云元数据地址；本机回环地址放行（工具的用途就是本地 API 调试）
_BLOCKED_HOSTS = {
    "169.254.169.254", "100.100.100.200",  # AWS/阿里云元数据
    "fd00:ec2::254",  # AWS IPv6 元数据
}


def _is_blocked_url(url: str) -> str | None:
    """检查 URL 是否指向禁止访问的地址，返回原因或 None。"""
    try:
        parsed = urlparse(url)
    except Exception:
        return "URL 解析失败"
    if parsed.scheme not in ("http", "https"):
        return f"不支持的协议: {parsed.scheme}（仅允许 http/https）"
    hostname = (parsed.hostname or "").lower()
    # 检查元数据地址
    if hostname in _BLOCKED_HOSTS:
        return f"禁止访问内网地址: {hostname}"
    # 检查私有 IP 段
    if hostname.startswith("10.") or hostname.startswith("192.168.") or hostname.startswith("172."):
        # 172.16-31.x.x 是私有段
        if hostname.startswith("172."):
            parts = hostname.split(".")
            if len(parts) >= 2 and parts[1].isdigit() and 16 <= int(parts[1]) <= 31:
                return f"禁止访问内网地址: {hostname}"
        else:
            return f"禁止访问内网地址: {hostname}"
    return None


def _truncate(text: str) -> str:
    limit = int(get_effective("tool_output_max_chars", 12000))
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n...[已截断，共 {len(text)} 字符]...\n" + text[-limit // 2 :]


def make_http_tools(ctx: ToolContext):

    @tool_spec(category="web", risk_level=RISK_MEDIUM, timeout=30, idempotent=False)
    @tool(parse_docstring=True)
    async def http_request(
        url: str,
        method: str = "GET",
        headers: str = "",
        body: str = "",
    ) -> str:
        """
        发送 HTTP 请求测试 API 端点（本地开发调试用）

        适用：验证本地后端接口、检查状态码/响应头/响应体、复现前端请求问题。
        支持 GET/POST/PUT/DELETE/PATCH/HEAD/OPTIONS；headers 用 JSON 字符串传入；
        body 用于 POST/PUT/PATCH，按 UTF-8 编码发送。

        安全限制：仅允许 http/https；拒绝云元数据地址和 10.x/192.168.x/172.16-31.x 私有网段，
        本机回环地址放行（工具用途就是本地 API 调试）；重定向手动跟随且每跳重新校验；
        响应上限 2MB，超时 30 秒。JSON 响应会格式化输出，超出 4000 字符时截断。

        Args:
            url: 请求 URL，例如 http://localhost:8000/api/health；不能为空。
            method: HTTP 方法，默认 GET；支持 GET/POST/PUT/DELETE/PATCH/HEAD/OPTIONS。
            headers: 可选，JSON 对象字符串，例如 '{"Authorization": "Bearer xxx"}'。
            body: 可选，请求体文本；JSON 请传合法 JSON 字符串，GET 请求会忽略。

        Returns:
            状态码、耗时、Content-Type、响应头与响应体；失败时返回原因。
        """
        ctx.check_stopped()
        if not url.strip():
            return "错误：url 不能为空。"

        # SSRF 防护：禁止访问内网/元数据地址
        blocked = _is_blocked_url(url.strip())
        if blocked:
            return f"安全限制：{blocked}"

        method = method.upper().strip()
        if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
            return f"错误：不支持的 HTTP 方法 {method}"

        parsed_headers: dict[str, str] = {"User-Agent": "CodingAgent/1.0"}
        if headers.strip():
            try:
                parsed_headers.update(json.loads(headers))
            except json.JSONDecodeError as e:
                return f"headers JSON 解析失败: {e}"

        try:
            import httpx
        except ImportError:
            return "httpx 未安装，无法发送 HTTP 请求。"

        started = time.time()
        current_url = url.strip()
        current_method = method
        current_body = body
        resp = None
        try:
            # 手动跟随重定向：每一跳都重新做安全校验，
            # 避免首跳公网、二跳 169.254.169.254 的 SSRF 绕过
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=30.0,
                verify=True,  # 启用 SSL 证书验证
            ) as client:
                redirects = 0
                while True:
                    blocked = _is_blocked_url(current_url)
                    if blocked:
                        return f"安全限制：{blocked}"
                    request = client.build_request(
                        current_method,
                        current_url,
                        headers=parsed_headers,
                        content=current_body.encode("utf-8") if current_body else None,
                    )
                    resp = await client.send(request)
                    if resp.status_code not in _REDIRECT_CODES:
                        break
                    location = resp.headers.get("location", "")
                    if not location:
                        break
                    if redirects >= _MAX_REDIRECTS:
                        return f"请求失败：重定向次数过多（>{_MAX_REDIRECTS}）"
                    # 浏览器语义：303 一律转 GET；301/302 的 POST 转 GET 并丢弃请求体
                    if (resp.status_code == 303 and current_method != "HEAD") or (
                        resp.status_code in (301, 302) and current_method == "POST"
                    ):
                        current_method, current_body = "GET", ""
                    current_url = str(resp.url.join(location))
                    redirects += 1
        except httpx.TimeoutException:
            return "请求失败：超时（30s）"
        except Exception as e:
            return f"请求失败: {e}"

        if resp is None:
            return "请求失败：未收到响应"

        # 响应大小限制
        if len(resp.content) > _MAX_RESPONSE_BYTES:
            return f"响应过大（{len(resp.content) / 1024 / 1024:.1f}MB），超过 {_MAX_RESPONSE_BYTES / 1024 / 1024:.0f}MB 限制"

        duration_ms = int((time.time() - started) * 1000)

        resp_headers = dict(resp.headers)
        content_type = resp.headers.get("content-type", "")

        try:
            resp_body = (
                json.dumps(resp.json(), indent=2, ensure_ascii=False)
                if "json" in content_type
                else resp.text
            )
        except Exception:
            resp_body = f"[二进制内容，{len(resp.content)} 字节]"

        parts = [
            f"状态码: {resp.status_code}",
            f"耗时: {duration_ms}ms",
            f"Content-Type: {content_type}",
        ]
        if len(resp_body) > 4000:
            parts.append(f"\n响应头:\n{json.dumps(resp_headers, indent=2, ensure_ascii=False)[:1000]}")
            parts.append(f"\n响应体（已截断）:\n{_truncate(resp_body)}")
        else:
            parts.append(f"\n响应头:\n{json.dumps(resp_headers, indent=2, ensure_ascii=False)[:2000]}")
            parts.append(f"\n响应体:\n{resp_body}")

        return "\n".join(parts)

    return http_request
