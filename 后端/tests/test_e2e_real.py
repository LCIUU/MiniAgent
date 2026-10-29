"""真实端到端测试：真实 LLM + 真实工具（需要已配置 API Key）。"""
import asyncio
import json
import tempfile

import httpx


async def main():
    ws = tempfile.mkdtemp(prefix="agent_e2e_")
    print("workspace:", ws)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=300) as c:
        r = await c.post(
            "/api/agent/chat",
            json={"message": "用 write_file 工具在当前工作区创建一个文件 hello.txt，内容为 'hello from agent'，然后读取它确认内容", "workspace": ws},
        )
        print("chat status:", r.status_code)
        run = r.json()
        print("run:", run)

        events = []
        async with c.stream("GET", f"/api/agent/runs/{run['run_id']}/stream") as resp:
            print("sse status:", resp.status_code)
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    events.append(ev)
                    if ev["type"] == "token":
                        continue
                    print("SSE:", ev["type"], str(ev)[:110])

        # 断言关键事件
        types = [e["type"] for e in events]
        print("\n== 结果 ==")
        print("token:", "token" in types)
        print("plan:", "plan" in types)
        print("tool_start:", "tool_start" in types)
        print("tool_end:", "tool_end" in types)
        print("error:", "error" in types)

        # 检查文件是否创建成功
        import os

        fpath = os.path.join(ws, "hello.txt")
        print("hello.txt exists:", os.path.exists(fpath))
        if os.path.exists(fpath):
            print("content:", open(fpath, encoding="utf-8").read()[:50])

        # 检查消息持久化
        msgs = await c.get(f"/api/sessions/{run['session_id']}/messages")
        data = msgs.json()
        print("persisted messages:", len(data))
        for m in data:
            print(" ", m["role"], ":", str(m["content"])[:60])
        print("\nE2E TEST FINISHED")


asyncio.run(main())
