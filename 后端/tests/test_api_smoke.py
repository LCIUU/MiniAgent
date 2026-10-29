"""API 冒烟测试：chat 启动 + SSE 流消费（无 API key 时验证错误分支）。"""
import asyncio
import json

import httpx


async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=60) as c:
        r = await c.post(
            "/api/agent/chat",
            json={"message": "测试消息", "workspace": "C:/Users/LCIU/Desktop/代码智能体"},
        )
        print("chat status:", r.status_code)
        run = r.json()
        print("run:", run)

        async with c.stream("GET", f"/api/agent/runs/{run['run_id']}/stream") as resp:
            print("sse status:", resp.status_code)
            count = 0
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    print("SSE:", ev["type"], str(ev)[:120])
                    count += 1
                    if ev["type"] in ("error", "run_done"):
                        break
            print("total events:", count)

        # WS 控制通道测试
        import websockets

        try:
            async with websockets.connect(f"ws://127.0.0.1:8000/api/agent/ws/{run['run_id']}") as ws:
                await ws.send(json.dumps({"type": "stop"}))
                reply = await ws.recv()
                print("WS stop reply:", reply)
        except Exception as e:  # noqa: BLE001
            print("WS test (expected if run ended):", str(e)[:80])

        print("API SMOKE TEST DONE")


asyncio.run(main())
