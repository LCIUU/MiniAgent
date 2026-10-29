// ===== SSE 事件流客户端 =====
// 用 fetch + ReadableStream 手动解析 SSE，支持 AbortController 中断。

import type { AgentEvent } from "../types";
import { BASE } from "./client";

const TOKEN = import.meta.env.VITE_API_TOKEN ?? "";

export interface StreamHandle {
  abort: () => void;
  done: Promise<void>;
}

export interface StreamOptions {
  /** 只回放 seq 大于该值的事件（断线重连用） */
  since?: number;
  /** 网络层断开（非主动 abort）时回调；不传则合成一条 error 事件 */
  onDrop?: (err: Error) => void;
}

export function openEventStream(
  runId: string,
  onEvent: (ev: AgentEvent) => void,
  onDone?: () => void,
  options: StreamOptions = {},
): StreamHandle {
  const controller = new AbortController();
  const query = options.since && options.since > 0 ? `?since=${options.since}` : "";

  const done = (async () => {
    try {
      const resp = await fetch(`${BASE}/api/agent/runs/${runId}/stream${query}`, {
        signal: controller.signal,
        headers: {
          Accept: "text/event-stream",
          ...(TOKEN ? { "X-API-Token": TOKEN } : {}),
        },
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`SSE 连接失败: ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done: readerDone, value } = await reader.read();
        if (readerDone) break;
        buffer += decoder.decode(value, { stream: true });

        // 按空行切分 SSE 消息（支持 \n\n、\r\n\r\n、\r\r）
        let idx: number;
        while (
          (idx = buffer.indexOf("\n\n")) !== -1 ||
          (idx = buffer.indexOf("\r\n\r\n")) !== -1 ||
          (idx = buffer.indexOf("\r\r")) !== -1
        ) {
          const raw = buffer.slice(0, idx);
          const delimiter = buffer.slice(idx, idx + 2);
          buffer = buffer.slice(idx + delimiter.length);
          // 按行解析，兼容 \r\n 和 \n
          for (const line of raw.split(/\r?\n/)) {
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (!payload) continue;
            try {
              onEvent(JSON.parse(payload) as AgentEvent);
            } catch {
              /* 忽略坏事件 */
            }
          }
        }
      }
      // 刷新 TextDecoder 内部缓冲（处理末尾不完整的多字节字符）
      const remaining = decoder.decode();
      if (remaining.trim()) {
        for (const line of remaining.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            onEvent(JSON.parse(payload) as AgentEvent);
          } catch {
            /* 忽略坏事件 */
          }
        }
      }
      onDone?.();
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        if (options.onDrop) options.onDrop(err as Error);
        else onEvent({ type: "error", message: `SSE 断开: ${(err as Error).message}` });
      }
    }
  })();

  return { abort: () => controller.abort(), done };
}
