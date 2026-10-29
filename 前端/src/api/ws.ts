// ===== WebSocket 控制通道 =====
// 用于向后端发送 停止 / 审批决策（后端→前端的流式数据走 SSE）。

export class WsControl {
  private ws: WebSocket | null = null;
  private runId: string | null = null;
  private retryTimer: number | null = null;
  private stopped = false;
  private sendQueue: unknown[] = [];  // CONNECTING 状态下的待发消息

  connect(runId: string) {
    if (this.runId === runId && this.ws?.readyState === WebSocket.OPEN) return;
    this.disconnect();
    this.runId = runId;
    this.stopped = false;
    this.sendQueue = [];
    this.open();
  }

  private open() {
    if (!this.runId || this.stopped) return;
    let url: string;
    if (import.meta.env.VITE_API_BASE) {
      const host = import.meta.env.VITE_API_BASE.replace(/^https?:\/\//, "");
      const proto = location.protocol === "https:" ? "wss" : "ws";
      url = `${proto}://${host}/api/agent/ws/${this.runId}`;
    } else {
      // 走 Vite 代理（ws: 同源 + ws:true）
      const proto = location.protocol === "https:" ? "wss" : "ws";
      url = `${proto}://${location.host}/api/agent/ws/${this.runId}`;
    }
    const token = import.meta.env.VITE_API_TOKEN ?? "";
    const suffix = token ? `?token=${encodeURIComponent(token)}` : "";
    try {
      this.ws = new WebSocket(`${url}${suffix}`);
    } catch {
      return;
    }
    this.ws.onopen = () => {
      // 连接成功后发送队列中的消息
      while (this.sendQueue.length > 0) {
        const data = this.sendQueue.shift();
        if (data && this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify(data));
        }
      }
    };
    this.ws.onclose = () => {
      if (!this.stopped && this.runId) {
        this.retryTimer = window.setTimeout(() => this.open(), 2000);
      }
    };
    this.ws.onerror = () => {
      // onerror 后通常会触发 onclose，由 onclose 处理重连
    };
  }

  sendStop() {
    this.send({ type: "stop" });
  }

  sendApproval(approvalId: string, approve: boolean) {
    this.send({ type: "approval", approval_id: approvalId, approve });
  }

  sendAnswer(questionId: string, answers: string[]) {
    this.send({ type: "answer", question_id: questionId, answers });
  }

  private send(data: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      // 未连接时入队，等 onopen 后统一发送
      this.sendQueue.push(data);
      if (this.ws?.readyState === WebSocket.CONNECTING) return;  // 等待 onopen
      this.open();
    }
  }

  disconnect() {
    this.stopped = true;
    if (this.retryTimer) window.clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.sendQueue = [];
    this.ws?.close();
    this.ws = null;
    this.runId = null;
  }
}

export const wsControl = new WsControl();
