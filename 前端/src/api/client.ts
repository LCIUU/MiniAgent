// ===== REST API 客户端 =====

import type {
  AttachmentMeta,
  ChatRequest,
  Message,
  RunInfo,
  Session,
  Settings,
  SettingsUpdate,
  WorkspaceList,
} from "../types";

// 开发环境走 Vite 代理（/api → 127.0.0.1:8000），避免 CORS；
// 也可用 VITE_API_BASE 指定后端直连地址。
const BASE = import.meta.env.VITE_API_BASE ?? "";
const TOKEN = import.meta.env.VITE_API_TOKEN ?? "";
const DEFAULT_TIMEOUT_MS = 30_000;  // 默认 30s 超时

async function request<T>(path: string, init?: RequestInit, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(`${BASE}${path}`, {
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(TOKEN ? { "X-API-Token": TOKEN } : {}),
      },
      ...init,
    });
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const body = await resp.json();
        detail = body.detail ?? JSON.stringify(body);
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    return resp.json() as Promise<T>;
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      throw new Error(`请求超时（${timeoutMs / 1000}s）`);
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

export const api = {
  // 会话
  listSessions: () => request<Session[]>("/api/sessions"),
  createSession: (workspace: string) =>
    request<Session>("/api/sessions", { method: "POST", body: JSON.stringify({ title: "新会话", workspace }) }),
  renameSession: (id: string, title: string) =>
    request<Session>(`/api/sessions/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteSession: (id: string) => request<{ ok: boolean }>(`/api/sessions/${id}`, { method: "DELETE" }),
  listMessages: (id: string, limit = 100, offset = 0) =>
    request<Message[]>(`/api/sessions/${id}/messages?limit=${limit}&offset=${offset}`),

  // 设置
  getSettings: () => request<Settings>("/api/settings"),
  updateSettings: (body: SettingsUpdate) =>
    request<Settings>("/api/settings", { method: "POST", body: JSON.stringify(body) }),
  testModel: (model = "") =>
    request<{ ok: boolean; latency_ms: number; model: string; reply: string }>(
      `/api/settings/test${model ? `?model=${encodeURIComponent(model)}` : ""}`,
      { method: "POST", body: JSON.stringify({}) },
    ),

  // 运行记录 / 用量
  listRuns: (sessionId = "", limit = 50) =>
    request<{ runs: Array<Record<string, unknown>> }>(
      `/api/runs?session_id=${encodeURIComponent(sessionId)}&limit=${limit}`,
    ),
  usage: (sessionId = "") =>
    request<{ runs: number; input_tokens: number; output_tokens: number; llm_calls: number; total_duration_ms: number }>(
      `/api/runs/usage?session_id=${encodeURIComponent(sessionId)}`,
    ),

  // 工作区
  listWorkspace: (path: string) =>
    request<WorkspaceList>(`/api/workspace?path=${encodeURIComponent(path)}`),
  selectWorkspace: (path: string) =>
    request<{ ok: boolean; workspace: string }>("/api/workspace/select", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  currentWorkspace: () => request<{ workspace: string }>("/api/workspace/current"),

  // 后台进程
  stopProcess: (pid: number) =>
    request<{ ok: boolean }>("/api/processes/stop", { method: "POST", body: JSON.stringify({ pid }) }),
  aliveProcesses: (pids: string) =>
    request<{ alive: number[] }>(`/api/processes/alive?pids=${encodeURIComponent(pids)}`),

  // Agent
  startChat: (body: ChatRequest) =>
    request<RunInfo>("/api/agent/chat", { method: "POST", body: JSON.stringify(body) }),
  stopRun: (runId: string) => request<{ ok: boolean }>(`/api/agent/runs/${runId}/stop`, { method: "POST" }),
  // 查询会话正在进行中的运行（切换会话/刷新后重连接管用）
  getActiveRun: (sessionId: string) =>
    request<{ run_id: string | null; started_at?: number }>(`/api/agent/sessions/${sessionId}/active_run`),
  decideApproval: (runId: string, approvalId: string, approve: boolean) =>
    request<{ ok: boolean }>(`/api/agent/runs/${runId}/approval`, {
      method: "POST",
      body: JSON.stringify({ approval_id: approvalId, approve }),
    }),
  answerQuestion: (runId: string, questionId: string, answers: string[]) =>
    request<{ ok: boolean }>(`/api/agent/runs/${runId}/answer`, {
      method: "POST",
      body: JSON.stringify({ question_id: questionId, answers }),
    }),

  // 附件上传（multipart，不能用 request 助手的 JSON 头）
  uploadFiles: async (workspace: string, files: File[]): Promise<AttachmentMeta[]> => {
    const fd = new FormData();
    fd.append("workspace", workspace);
    for (const f of files) fd.append("files", f);
    const resp = await fetch(`${BASE}/api/agent/upload`, {
      method: "POST",
      body: fd,
      headers: TOKEN ? { "X-API-Token": TOKEN } : undefined,
    });
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const body = await resp.json();
        detail = body.detail ?? JSON.stringify(body);
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    const data = (await resp.json()) as { files: AttachmentMeta[]; errors?: string[] };
    if (data.errors?.length) throw new Error(data.errors.join("；"));
    return data.files;
  },
};

export { BASE };
