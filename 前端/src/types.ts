// ===== 全局类型定义 =====

export interface Session {
  id: string;
  title: string;
  workspace: string;
  provider: string;
  model: string;
  created_at: string;
  updated_at: string;
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  result?: string;
  status: string; // running | success | error | rejected | cancelled
  duration_ms?: number;
  started_at?: string; // 调用时刻 HH:MM:SS
  call_id?: string;
  diff?: string; // 文件修改的 unified diff
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: AttachmentMeta[];
  tool_calls: ToolCall[];
  timeline?: TimelineItem[];
  token_usage?: { input_tokens?: number; output_tokens?: number; llm_calls?: number } | null;
  duration_ms?: number; // 本轮运行总耗时（毫秒），刷新后仍可展示
  created_at: string;
}

// SSE 事件类型
export type AgentEvent =
  | { type: "status"; status: string; detail?: string }
  | { type: "thinking"; delta: string }
  | { type: "token"; delta: string }
  | { type: "plan"; steps: string[]; todos?: { content: string; status: string }[] }
  | { type: "tool_start"; tool: string; args: Record<string, unknown>; call_id: string; started_at?: string }
  | { type: "tool_output"; call_id: string; stream?: string; delta: string }
  | { type: "diff"; call_id?: string; path: string; diff: string; added?: number; removed?: number }
  | { type: "tool_end"; tool: string; call_id: string; result: string; status: string; duration_ms?: number; exit_code?: number; started_at?: string }
  | { type: "approval"; approval_id: string; tool: string; reason: string; command?: string; cwd?: string }
  | { type: "approval_done"; approval_id: string; approved?: boolean }
  | { type: "question"; question_id: string; tool: string; question: string; options?: string[]; allow_multiple?: boolean }
  | { type: "question_done"; question_id: string; answers?: string[] }
  | { type: "summary"; summary: string }
  | { type: "summary_start"; before_tokens?: number; messages_before?: number }
  | { type: "summary_done"; ok: boolean; before_tokens?: number; after_tokens?: number; messages_before?: number; messages_after?: number; duration_ms?: number; summary?: string }
  | { type: "usage"; input_tokens?: number; output_tokens?: number; llm_calls?: number }
  | { type: "message_done"; message_id: string }
  | { type: "run_start"; run_id: string; started_at?: number }
  | { type: "run_done"; session_id: string; message_id?: string; ended_at?: number; duration_ms?: number; token_usage?: Record<string, number> }
  | { type: "error"; message: string; fatal?: boolean }
  | { type: "mcp_loaded"; detail: string };

export interface ModelCatalogItem {
  name: string;
  label: string;
  base_url: string;
  configured: boolean;        // 是否已有可用 API Key（专属 Key / env / 全局回退）
  supports_thinking: boolean; // 是否具备思考能力（决定推理档位可用性）
  supports_vision: boolean;   // 是否支持图片输入
  key_env: string;            // 未配置时提示用的环境变量名
}

export interface Settings {
  base_url: string;
  model: string;
  has_api_key: boolean;
  current_model_configured: boolean;
  has_tavily_key: boolean;
  command_confirm_mode: "auto" | "high_risk" | "all";
  workspace: string;
  model_options: string[];
  model_catalog: ModelCatalogItem[];
  reasoning_effort: string | null; // 推理等级（null=跟随模型默认，持久化）
}

export interface SettingsUpdate {
  base_url?: string;
  model?: string;
  api_key?: string;
  model_api_key?: string; // 当前所选模型的专属 API Key（与 model 搭配保存）
  model_base_url?: string; // 当前所选模型的专属端点
  clear_model_api_key?: boolean;
  tavily_api_key?: string;
  command_confirm_mode?: "auto" | "high_risk" | "all";
  workspace?: string;
  reasoning_effort?: string | null;
}

export interface WorkspaceEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
}

export interface WorkspaceList {
  path: string;
  parent: string | null;
  entries: WorkspaceEntry[];
}

export interface RunInfo {
  run_id: string;
  session_id: string;
  status: string;
}

// 附件（上传接口返回的元信息；发送消息时原样回传）
export interface AttachmentMeta {
  name: string;
  path: string;
  size: number;
  kind: string; // pdf | word | text | image
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  workspace: string;
  attachments?: AttachmentMeta[];
  /** 思考强度档位：off | low | medium | high | xhigh，不传=跟随模型默认 */
  reasoning_effort?: string | null;
}

// 时间轴条目：执行过程流的数据单元（思考/工具/计划/阶段正文按时间交错）
export type TimelineItem =
  | { kind: "thinking"; id: string; start: number; text: string; duration_ms?: number } // 模型推理（连续增量合并，阶段切换新开；结束后补时长）
  | { kind: "text"; id: string; start: number; text: string } // 模型输出的阶段说明正文（工具轮之间的文本）
  | {
      kind: "tool";
      id: string;
      call_id: string;
      name: string;
      args: Record<string, unknown>;
      status: string; // running | success | error | rejected | cancelled
      result?: string;
      diff?: string; // 文件修改的 unified diff
      duration_ms?: number;
      start: number;
      started_at?: string; // HH:MM:SS
    }
  | { kind: "plan"; id: string; start: number; steps: string[]; todos?: { content: string; status: string }[] } // 执行计划（更新时覆盖）
  | { kind: "compress"; id: string; start: number; status: "running" | "done"; failed?: boolean; duration_ms?: number; before_tokens?: number; after_tokens?: number; summary?: string }; // 上下文压缩（触发摘要时展示）

// 运行中的流式消息（前端状态）
export interface StreamMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: AttachmentMeta[]; // 用户消息附件（图片渲染缩略图）
  tool_calls: ToolCall[];
  status: "streaming" | "done" | "error";
  session_id: string;
  pendingApproval?: ApprovalRequest | null;
  pendingQuestion?: QuestionRequest | null;
  reasoning?: string; // 模型思考过程（纯文本累计，兼容字段）
  plan?: string[]; // 模型给出的执行计划
  timeline?: TimelineItem[]; // 思考过程时间轴（按事件到达顺序）
  duration_ms?: number; // 本轮运行总耗时（毫秒），来自历史消息持久化
}

export interface ApprovalRequest {
  approval_id: string;
  tool: string;
  reason: string;
  command?: string;
  cwd?: string;
}

export interface QuestionRequest {
  question_id: string;
  tool: string;
  question: string;
  options: string[];
  allow_multiple: boolean;
}
