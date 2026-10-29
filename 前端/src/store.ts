// ===== 轻量全局状态 =====
// 极简的发布订阅 store，避免引入 zustand。

import type { Session, Settings, StreamMessage } from "./types";

export interface AppState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: StreamMessage[];
  settings: Settings | null;
  workspace: string;
  running: boolean;
  runId: string | null;
  connecting: boolean;
  sidebarOpen: boolean;
  settingsOpen: boolean;
  workspaceOpen: boolean;
  currentStream: StreamMessage | null;
  error: string | null;
  expandedWorkspaces: Record<string, boolean>; // 侧栏工作区分组展开状态
  runStartedAt: number | null; // 本次运行开始时间（epoch ms）
  runDurationMs: number | null; // 本次运行总耗时（结束后的最终值）
  hasMoreEarlierMessages: boolean; // 会话是否还有更早的历史消息可加载
}

type Listener = () => void;

const state: AppState = {
  sessions: [],
  activeSessionId: null,
  messages: [],
  settings: null,
  workspace: "",
  running: false,
  runId: null,
  connecting: false,
  sidebarOpen: true,
  settingsOpen: false,
  workspaceOpen: false,
  currentStream: null,
  error: null,
  expandedWorkspaces: {},
  runStartedAt: null,
  runDurationMs: null,
  hasMoreEarlierMessages: false,
};

const listeners = new Set<Listener>();

export function getState(): AppState {
  return state;
}

export function setState(patch: Partial<AppState>) {
  Object.assign(state, patch);
  listeners.forEach((l) => l());
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** 展开/折叠某个工作区分组（默认展开）。 */
export function toggleWorkspace(key: string) {
  const cur = state.expandedWorkspaces[key] ?? true;
  Object.assign(state, { expandedWorkspaces: { ...state.expandedWorkspaces, [key]: !cur } });
  listeners.forEach((l) => l());
}

/** 强制设置某工作区分组的展开状态。 */
export function setWorkspaceExpanded(key: string, expanded: boolean) {
  Object.assign(state, { expandedWorkspaces: { ...state.expandedWorkspaces, [key]: expanded } });
  listeners.forEach((l) => l());
}
