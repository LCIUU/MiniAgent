import { useMemo, useState } from "react";
import { api } from "../api/client";
import { getState, setState, toggleWorkspace } from "../store";
import { refreshSessions, useAgent, useStore } from "../useAgent";
import type { Session } from "../types";
import settingsIconUrl from "../assets/设置图标.svg";
import searchIconUrl from "../assets/搜索图标.svg";
import folderPlusIconUrl from "../assets/新增工作区图标.svg";
import newSessionIconUrl from "../assets/新会话图标.svg";
import trashIconUrl from "../assets/垃圾桶图标.svg";
import sidebarIconUrl from "../assets/侧边栏图标.svg";

/** 把绝对路径压成可读的名称：取最后一层目录名。 */
function wsLabel(path: string): string {
  if (!path) return "未分配";
  const norm = path.replace(/[\\/]+$/, "");
  const parts = norm.split(/[\\/]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : path;
}

const SearchIcon = (
  <img src={searchIconUrl} alt="" width={30} height={30} aria-hidden />
);

const FolderPlusIcon = (
  <img src={folderPlusIconUrl} alt="" width={30} height={30} aria-hidden />
);

const FolderIcon = (
  <svg width="30" height="30" viewBox="0 0 800.000000 800.000000" preserveAspectRatio="xMidYMid meet" aria-hidden>
    <g transform="translate(-43.155319,816.278051) scale(0.098712,-0.098712)" fill="currentColor" stroke="none">
      <path d="M3145 6003 c-168 -22 -298 -60 -397 -118 -229 -134 -329 -295 -358 -578 -21 -206 -33 -830 -26 -1402 9 -863 23 -1058 87 -1194 56 -122 138 -195 265 -239 128 -44 172 -47 704 -46 821 1 2128 22 2195 34 165 30 282 88 391 195 121 119 183 254 239 520 23 109 25 115 39 115 7 0 21 9 31 20 18 20 18 21 -4 44 -19 20 -20 24 -6 36 22 18 55 -13 55 -52 0 -52 50 -28 103 51 15 22 15 26 0 52 -16 27 -16 30 4 51 36 40 44 63 28 84 -13 18 -16 17 -62 -27 -29 -28 -61 -49 -80 -52 -30 -6 -33 -4 -33 17 0 13 8 34 18 47 32 43 69 144 112 306 l33 123 41 10 c115 30 122 47 29 67 -71 16 -80 30 -70 108 7 45 -12 115 -48 185 -24 46 -111 136 -154 161 l-36 20 3 143 c6 216 -15 318 -101 496 -53 111 -169 231 -270 279 -142 68 -191 75 -497 75 -170 0 -321 -6 -395 -14 -74 -9 -223 -15 -390 -15 -260 0 -272 1 -328 23 -69 28 -159 108 -224 199 -112 157 -269 246 -483 272 -79 10 -348 13 -415 4z m474 -168 c159 -37 223 -83 346 -245 154 -204 200 -221 570 -224 127 0 379 8 561 18 271 16 352 17 441 8 173 -18 290 -70 396 -178 58 -59 130 -202 145 -289 18 -99 24 -202 13 -210 -6 -3 -11 2 -11 14 0 11 -5 32 -10 47 -10 25 -12 26 -29 10 -9 -9 -19 -16 -22 -16 -2 0 -4 36 -4 79 0 73 -11 114 -23 83 -13 -40 -96 -113 -142 -127 -20 -6 -12 6 38 59 40 44 62 75 62 91 0 13 5 27 10 30 16 10 12 19 -12 30 -12 6 -26 24 -30 41 l-8 32 -64 -5 c-35 -3 -76 -3 -90 1 -28 7 -36 26 -10 26 8 0 25 5 37 12 12 6 41 10 65 9 23 -1 42 0 42 3 0 18 -89 83 -165 121 -161 80 -143 79 -890 79 l-660 1 -45 22 c-60 30 -88 57 -169 166 -104 138 -199 205 -371 258 -69 22 -92 24 -290 24 -190 -1 -224 -3 -296 -23 -101 -27 -144 -45 -212 -88 -71 -46 -89 -37 -32 15 77 73 192 116 366 140 109 15 406 7 493 -14z m-454 -145 c-3 -5 -17 -10 -30 -10 -59 0 -99 -39 -42 -41 21 0 21 -1 2 -9 -11 -5 -52 -21 -92 -36 -58 -21 -72 -31 -70 -46 3 -28 -95 -77 -185 -92 -40 -7 -76 -11 -80 -9 -13 7 26 58 79 104 70 61 162 109 248 130 78 20 180 25 170 9z m319 -40 c5 -5 6 -16 2 -25 -9 -24 -80 -17 -84 9 -2 10 2 23 10 27 14 10 57 3 72 -11z m1989 -387 c-7 -2 -19 -2 -25 0 -7 3 -2 5 12 5 14 0 19 -2 13 -5z m198 -80 c43 -26 30 -38 -42 -38 -109 1 -152 22 -86 43 49 15 96 14 128 -5z m-2101 -622 l75 -7 -100 -7 c-308 -22 -442 -58 -559 -149 -83 -65 -144 -166 -206 -343 l-21 -60 5 60 c7 82 41 209 72 273 89 183 351 266 734 233z m2283 -101 c114 -12 175 -32 239 -81 65 -48 102 -103 128 -187 28 -90 25 -140 -19 -367 -22 -110 -51 -267 -66 -350 -94 -535 -113 -593 -240 -720 -80 -81 -145 -118 -260 -147 -77 -20 -119 -21 -1070 -30 -1128 -10 -1326 -8 -1435 17 -131 31 -217 83 -286 178 -83 113 -124 284 -104 441 24 186 120 639 177 826 35 119 63 169 127 227 74 68 129 97 241 128 126 34 283 42 1070 55 598 10 745 12 1225 18 102 1 225 -3 273 -8z m-3343 -995 c5 -214 12 -421 15 -460 7 -103 -10 -50 -24 70 -15 147 -25 864 -15 1120 l9 220 2 -280 c1 -154 7 -455 13 -670z" />
      <path d="M3383 4319 c-188 -31 -298 -139 -368 -360 -33 -106 -67 -250 -60 -257 3 -3 5 0 5 7 0 12 24 69 58 141 28 58 100 157 130 178 15 11 50 27 77 36 55 18 297 66 333 66 12 0 22 4 22 9 0 12 -111 25 -202 25 -81 0 -71 11 42 49 150 50 282 69 674 99 12 0 -8 4 -44 8 -91 11 -604 10 -667 -1z" />
      <path d="M5924 4296 c92 -28 121 -53 152 -133 14 -36 23 -49 23 -35 1 26 16 29 30 5 9 -17 10 -17 17 0 10 28 -24 105 -60 132 -38 29 -132 55 -196 54 -44 0 -44 0 34 -23z" />
      <path d="M6013 3755 c-7 -22 -24 -85 -37 -140 -14 -55 -28 -111 -32 -125 -39 -156 -109 -332 -158 -399 -102 -138 -210 -196 -638 -342 -158 -53 -190 -70 -101 -51 26 6 140 22 253 37 204 25 324 51 383 81 49 26 126 104 161 164 18 30 47 96 64 145 56 159 127 529 120 625 l-3 45 -12 -40z" />
      <path d="M2890 2943 c0 -95 84 -203 157 -203 15 0 13 4 -9 23 -35 28 -58 84 -58 142 0 38 -2 44 -13 33 -7 -7 -16 -21 -20 -30 -10 -29 -26 -21 -33 17 -8 44 -24 56 -24 18z" />
    </g>
  </svg>
);

const SidebarIcon = (
  <img src={sidebarIconUrl} alt="" width={30} height={30} aria-hidden />
);

const SettingsIcon = (
  <img src={settingsIconUrl} alt="" width={30} height={30} aria-hidden />
);

interface WsGroup {
  key: string; // workspace 路径（"" 表示未分配）
  label: string;
  sessions: Session[];
}

export function Sidebar({
  onSelect,
  onNew,
  onNewCurrent,
  onSelectWorkspace,
  onDeleteWorkspace,
}: {
  onSelect: (id: string) => void;
  onNew: (workspace: string) => void;
  onNewCurrent: () => void;
  onSelectWorkspace: (key: string) => void;
  onDeleteWorkspace: (key: string) => void;
}) {
  const { sessions, activeSessionId, expandedWorkspaces, workspace, sidebarOpen } = useStore((s) => ({
    sessions: s.sessions,
    activeSessionId: s.activeSessionId,
    expandedWorkspaces: s.expandedWorkspaces,
    workspace: s.workspace,
    sidebarOpen: s.sidebarOpen,
  }));
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const agent = useAgent();

  // 按工作区分组（未分配会话归入 "未分配"；当前工作区即使无会话也常驻显示）
  const groups = useMemo<WsGroup[]>(() => {
    const q = query.trim().toLowerCase();
    const map = new Map<string, Session[]>();
    for (const s of sessions) {
      if (q && !s.title.toLowerCase().includes(q)) continue;
      const key = s.workspace || "";
      const arr = map.get(key) ?? [];
      arr.push(s);
      map.set(key, arr);
    }
    if (workspace && !map.has(workspace)) {
      map.set(workspace, []);
    }
    const out: WsGroup[] = [];
    for (const [key, list] of map.entries()) {
      // 按更新时间倒序
      list.sort((a, b) => (a.updated_at < b.updated_at ? 1 : a.updated_at > b.updated_at ? -1 : 0));
      out.push({ key, label: wsLabel(key), sessions: list });
    }
    // 未分配组放最后，其余按名称排序
    out.sort((a, b) => {
      if (!a.key) return 1;
      if (!b.key) return -1;
      return a.label.localeCompare(b.label, "zh-Hans-CN");
    });
    return out;
  }, [sessions, query, workspace]);

  const toggleSearch = () => {
    setSearchOpen((o) => {
      const next = !o;
      if (!next) setQuery(""); // 关闭时清空过滤
      return next;
    });
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm("删除该会话及其全部消息？")) return;
    try {
      await api.deleteSession(id);
      // 删除的是当前会话：立即重置视图（含运行中的连接），
      // 否则输入框还会向已删除的会话发消息
      if (getState().activeSessionId === id) {
        await agent.newSession();
      }
      refreshSessions();
    } catch (err) {
      alert(`删除失败: ${(err as Error).message}`);
    }
  };

  const handleDeleteWorkspace = (key: string) => {
    const name = wsLabel(key);
    if (!confirm(`确定删除工作区「${name}」？该工作区下的所有会话将被一并删除。`)) return;
    onDeleteWorkspace(key);
  };

  return (
    <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
      <div className="sidebar-body">
      <div className="sidebar-top">
        <div className="sidebar-brand">
          <div className="logo-mark" aria-hidden>
            {'</>'}
          </div>
          <div className="logo-text">
            <span className="logo-title">LCIU</span>
            <span className="logo-sub">CODEX</span>
          </div>
          <button
            className="sidebar-close-btn"
            onClick={() => setState({ sidebarOpen: false })}
            aria-label="收起侧边栏"
          >
            {SidebarIcon}
          </button>
        </div>
        <button className="new-session-btn" onClick={onNewCurrent}>
          <img src={newSessionIconUrl} alt="" width={30} height={30} aria-hidden />
          <span>新会话</span>
        </button>
      </div>

      <div className="ws-section-header">
        <span className="ws-section-title">工作区</span>
        <div className="ws-section-actions">
          <button
            className={`ws-action-btn ${searchOpen ? "active" : ""}`}
            onClick={toggleSearch}
            aria-label="搜索对话"
          >
            {SearchIcon}
          </button>
          <button
            className="ws-action-btn"
            onClick={() => setState({ workspaceOpen: true })}
            aria-label="新增工作区"
          >
            {FolderPlusIcon}
          </button>
        </div>
      </div>

      {searchOpen && (
        <div className="sidebar-search">
          <input
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索对话"
            autoFocus
          />
        </div>
      )}

      <div className="session-list">
        {groups.length === 0 && (
          <div className="sidebar-empty">
            {sessions.length === 0 ? "暂无会话" : "无匹配结果"}
          </div>
        )}
        {groups.map((g) => {
          const expanded = expandedWorkspaces[g.key] ?? true; // 默认展开
          const isActiveWs = g.key === workspace && !!workspace;
          return (
            <div className="ws-group" key={g.key}>
              <div className="ws-group-header">
                <button
                  className={`ws-folder-btn ${expanded ? "expanded" : ""}`}
                  onClick={() => toggleWorkspace(g.key)}
                  aria-label={expanded ? "折叠会话列表" : "展开会话列表"}
                >
                  {FolderIcon}
                </button>
                <div
                  className={`ws-name ${isActiveWs ? "active" : ""}`}
                  onClick={() => onSelectWorkspace(g.key)}
                  title={g.key || g.label}
                >
                  <span className="ws-name-text">{g.label}</span>
                </div>
                {g.key && (
                  <button
                    className="ws-new-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onNew(g.key);
                    }}
                    aria-label={`在「${g.label}」新建会话`}
                  >
                    <img src={newSessionIconUrl} alt="" width={30} height={30} aria-hidden />
                  </button>
                )}
                {g.key && (
                  <button
                    className="ws-del-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteWorkspace(g.key);
                    }}
                    aria-label={`删除工作区「${g.label}」及其全部会话`}
                  >
                    <img src={trashIconUrl} alt="" width={30} height={30} aria-hidden />
                  </button>
                )}
              </div>
              {expanded && (
                <div className="ws-group-sessions">
                  {g.sessions.length === 0 && <div className="ws-group-empty">暂无会话</div>}
                  {g.sessions.map((s) => (
                    <div
                      key={s.id}
                      className={`session-item ${s.id === activeSessionId ? "active" : ""}`}
                      onClick={() => onSelect(s.id)}
                      title={s.workspace}
                    >
                      <span className="title">{s.title}</span>
                      <button className="del" onClick={(e) => handleDelete(e, s.id)} aria-label="删除">
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
        </div>

        <div className="sidebar-footer">
          <button
            className="settings-btn"
            onClick={() => setState({ settingsOpen: true })}
          >
            {SettingsIcon}
            <span>设置</span>
          </button>
        </div>
      </div>

      <div className="sidebar-rail-body">
        <button
          className="rail-logo"
          onClick={() => setState({ sidebarOpen: true })}
          aria-label="展开侧边栏"
        >
          {'</>'}
        </button>
        <div className="rail-actions">
          <button className="rail-btn" onClick={onNewCurrent} aria-label="新会话">
            <img src={newSessionIconUrl} alt="" width={30} height={30} aria-hidden />
          </button>
          <button
            className="rail-btn"
            onClick={() => setState({ workspaceOpen: true })}
            aria-label="新增工作区"
          >
            {FolderPlusIcon}
          </button>
          <button
            className="rail-btn"
            onClick={() => {
              setState({ sidebarOpen: true });
              setSearchOpen(true);
            }}
            aria-label="搜索"
          >
            {SearchIcon}
          </button>
        </div>
        <button
          className="rail-btn rail-settings"
          onClick={() => setState({ settingsOpen: true })}
          aria-label="设置"
        >
          {SettingsIcon}
        </button>
      </div>
    </aside>
  );
}
