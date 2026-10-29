import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAgent, useStore } from "../useAgent";
import type { WorkspaceList } from "../types";

export function WorkspaceModal({ onClose }: { onClose: () => void }) {
  const { workspace } = useStore();
  const { selectWorkspace } = useAgent();
  const [current, setCurrent] = useState(workspace || "");
  const [data, setData] = useState<WorkspaceList | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const load = async (path: string) => {
    setLoading(true);
    setError("");
    try {
      const d = await api.listWorkspace(path);
      setData(d);
      setCurrent(d.path);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load(current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const select = async (path: string) => {
    try {
      await selectWorkspace(path);
      onClose();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const goUp = () => {
    if (data?.parent) void load(data.parent);
  };

  const home = () => {
    void load("");
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>选择工作区</h2>
          <button className="modal-close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="modal-body">
          <div className="ws-path">📁 {current || "加载中…"}</div>
          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            <button className="btn" onClick={goUp} disabled={!data?.parent}>
              ↑ 上级
            </button>
            <button className="btn" onClick={home}>
              🏠 默认
            </button>
            <button
              className="btn primary"
              style={{ marginLeft: "auto" }}
              onClick={() => select(current)}
            >
              选择此目录
            </button>
          </div>
          {error && <div className="error-banner">{error}</div>}
          {loading ? (
            <div style={{ color: "var(--text-40)", padding: 20 }}>加载中…</div>
          ) : (
            <div className="ws-list">
              {data?.entries.map((e) => (
                <div
                  key={e.path}
                  className="ws-entry"
                  onClick={() => (e.is_dir ? void load(e.path) : undefined)}
                >
                  <span className={e.is_dir ? "icon" : "file"}>
                    {e.is_dir ? "📁" : "📄"}
                  </span>
                  <span>{e.name}</span>
                  {!e.is_dir && (
                    <span className="size">
                      {e.size >= 1024 ? `${(e.size / 1024).toFixed(1)}KB` : `${e.size}B`}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn" onClick={onClose}>
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}
