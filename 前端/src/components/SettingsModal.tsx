import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useStore } from "../useAgent";
import { setState } from "../store";
import type { SettingsUpdate } from "../types";

export function SettingsModal({ onClose }: { onClose: () => void }) {
  const { settings } = useStore();
  const catalog = settings?.model_catalog ?? [];
  const [selected, setSelected] = useState(settings?.model || catalog[0]?.name || "");
  const [modelKey, setModelKey] = useState("");
  const [modelBaseUrl, setModelBaseUrl] = useState("");
  const [clearModelKey, setClearModelKey] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [tavilyKey, setTavilyKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [msg, setMsg] = useState("");

  // settings 异步到达时同步 selected（useState 初始值只在挂载时取一次）
  useEffect(() => {
    if (settings?.model && settings.model !== selected) {
      setSelected(settings.model);
    }
  }, [settings?.model]);

  const current = catalog.find((m) => m.name === selected);

  const save = async () => {
    setSaving(true);
    setMsg("");
    const body: SettingsUpdate = { model: selected };
    if (modelKey.trim()) body.model_api_key = modelKey.trim();
    if (modelBaseUrl.trim()) body.model_base_url = modelBaseUrl.trim();
    if (clearModelKey) body.clear_model_api_key = true;
    if (apiKey.trim()) body.api_key = apiKey.trim();
    if (tavilyKey.trim()) body.tavily_api_key = tavilyKey.trim();
    try {
      const s = await api.updateSettings(body);
      setState({ settings: s });
      setMsg("✅ 已保存，模型切换即时生效");
      setTimeout(onClose, 600);
    } catch (err) {
      setMsg(`❌ 保存失败: ${(err as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  const test = async () => {
    setTesting(true);
    setMsg("");
    try {
      const r = await api.testModel(selected);
      setMsg(`✅ 连接成功（${r.latency_ms}ms，模型 ${r.model}）`);
    } catch (err) {
      setMsg(`❌ 连接失败: ${(err as Error).message}`);
    } finally {
      setTesting(false);
    }
  };

  // 专属 Key / Base URL 都绑定到“当前选中的模型”，切换模型时必须清空输入，
  // 否则会把为 A 填的凭证保存到 B 名下。
  const selectModel = (name: string) => {
    if (name === selected) return;
    setSelected(name);
    setModelKey("");
    setModelBaseUrl("");
    setClearModelKey(false);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>设置</h2>
          <button className="modal-close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="modal-body">
          <div className="field">
            <label>模型（切换即时生效，无需重启）</label>
            <div className="provider-list" style={{ flexWrap: "wrap", gap: "12px 20px" }}>
              {catalog.map((m) => (
                <label className="provider-item" key={m.name}>
                  <input
                    type="checkbox"
                    checked={selected === m.name}
                    onChange={() => selectModel(m.name)}
                  />
                  <svg viewBox="0 0 64 64" width="20" height="20">
                    <path
                      d="M 0 16 V 56 A 8 8 90 0 0 8 64 H 56 A 8 8 90 0 0 64 56 V 8 A 8 8 90 0 0 56 0 H 8 A 8 8 90 0 0 0 8 V 16 L 32 48 L 64 16 V 8 A 8 8 90 0 0 56 0 H 8 A 8 8 90 0 0 0 8 V 56 A 8 8 90 0 0 8 64 H 56 A 8 8 90 0 0 64 56 V 16"
                      pathLength="575.0541381835938"
                      className="path"
                    />
                  </svg>
                  <span>
                    {m.label}
                    <span style={{ fontSize: 12, color: "var(--ds-text-2)", marginLeft: 6 }}>
                      {m.name}
                    </span>
                  </span>
                  <span
                    title={m.configured ? "API Key 已配置" : `未配置 API Key（可设置 ${m.key_env}）`}
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background: m.configured ? "var(--green)" : "var(--red)",
                      display: "inline-block",
                      flexShrink: 0,
                    }}
                  />
                </label>
              ))}
            </div>
            {current && (
              <div className="hint" style={{ marginTop: 8 }}>
                端点：{current.base_url}
                {current.supports_thinking ? "　·　支持思考（可调推理档位）" : ""}
              </div>
            )}
            {current && !current.configured && (
              <div className="hint" style={{ marginTop: 4, color: "var(--red)" }}>
                该模型尚未配置 API Key：请在下方填写「模型专属 API Key」，或在后端 .env 中设置{" "}
                {current.key_env}
              </div>
            )}
          </div>

          <div className="field">
            <div className="md-group">
              <input
                className="md-input"
                type="password"
                value={modelKey}
                onChange={(e) => {
                  setModelKey(e.target.value);
                  if (e.target.value.trim()) setClearModelKey(false);
                }}
                placeholder={
                  current?.configured ? "已配置，输入以覆盖" : "仅对当前选择的模型生效"
                }
              />
              <span className="md-highlight" />
              <span className="md-bar" />
              <label>模型专属 API Key{current ? `（${current.name}）` : ""}</label>
            </div>
            <label style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8, fontSize: 12 }}>
              <input type="checkbox" checked={clearModelKey} onChange={(e) => setClearModelKey(e.target.checked)} />
              清除该模型的专属 Key（回退到全局 Key / .env）
            </label>
          </div>

          <div className="field">
            <div className="md-group">
              <input
                className="md-input"
                type="text"
                value={modelBaseUrl}
                onChange={(e) => setModelBaseUrl(e.target.value)}
                placeholder={current?.base_url ? `留空使用 ${current.base_url}` : "https://.../v1"}
              />
              <span className="md-highlight" />
              <span className="md-bar" />
              <label>模型专属 Base URL（自定义兼容端点）</label>
            </div>
          </div>

          <div className="field">
            <div className="md-group">
              <input
                className="md-input"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={settings?.has_api_key ? "已配置，输入以覆盖" : "sk-...（可选，所有模型共用回退）"}
              />
              <span className="md-highlight" />
              <span className="md-bar" />
              <label>全局 API Key（可选回退）</label>
            </div>
          </div>

          <div className="field">
            <div className="md-group">
              <input
                className="md-input"
                type="password"
                value={tavilyKey}
                onChange={(e) => setTavilyKey(e.target.value)}
                placeholder={settings?.has_tavily_key ? "已配置，输入以覆盖" : "tvly-..."}
              />
              <span className="md-highlight" />
              <span className="md-bar" />
              <label>Tavily API Key</label>
            </div>
          </div>

          {msg && (
            <div
              style={{
                fontSize: 13,
                color: msg.startsWith("✅") ? "var(--green)" : "var(--red)",
              }}
            >
              {msg}
            </div>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn" onClick={onClose}>
            取消
          </button>
          <button
            className="btn"
            onClick={test}
            disabled={testing || saving}
          >
            {testing ? "测试中…" : "测试连接"}
          </button>
          <button className="btn primary" onClick={save} disabled={saving || testing}>
            {saving ? "保存中…" : "保存"}
          </button>
        </div>
      </div>
    </div>
  );
}
