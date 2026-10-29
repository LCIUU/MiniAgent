import { useEffect } from "react";
import { ChatView } from "./components/ChatView";
import { BackgroundTasks } from "./components/BackgroundTasks";
import { InputBox } from "./components/InputBox";
import { SettingsModal } from "./components/SettingsModal";
import { Sidebar } from "./components/Sidebar";
import { WorkspaceModal } from "./components/WorkspaceModal";
import { setState } from "./store";
import { initApp, useAgent, useStore } from "./useAgent";

export default function App() {
  const {
    settingsOpen,
    workspaceOpen,
    running,
    connecting,
    workspace,
  } = useStore();
  const agent = useAgent();

  useEffect(() => {
    initApp();
  }, []);

  return (
    <div className="app">
      <Sidebar
        onSelect={(id) => void agent.loadSession(id)}
        onNew={(ws) => void agent.newSession(ws)}
        onNewCurrent={() => {
          if (!workspace) {
            setState({ workspaceOpen: true });
            return;
          }
          void agent.newSession();
        }}
        onSelectWorkspace={(key) => void agent.selectWorkspace(key)}
        onDeleteWorkspace={(key) => void agent.deleteWorkspace(key)}
      />

      <div className="main">
        <div className="topbar">
          <div className="session-title">
            <span className="session-label">工作区</span>
            <span className="session-sep" aria-hidden="true">|</span>
            <span className="session-path">{workspace || "未选择"}</span>
          </div>
          <BackgroundTasks />
        </div>

        <ChatView
          onApproval={(id, approve) => agent.decideApproval(id, approve)}
          onAnswer={(id, answers) => agent.answerQuestion(id, answers)}
        />

        <InputBox
          disabled={running}
          running={running}
          onStop={agent.stopRun}
          onSend={(text, attachments, effort) => void agent.sendMessage(text, attachments, effort)}
          placeholder={
            connecting
              ? "正在连接…"
              : workspace
                ? "描述你想实现的功能…"
                : "请先在左侧选择一个工作区"
          }
        />
      </div>

      {settingsOpen && <SettingsModal onClose={() => setState({ settingsOpen: false })} />}
      {workspaceOpen && <WorkspaceModal onClose={() => setState({ workspaceOpen: false })} />}
    </div>
  );
}
