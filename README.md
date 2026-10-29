# AI Coding Agent（类 Codex 编程智能体）

一个类似 OpenAI Codex / Claude Code 的 AI 编程助手：你在前端描述需求，Agent 在指定工作区内自主规划、读写代码、执行命令、运行测试。

## 架构

```
┌─────────────────┐   SSE 事件流    ┌──────────────────────────────┐
│  前端 (React)    │ ─────────────▶ │  后端 (FastAPI + LangGraph)    │
│  Codex 风格 UI  │ ◀───────────── │  ├─ /api/agent/chat     发起运行│
│                 │   WebSocket     │  ├─ /api/agent/runs/{id}/stream  SSE
│  会话列表        │   (停止/审批)   │  ├─ /api/agent/ws/{id}   控制通道│
│  流式输出        │                 │  ├─ /api/sessions       会话 CRUD│
│  工具卡片+diff   │                 │  ├─ /api/settings       设置     │
│  工作区选择      │                 │  ├─ /api/workspace      目录浏览  │
└─────────────────┘                 │  └─ /api/runs           运行审计  │
                                               │
                                  PostgreSQL (会话/消息/运行/缓存)
```

## 目录

| 目录 | 说明 |
|------|------|
| `后端/` | FastAPI 异步后端，Python 3.13 + uv 管理依赖 |
| `前端/` | React 19 + Vite + TypeScript，Codex 风格深色 UI |

## 快速开始

### 1. 后端（二选一）

**方式 A：双击启动（Windows）**

```
后端/start.bat
```

**方式 B：命令行启动**

```bash
cd 后端
uv sync                 # 首次需要安装依赖
uv run python main.py   # 或 python main.py
```

`main.py` 启动时会自动完成：

- 读取 `.env` 配置
- 检查 PostgreSQL 数据库，不存在则自动创建 `coding_agent`
- 建表 + 幂等迁移（sessions / messages / tool_cache / runs / schema_migrations）
- 把历史明文 API Key 迁移为加密存储
- 启动后台清理任务（过期工具缓存）
- 打印当前模型配置状态，然后启动服务（http://127.0.0.1:8000，文档在 `/docs`）

`.env` 关键项：

```ini
DATABASE_URL=postgresql+asyncpg://postgres:密码@localhost:5432/coding_agent
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # 百炼 / DeepSeek / OpenAI 端点
MODEL=qwen-max
OPENAI_API_KEY=你的Key
TAVILY_API_KEY=可选，网络搜索用
COMMAND_CONFIRM_MODE=auto      # auto | high_risk | all
```

数据库需要预先存在：`CREATE DATABASE coding_agent;`（表结构启动时自动创建）。

> API Key / 模型 / 服务商也可以完全在前端「设置」页配置（写入 `后端/agent_settings.json`，优先级高于 `.env`）。设置文件中的密钥使用 Fernet 加密，密钥文件为 `后端/.agent_secret_key`（两者都已加入 `.gitignore`）。

### 2. 前端

```bash
cd 前端
pnpm install
pnpm dev
```

打开 http://localhost:5173 —— Vite 已配置代理，`/api` 与 WebSocket 自动转发到后端 8000 端口。

### 3. 使用

1. 点击右上角 **📁 选择工作区**，选择 Agent 的工作目录（浏览目录树）。
2. 点击 **⚙ 设置**，选择模型服务商（阿里百炼 / DeepSeek / OpenAI）、填写 API Key、选择命令确认模式，可点击「测试连接」验证端点。
3. 在底部输入框描述任务，例如：
   - 「查看当前项目结构，并说明主要功能」
   - 「帮我写一个 Python 脚本，批量重命名当前目录下的图片文件」
   - 「运行测试并修复失败用例」
4. Agent 会自行判断是否需要先列计划（多步骤任务会展示任务规划），随后逐步调用工具执行；工具调用以卡片展示（命令输出、文件 diff 均可展开），可以随时点 **■ 停止**。

## 功能清单

### Agent 核心（LangGraph + DeepAgents 思路）

- **ReAct 单循环**：不设独立规划阶段，模型自主决定是否调用 `write_todos` 制定/更新计划，简单问答直接作答
- **LangGraph 状态图**：`execute → context` 两节点，自动上下文压缩，全程异步
- **流式输出**：token 增量 + 工具实时输出 + 文件 diff，通过 SSE 推送
- **停止语义**：停止会取消审批等待、终止前台命令进程树，5 秒未退出则强制取消任务
- **上下文管理**：中英文感知 token 计数，超阈值生成结构化交接摘要；摘要跨 run 恢复
- **项目上下文**：自动注入 `AGENTS.md` / `.agent/rules.md` / `CLAUDE.md` / `.cursorrules` 与 `.agent/memory.md`
- **可靠性**：LLM 原生重试 + 可选降级模型；同一会话运行互斥

### 工具集

| 工具 | 说明 |
|------|------|
| `terminal` | 在工作区内执行 shell 命令，流式输出、高危审批、超时/停止杀进程树 |
| `process` | 查看/列出/停止由 Agent 启动的后台进程 |
| `read_file` / `write_file` / `edit_file` / `list_dir` | 文件读写编辑（编码/BOM/换行保持，diff 事件） |
| `delete_file` / `move_file` / `copy_file` | 删除、移动/重命名、复制（删除/移动需审批） |
| `restore_file` | 查看/恢复修改前自动备份（`.agent_backups`，非 Git 项目的兜底撤销） |
| `apply_patch` | 标准 unified diff 多文件、多 hunk 原子修改 |
| `grep_search` / `list_files` | 代码检索（有 ripgrep 用 rg，否则纯 Python 回退，带超时） |
| `project_outline` / `find_symbol` | 轻量代码索引：Python AST + JS/TS/Go/Rust/Java 正则 |
| `git_status` / `git_diff` / `git_log` / `git_commit` | Git 状态、差异、日志与提交（提交需审批） |
| `project_memory` | 读写工作区 `.agent/memory.md` 长期记忆 |
| `format_code` | 按项目类型调用 ruff / black / prettier / gofmt / cargo fmt |
| `web_search` | Tavily 网络搜索（需配置 TAVILY_API_KEY） |
| `web_fetch` | 抓取网页正文（SSRF 防护、2MB/超时限制） |
| `run_checks` | 自动识别项目类型并运行 test/build/lint |
| MCP 工具 | 读取 `后端/mcp.json`，动态加载 MCP 服务器工具（stdio/sse/websocket/http） |

### 命令与写操作审批（可配置）

- `auto`：全部自动执行
- `high_risk`：高危命令（`rm -rf`、`git push`、`Remove-Item -Recurse` 等）需前端确认
- `all`：所有终端命令与写文件操作需确认
- 确认卡片直接显示在消息流中，允许/拒绝即时生效；停止运行会取消待审批请求

### 多模型

统一走 **OpenAI 兼容协议**（`langchain-openai`）：

- 阿里百炼：`https://dashscope.aliyuncs.com/compatible-mode/v1`（qwen 系列）
- DeepSeek：`https://api.deepseek.com/v1`（deepseek 系列）
- OpenAI / 任意兼容端点：设置页自由填写 Base URL
- 每个模型可保存专属 Key / 专属 Base URL；支持清除 Key 回退到全局

## 安全说明

- 默认只监听 `127.0.0.1`，CORS 默认仅允许本机 Vite
- API Key 加密落盘；日志自动脱敏 `sk-*` / `tvly-*`
- 所有文件工具与 terminal `cwd` 都做工作区边界校验（含符号链接解析）
- 上传接口只允许写入当前工作区或已有会话绑定的工作区
- 设置 `API_TOKEN` 后，所有 `/api` 请求需带 `X-API-Token`（前端设置 `VITE_API_TOKEN`）
- `RATE_LIMIT_PER_MUTATION` 可开启每 IP 每分钟写请求限流

## MCP 配置

复制 `后端/mcp.example.json` 为 `后端/mcp.json`，按需填入 MCP 服务器：

```json
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/你的目录"]
    },
    "远程服务器": {
      "transport": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

`transport` 支持：`stdio` / `sse` / `websocket` / `http`。MCP 工具名会带服务器前缀；与内置工具重名时自动重命名为 `mcp__<原名>`，不会静默覆盖内置工具。

## API 速览

- `POST /api/agent/chat`：发起运行，返回 run_id
- `GET /api/agent/runs/{run_id}/stream`：SSE（支持 `Last-Event-ID` / `since` 断线续传、心跳）
- `POST /api/agent/runs/{run_id}/stop`：停止
- `POST /api/agent/runs/{run_id}/approval`：审批
- `POST /api/agent/upload`：上传附件（multipart，`workspace` 为表单字段）
- `GET /api/runs?session_id=`：运行记录；`GET /api/runs/usage`：token 用量聚合
- `POST /api/settings/test`：测试模型连接
- `GET /api/health` / `GET /api/health/ready`：存活 / 数据库就绪

## 开发说明

- 后端入口 `后端/app/main.py`，Agent 图在 `后端/app/agent/graph.py`，工具在 `后端/app/tools/`
- 前端入口 `前端/src/App.tsx`，事件流处理在 `前端/src/useAgent.ts`
- SSE 事件协议见 `后端/app/agent/events.py`
- 数据库迁移见 `后端/app/services/migrations.py`（幂等 DDL + 版本记录）
- 设置持久化在 `后端/agent_settings.json`（密钥加密，API 仅返回是否已配置）

## 已知限制

- 单用户、无内置账号体系；`API_TOKEN` 只是可选的本地访问口令
- 代码索引是轻量正则/AST，不是 LSP/tree-sitter 级别的精确语义索引
- 长任务中断后消息由事件缓冲重建；极端情况下仍可能丢失部分工具实时输出
- 工具缓存默认 TTL 600s，跨 run 读取仍建议以文件 mtime 为最终依据
- 当前没有 Docker / 多用户配额；生产化需要额外的进程隔离与审计
