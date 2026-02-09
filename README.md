# OpenClaw Plugin For NcatBot

`openclaw` 插件用于将 NcatBot 与 OpenClaw 打通，支持两种调用方式：

- `cli`：本机调用 `openclaw agent`（功能最完整，支持桥接工具）
- `remote`：通过远端 OpenClaw `/tools/invoke` 调用（桥接工具不可用）

还支持 `auto` 模式：优先 `cli`，若本机找不到 `openclaw` 命令则自动切换到 `remote`。

## 文件结构

- `plugins/openclaw/openclaw.py`：插件主逻辑
- `plugins/openclaw/openclaw.yaml`：插件配置
- `plugins/openclaw/openclaw_prompt.md`：运行时 Prompt 模板
- `plugins/openclaw/tools/pdf_extract.py`：PDF 提取工具

## 快速配置

配置文件：`plugins/openclaw/openclaw.yaml`

```yaml
# 群聊白名单。为空时所有群可用；非空时仅列表群可触发。
group_pool:
  - 1080141352

# 次数限流（群聊非 root 用户）。0 表示关闭。
rate_limit:
  window_seconds: 0
  max_questions: 0

# token 限流（群聊非 root 用户）。0 表示关闭。
token_limit:
  window_seconds: 0
  max_tokens: 0

# root 用户（私聊直通、群聊模式控制等）。
root_ids:
  - 3284552971

remote:
  host: 127.0.0.1
  port: 18789
  token: "YOUR_OPENCLAW_TOKEN"
  timeout_seconds: 120

runtime:
  invoke_mode: auto
  cli_cmd: openclaw
  cli_timeout_seconds: 180
  cli_stateless: false

bridge:
  enabled: true
  host: 127.0.0.1
  port: 18999
  token: "YOUR_BRIDGE_TOKEN"
  allowed_paths:
    - ./data

context:
  enabled: true
  max_messages_per_channel: 300
  default_recent_limit: 20
  default_recent_max_chars: 4000
  prompt_recent_limit: 12
  prompt_recent_max_chars: 1500

image_tool:
  download_root: data/downloads/images
  download_timeout_seconds: 30
  download_max_mb: 20
  read_allow_base64: false
  read_max_base64_chars: 200000

file_tool:
  download_root: data/downloads/files
  download_timeout_seconds: 60
  download_max_mb: 50
```

## 配置项说明

### `runtime.invoke_mode`

- `auto`：优先 CLI，CLI 不可用时回退 remote
- `cli`：强制本机 `openclaw agent`
- `remote`：强制远程 `/tools/invoke`

### `runtime.cli_stateless`

- `false`：复用会话（推荐）
- `true`：每次请求使用新会话，接近单轮调用

### `bridge`（仅 CLI 模式可用）

- 为 OpenClaw 提供本地工具能力（如发送消息、读文件、下载文件/图片等）
- `allowed_paths` 必须限制在安全目录（推荐 `./data`）

### `context`

- 控制本地上下文缓存上限，以及注入 Prompt 的最近窗口大小

## 触发规则

- 群聊：需要 `@机器人` 才触发 openclaw
- 私聊：仅 `root_ids` 中用户直通 openclaw
- 群内 `#` 命令由主程序自定义命令分发处理

## 模式差异

- `cli` 模式：可用桥接工具，适合强耦合联动
- `remote` 模式：不注入桥接工具，能力更受限但部署简单

## 启动与重载

```bash
python main.py
```

配置修改后可使用群内 `#重载`，会同时重载：

- 你的自定义插件分发
- 官方 `openclaw` 插件实例

## 发布到 GitHub（建议）

如果你要把插件单独发布成仓库，建议把 `plugins/openclaw` 目录作为仓库根目录。

```bash
cd plugins/openclaw
git init
git add .
git commit -m "feat: initial openclaw plugin release"
```

然后在 GitHub 创建空仓库（网页或 `gh`），再执行：

```bash
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
