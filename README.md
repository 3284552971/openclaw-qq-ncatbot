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
  - 1080  #群聊QQ号

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
  - 3284552971  #这是煮包的QQ号

remote:
  host: 127.0.0.1
  port: 18789
  token: "YOUR_OPENCLAW_TOKEN"  #煮包是本地部署openclaw所以token给你们看了也没事
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

## 会话重置（上下文清空）

支持通过自然语言命令重置当前会话上下文：

- 命令词：`新建对话`
- 私聊触发：root 用户在私聊直接发送 `新建对话`
- 群聊触发：在群里 `@机器人 新建对话`

行为说明：

- 插件会向 OpenClaw 当前会话发送 `/new`，清空该会话上下文并继续复用同一会话键
- 若 OpenClaw 侧历史被手动删除导致会话不存在，插件会自动重建会话并提示用户
- 若当前还没有可重置的会话，会返回提示而不会报错

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

### 在 `main.py` 中重载插件配置

如果你使用的是当前仓库的启动方式，可在 `main.py` 保留以下重载流程：

```python
def reload():
    global plugins, root_transfer, transfer, task_dict
    plugins = plugins_loader.load_plugins()  # 重新加载自定义插件映射
    with open("commands.json", "r", encoding="utf-8") as f:
        commands = json.load(f)
    root_transfer = commands["root_transfer"]
    transfer = commands["transfer"]
    task_dict = commands["task_dict"]

async def _reload_official_openclaw_plugin():
    loader = getattr(bot, "plugin_loader", None)
    if loader is None:
        return False
    if "openclaw" in loader.plugins:
        return await loader.reload_plugin("openclaw")  # 官方插件热重载
    return (await loader.load_plugin("openclaw")) is not None
```

在群消息处理里监听 root 用户的 `#重载`：

```python
if str(msg.user_id) == root_id and message.startswith("#重载"):
    reload()
    ok = await _reload_official_openclaw_plugin()
    await msg.reply(text="重载成功" if ok else "重载完成，但 openclaw 重载失败", at=msg.user_id)
    return
```

说明：

- `reload()` 负责更新 `commands.json` 与你自定义插件路由。
- `_reload_official_openclaw_plugin()` 负责刷新官方插件系统中的 `openclaw` 实例。
- 这样改完 `openclaw.yaml`、`openclaw_prompt.md` 后，无需重启进程即可生效。
