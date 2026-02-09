import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import shlex
import shutil
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request
from xml.etree import ElementTree as ET
from ncatbot.plugin_system import NcatBotPlugin, on_message
from ncatbot.core.event import BaseMessageEvent

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = Path(__file__).resolve().parent / "openclaw.yaml"
PROMPT_MD_PATH = Path(__file__).resolve().parent / "openclaw_prompt.md"
STATE_PATH = DATA_DIR / "openclaw_state.json"

OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789").strip()
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "").strip()
OPENCLAW_TIMEOUT = float(os.getenv("OPENCLAW_TIMEOUT", "120"))
OPENCLAW_BRIDGE_HOST = os.getenv("OPENCLAW_BRIDGE_HOST", "127.0.0.1").strip()
OPENCLAW_BRIDGE_PORT = int(os.getenv("OPENCLAW_BRIDGE_PORT", "18999"))
OPENCLAW_BRIDGE_TOKEN = os.getenv("OPENCLAW_BRIDGE_TOKEN", "").strip()
OPENCLAW_DEBUG = os.getenv("OPENCLAW_DEBUG", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OPENCLAW_MAX_RETRIES = max(1, int(os.getenv("OPENCLAW_MAX_RETRIES", "3")))
OPENCLAW_RETRY_DELAY = max(0.0, float(os.getenv("OPENCLAW_RETRY_DELAY", "1.0")))


def _debug_log(message: str):
    if OPENCLAW_DEBUG:
        print(f"[openclaw][debug] {message}")

DEFAULT_CONFIG = {
    "group_pool": [],
    "rate_limit": {"window_seconds": 5 * 60 * 60, "max_questions": 10},
    "token_limit": {"window_seconds": 5 * 60 * 60, "max_tokens": 20000},
    "root_ids": [3284552971],
    "image_watch": {
        "enabled": False,
        "root_dir": str(DATA_DIR / "imagedata"),
        "poll_seconds": 2,
    },
    "remote": {
        "host": "127.0.0.1",
        "port": 18789,
        "token": OPENCLAW_GATEWAY_TOKEN,
        "timeout_seconds": OPENCLAW_TIMEOUT,
    },
    "runtime": {
        # auto: prefer local CLI, fallback to remote mode when CLI is unavailable
        # cli: call `openclaw agent` directly per turn
        # remote: call remote OpenClaw /tools/invoke sessions_send
        "invoke_mode": os.getenv("OPENCLAW_INVOKE_MODE", "auto"),
        "cli_cmd": os.getenv("OPENCLAW_CLI_CMD", "openclaw"),
        "cli_timeout_seconds": OPENCLAW_TIMEOUT,
        # When true under cli mode, always create a fresh session-id per request.
        # This gives strict single-turn behavior and avoids stale-session pollution.
        "cli_stateless": False,
    },
    "bridge": {
        "enabled": True,
        "host": OPENCLAW_BRIDGE_HOST,
        "port": OPENCLAW_BRIDGE_PORT,
        "token": OPENCLAW_BRIDGE_TOKEN,
        "allowed_paths": ["./data"],
    },
    "context": {
        "enabled": True,
        "max_messages_per_channel": 300,
        "default_recent_limit": 20,
        "default_recent_max_chars": 4000,
        "prompt_recent_limit": 12,
        "prompt_recent_max_chars": 1500,
    },
    "image_tool": {
        "download_root": "data/downloads/images",
        "download_timeout_seconds": 30,
        "download_max_mb": 20,
        "read_allow_base64": False,
        "read_max_base64_chars": 200000,
    },
    "file_tool": {
        "download_root": "data/downloads/files",
        "download_timeout_seconds": 60,
        "download_max_mb": 50,
    },
    "delivery": {
        # When False, plugin does not forward model reply to QQ.
        # User-visible messages should be sent by the model itself via bridge APIs.
        "fallback_send_reply": False,
        # When True, JSON-looking reply chunks are dropped from fallback send path.
        "ignore_json_reply": True,
    },
}

PROMPT_TEMPLATE = (
    "【提问人】QQ:{user_id}\n"
    "【群聊】{group_label}\n"
    "【时间】{time}\n"
    "【问题】{question}\n"
    "【桥接工具】\n{bridge_info}"
)

IMAGE_SEND_GUIDE = (
    "图片任务执行规则:\n"
    "1) 当你生成了图片文件路径(例如 .png/.jpg)且目标是把图片发到QQ时，必须调用桥接 /send_image。\n"
    "2) 调用 /send_image 成功后，可按需要继续补充文字说明。\n"
    "3) 若图片尚未生成，先完成生成并确保路径在允许目录内，再调用 /send_image。\n"
)

def _strip_yaml_comment(line):
    if "#" not in line:
        return line
    idx = line.find("#")
    return line[:idx]


def _parse_scalar(value):
    raw = value.strip()
    if raw == "":
        return ""
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        pass
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _load_simple_yaml(text):
    root = {}
    stack = [(-1, root, None)]  # (indent, container, last_key)
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        last_key = stack[-1][2]

        if content.startswith("- "):
            value = _parse_scalar(content[2:])
            if isinstance(container, list):
                container.append(value)
            elif isinstance(container, dict):
                # Handle lists declared under a key (e.g. key: \n  - item)
                if last_key:
                    if not isinstance(container.get(last_key), list):
                        container[last_key] = []
                    container[last_key].append(value)
                else:
                    # Convert the parent key's container to a list if needed
                    if len(stack) >= 2:
                        parent_indent, parent_container, parent_last_key = stack[-2]
                        if isinstance(parent_container, dict) and parent_last_key:
                            if not isinstance(parent_container.get(parent_last_key), list):
                                parent_container[parent_last_key] = []
                            parent_container[parent_last_key].append(value)
                            # Point current frame to the list for subsequent items
                            stack[-1] = (stack[-1][0], parent_container[parent_last_key], None)
            continue

        if ":" in content:
            key, rest = content.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            if rest == "":
                new_container = {}
                if isinstance(container, dict):
                    container[key] = new_container
                    stack[-1] = (stack[-1][0], container, key)
                    stack.append((indent, new_container, None))
                elif isinstance(container, list):
                    new_item = {key: new_container}
                    container.append(new_item)
                    stack.append((indent, new_container, None))
            else:
                if isinstance(container, dict):
                    container[key] = _parse_scalar(rest)
                    stack[-1] = (stack[-1][0], container, key)
                elif isinstance(container, list):
                    container.append({key: _parse_scalar(rest)})
    return root


def _load_config():
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = f.read()
        data = _load_simple_yaml(raw)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data or {})
        if "rate_limit" in data:
            merged = DEFAULT_CONFIG["rate_limit"].copy()
            merged.update(data.get("rate_limit") or {})
            cfg["rate_limit"] = merged
        if "token_limit" in data:
            merged = DEFAULT_CONFIG["token_limit"].copy()
            merged.update(data.get("token_limit") or {})
            cfg["token_limit"] = merged
        if "image_watch" in data:
            merged = DEFAULT_CONFIG["image_watch"].copy()
            merged.update(data.get("image_watch") or {})
            cfg["image_watch"] = merged
        if "gateway" in data:
            # Backward compatibility: migrate legacy `gateway` section into `remote`.
            merged = DEFAULT_CONFIG["remote"].copy()
            legacy = data.get("gateway") or {}
            url = str(legacy.get("url") or "").strip()
            if url.startswith("http://") or url.startswith("https://"):
                # Parse host:port from url (very lightweight).
                no_scheme = url.split("://", 1)[1]
                host_port = no_scheme.split("/", 1)[0]
                if ":" in host_port:
                    host, port = host_port.rsplit(":", 1)
                    merged["host"] = host.strip() or merged["host"]
                    if str(port).isdigit():
                        merged["port"] = int(port)
                elif host_port:
                    merged["host"] = host_port.strip()
            if "token" in legacy:
                merged["token"] = legacy.get("token")
            if "timeout_seconds" in legacy:
                merged["timeout_seconds"] = legacy.get("timeout_seconds")
            cfg["remote"] = merged
        if "remote" in data:
            merged = DEFAULT_CONFIG["remote"].copy()
            merged.update(data.get("remote") or {})
            cfg["remote"] = merged
        if "runtime" in data:
            merged = DEFAULT_CONFIG["runtime"].copy()
            merged.update(data.get("runtime") or {})
            cfg["runtime"] = merged
        if "bridge" in data:
            merged = DEFAULT_CONFIG["bridge"].copy()
            merged.update(data.get("bridge") or {})
            cfg["bridge"] = merged
        if "context" in data:
            merged = DEFAULT_CONFIG["context"].copy()
            merged.update(data.get("context") or {})
            cfg["context"] = merged
        if "image_tool" in data:
            merged = DEFAULT_CONFIG["image_tool"].copy()
            merged.update(data.get("image_tool") or {})
            cfg["image_tool"] = merged
        if "file_tool" in data:
            merged = DEFAULT_CONFIG["file_tool"].copy()
            merged.update(data.get("file_tool") or {})
            cfg["file_tool"] = merged
        if "delivery" in data:
            merged = DEFAULT_CONFIG["delivery"].copy()
            merged.update(data.get("delivery") or {})
            cfg["delivery"] = merged
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()


def _load_state():
    if not STATE_PATH.exists():
        return {
            "version": 1,
            "sessions": {},
            "gateway_sessions": {},
            "usage": {},
            "token_usage": {},
            "group_mode": {},
            "context": {"group": {}, "private": {}},
        }
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("state invalid")
        data.setdefault("sessions", {})
        data.setdefault("gateway_sessions", {})
        data.setdefault("usage", {})
        data.setdefault("token_usage", {})
        data.setdefault("group_mode", {})
        data.setdefault("context", {"group": {}, "private": {}})
        return data
    except Exception:
        return {
            "version": 1,
            "sessions": {},
            "gateway_sessions": {},
            "usage": {},
            "token_usage": {},
            "group_mode": {},
            "context": {"group": {}, "private": {}},
        }


def _save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _session_key(group_id, user_id, is_private):
    if is_private:
        return f"private:{user_id}"
    return f"group:{group_id}:{user_id}"


def _get_session_id(state, key, reset=False):
    sessions = state.setdefault("sessions", {})
    if reset or key not in sessions:
        sessions[key] = uuid.uuid4().hex
    return sessions[key]


def _reset_dialog_state(state, group_id, user_id, is_private):
    """
    Reset conversation state for one dialog scope to reduce state file size.
    It clears local session ids, gateway session bindings, cached context, and
    per-user/group usage counters.
    """
    key = _session_key(group_id, user_id, is_private)

    sessions = state.setdefault("sessions", {})
    sessions.pop(key, None)

    gateway_sessions = state.setdefault("gateway_sessions", {})
    gateway_sessions.pop(_gateway_session_slot(key, "chat"), None)

    context_root = state.setdefault("context", {"group": {}, "private": {}})
    if is_private:
        (context_root.setdefault("private", {})).pop(str(user_id), None)
    else:
        (context_root.setdefault("group", {})).pop(str(group_id), None)

    usage = state.setdefault("usage", {})
    token_usage = state.setdefault("token_usage", {})
    if not is_private:
        usage.pop(_usage_key(group_id, user_id), None)
        token_usage.pop(_token_usage_key(group_id, user_id), None)

    # Recreate a fresh session id immediately, so next run uses a new isolated context.
    _get_session_id(state, key, reset=True)


def _usage_key(group_id, user_id):
    return f"group:{group_id}:{user_id}"


def _check_rate_limit(state, key, window_seconds, max_questions, now_ts):
    usage = state.setdefault("usage", {})
    timestamps = usage.get(key, [])
    if not isinstance(timestamps, list):
        timestamps = []
    timestamps = [t for t in timestamps if now_ts - float(t) <= window_seconds]
    if len(timestamps) >= max_questions:
        usage[key] = timestamps
        return False
    timestamps.append(now_ts)
    usage[key] = timestamps
    return True


def _token_usage_key(group_id, user_id):
    return f"group:{group_id}:{user_id}"


def _token_usage_total(entries, now_ts, window_seconds):
    total = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        ts = float(item.get("ts", 0))
        if now_ts - ts <= window_seconds:
            total += int(item.get("tokens", 0))
    return total


def _check_token_limit(state, key, window_seconds, max_tokens, now_ts):
    usage = state.setdefault("token_usage", {})
    entries = usage.get(key, [])
    if not isinstance(entries, list):
        entries = []
    entries = [e for e in entries if now_ts - float(e.get("ts", 0)) <= window_seconds]
    total = _token_usage_total(entries, now_ts, window_seconds)
    usage[key] = entries
    if total >= max_tokens:
        return False, total
    return True, total


def _record_token_usage(state, key, tokens, now_ts):
    if tokens is None:
        return
    usage = state.setdefault("token_usage", {})
    entries = usage.get(key, [])
    if not isinstance(entries, list):
        entries = []
    entries.append({"ts": now_ts, "tokens": int(tokens)})
    usage[key] = entries


def _extract_ids_from_path(path: Path, root_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        rel = path.relative_to(root_dir)
    except ValueError:
        return None, None
    group_id = None
    user_id = None
    for part in rel.parts:
        if group_id is None and part.startswith("g_"):
            group_id = part[2:]
        if user_id is None and part.startswith("u_"):
            user_id = part[2:]
    return group_id, user_id


async def _send_image(bot, group_id: str, image_path: Path):
    path_str = str(image_path)
    _debug_log(f"send_image group_id={group_id} path={path_str}")
    await bot.api.post_group_msg(
        group_id=group_id,
        image=path_str,
    )


async def _send_private_image(bot, user_id: str, image_path: Path):
    path_str = str(image_path)
    _debug_log(f"send_private_image user_id={user_id} path={path_str}")
    await bot.api.post_private_msg(
        user_id=user_id,
        image=path_str,
    )


_WATCH_TASK = None
_WATCH_SEEN = set()
_BRIDGE_SERVER = None
_BRIDGE_TASK = None
_BRIDGE_BOT = None


async def _image_watch_loop(bot, root_dir: Path, poll_seconds: float):
    root_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    if not _WATCH_SEEN:
        _debug_log(f"image_watch_start root_dir={root_dir} poll_seconds={poll_seconds}")
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                _WATCH_SEEN.add(str(p))
        _debug_log(f"image_watch_seeded count={len(_WATCH_SEEN)}")

    while True:
        for p in root_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            key = str(p)
            if key in _WATCH_SEEN:
                continue
            try:
                mtime = p.stat().st_mtime
                if time.time() - mtime < 1.0:
                    continue
            except FileNotFoundError:
                continue

            group_id, user_id = _extract_ids_from_path(p, root_dir)
            if not group_id:
                _debug_log(f"image_watch_skip_no_group path={p}")
                _WATCH_SEEN.add(key)
                continue
            try:
                if group_id == "private":
                    if not user_id:
                        _debug_log(f"image_watch_skip_no_user path={p}")
                    else:
                        _debug_log(f"image_watch_send_private path={p} user_id={user_id}")
                        await _send_private_image(bot, user_id, p)
                else:
                    _debug_log(f"image_watch_send path={p} group_id={group_id}")
                    await _send_image(bot, group_id, p)
            except Exception as exc:
                _debug_log(f"image_watch_send_failed path={p} group_id={group_id} err={exc}")
            finally:
                _WATCH_SEEN.add(key)

        await asyncio.sleep(poll_seconds)


def start_image_watcher(bot):
    global _WATCH_TASK
    cfg = _load_config()
    watch_cfg = cfg.get("image_watch") or {}
    if not watch_cfg.get("enabled"):
        stop_image_watcher()
        return
    root_dir = Path(str(watch_cfg.get("root_dir") or (DATA_DIR / "imagedata")))
    poll_seconds = float(watch_cfg.get("poll_seconds", 2))
    if _WATCH_TASK and not _WATCH_TASK.done():
        return
    _WATCH_TASK = asyncio.create_task(
        _image_watch_loop(bot, root_dir, poll_seconds),
        name="openclaw_image_watch",
    )


def stop_image_watcher():
    global _WATCH_TASK
    task = _WATCH_TASK
    _WATCH_TASK = None
    _WATCH_SEEN.clear()
    if task and not task.done():
        task.cancel()
    try:
        loop = asyncio.get_running_loop()
        for t in asyncio.all_tasks(loop):
            if t.done():
                continue
            if task is not None and t is task:
                continue
            cancel_hit = False
            try:
                t_name = t.get_name()
                if t_name == "openclaw_image_watch":
                    cancel_hit = True
            except Exception:
                pass
            if not cancel_hit:
                try:
                    coro_name = str(t.get_coro())
                    if "_image_watch_loop" in coro_name:
                        cancel_hit = True
                except Exception:
                    pass
            if cancel_hit:
                t.cancel()
    except RuntimeError:
        # No running loop at call site.
        pass


def _extract_reply_and_usage(stdout_text):
    text = stdout_text.strip()
    if not text:
        return "", None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return text, None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return text, None

    usage_total = None
    if isinstance(data, dict):
        meta = (data.get("result") or {}).get("meta") or data.get("meta") or {}
        agent_meta = meta.get("agentMeta") or {}
        usage = agent_meta.get("usage") or {}
        if isinstance(usage, dict) and usage:
            total = usage.get("total")
            if isinstance(total, (int, float)) and total > 0:
                usage_total = int(total)
            else:
                usage_total = int(usage.get("input", 0)) + int(usage.get("output", 0))

    payloads = (data.get("result") or {}).get("payloads") or []
    lines = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        body = (payload.get("text") or "").strip()
        if body:
            lines.append(body)
        media = payload.get("mediaUrls") or []
        media_url = payload.get("mediaUrl")
        if isinstance(media_url, str) and media_url.strip():
            media = [media_url.strip(), *media]
        for url in media:
            if isinstance(url, str) and url.strip():
                lines.append(f"MEDIA:{url.strip()}")
    if lines:
        return "\n".join(lines).strip(), usage_total
    summary = (data.get("summary") or "").strip()
    return summary or text, usage_total


def _parse_json_from_text(raw_text):
    text = str(raw_text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _cli_error_kind_from_stdout(stdout_text):
    parsed = _parse_json_from_text(stdout_text)
    if not isinstance(parsed, dict):
        raw = str(stdout_text or "")
        if "No session found" in raw:
            return "session_missing", raw.strip()[-1200:]
        return "", ""

    details = (parsed.get("result") or {}).get("details") or {}
    status = str(
        parsed.get("status")
        or (details.get("status") if isinstance(details, dict) else "")
        or ""
    ).strip().lower()
    error_text = str(
        parsed.get("error")
        or (details.get("error") if isinstance(details, dict) else "")
        or ""
    ).strip()

    if "No session found" in error_text:
        return "session_missing", error_text[-1200:]
    if status == "error":
        return "cli_error", (error_text or "status_error")[-1200:]
    if error_text:
        return "cli_error", error_text[-1200:]
    return "", ""


def _strip_markdown_basic(text: str) -> str:
    # Remove bold markers like **text**
    return text.replace("**", "")


def _split_reply_parts(text: str):
    raw = str(text or "").strip()
    if not raw:
        return []
    delim = None
    for token in ("[[SEND_SPLIT]]", "<<SEND_SPLIT>>", "<>"):
        if token in raw:
            delim = token
            break
    if delim is None:
        return [raw]
    parts = [x.strip() for x in raw.split(delim)]
    return [x for x in parts if x]


def _is_control_noise_line(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s:
        return True
    if "agent-to-agent announce step" in s:
        return True
    if s in {"no_reply", "`no_reply`"}:
        return True
    if s.startswith("agent-to-agent"):
        return True
    return False


def _sanitize_reply_parts(parts):
    clean = []
    for part in parts:
        # Drop explicit control/noise lines while keeping user-visible content.
        lines = [ln for ln in str(part).splitlines() if not _is_control_noise_line(ln)]
        text = "\n".join(lines).strip()
        if not text:
            continue
        clean.append(text)
    return clean


def _has_control_noise(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    for line in raw.splitlines():
        if _is_control_noise_line(line):
            return True
    return False


def _looks_like_json_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    # Support fenced json blocks.
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
    if not ((raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]"))):
        return False
    try:
        parsed = json.loads(raw)
    except Exception:
        return False
    return isinstance(parsed, (dict, list))


async def _send_text(bot, group_id, user_id, text):
    if group_id:
        await bot.api.post_group_msg(group_id=group_id, text=text)
        _record_context_message(group_id=group_id, user_id=user_id, is_private=False, role="assistant", text=text)
    else:
        await bot.api.post_private_msg(user_id, text=text)
        _record_context_message(group_id=None, user_id=user_id, is_private=True, role="assistant", text=text)


def _clamp_int(raw, default_value, min_value, max_value):
    try:
        value = int(raw)
    except Exception:
        value = int(default_value)
    return max(min_value, min(value, max_value))


def _context_scope_id(group_id, user_id, is_private):
    if is_private:
        uid = str(user_id).strip()
        return "private", uid
    gid = str(group_id).strip()
    return "group", gid


def _context_text_compact(text, max_len=1000):
    raw = str(text or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"\s+", " ", raw)
    if len(compact) > max_len:
        return compact[: max_len - 3] + "..."
    return compact


def _extract_cq_image_urls(text):
    raw = str(text or "")
    urls = []
    # CQ image segment format: [CQ:image,file=...,url=...]
    for matched in re.findall(r"\[CQ:image,([^\]]+)\]", raw, flags=re.IGNORECASE):
        m = re.search(r"(?:^|,)url=([^,\]]+)", matched, flags=re.IGNORECASE)
        if m:
            u = str(m.group(1)).strip()
            if u.startswith("http://") or u.startswith("https://"):
                urls.append(u)
    # Also accept plain image links in text.
    for m in re.findall(r"https?://[^\s\]]+\.(?:png|jpg|jpeg|webp|gif|bmp)", raw, flags=re.IGNORECASE):
        urls.append(str(m).strip())
    seen = set()
    result = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        result.append(u)
    return result


def _extract_cq_file_refs(text):
    raw = str(text or "")
    refs = []
    for matched in re.findall(r"\[CQ:file,([^\]]+)\]", raw, flags=re.IGNORECASE):
        item = {"url": "", "file_id": "", "file_name": ""}
        m_url = re.search(r"(?:^|,)url=([^,\]]+)", matched, flags=re.IGNORECASE)
        if m_url:
            item["url"] = str(m_url.group(1)).strip()
        m_id = re.search(r"(?:^|,)file_id=([^,\]]+)", matched, flags=re.IGNORECASE)
        if m_id:
            item["file_id"] = str(m_id.group(1)).strip()
        m_name = re.search(r"(?:^|,)file=([^,\]]+)", matched, flags=re.IGNORECASE)
        if m_name:
            item["file_name"] = str(m_name.group(1)).strip()
        if item["url"] or item["file_id"] or item["file_name"]:
            refs.append(item)

    # Runtime event-style format, e.g.:
    # File(file="denseTNT.pdf", url="https://.../?fname=", file_id="/uuid", file_size="3058247")
    for matched in re.findall(r"File\(([^)]*)\)", raw):
        item = {"url": "", "file_id": "", "file_name": ""}
        m_name = re.search(r'file="([^"]*)"', matched)
        if m_name:
            item["file_name"] = str(m_name.group(1)).strip()
        m_url = re.search(r'url="([^"]*)"', matched)
        if m_url:
            item["url"] = str(m_url.group(1)).strip()
        m_id = re.search(r'file_id="([^"]*)"', matched)
        if m_id:
            item["file_id"] = str(m_id.group(1)).strip()
        if item["url"] or item["file_id"] or item["file_name"]:
            refs.append(item)

    for u in re.findall(r"https?://[^\s\]]+", raw, flags=re.IGNORECASE):
        url = str(u).strip()
        lower = url.lower()
        if any(lower.endswith(ext) for ext in [".pdf", ".txt", ".md", ".doc", ".docx"]):
            refs.append({"url": url, "file_id": "", "file_name": Path(url).name})

    seen = set()
    unique = []
    for item in refs:
        key = (item.get("url") or "", item.get("file_id") or "", item.get("file_name") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _load_prompt_template_md():
    try:
        if PROMPT_MD_PATH.exists():
            return PROMPT_MD_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        _debug_log(f"load_prompt_md_failed err={exc}")
    return ""


def _render_prompt_md(template_text, values):
    txt = str(template_text or "")
    if not txt.strip():
        return ""
    for key, value in values.items():
        txt = txt.replace("{{" + str(key) + "}}", str(value))
    return txt.strip()


def _guess_image_ext(url_or_name):
    raw = str(url_or_name or "").strip().lower()
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
        if ext in raw:
            return ext
    return ".jpg"


def _safe_filename(name, fallback_stem="image", fallback_ext=".jpg"):
    raw = str(name or "").strip()
    if not raw:
        raw = fallback_stem + fallback_ext
    raw = raw.replace("\\", "_").replace("/", "_")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    if "." not in raw:
        raw += fallback_ext
    if len(raw) > 120:
        base, dot, ext = raw.rpartition(".")
        if not dot:
            base = raw[:115]
            ext = "dat"
        raw = f"{base[:100]}.{ext[:12]}"
    return raw


def _image_size_from_bytes(path_obj):
    # Return width,height when parsable from common formats, otherwise (None, None).
    try:
        with open(path_obj, "rb") as f:
            head = f.read(64)
            if len(head) < 10:
                return None, None
            # PNG
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                f.seek(16)
                wh = f.read(8)
                if len(wh) == 8:
                    return int.from_bytes(wh[0:4], "big"), int.from_bytes(wh[4:8], "big")
            # GIF
            if head[:6] in {b"GIF87a", b"GIF89a"}:
                return int.from_bytes(head[6:8], "little"), int.from_bytes(head[8:10], "little")
            # JPEG
            if head.startswith(b"\xff\xd8"):
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2:
                        break
                    if marker[0] != 0xFF:
                        continue
                    while marker[1] == 0xFF:
                        marker = bytes([0xFF]) + f.read(1)
                        if len(marker) < 2:
                            break
                    if len(marker) < 2:
                        break
                    if marker[1] in {0xD8, 0xD9}:
                        continue
                    seg_len_b = f.read(2)
                    if len(seg_len_b) < 2:
                        break
                    seg_len = int.from_bytes(seg_len_b, "big")
                    if seg_len < 2:
                        break
                    if marker[1] in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                        data = f.read(5)
                        if len(data) == 5:
                            h = int.from_bytes(data[1:3], "big")
                            w = int.from_bytes(data[3:5], "big")
                            return w, h
                        break
                    f.seek(seg_len - 2, 1)
            # WebP (basic parse)
            if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
                chunk = head[12:16]
                if chunk == b"VP8X":
                    f.seek(24)
                    d = f.read(6)
                    if len(d) == 6:
                        w = 1 + int.from_bytes(d[0:3], "little")
                        h = 1 + int.from_bytes(d[3:6], "little")
                        return w, h
        return None, None
    except Exception:
        return None, None


def _image_meta(path_obj):
    p = Path(path_obj).resolve()
    stat = p.stat()
    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    width, height = _image_size_from_bytes(p)
    return {
        "path": str(p),
        "size_bytes": int(stat.st_size),
        "sha256": h.hexdigest(),
        "mime": mime,
        "width": width,
        "height": height,
    }


def _append_context_entry(state, cfg, group_id, user_id, is_private, role, text, nickname=""):
    context_cfg = cfg.get("context") or {}
    if not bool(context_cfg.get("enabled", True)):
        return
    scope, scope_id = _context_scope_id(group_id, user_id, is_private)
    if not scope_id:
        return
    msg = _context_text_compact(text, max_len=1200)
    if not msg:
        return

    max_messages = _clamp_int(
        (context_cfg.get("max_messages_per_channel", 300)),
        300,
        50,
        5000,
    )
    context_root = state.setdefault("context", {"group": {}, "private": {}})
    bucket = context_root.setdefault(scope, {})
    entries = bucket.setdefault(scope_id, [])
    now_ts = time.time()

    # Basic dedupe for hot paths where the same event can be recorded twice.
    if entries:
        last = entries[-1]
        if (
            isinstance(last, dict)
            and str(last.get("role") or "") == str(role)
            and str(last.get("user_id") or "") == str(user_id)
            and str(last.get("text") or "") == msg
            and now_ts - float(last.get("ts") or 0) <= 3
        ):
            return

    entries.append(
        {
            "ts": now_ts,
            "role": str(role),
            "user_id": str(user_id),
            "nickname": str(nickname or "").strip(),
            "text": msg,
            "image_urls": _extract_cq_image_urls(text),
            "file_refs": _extract_cq_file_refs(text),
        }
    )
    if len(entries) > max_messages:
        del entries[: len(entries) - max_messages]


def _recent_context_items_text(state, cfg, group_id, user_id, is_private, limit=None, max_chars=None):
    context_cfg = cfg.get("context") or {}
    scope, scope_id = _context_scope_id(group_id, user_id, is_private)
    context_root = state.setdefault("context", {"group": {}, "private": {}})
    entries = ((context_root.get(scope) or {}).get(scope_id) or [])
    if not isinstance(entries, list):
        entries = []

    use_limit = _clamp_int(
        limit if limit is not None else context_cfg.get("default_recent_limit", 20),
        20,
        1,
        200,
    )
    use_max_chars = _clamp_int(
        max_chars if max_chars is not None else context_cfg.get("default_recent_max_chars", 4000),
        4000,
        200,
        50000,
    )
    selected = entries[-use_limit:]

    def _line_for(item):
        ts = float(item.get("ts") or 0)
        when = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts > 0 else "--:--"
        role = str(item.get("role") or "").strip().lower()
        uid = str(item.get("user_id") or "").strip()
        nick = str(item.get("nickname") or "").strip()
        if role == "assistant":
            who = "助手"
        elif nick:
            who = f"{nick}({uid})"
        else:
            who = uid or "用户"
        txt = _context_text_compact(item.get("text") or "", max_len=1200)
        return f"[{when}] {who}: {txt}"

    lines = [_line_for(item) for item in selected if isinstance(item, dict)]
    if not lines:
        return [], ""

    used = 0
    kept_lines = []
    for line in reversed(lines):
        size = len(line) + 1
        if kept_lines and used + size > use_max_chars:
            break
        if not kept_lines and size > use_max_chars:
            kept_lines.append(line[-use_max_chars:])
            used = use_max_chars
            break
        kept_lines.append(line)
        used += size
    kept_lines.reverse()

    line_set = set(kept_lines)
    kept_items = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        if _line_for(item) in line_set:
            kept_items.append(item)
    return kept_items, "\n".join(kept_lines).strip()


def _record_context_message(group_id, user_id, is_private, role, text, nickname=""):
    try:
        cfg = _load_config()
        state = _load_state()
        _append_context_entry(
            state=state,
            cfg=cfg,
            group_id=group_id,
            user_id=user_id,
            is_private=is_private,
            role=role,
            text=text,
            nickname=nickname,
        )
        _save_state(state)
    except Exception as exc:
        _debug_log(f"context_record_failed err={exc}")


def record_incoming_context(group_id, user_id, text, is_private, nickname=""):
    _record_context_message(
        group_id=group_id,
        user_id=user_id,
        is_private=bool(is_private),
        role="user",
        text=text,
        nickname=nickname,
    )


def _http_json_response(status_code, payload):
    reason = {
        200: "OK",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        500: "Internal Server Error",
    }.get(status_code, "OK")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    head = [
        f"HTTP/1.1 {status_code} {reason}",
        "Content-Type: application/json; charset=utf-8",
        f"Content-Length: {len(body)}",
        "Connection: close",
        "",
        "",
    ]
    return ("\r\n".join(head)).encode("utf-8") + body


def _is_path_allowed(path_obj, cfg):
    bridge_cfg = cfg.get("bridge") or {}
    roots = bridge_cfg.get("allowed_paths") or []
    if not isinstance(roots, list) or not roots:
        return False
    target = path_obj.resolve()
    for root in roots:
        root_str = str(root).strip()
        if not root_str:
            continue
        root_path = Path(root_str).expanduser()
        if not root_path.is_absolute():
            root_path = (BASE_DIR / root_path).resolve()
        else:
            root_path = root_path.resolve()
        try:
            target.relative_to(root_path)
            return True
        except ValueError:
            continue
    return False


def _read_token_from_headers(headers):
    x_token = str(headers.get("x-bridge-token", "")).strip()
    if x_token:
        return x_token
    auth = str(headers.get("authorization", "")).strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


async def _bridge_handle_send_text(payload):
    target = str(payload.get("target") or "").strip().lower()
    text = str(payload.get("text") or "").strip()
    if not text:
        return 400, {"ok": False, "error": "text_required"}
    if target == "private":
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            return 400, {"ok": False, "error": "user_id_required"}
        await _send_text(_BRIDGE_BOT, None, user_id, text=text)
        return 200, {"ok": True}
    if target == "group":
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            return 400, {"ok": False, "error": "group_id_required"}
        await _send_text(_BRIDGE_BOT, group_id, "", text=text)
        return 200, {"ok": True}
    return 400, {"ok": False, "error": "target_must_be_private_or_group"}


async def _bridge_handle_send_file(payload, cfg):
    target = str(payload.get("target") or "").strip().lower()
    path_raw = str(payload.get("path") or "").strip()
    if not path_raw:
        return 400, {"ok": False, "error": "path_required"}
    file_path = Path(path_raw).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        return 400, {"ok": False, "error": "file_not_found"}
    if not _is_path_allowed(file_path, cfg):
        return 403, {"ok": False, "error": "path_not_allowed"}
    if target == "private":
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            return 400, {"ok": False, "error": "user_id_required"}
        await _BRIDGE_BOT.api.post_private_file(user_id=user_id, file=str(file_path))
        return 200, {"ok": True}
    if target == "group":
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            return 400, {"ok": False, "error": "group_id_required"}
        await _BRIDGE_BOT.api.post_group_file(group_id=group_id, file=str(file_path))
        return 200, {"ok": True}
    return 400, {"ok": False, "error": "target_must_be_private_or_group"}


async def _bridge_handle_send_image(payload, cfg):
    target = str(payload.get("target") or "").strip().lower()
    path_raw = str(payload.get("path") or "").strip()
    if not path_raw:
        return 400, {"ok": False, "error": "path_required"}
    image_path = Path(path_raw).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        return 400, {"ok": False, "error": "file_not_found"}
    if not _is_path_allowed(image_path, cfg):
        return 403, {"ok": False, "error": "path_not_allowed"}
    ext = image_path.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return 400, {"ok": False, "error": "not_image_file"}

    if target == "private":
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            return 400, {"ok": False, "error": "user_id_required"}
        await _BRIDGE_BOT.api.post_private_msg(user_id=user_id, image=str(image_path))
        _record_context_message(group_id=None, user_id=user_id, is_private=True, role="assistant", text=f"[image] {image_path}")
        return 200, {"ok": True}

    if target == "group":
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            return 400, {"ok": False, "error": "group_id_required"}
        await _BRIDGE_BOT.api.post_group_msg(group_id=group_id, image=str(image_path))
        _record_context_message(group_id=group_id, user_id="", is_private=False, role="assistant", text=f"[image] {image_path}")
        return 200, {"ok": True}

    return 400, {"ok": False, "error": "target_must_be_private_or_group"}


async def _bridge_handle_get_file_url(payload):
    target = str(payload.get("target") or "").strip().lower()
    file_id = str(payload.get("file_id") or "").strip()
    if not file_id:
        return 400, {"ok": False, "error": "file_id_required"}
    if target == "private":
        url = await _BRIDGE_BOT.api.get_private_file_url(file_id=file_id)
        return 200, {"ok": True, "url": url}
    if target == "group":
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            return 400, {"ok": False, "error": "group_id_required"}
        url = await _BRIDGE_BOT.api.get_group_file_url(group_id=group_id, file_id=file_id)
        return 200, {"ok": True, "url": url}
    return 400, {"ok": False, "error": "target_must_be_private_or_group"}


async def _bridge_handle_context_recent(payload, cfg, is_private):
    if is_private:
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            return 400, {"ok": False, "error": "user_id_required"}
        group_id = None
        scope = "private"
        scope_id = user_id
    else:
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            return 400, {"ok": False, "error": "group_id_required"}
        user_id = ""
        scope = "group"
        scope_id = group_id

    items, text = _recent_context_items_text(
        state=_load_state(),
        cfg=cfg,
        group_id=group_id,
        user_id=user_id,
        is_private=is_private,
        limit=payload.get("limit"),
        max_chars=payload.get("max_chars"),
    )
    return 200, {
        "ok": True,
        "scope": scope,
        "scope_id": scope_id,
        "count": len(items),
        "items": items,
        "text": text,
    }


async def _bridge_handle_context_files(payload, is_private):
    if is_private:
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            return 400, {"ok": False, "error": "user_id_required"}
        scope = "private"
        scope_id = user_id
    else:
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            return 400, {"ok": False, "error": "group_id_required"}
        scope = "group"
        scope_id = group_id

    limit = _clamp_int(payload.get("limit"), 20, 1, 200)
    state = _load_state()
    context_root = state.setdefault("context", {"group": {}, "private": {}})
    entries = ((context_root.get(scope) or {}).get(scope_id) or [])
    if not isinstance(entries, list):
        entries = []

    out = []
    for item in reversed(entries):
        if not isinstance(item, dict):
            continue
        file_refs = item.get("file_refs") or []
        if not isinstance(file_refs, list) or not file_refs:
            continue
        for ref in file_refs:
            if not isinstance(ref, dict):
                continue
            out.append(
                {
                    "ts": float(item.get("ts") or 0),
                    "role": str(item.get("role") or ""),
                    "user_id": str(item.get("user_id") or ""),
                    "nickname": str(item.get("nickname") or ""),
                    "file_ref": {
                        "url": str(ref.get("url") or ""),
                        "file_id": str(ref.get("file_id") or ""),
                        "file_name": str(ref.get("file_name") or ""),
                    },
                    "text": str(item.get("text") or ""),
                }
            )
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    out.reverse()

    return 200, {
        "ok": True,
        "scope": scope,
        "scope_id": scope_id,
        "count": len(out),
        "files": out,
    }


def _image_download_target(cfg, payload):
    image_cfg = cfg.get("image_tool") or {}
    raw_root = str(image_cfg.get("download_root") or (DATA_DIR / "downloads" / "images")).strip()
    root = Path(raw_root)
    if not root.is_absolute():
        root = (BASE_DIR / root).resolve()

    gid = str(payload.get("group_id") or "").strip()
    uid = str(payload.get("user_id") or "").strip()
    if gid:
        sub = root / f"g_{gid}"
        if uid:
            sub = sub / f"u_{uid}"
    elif uid:
        sub = root / "g_private" / f"u_{uid}"
    else:
        sub = root / "misc"
    return root, sub


def _file_download_target(cfg, payload):
    file_cfg = cfg.get("file_tool") or {}
    raw_root = str(file_cfg.get("download_root") or (DATA_DIR / "downloads" / "files")).strip()
    root = Path(raw_root)
    if not root.is_absolute():
        root = (BASE_DIR / root).resolve()

    gid = str(payload.get("group_id") or "").strip()
    uid = str(payload.get("user_id") or "").strip()
    if gid:
        sub = root / f"g_{gid}"
        if uid:
            sub = sub / f"u_{uid}"
    elif uid:
        sub = root / "g_private" / f"u_{uid}"
    else:
        sub = root / "misc"
    return root, sub


def _download_file_sync(url, target_path, timeout_seconds, max_bytes):
    req = urllib_request.Request(
        url,
        headers={
            "User-Agent": "ncatbot-openclaw-bridge/1.0",
            "Accept": "*/*",
        },
        method="GET",
    )
    total = 0
    content_type = ""
    with urllib_request.urlopen(req, timeout=timeout_seconds) as resp:
        content_type = str(resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        with open(target_path, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"file_too_large>{max_bytes}")
                f.write(chunk)
    return {"size_bytes": total, "content_type": content_type}


def _download_image_sync(url, target_path, timeout_seconds, max_bytes):
    req = urllib_request.Request(
        url,
        headers={
            "User-Agent": "ncatbot-openclaw-bridge/1.0",
            "Accept": "image/*,*/*;q=0.8",
        },
        method="GET",
    )
    total = 0
    content_type = ""
    with urllib_request.urlopen(req, timeout=timeout_seconds) as resp:
        content_type = str(resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        with open(target_path, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"file_too_large>{max_bytes}")
                f.write(chunk)
    return {"size_bytes": total, "content_type": content_type}


def _is_image_path(path_obj):
    try:
        with open(path_obj, "rb") as f:
            head = f.read(16)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return True
        if head.startswith(b"\xff\xd8\xff"):
            return True
        if head[:6] in {b"GIF87a", b"GIF89a"}:
            return True
        if head.startswith(b"RIFF") and len(head) >= 12 and head[8:12] == b"WEBP":
            return True
        if head.startswith(b"BM"):
            return True
    except Exception:
        pass
    ext = Path(path_obj).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return True
    mime = mimetypes.guess_type(str(path_obj))[0] or ""
    return mime.startswith("image/")


async def _bridge_handle_image_download(payload, cfg):
    url = str(payload.get("url") or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return 400, {"ok": False, "error": "url_required_http_https"}

    root, target_dir = _image_download_target(cfg, payload)
    if not _is_path_allowed(root, cfg):
        return 403, {"ok": False, "error": "download_root_not_allowed"}
    target_dir.mkdir(parents=True, exist_ok=True)

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = str(payload.get("filename") or "").strip()
    if filename:
        safe_name = _safe_filename(filename, fallback_stem=f"img_{now_tag}", fallback_ext=_guess_image_ext(url))
    else:
        safe_name = _safe_filename(f"img_{now_tag}_{uuid.uuid4().hex[:8]}{_guess_image_ext(url)}")
    final_path = (target_dir / safe_name).resolve()
    if not _is_path_allowed(final_path, cfg):
        return 403, {"ok": False, "error": "path_not_allowed"}

    image_cfg = cfg.get("image_tool") or {}
    timeout_seconds = max(3, int(image_cfg.get("download_timeout_seconds", 30)))
    max_bytes = max(1, int(image_cfg.get("download_max_mb", 20))) * 1024 * 1024
    tmp_path = final_path.with_suffix(final_path.suffix + ".part")
    try:
        info = await asyncio.to_thread(_download_image_sync, url, tmp_path, timeout_seconds, max_bytes)
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        if not _is_image_path(final_path):
            final_path.unlink(missing_ok=True)
            return 400, {"ok": False, "error": "downloaded_file_not_image"}
        meta = _image_meta(final_path)
        meta["source_url"] = url
        meta["content_type"] = info.get("content_type") or ""
        return 200, {"ok": True, "saved": meta}
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return 500, {"ok": False, "error": f"download_failed:{exc}"}


async def _bridge_handle_file_download(payload, cfg):
    url = str(payload.get("url") or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return 400, {"ok": False, "error": "url_required_http_https"}

    root, target_dir = _file_download_target(cfg, payload)
    if not _is_path_allowed(root, cfg):
        return 403, {"ok": False, "error": "download_root_not_allowed"}
    target_dir.mkdir(parents=True, exist_ok=True)

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = str(payload.get("filename") or "").strip()
    if filename:
        safe_name = _safe_filename(filename, fallback_stem=f"file_{now_tag}", fallback_ext=".bin")
    else:
        guessed = Path(url.split("?", 1)[0]).name
        safe_name = _safe_filename(guessed, fallback_stem=f"file_{now_tag}_{uuid.uuid4().hex[:8]}", fallback_ext=".bin")
    final_path = (target_dir / safe_name).resolve()
    if not _is_path_allowed(final_path, cfg):
        return 403, {"ok": False, "error": "path_not_allowed"}

    file_cfg = cfg.get("file_tool") or {}
    timeout_seconds = max(3, int(file_cfg.get("download_timeout_seconds", 60)))
    max_bytes = max(1, int(file_cfg.get("download_max_mb", 50))) * 1024 * 1024
    tmp_path = final_path.with_suffix(final_path.suffix + ".part")
    try:
        info = await asyncio.to_thread(_download_file_sync, url, tmp_path, timeout_seconds, max_bytes)
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        data = {
            "path": str(final_path),
            "size_bytes": int(info.get("size_bytes") or final_path.stat().st_size),
            "content_type": info.get("content_type") or "",
            "source_url": url,
        }
        return 200, {"ok": True, "saved": data}
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return 500, {"ok": False, "error": f"download_failed:{exc}"}


async def _bridge_handle_image_read(payload, cfg):
    path_raw = str(payload.get("path") or "").strip()
    if not path_raw:
        return 400, {"ok": False, "error": "path_required"}
    image_path = Path(path_raw).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        return 400, {"ok": False, "error": "file_not_found"}
    if not _is_path_allowed(image_path, cfg):
        return 403, {"ok": False, "error": "path_not_allowed"}
    if not _is_image_path(image_path):
        return 400, {"ok": False, "error": "not_image_file"}

    meta = _image_meta(image_path)
    with_base64 = bool(payload.get("with_base64", False))
    image_cfg = cfg.get("image_tool") or {}
    allow_b64 = bool(image_cfg.get("read_allow_base64", False))
    if with_base64 and allow_b64:
        max_chars = _clamp_int(
            payload.get("max_base64_chars", image_cfg.get("read_max_base64_chars", 200000)),
            int(image_cfg.get("read_max_base64_chars", 200000)),
            1000,
            2_000_000,
        )
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        truncated = False
        if len(b64) > max_chars:
            b64 = b64[:max_chars]
            truncated = True
        return 200, {"ok": True, "meta": meta, "base64": b64, "base64_truncated": truncated}
    return 200, {"ok": True, "meta": meta}


async def _bridge_handle_file_read(payload, cfg):
    path_raw = str(payload.get("path") or "").strip()
    if not path_raw:
        return 400, {"ok": False, "error": "path_required"}
    file_path = Path(path_raw).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        return 400, {"ok": False, "error": "file_not_found"}
    if not _is_path_allowed(file_path, cfg):
        return 403, {"ok": False, "error": "path_not_allowed"}

    max_chars = _clamp_int(payload.get("max_chars"), 12000, 200, 200000)
    tail = bool(payload.get("tail", False))
    decode_encoding = str(payload.get("encoding") or "utf-8").strip() or "utf-8"
    byte_hint = max_chars * 4
    truncated = False
    data = b""

    try:
        fsize = file_path.stat().st_size
        with open(file_path, "rb") as f:
            if tail and fsize > byte_hint:
                f.seek(max(0, fsize - byte_hint))
                data = f.read(byte_hint)
                truncated = True
            else:
                data = f.read(byte_hint + 1)
                if len(data) > byte_hint:
                    data = data[:byte_hint]
                    truncated = True
        text = data.decode(decode_encoding, errors="replace")
    except Exception as exc:
        return 500, {"ok": False, "error": f"read_failed:{exc}"}

    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return 200, {
        "ok": True,
        "path": str(file_path),
        "encoding": decode_encoding,
        "truncated": truncated,
        "text": text,
    }


def _read_docx_text(path_obj):
    """
    Read .docx plain text with stdlib only.
    Note: legacy .doc binary files are not supported.
    """
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        with zipfile.ZipFile(path_obj, "r") as zf:
            xml_bytes = zf.read("word/document.xml")
    except Exception as exc:
        raise RuntimeError(f"docx_open_failed:{exc}")

    try:
        root = ET.fromstring(xml_bytes)
    except Exception as exc:
        raise RuntimeError(f"docx_parse_failed:{exc}")

    paragraphs = []
    for para in root.findall(".//w:p", ns):
        parts = []
        for node in para.findall(".//w:t", ns):
            if node.text:
                parts.append(node.text)
        if parts:
            paragraphs.append("".join(parts))
    return "\n".join(paragraphs).strip()


async def _bridge_handle_word_read(payload, cfg):
    path_raw = str(payload.get("path") or "").strip()
    if not path_raw:
        return 400, {"ok": False, "error": "path_required"}
    file_path = Path(path_raw).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        return 400, {"ok": False, "error": "file_not_found"}
    if not _is_path_allowed(file_path, cfg):
        return 403, {"ok": False, "error": "path_not_allowed"}

    suffix = file_path.suffix.lower()
    if suffix != ".docx":
        if suffix == ".doc":
            return 400, {"ok": False, "error": "doc_not_supported_convert_to_docx"}
        return 400, {"ok": False, "error": "only_docx_supported"}

    max_chars = _clamp_int(payload.get("max_chars"), 20000, 200, 300000)
    tail = bool(payload.get("tail", False))
    try:
        text = await asyncio.to_thread(_read_docx_text, file_path)
    except Exception as exc:
        return 500, {"ok": False, "error": f"word_read_failed:{exc}"}

    truncated = False
    if len(text) > max_chars:
        text = text[-max_chars:] if tail else text[:max_chars]
        truncated = True
    return 200, {
        "ok": True,
        "path": str(file_path),
        "format": "docx",
        "truncated": truncated,
        "text": text,
    }


async def _bridge_handle_client(reader, writer):
    try:
        raw_head = await reader.readuntil(b"\r\n\r\n")
    except Exception:
        writer.close()
        await writer.wait_closed()
        return
    try:
        header_text = raw_head.decode("utf-8", errors="replace")
        lines = header_text.split("\r\n")
        request_line = lines[0].strip()
        parts = request_line.split(" ")
        if len(parts) < 2:
            resp = _http_json_response(400, {"ok": False, "error": "bad_request_line"})
            writer.write(resp)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        method = parts[0].upper()
        path = parts[1].split("?", 1)[0]
        headers = {}
        for line in lines[1:]:
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
        content_length = int(headers.get("content-length", "0") or "0")
        body_bytes = b""
        if content_length > 0:
            body_bytes = await reader.readexactly(content_length)
        cfg = _load_config()
        bridge_cfg = cfg.get("bridge") or {}
        expected_token = str(bridge_cfg.get("token") or "").strip()
        if path == "/health" and method == "GET":
            resp = _http_json_response(200, {"ok": True, "service": "ncatbot-bridge"})
            writer.write(resp)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        if method != "POST":
            resp = _http_json_response(405, {"ok": False, "error": "method_not_allowed"})
            writer.write(resp)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        provided_token = _read_token_from_headers(headers)
        if expected_token and provided_token != expected_token:
            resp = _http_json_response(401, {"ok": False, "error": "unauthorized"})
            writer.write(resp)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        try:
            payload = json.loads(body_bytes.decode("utf-8", errors="replace") or "{}")
        except Exception:
            resp = _http_json_response(400, {"ok": False, "error": "invalid_json"})
            writer.write(resp)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        if not isinstance(payload, dict):
            resp = _http_json_response(400, {"ok": False, "error": "payload_must_be_object"})
            writer.write(resp)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        if path == "/send_text":
            status, data = await _bridge_handle_send_text(payload)
        elif path == "/send_file":
            status, data = await _bridge_handle_send_file(payload, cfg)
        elif path == "/send_image":
            status, data = await _bridge_handle_send_image(payload, cfg)
        elif path == "/get_file_url":
            status, data = await _bridge_handle_get_file_url(payload)
        elif path == "/context/private_recent":
            status, data = await _bridge_handle_context_recent(payload, cfg, is_private=True)
        elif path == "/context/group_recent":
            status, data = await _bridge_handle_context_recent(payload, cfg, is_private=False)
        elif path == "/context/private_files":
            status, data = await _bridge_handle_context_files(payload, is_private=True)
        elif path == "/context/group_files":
            status, data = await _bridge_handle_context_files(payload, is_private=False)
        elif path == "/file/download":
            status, data = await _bridge_handle_file_download(payload, cfg)
        elif path == "/image/download":
            status, data = await _bridge_handle_image_download(payload, cfg)
        elif path == "/image/read":
            status, data = await _bridge_handle_image_read(payload, cfg)
        elif path == "/file/read":
            status, data = await _bridge_handle_file_read(payload, cfg)
        elif path in {"/file/read_word", "/word/read"}:
            status, data = await _bridge_handle_word_read(payload, cfg)
        else:
            status, data = 404, {"ok": False, "error": "not_found"}
        writer.write(_http_json_response(status, data))
        await writer.drain()
    except Exception as exc:
        writer.write(_http_json_response(500, {"ok": False, "error": str(exc)}))
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def _bridge_server_loop(host, port):
    global _BRIDGE_SERVER
    try:
        _BRIDGE_SERVER = await asyncio.start_server(_bridge_handle_client, host, port)
    except OSError as exc:
        _debug_log(f"bridge_server_bind_failed host={host} port={port} err={exc}")
        return
    _debug_log(f"bridge_server_started host={host} port={port}")
    async with _BRIDGE_SERVER:
        await _BRIDGE_SERVER.serve_forever()


def start_bridge_server(bot):
    global _BRIDGE_TASK, _BRIDGE_BOT
    cfg = _load_config()
    bridge_cfg = cfg.get("bridge") or {}
    if _resolve_effective_invoke_mode(cfg) != "cli":
        stop_bridge_server()
        return
    if not bool(bridge_cfg.get("enabled", True)):
        stop_bridge_server()
        return
    host = str(bridge_cfg.get("host") or "127.0.0.1").strip()
    port = int(bridge_cfg.get("port") or 18999)
    _BRIDGE_BOT = bot
    if _BRIDGE_TASK and not _BRIDGE_TASK.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _debug_log("bridge_server_start_skipped_no_loop")
        return
    _BRIDGE_TASK = loop.create_task(
        _bridge_server_loop(host, port),
        name="openclaw_bridge_server",
    )


def stop_bridge_server():
    global _BRIDGE_TASK, _BRIDGE_SERVER, _BRIDGE_BOT
    task = _BRIDGE_TASK
    _BRIDGE_TASK = None
    _BRIDGE_BOT = None
    server = _BRIDGE_SERVER
    _BRIDGE_SERVER = None
    if server is not None:
        try:
            server.close()
        except Exception:
            pass
    if task and not task.done():
        task.cancel()
    try:
        loop = asyncio.get_running_loop()
        for t in asyncio.all_tasks(loop):
            if t.done():
                continue
            if task is not None and t is task:
                continue
            cancel_hit = False
            try:
                t_name = t.get_name()
                if t_name == "openclaw_bridge_server":
                    cancel_hit = True
            except Exception:
                pass
            if not cancel_hit:
                try:
                    coro_name = str(t.get_coro())
                    if "_bridge_server_loop" in coro_name:
                        cancel_hit = True
                except Exception:
                    pass
            if cancel_hit:
                t.cancel()
    except RuntimeError:
        pass


def _resolve_remote_runtime(cfg):
    remote_cfg = cfg.get("remote") or {}
    legacy_url = str(os.getenv("OPENCLAW_GATEWAY_URL") or "").strip()
    host = str(
        os.getenv("OPENCLAW_REMOTE_HOST")
        or remote_cfg.get("host")
        or "127.0.0.1"
    ).strip()
    port_raw = str(
        os.getenv("OPENCLAW_REMOTE_PORT")
        or remote_cfg.get("port")
        or "18789"
    ).strip()
    port = int(port_raw) if port_raw.isdigit() else 18789
    if legacy_url.startswith("http://") or legacy_url.startswith("https://"):
        try:
            no_scheme = legacy_url.split("://", 1)[1]
            host_port = no_scheme.split("/", 1)[0]
            if ":" in host_port:
                legacy_host, legacy_port = host_port.rsplit(":", 1)
                if legacy_host.strip():
                    host = legacy_host.strip()
                if legacy_port.isdigit():
                    port = int(legacy_port)
            elif host_port.strip():
                host = host_port.strip()
        except Exception:
            pass
    token = str(
        os.getenv("OPENCLAW_REMOTE_TOKEN")
        or os.getenv("OPENCLAW_GATEWAY_TOKEN")
        or remote_cfg.get("token")
        or OPENCLAW_GATEWAY_TOKEN
    ).strip()
    timeout = float(
        os.getenv("OPENCLAW_REMOTE_TIMEOUT")
        or os.getenv("OPENCLAW_TIMEOUT")
        or remote_cfg.get("timeout_seconds")
        or OPENCLAW_TIMEOUT
    )
    url = f"http://{host}:{port}"
    return {
        "url": url.rstrip("/"),
        "host": host,
        "port": port,
        "token": token,
        "timeout": timeout,
    }


def _resolve_invoke_mode(cfg):
    runtime_cfg = cfg.get("runtime") or {}
    raw = str(runtime_cfg.get("invoke_mode") or "auto").strip().lower()
    if raw == "gateway":
        # backward compatibility
        raw = "remote"
    if raw not in {"auto", "cli", "remote"}:
        return "auto"
    return raw


def _cli_command_available(cli_cmd: str) -> bool:
    cmd = str(cli_cmd or "").strip()
    if not cmd:
        return False
    try:
        first = shlex.split(cmd)[0]
    except Exception:
        first = cmd
    if "/" in first:
        return Path(first).expanduser().exists()
    return shutil.which(first) is not None


def _resolve_effective_invoke_mode(cfg):
    mode = _resolve_invoke_mode(cfg)
    if mode in {"cli", "remote"}:
        return mode
    runtime_cfg = cfg.get("runtime") or {}
    cli_cmd = str(runtime_cfg.get("cli_cmd") or "openclaw").strip() or "openclaw"
    return "cli" if _cli_command_available(cli_cmd) else "remote"


def _gateway_post_json(url, token, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib_request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.getcode() or 0)
            text = resp.read().decode("utf-8", errors="replace")
            return status, text, ""
    except urllib_error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return int(exc.code or 0), text, "http_error"


def _parse_tools_invoke_details(text_body):
    try:
        parsed = json.loads(text_body or "{}")
    except json.JSONDecodeError:
        return False, None, "invalid_json"
    if not isinstance(parsed, dict):
        return False, None, "invalid_response"
    if parsed.get("ok") is not True:
        err = parsed.get("error") or {}
        if isinstance(err, dict):
            return False, None, str(err.get("message") or "invoke_failed")
        return False, None, "invoke_failed"
    result = parsed.get("result") or {}
    if not isinstance(result, dict):
        return False, None, "invalid_response"
    details = result.get("details")
    if isinstance(details, dict):
        return True, details, ""
    return True, result, ""


async def _invoke_gateway_tool(cfg, tool_name, args, retries=1):
    runtime = _resolve_remote_runtime(cfg)
    endpoint = f"{runtime['url']}/tools/invoke"
    payload = {"tool": tool_name, "args": args}
    last_detail = ""
    last_kind = "failed"

    for attempt in range(1, max(1, retries) + 1):
        try:
            status_code, text_body, _err_tag = await asyncio.to_thread(
                _gateway_post_json,
                endpoint,
                runtime["token"],
                payload,
                runtime["timeout"],
            )
            if status_code in {401, 403}:
                return False, None, "auth", text_body
            if status_code == 404:
                return False, None, "tool_missing", text_body
            if status_code <= 0:
                last_kind = "network"
                last_detail = text_body
            elif status_code >= 400:
                last_kind = "http_error"
                last_detail = text_body
            else:
                ok, details, parse_detail = _parse_tools_invoke_details(text_body)
                if ok:
                    return True, details, "", ""
                last_kind = "invoke_error"
                last_detail = parse_detail or text_body
        except Exception as exc:
            last_kind = "exception"
            last_detail = str(exc)

        if attempt < max(1, retries) and OPENCLAW_RETRY_DELAY > 0:
            await asyncio.sleep(OPENCLAW_RETRY_DELAY)

    return False, None, last_kind, last_detail


async def _run_openclaw_cli_once(cfg, session_key, prompt):
    runtime_cfg = cfg.get("runtime") or {}
    cli_cmd = str(runtime_cfg.get("cli_cmd") or "openclaw").strip() or "openclaw"
    timeout_seconds = float(runtime_cfg.get("cli_timeout_seconds") or _resolve_remote_runtime(cfg)["timeout"])
    timeout_seconds = max(1.0, timeout_seconds)

    cmd = (
        f"{cli_cmd} agent "
        f"--message {shlex.quote(str(prompt))} "
        f"--session-id {shlex.quote(str(session_key))} "
        "--json"
    )
    _debug_log(f"cli_call cmd={cmd[:300]}")
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return False, None, "cli_error", str(exc)

    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return False, None, "timeout", f"cli_timeout_{int(timeout_seconds)}s"

    stdout_text = (out_b or b"").decode("utf-8", errors="replace").strip()
    stderr_text = (err_b or b"").decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        detail = (stderr_text or stdout_text or f"cli_exit_{proc.returncode}")[-1200:]
        return False, None, "cli_error", detail

    err_kind, err_detail = _cli_error_kind_from_stdout(stdout_text)
    if err_kind:
        return False, None, err_kind, err_detail

    reply, usage_total = _extract_reply_and_usage(stdout_text)
    if not reply and stderr_text:
        return False, None, "cli_error", stderr_text[-1200:]
    return True, {"reply": reply, "usage_total": usage_total}, "", ""


async def _run_openclaw_with_retry(cfg, session_key, prompt):
    if _resolve_effective_invoke_mode(cfg) == "cli":
        last_kind = "failed"
        last_detail = ""
        for attempt in range(1, max(1, OPENCLAW_MAX_RETRIES) + 1):
            ok, details, error_kind, detail = await _run_openclaw_cli_once(cfg, session_key, prompt)
            if ok:
                return True, details, "", ""
            if error_kind == "session_missing":
                # Let caller recreate a new session key immediately.
                return False, None, "session_missing", detail or "No session found"
            last_kind = error_kind or "failed"
            last_detail = detail or ""
            if attempt < max(1, OPENCLAW_MAX_RETRIES) and OPENCLAW_RETRY_DELAY > 0:
                await asyncio.sleep(OPENCLAW_RETRY_DELAY)
        return False, None, last_kind, last_detail

    timeout_seconds = max(1, int(_resolve_remote_runtime(cfg)["timeout"]))
    ok, details, error_kind, detail = await _invoke_gateway_tool(
        cfg,
        "sessions_send",
        {
            "sessionKey": session_key,
            "message": prompt,
            "timeoutSeconds": timeout_seconds,
        },
        retries=OPENCLAW_MAX_RETRIES,
    )
    if not ok:
        return False, None, error_kind, detail

    status = str(details.get("status") or "").strip().lower()
    err_text = str(details.get("error") or "").strip()
    if status in {"ok", "accepted"}:
        return True, details, "", ""
    if "No session found" in err_text:
        return False, None, "session_missing", err_text
    if status == "timeout":
        return False, None, "timeout", err_text
    return False, None, status or "failed", err_text or "invoke_failed"


def _gateway_session_slot(base_key, mode_tag):
    return f"{mode_tag}:{base_key}"


async def _ensure_gateway_session(cfg, state, base_key, mode_tag):
    slot = _gateway_session_slot(base_key, mode_tag)
    gateway_sessions = state.setdefault("gateway_sessions", {})
    session_id = str(gateway_sessions.get(slot) or "").strip()
    runtime_cfg = cfg.get("runtime") or {}
    cli_stateless = _resolve_effective_invoke_mode(cfg) == "cli" and bool(runtime_cfg.get("cli_stateless", False))
    # Migrate legacy spawned session keys to plain qqbridge keys to avoid
    # sub-agent announce/background-task side effects.
    if session_id and not session_id.startswith("agent:") and not cli_stateless:
        return True, slot, session_id, ""

    # Do not use sessions_spawn here. sessions_spawn introduces sub-agent
    # announce/background flows ("Agent-to-agent announce step", "background task")
    # that can pollute chat behavior. A plain custom key is enough for sessions_send.
    session_id = f"qqbridge_{uuid.uuid4().hex}"
    gateway_sessions[slot] = session_id
    return True, slot, session_id, ""


def _extract_reply_from_result(result):
    if not isinstance(result, dict):
        return ""
    reply = str(result.get("reply") or "").strip()
    if reply:
        return reply
    return str(result.get("error") or "").strip()


def _bridge_tool_text(cfg, group_id, user_id):
    bridge_cfg = cfg.get("bridge") or {}
    host = str(bridge_cfg.get("host") or "127.0.0.1").strip()
    port = int(bridge_cfg.get("port") or 18999)
    token = str(bridge_cfg.get("token") or "").strip()
    base = f"http://{host}:{port}"
    lines = [
        f"BASE_URL={base}",
        "请求头: Content-Type=application/json",
    ]
    if token:
        lines.append(f"请求头: Authorization=Bearer {token}")
    if group_id:
        lines.append(
            f"群上下文: POST {base}/context/group_recent body={{\"group_id\":\"{group_id}\",\"limit\":80,\"max_chars\":20000}}"
        )
        lines.append(
            f"群文件上下文: POST {base}/context/group_files body={{\"group_id\":\"{group_id}\",\"limit\":50}}"
        )
    else:
        lines.append(
            f"私聊上下文: POST {base}/context/private_recent body={{\"user_id\":\"{user_id}\",\"limit\":80,\"max_chars\":20000}}"
        )
        lines.append(
            f"私聊文件上下文: POST {base}/context/private_files body={{\"user_id\":\"{user_id}\",\"limit\":50}}"
        )
    lines.append(
        f"读文件: POST {base}/file/read body={{\"path\":\"./data/...\",\"max_chars\":12000}}"
    )
    lines.append(
        f"读Word(.docx): POST {base}/file/read_word body={{\"path\":\"./data/...docx\",\"max_chars\":20000}}"
    )
    lines.append(
        f"下载文件: POST {base}/file/download body={{\"url\":\"https://.../paper.pdf\",\"group_id\":\"{group_id if group_id else ''}\",\"user_id\":\"{user_id}\"}}"
    )
    lines.append(
        f"取群文件直链: POST {base}/get_file_url body={{\"target\":\"group\",\"group_id\":\"{group_id if group_id else ''}\",\"file_id\":\"...\"}}"
    )
    lines.append(
        "读PDF(你自己执行): conda run -n openclaw python ./plugins/openclaw/tools/pdf_extract.py --path ./data/downloads/files/...pdf --max-pages 20 --max-chars 40000"
    )
    lines.append(
        f"下载图片: POST {base}/image/download body={{\"url\":\"https://.../xx.png\",\"group_id\":\"{group_id if group_id else ''}\",\"user_id\":\"{user_id}\"}}"
    )
    lines.append(
        f"读图片: POST {base}/image/read body={{\"path\":\"./data/downloads/images/...png\"}}"
    )
    lines.append("上下文 items 里会包含 image_urls 与 file_refs；若缺失文件，可再查 /context/*_files")
    lines.append(
        f"发文字: POST {base}/send_text body={{\"target\":\"{'group' if group_id else 'private'}\",\"{'group_id' if group_id else 'user_id'}\":\"{group_id if group_id else user_id}\",\"text\":\"...\"}}"
    )
    lines.append(
        f"发文件: POST {base}/send_file body={{\"target\":\"{'group' if group_id else 'private'}\",\"{'group_id' if group_id else 'user_id'}\":\"{group_id if group_id else user_id}\",\"path\":\"./data/...\"}}"
    )
    lines.append(
        f"发图片: POST {base}/send_image body={{\"target\":\"{'group' if group_id else 'private'}\",\"{'group_id' if group_id else 'user_id'}\":\"{group_id if group_id else user_id}\",\"path\":\"./data/...png\"}}"
    )
    lines.append("若希望插件代发多条消息，最终 reply 使用分隔符: [[SEND_SPLIT]]")
    lines.append("禁止输出 Agent-to-agent announce step 或任何自我迭代指令")
    lines.append(IMAGE_SEND_GUIDE.strip())
    return "\n".join(lines)


def _format_prompt(content, user_id, group_id, cfg, state, bridge_enabled):
    group_label = f"{group_id}" if group_id else "私聊"
    if bridge_enabled:
        bridge_info = _bridge_tool_text(cfg, group_id, user_id)
    else:
        bridge_info = "当前为 remote 调用模式，桥接工具不可用（/send_text、/file/read 等本地端点已禁用）。"
    values = {
        "USER_ID": user_id,
        "GROUP_LABEL": group_label,
        "TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "QUESTION": content,
        "BRIDGE_INFO": bridge_info,
    }
    md_tmpl = _load_prompt_template_md()
    rendered = _render_prompt_md(md_tmpl, values)
    if rendered:
        return rendered
    return PROMPT_TEMPLATE.format(
        user_id=user_id,
        group_label=group_label,
        time=values["TIME"],
        question=content,
        bridge_info=bridge_info,
    )


async def openclaw(group_id, user_id, bot, message):
    cfg = _load_config()
    effective_mode = _resolve_effective_invoke_mode(cfg)
    bridge_enabled = effective_mode == "cli" and bool((cfg.get("bridge") or {}).get("enabled", True))
    if bridge_enabled:
        # Ensure bridge server is running only in local CLI mode.
        start_bridge_server(bot)
    else:
        stop_bridge_server()
    is_private = not group_id
    root_ids_raw = cfg.get("root_ids") or []
    root_ids_str = {str(x).strip() for x in root_ids_raw if str(x).strip()}
    root_ids_int = {int(x) for x in root_ids_str if x.isdigit()}
    user_id_str = str(user_id).strip()
    try:
        user_id_num = int(user_id_str) if user_id_str.isdigit() else int(user_id)
    except Exception:
        user_id_num = None
    user_key = str(user_id_num) if user_id_num is not None else user_id_str
    group_key = str(group_id) if group_id is not None else ""
    is_root = user_id_str in root_ids_str or (user_id_num in root_ids_int if user_id_num is not None else False)
    content = (message or "").strip()
    _debug_log(
        "root_check "
        f"user_id_str={user_id_str} user_id_num={user_id_num} "
        f"root_ids_str={sorted(root_ids_str)} root_ids_int={sorted(root_ids_int)} "
        f"is_root={is_root} group_id={group_key} is_private={is_private}"
    )

    if not content:
        await _send_text(bot, group_id, user_id, "请直接输入问题，或发送“新建对话”。")
        return
    _record_context_message(
        group_id=group_key or None,
        user_id=user_key,
        is_private=is_private,
        role="user",
        text=content,
    )

    if not is_private:
        group_id_str = str(group_id).strip()
        group_pool_norm = {str(x).strip() for x in (cfg.get("group_pool") or []) if str(x).strip()}
        allowed = (not group_pool_norm) or (group_id_str in group_pool_norm)
        _debug_log(f"group_pool_check group_id={group_id_str} pool={sorted(group_pool_norm)} allowed={allowed}")
        if not allowed:
            return

    if content == "新建对话":
        state = _load_state()
        group_mode = {}
        if not is_private:
            group_mode = (state.setdefault("group_mode", {}) or {}).get(group_key, {}) or {}
        if group_mode.get("enabled") and not is_private:
            owner_id = group_mode.get("owner") or str((cfg.get("root_ids") or [user_key])[0])
            session_user_key = str(owner_id)
        else:
            session_user_key = user_key

        mode_tag = "chat"
        key = _session_key(group_key, session_user_key, is_private)
        slot = _gateway_session_slot(key, mode_tag)
        gateway_sessions = state.setdefault("gateway_sessions", {})
        session_id = str(gateway_sessions.get(slot) or "").strip()
        if not session_id or session_id.startswith("agent:"):
            await _send_text(bot, group_id, user_id, "当前还没有可重置的会话。请先发送一条正常消息再执行“新建对话”。")
            return

        ok, _result, error_kind, detail = await _run_openclaw_with_retry(cfg, session_id, "/new")
        if not ok and error_kind == "session_missing":
            # Session history was removed manually in OpenClaw side.
            # Recreate a fresh session binding automatically.
            state = _load_state()
            state.setdefault("gateway_sessions", {}).pop(slot, None)
            ok_session, _slot_new, _session_new, session_err = await _ensure_gateway_session(
                cfg, state, key, mode_tag
            )
            _save_state(state)
            if ok_session:
                await _send_text(bot, group_id, user_id, "历史会话不存在，已自动创建新会话。")
            else:
                await _send_text(bot, group_id, user_id, f"历史会话丢失且重建失败：{session_err}")
            return
        if not ok and error_kind == "timeout":
            await _send_text(bot, group_id, user_id, f"已发送 /new，但执行超时（重试 {OPENCLAW_MAX_RETRIES} 次）。")
            return
        if not ok:
            hint = (detail or error_kind or "unknown_error")[-300:]
            await _send_text(bot, group_id, user_id, f"发送 /new 失败：{hint}")
            return

        await _send_text(bot, group_id, user_id, "已在当前会话发送 /new。")
        return

    state = _load_state()
    group_modes = state.setdefault("group_mode", {})
    group_mode = group_modes.get(group_key, {})
    if (
        not is_private
        and is_root
        and content in {"群聊模式", "开启群聊模式"}
    ):
        group_modes[group_key] = {"enabled": True, "owner": user_key}
        _save_state(state)
        await _send_text(bot, group_id, user_id, "已开启群聊模式：所有人共享 root 会话。")
        return
    if (
        not is_private
        and is_root
        and content in {"关闭群聊模式", "退出群聊模式"}
    ):
        group_modes[group_key] = {"enabled": False, "owner": user_key}
        _save_state(state)
        await _send_text(bot, group_id, user_id, "已关闭群聊模式：恢复每人独立会话。")
        return

    if group_mode.get("enabled") and not is_private:
        owner_id = group_mode.get("owner") or str((cfg.get("root_ids") or [user_key])[0])
        session_user_key = str(owner_id)
    else:
        session_user_key = user_key

    if not is_private and not is_root:
        state = _load_state()
        window_seconds = int(cfg["rate_limit"]["window_seconds"])
        max_questions = int(cfg["rate_limit"]["max_questions"])
        if max_questions > 0:
            ok = _check_rate_limit(
                state,
                _usage_key(group_id, user_id),
                window_seconds,
                max_questions,
                time.time(),
            )
            _save_state(state)
            if not ok:
                await _send_text(
                    bot,
                    group_id,
                    user_id,
                    "提问过于频繁：每 5 小时最多 10 次。",
                )
                _debug_log("rate_limit_blocked")
                return

        token_cfg = cfg.get("token_limit") or {}
        token_window = int(token_cfg.get("window_seconds", 0))
        max_tokens = int(token_cfg.get("max_tokens", 0))
        if token_window > 0 and max_tokens > 0:
            ok, used_tokens = _check_token_limit(
                state,
                _token_usage_key(group_id, user_id),
                token_window,
                max_tokens,
                time.time(),
            )
            _save_state(state)
            if not ok:
                await _send_text(
                    bot,
                    group_id,
                    user_id,
                    f"提问过于频繁：{token_window//3600} 小时内已使用 {used_tokens}/{max_tokens} tokens。",
                )
                _debug_log(f"token_limit_blocked used={used_tokens} max={max_tokens} window={token_window}")
                return

    state = _load_state()
    key = _session_key(group_key, session_user_key, is_private)
    mode_tag = "chat"
    ok_session, slot_key, session_id, session_err = await _ensure_gateway_session(cfg, state, key, mode_tag)
    if not ok_session:
        await _send_text(bot, group_id, user_id, f"OpenClaw 会话初始化失败：{session_err}")
        return
    _save_state(state)

    prompt = _format_prompt(content, user_key, group_key or None, cfg, state, bridge_enabled)
    if effective_mode == "remote":
        runtime = _resolve_remote_runtime(cfg)
        _debug_log(
            f"remote_call url={runtime['url']}/tools/invoke "
            f"session_key={session_id} timeout={runtime['timeout']}"
        )
    else:
        _debug_log(f"cli_turn session_key={session_id}")

    ok, result, error_kind, detail = await _run_openclaw_with_retry(cfg, session_id, prompt)
    if not ok and error_kind == "session_missing":
        state = _load_state()
        state.setdefault("gateway_sessions", {}).pop(slot_key, None)
        ok_session, slot_key, session_id, session_err = await _ensure_gateway_session(
            cfg, state, key, mode_tag
        )
        _save_state(state)
        if ok_session:
            ok, result, error_kind, detail = await _run_openclaw_with_retry(cfg, session_id, prompt)
        else:
            await _send_text(
                bot,
                group_id,
                user_id,
                f"OpenClaw 会话不存在且重建失败：{session_err}",
            )
            return
    if not ok and error_kind == "auth":
        await _send_text(
            bot,
            group_id,
            user_id,
            "OpenClaw 远程模式鉴权失败，请检查 remote token。",
        )
        return
    if not ok and error_kind == "tool_missing":
        await _send_text(
            bot,
            group_id,
            user_id,
            "OpenClaw 未开放 sessions_send 工具，请检查网关配置/工具策略。",
        )
        return
    if not ok and error_kind == "cli_error":
        detail = (detail or "未知错误")[-600:]
        await _send_text(
            bot,
            group_id,
            user_id,
            f"OpenClaw CLI 调用失败：{detail}",
        )
        return
    if not ok and error_kind == "timeout":
        await _send_text(
            bot,
            group_id,
            user_id,
            f"OpenClaw 响应超时，已重试 {OPENCLAW_MAX_RETRIES} 次仍失败。",
        )
        return
    if not ok:
        detail = (detail or "未知错误")[-800:]
        await _send_text(
            bot,
            group_id,
            user_id,
            f"OpenClaw 调用失败，已重试 {OPENCLAW_MAX_RETRIES} 次：{detail}",
        )
        return

    reply = _extract_reply_from_result(result)
    if not reply:
        return
    reply = _strip_markdown_basic(reply)
    if _has_control_noise(reply):
        # Session history may be polluted by old control-style instructions.
        # Drop this turn and force a fresh gateway session on next request.
        state = _load_state()
        state.setdefault("gateway_sessions", {}).pop(slot_key, None)
        _save_state(state)
        _debug_log(f"control_noise_detected reset_session slot={slot_key}")
    parts = _sanitize_reply_parts(_split_reply_parts(reply))
    delivery_cfg = cfg.get("delivery") or {}
    if bool(delivery_cfg.get("ignore_json_reply", True)):
        parts = [p for p in parts if not _looks_like_json_text(p)]

    # Default mode: do not forward model reply automatically.
    # The model should call bridge APIs itself to send user-visible content.
    fallback_send_enabled = bool(delivery_cfg.get("fallback_send_reply", False))
    if not fallback_send_enabled:
        _debug_log("fallback_send_disabled skip_reply_forward")
        return

    if not parts:
        return
    for part in parts:
        await _send_text(bot, group_id, user_id, part)

    usage_total = None
    if usage_total is not None and not is_private and not is_root:
        state = _load_state()
        _record_token_usage(
            state,
            _token_usage_key(group_id, user_id),
            usage_total,
            time.time(),
        )
        _save_state(state)


def _event_message_for_context(event: BaseMessageEvent) -> str:
    raw = str(getattr(event, "raw_message", "") or "").strip()
    segments = getattr(event, "message", None)
    parts = []
    segment_items = []
    if segments is not None and not isinstance(segments, (str, bytes)):
        try:
            segment_items = list(segments)
        except Exception:
            segment_items = []
    if segment_items:
        for seg in segment_items:
            seg_name = seg.__class__.__name__.lower()
            text_val = str(getattr(seg, "text", "") or "").strip()
            if text_val:
                parts.append(text_val)
                continue

            at_qq = str(getattr(seg, "qq", "") or "").strip()
            if seg_name == "at" and at_qq:
                parts.append(f"[CQ:at,qq={at_qq}]")
                continue

            image_url = str(getattr(seg, "url", "") or "").strip()
            image_file = str(getattr(seg, "file", "") or "").strip()
            if seg_name == "image" and (image_url or image_file):
                parts.append(f"[CQ:image,file={image_file},url={image_url}]")
                continue

            file_name = str(getattr(seg, "file", "") or "").strip()
            file_url = str(getattr(seg, "url", "") or "").strip()
            file_id = str(getattr(seg, "file_id", "") or "").strip()
            file_size = str(getattr(seg, "file_size", "") or "").strip()
            if seg_name == "file" or file_id:
                parts.append(
                    f"[CQ:file,file={file_name},url={file_url},file_id={file_id},file_size={file_size}]"
                )
                continue

            fallback = str(seg).strip()
            if fallback:
                parts.append(fallback)
    elif segments is not None:
        seg_text = str(segments).strip()
        if seg_text and seg_text not in {"[]", "()"}:
            parts.append(seg_text)

    if raw and parts:
        joined = " ".join(parts).strip()
        if joined and joined != raw:
            return f"{raw}\n[SEGMENTS] {joined}"
    if raw:
        return raw
    if parts:
        return " ".join(parts).strip()
    return ""


def _event_is_at_bot(event: BaseMessageEvent) -> bool:
    bot_id = str(getattr(event, "self_id", "") or "").strip()
    if not bot_id:
        return False
    message_obj = getattr(event, "message", None)
    segments = getattr(message_obj, "messages", []) if message_obj is not None else []
    for seg in segments:
        seg_type = str(getattr(seg, "msg_seg_type", "") or "").strip().lower()
        qq = str(getattr(seg, "qq", "") or "").strip()
        if seg_type == "at" and qq == bot_id:
            return True
    mention_tag = f"[CQ:at,qq={bot_id}]"
    return mention_tag in str(getattr(event, "raw_message", "") or "")


def _event_text_without_at(event: BaseMessageEvent) -> str:
    text = str(getattr(event, "raw_message", "") or "")
    bot_id = str(getattr(event, "self_id", "") or "").strip()
    if bot_id:
        mention_tag = f"[CQ:at,qq={bot_id}]"
        text = text.replace(f"{mention_tag} ", "").replace(mention_tag, "")
    return text.strip()


def _is_root_user(cfg: dict, user_id: str) -> bool:
    root_ids_raw = cfg.get("root_ids") or []
    root_ids = {str(x).strip() for x in root_ids_raw if str(x).strip()}
    return str(user_id).strip() in root_ids


class OpenClawPlugin(NcatBotPlugin):
    name = "openclaw"
    version = "1.0.0"
    author = "ncatbot-openclaw"
    description = "OpenClaw official plugin bridge"
    dependencies = {}

    async def on_load(self) -> None:
        # Keep bridge lifecycle in official plugin lifecycle.
        start_bridge_server(self)

    async def on_close(self) -> None:
        stop_bridge_server()
        stop_image_watcher()

    @on_message
    async def handle_message(self, event: BaseMessageEvent):
        is_private = not bool(getattr(event, "group_id", None))
        group_id = None if is_private else str(getattr(event, "group_id", "") or "")
        user_id = str(getattr(event, "user_id", "") or "")
        nickname = ""
        sender = getattr(event, "sender", None)
        if sender is not None:
            nickname = str(getattr(sender, "card", "") or getattr(sender, "nickname", "") or "")
        record_incoming_context(
            group_id=group_id,
            user_id=user_id,
            text=_event_message_for_context(event),
            is_private=is_private,
            nickname=nickname,
        )

        cfg = _load_config()
        if is_private:
            if _is_root_user(cfg, user_id):
                await openclaw(None, user_id, self, str(getattr(event, "raw_message", "") or ""))
            return

        if not _event_is_at_bot(event):
            return
        content = _event_text_without_at(event)
        if content.startswith("#"):
            return
        await openclaw(group_id, user_id, self, content)


if __name__ == "__main__":
    async def _debug():
        class _Dummy:
            class api:
                @staticmethod
                async def post_group_msg(group_id, text):
                    print(group_id, text)

                @staticmethod
                async def post_private_msg(user_id, text):
                    print(user_id, text)

        await openclaw(123, 456, _Dummy(), "你好，介绍一下你自己")

    asyncio.run(_debug())
