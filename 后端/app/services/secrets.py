"""敏感设置项加密与脱敏。

设计目标：
- 设置文件（agent_settings.json）中的 API Key 不再以明文落盘
- 密钥优先级：环境变量 AGENT_SETTINGS_KEY > 本地密钥文件 .agent_secret_key
- 使用 Fernet（AES128-CBC + HMAC）做对称加密；密钥文件权限尽量收紧
- cryptography 不可用时优雅降级为明文（只记录警告，不阻塞主流程）

注意：本地密钥文件与设置文件在同一台机器上，属于"防误提交/防随手查看"级别的保护，
不是对抗本机攻击者的强隔离；生产环境建议改用系统 keyring 或外部 Secret Manager。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# app/services/secrets.py -> 后端根目录
BACKEND_DIR = Path(__file__).resolve().parents[2]

ENCRYPTED_PREFIX = "enc:v1:"
SENSITIVE_TOP_KEYS = {
    "openai_api_key",
    "dashscope_api_key",
    "deepseek_api_key",
    "tavily_api_key",
    "api_key",
}
SENSITIVE_NESTED_KEYS = {"model_api_keys"}

_fernet_cache: Any = None
_fernet_ready = False


def restrict_file_permissions(path: Path) -> None:
    """尽量把敏感文件权限收紧到当前用户（Windows 用 icacls，POSIX 用 chmod 600）。"""
    try:
        if os.name == "nt":
            user = os.environ.get("USERNAME") or ""
            if not user:
                return
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
        else:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


def _key_file() -> Path:
    raw = os.environ.get("AGENT_SECRET_KEY_FILE", "").strip()
    return Path(raw) if raw else BACKEND_DIR / ".agent_secret_key"


def _load_fernet() -> Any:
    """懒加载 Fernet；密钥来自环境变量或本地文件，首次使用时生成。"""
    global _fernet_cache, _fernet_ready
    if _fernet_ready:
        return _fernet_cache
    _fernet_ready = True
    try:
        from cryptography.fernet import Fernet
    except Exception as e:
        logger.warning("cryptography 不可用，设置文件将保持明文: %s", e)
        return None

    env_key = os.environ.get("AGENT_SETTINGS_KEY", "").strip()
    if env_key:
        try:
            _fernet_cache = Fernet(env_key.encode("utf-8"))
            return _fernet_cache
        except Exception as e:
            logger.warning("AGENT_SETTINGS_KEY 无效，忽略该环境变量: %s", e)

    path = _key_file()
    try:
        if path.exists():
            key = path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(key)
            restrict_file_permissions(path)
            logger.info("已生成本地设置加密密钥: %s", path)
        _fernet_cache = Fernet(key)
    except Exception as e:
        logger.warning("设置加密密钥加载失败，降级明文: %s", e)
        _fernet_cache = None
    return _fernet_cache


def encrypt_value(value: str) -> str:
    """加密单个字符串；空值/已加密值原样返回。"""
    if not isinstance(value, str) or not value or value.startswith(ENCRYPTED_PREFIX):
        return value
    fernet = _load_fernet()
    if fernet is None:
        return value
    try:
        token = fernet.encrypt(value.encode("utf-8")).decode("ascii")
        return ENCRYPTED_PREFIX + token
    except Exception as e:
        logger.warning("敏感值加密失败（保持明文）: %s", e)
        return value


def decrypt_value(value: str) -> str:
    """解密单个字符串；非加密值原样返回。"""
    if not isinstance(value, str) or not value.startswith(ENCRYPTED_PREFIX):
        return value
    fernet = _load_fernet()
    if fernet is None:
        logger.error("敏感值解密失败：cryptography 未安装或密钥未加载，所有加密值将返回空串")
        return ""
    try:
        return fernet.decrypt(value[len(ENCRYPTED_PREFIX) :].encode("ascii")).decode("utf-8")
    except Exception as e:
        logger.error("敏感值解密失败（密钥可能已更换，所有 API Key 需重新配置）: %s", e)
        return ""


def is_encrypted(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX)


def redact_secret(value: str, *, keep: int = 4) -> str:
    """日志/展示用脱敏：只保留末尾若干位。"""
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return "*" * max(4, len(value) - keep) + value[-keep:]


def encrypt_settings(data: dict[str, Any]) -> dict[str, Any]:
    """返回加密后的设置副本（不修改入参）。"""
    out = copy.deepcopy(data)
    for key in SENSITIVE_TOP_KEYS:
        if isinstance(out.get(key), str) and out[key]:
            out[key] = encrypt_value(out[key])
    for key in SENSITIVE_NESTED_KEYS:
        nested = out.get(key)
        if isinstance(nested, dict):
            out[key] = {
                str(k): encrypt_value(v) if isinstance(v, str) else v
                for k, v in nested.items()
            }
    return out


def decrypt_settings(data: dict[str, Any]) -> dict[str, Any]:
    """返回解密后的设置副本（不修改入参）。"""
    out = copy.deepcopy(data)
    for key in SENSITIVE_TOP_KEYS:
        if is_encrypted(out.get(key)):
            out[key] = decrypt_value(out[key])
    for key in SENSITIVE_NESTED_KEYS:
        nested = out.get(key)
        if isinstance(nested, dict):
            out[key] = {
                str(k): decrypt_value(v) if is_encrypted(v) else v
                for k, v in nested.items()
            }
    return out


def merge_preserving_unreadable(data: dict[str, Any], raw: dict[str, Any]) -> None:
    """保存前用磁盘上的原密文补齐解密失败的敏感项（原地修改 data）。

    密钥文件丢失/更换后，``decrypt_settings`` 会把无法解密的 Key 变成空串；
    若此时保存任意设置（读改写），空串会重新加密落盘，等于把 Key 永久清空。
    """
    for key in SENSITIVE_TOP_KEYS:
        incoming = data.get(key)
        original = raw.get(key)
        if incoming in (None, "") and is_encrypted(original):
            data[key] = original
            logger.warning(
                "设置项 %s 无法解密，保存时保留原密文（如需更新请重新填写）", key
            )
    for key in SENSITIVE_NESTED_KEYS:
        original = raw.get(key)
        if not isinstance(original, dict):
            continue
        incoming = data.get(key)
        if not isinstance(incoming, dict):
            incoming = {}
            data[key] = incoming
        for sub, value in original.items():
            if is_encrypted(value) and incoming.get(sub) in (None, ""):
                incoming[sub] = value
                logger.warning(
                    "设置项 %s.%s 无法解密，保存时保留原密文（如需更新请重新填写）", key, sub
                )


def has_plaintext_secrets(data: dict[str, Any]) -> bool:
    """判断设置里是否仍有明文敏感项（用于启动时一次迁移）。"""
    for key in SENSITIVE_TOP_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value and not is_encrypted(value):
            return True
    for key in SENSITIVE_NESTED_KEYS:
        nested = data.get(key)
        if isinstance(nested, dict):
            for value in nested.values():
                if isinstance(value, str) and value and not is_encrypted(value):
                    return True
    return False


def dump_for_debug(data: dict[str, Any]) -> str:
    """安全调试输出：敏感项脱敏后序列化。"""
    safe = copy.deepcopy(data)
    for key in SENSITIVE_TOP_KEYS:
        if isinstance(safe.get(key), str):
            safe[key] = redact_secret(safe[key])
    for key in SENSITIVE_NESTED_KEYS:
        nested = safe.get(key)
        if isinstance(nested, dict):
            safe[key] = {
                k: redact_secret(v) if isinstance(v, str) else v
                for k, v in nested.items()
            }
    return json.dumps(safe, ensure_ascii=False)
