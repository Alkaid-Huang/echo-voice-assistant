"""
极简 .env 加载器（不引入额外依赖）

目的：API Key 这类敏感信息写在本地的 .env 文件里（已被 .gitignore 忽略），
不进入版本库；同时保留"环境变量优先"的语义，方便 CI / 服务器注入。
"""
import os
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """把 .env 里的键值对写入环境变量；已存在的环境变量不会被覆盖"""
    env_file = Path(path)
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
