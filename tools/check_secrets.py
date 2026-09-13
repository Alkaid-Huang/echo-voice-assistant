"""
提交前密钥扫描：检查受版本控制的文件里是否出现明文 API Key。

用法（产品仓库根目录）：
    python tools/check_secrets.py

为什么需要它：`.env` 已被 .gitignore 忽略，但模板文件、文档、脚本都可能被误填真实 Key
（本项目就发生过一次，见 docs/真实Bug笔记.md 的事故记录 #1）。
"""
import re
import subprocess
from pathlib import Path

SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9]{20,}")
SKIP_DIRS = {".venv", "models", "outputs", ".git", "__pycache__", ".pytest_cache"}


def tracked_files() -> list:
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def main() -> None:
    hits = []
    for path in tracked_files():
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if SECRET_PATTERN.search(text):
            hits.append(str(path))

    if hits:
        print("[!] 检测到疑似明文 API Key，提交前请移除：")
        for item in hits:
            print("   " + item)
        raise SystemExit(1)
    print("[OK] 未检测到明文密钥")


if __name__ == "__main__":
    main()
