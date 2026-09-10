"""
ChatHistory —— 多轮对话上下文 + JSON 持久化

对照 Open-LLM-VTuber 的 ChatHistoryManager 思路（简化版）：
只保留最近 N 轮，避免上下文无限增长；退出后可恢复。
"""
import json
import os
from collections import deque
from typing import List, Optional


class ChatHistory:
    """定长对话历史（按消息条数裁剪）"""

    def __init__(self, path: Optional[str] = None, max_messages: int = 12):
        self.path = path
        self.max_messages = max_messages
        self._messages: deque = deque(maxlen=max_messages)
        if path:
            self.load()

    def append(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def messages(self) -> List[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.messages(), f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data[-self.max_messages :]:
                self._messages.append(
                    {"role": item["role"], "content": item["content"]}
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            # 历史文件损坏时不让程序起不来，直接当空历史
            self._messages.clear()
