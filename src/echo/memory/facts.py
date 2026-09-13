"""
FactStore —— 长期记忆（简化版）

把用户的事实记成条目存 JSON，检索用关键词重合度打分。
这是"记住你"的最小可用版本；后续可换成向量检索（M3+）。
"""
import json
import os
import time
from typing import List, Optional


class FactStore:
    """事实条目存储 + 关键词检索"""

    def __init__(self, path: Optional[str] = None, max_facts: int = 200):
        self.path = path
        self.max_facts = max_facts
        self._facts: List[dict] = []
        if path:
            self.load()

    def add(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        for item in self._facts:  # 去重
            if item["fact"] == fact:
                return
        self._facts.append({"fact": fact, "ts": time.time()})
        self._facts = self._facts[-self.max_facts :]
        self.save()

    def search(self, query: str, limit: int = 3) -> List[str]:
        query = (query or "").strip()
        if not query:
            return [item["fact"] for item in self._facts[-limit:]]
        keywords = [k for k in query.replace("，", " ").replace(",", " ").split() if k]
        scored = []
        for item in self._facts:
            text = item["fact"]
            score = sum(1 for k in keywords if k in text)
            if score == 0 and query in text:
                score = 1
            if score:
                scored.append((score, item["ts"], text))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [text for _, _, text in scored[:limit]]

    def all(self) -> List[str]:
        return [item["fact"] for item in self._facts]

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._facts, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._facts = [item for item in data if isinstance(item, dict) and "fact" in item]
        except (json.JSONDecodeError, TypeError):
            self._facts = []
