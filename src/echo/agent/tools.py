"""
Agent 工具集 —— 让 Echo 能"动手做事"，而不只是聊天

设计对照 OpenAI function calling 规范：
- ToolSpec：名字 + 说明 + JSON Schema 参数 + 实现函数
- ToolRegistry：统一注册、导出模型可读的工具描述、按名调用并兜底异常

工具执行失败不抛异常，而是把可读错误当结果回给模型（模型可以据此重试或解释）。
"""
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..memory.facts import FactStore


@dataclass
class ToolSpec:
    """一个可被模型调用的工具"""

    name: str
    description: str
    parameters: dict
    func: Callable[..., str]

    def to_openai(self) -> dict:
        """转成 OpenAI / DeepSeek 兼容的 tools 描述"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：注册、导出、调用"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self, enabled: Optional[List[str]] = None) -> List[dict]:
        """导出给模型看的工具描述；enabled 为 None 表示全部"""
        names = enabled if enabled is not None else self.names()
        return [self._tools[n].to_openai() for n in names if n in self._tools]

    def call(self, name: str, arguments) -> str:
        """执行工具；任何异常都转成可读文本，避免 Agent 因工具报错而中断"""
        spec = self._tools.get(name)
        if spec is None:
            return f"[工具错误] 未注册的工具：{name}"
        try:
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments.strip() else {}
            else:
                args = arguments or {}
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            return f"[工具错误] 参数不是合法 JSON：{arguments}"
        try:
            return str(spec.func(**args))
        except TypeError as e:
            return f"[工具错误] 参数不匹配：{e}"
        except Exception as e:  # 工具内部异常不该拖垮对话
            return f"[工具错误] 执行失败：{e}"


def build_default_tools(
    facts: Optional[FactStore] = None, enabled: Optional[List[str]] = None
) -> ToolRegistry:
    """内置工具：时间、天气、记住/回忆"""
    registry = ToolRegistry()

    def get_current_time(timezone: str = "Asia/Shanghai") -> str:
        from datetime import datetime

        try:
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo(timezone))
        except Exception:
            now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S") + f"（{timezone}）"

    def get_weather(city: str) -> str:
        """Open-Meteo 免费接口（无需 API Key）：先地理编码，再查当前天气"""
        import requests

        try:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "zh"},
                timeout=8,
            ).json()
            results = geo.get("results") or []
            if not results:
                return f"没找到城市：{city}"
            place = results[0]
            weather = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current_weather": True,
                },
                timeout=8,
            ).json()
            current = weather.get("current_weather") or {}
            return (
                f"{place.get('name', city)} 当前 {current.get('temperature')}℃，"
                f"风速 {current.get('windspeed')} km/h"
            )
        except Exception as e:
            return f"[工具错误] 天气查询失败：{e}"

    def remember(fact: str) -> str:
        if facts is None:
            return "[工具错误] 记忆库未启用"
        facts.add(fact)
        return f"已记住：{fact}"

    def recall(query: str) -> str:
        if facts is None:
            return "[工具错误] 记忆库未启用"
        found = facts.search(query)
        return "；".join(found) if found else "没有找到相关记忆"

    specs = [
        ToolSpec(
            name="get_current_time",
            description="查询当前日期和时间",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区名，如 Asia/Shanghai",
                    }
                },
            },
            func=get_current_time,
        ),
        ToolSpec(
            name="get_weather",
            description="查询某个城市的当前天气",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京"}
                },
                "required": ["city"],
            },
            func=get_weather,
        ),
        ToolSpec(
            name="remember",
            description="记住一条关于用户的事实，供以后回忆",
            parameters={
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "要记住的事实"}
                },
                "required": ["fact"],
            },
            func=remember,
        ),
        ToolSpec(
            name="recall",
            description="回忆之前记住的与关键词相关的事实",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词"}
                },
                "required": ["query"],
            },
            func=recall,
        ),
    ]

    for spec in specs:
        if enabled is None or spec.name in enabled:
            registry.register(spec)
    return registry
