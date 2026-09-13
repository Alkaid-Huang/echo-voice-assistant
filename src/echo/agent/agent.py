"""
Agent —— Echo 的大脑：工具调用循环 + 记忆 + 情绪输出

一轮处理的流程：
    用户文本 → 模型（带工具清单）→ 若请求工具：执行 → 结果回填 → 再问模型
                     ↓ 直到模型给出最终回答
    最终回答 = JSON：{"reply": "...", "emotion": "happy"}
                reply 交给 TTS 朗读，emotion 交给 Live2D 做表情

事件回调 on_event(kind, payload) 供上层（管线/WebSocket）转发：
    thinking / tool_call / tool_result / speech / emotion / error
"""
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..llm.llm_interface import LLMInterface
from ..memory.chat_history import ChatHistory
from ..memory.facts import FactStore
from .tools import ToolRegistry

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "shy"]

_EMOTION_KEYWORDS = {
    "happy": ["开心", "高兴", "太好了", "哈哈", "棒", "喜欢"],
    "sad": ["抱歉", "难过", "遗憾", "可惜"],
    "angry": ["生气", "讨厌", "烦"],
    "surprised": ["居然", "竟然", "哇", "真的吗"],
    "shy": ["害羞", "不好意思", "别夸"],
}


@dataclass
class AgentReply:
    """一次 Agent 应答的结果"""

    text: str
    emotion: str = "neutral"
    tool_calls: List[dict] = field(default_factory=list)


def guess_emotion(text: str) -> str:
    """没拿到结构化情绪时，用关键词兜底（保证 Live2D 总有表情可用）"""
    for emotion, words in _EMOTION_KEYWORDS.items():
        if any(w in text for w in words):
            return emotion
    return "neutral"


def parse_agent_output(raw: str) -> tuple:
    """
    解析模型输出 → (回复文本, 情绪)。

    优先按 JSON 解析；解析不出来就把整段当文本，用关键词猜情绪。
    这样即使模型不听话（没按格式输出），链路也不会崩。
    """
    text = (raw or "").strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict) and data.get("reply"):
        emotion = str(data.get("emotion", "neutral")).strip().lower()
        return str(data["reply"]).strip(), (emotion if emotion in EMOTIONS else "neutral")
    return text, guess_emotion(text)


class Agent:
    """带工具调用与情绪输出的对话大脑"""

    def __init__(
        self,
        llm: LLMInterface,
        tools: ToolRegistry,
        history: Optional[ChatHistory] = None,
        facts: Optional[FactStore] = None,
        persona: str = "你是 Echo，一个住在桌面上的 AI 伴侣，说话口语化、简短、有温度。",
        max_iterations: int = 3,
        enabled_tools: Optional[List[str]] = None,
        on_event: Optional[Callable[[str, object], None]] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.history = history
        self.facts = facts
        self.persona = persona
        self.max_iterations = max_iterations
        self.enabled_tools = enabled_tools
        self.on_event = on_event or (lambda kind, payload=None: None)

    # ── 事件 ──
    def _emit(self, kind: str, payload=None) -> None:
        try:
            self.on_event(kind, payload)
        except Exception:
            pass

    # ── 提示词 ──
    def _system_prompt(self) -> str:
        tool_names = ", ".join(self.tools.names())
        return (
            f"{self.persona}\n"
            f"你可以使用这些工具：{tool_names or '无'}。需要事实信息（时间、天气、用户偏好）时先调用工具。\n"
            "【输出格式】最终回答必须只输出一个 JSON，不要多余文字：\n"
            '{"reply": "要对用户说的话", "emotion": "neutral|happy|sad|angry|surprised|shy"}\n'
            "emotion 用于驱动你的虚拟形象表情，请按这句话的情绪选择。"
        )

    def build_messages(self, user_text: str) -> List[dict]:
        messages = [{"role": "system", "content": self._system_prompt()}]
        if self.history is not None:
            messages.extend(self.history.messages())
        if self.facts is not None and self.facts.all():
            # 把长期记忆作为背景注入（数量少时直接全给）
            memories = "；".join(self.facts.all()[-10:])
            messages.append({"role": "system", "content": f"你记得这些关于用户的事：{memories}"})
        messages.append({"role": "user", "content": user_text})
        return messages

    # ── 主循环 ──
    async def respond(self, user_text: str) -> AgentReply:
        """处理一轮用户输入，必要时调用工具，返回 (回复文本, 情绪, 工具轨迹)"""
        messages = self.build_messages(user_text)
        tool_trace: List[dict] = []

        for step in range(self.max_iterations):
            self._emit("thinking", {"step": step + 1})
            response = await self.llm.async_chat_with_tools(
                messages, self.tools.specs(self.enabled_tools)
            )

            if not response.tool_calls:
                text, emotion = parse_agent_output(response.content)
                if self.history is not None:
                    self.history.append("user", user_text)
                    self.history.append("assistant", text)
                    self.history.save()
                self._emit("speech", {"text": text, "tools": len(tool_trace)})
                self._emit("emotion", {"emotion": emotion})
                return AgentReply(text=text, emotion=emotion, tool_calls=tool_trace)

            # 模型请求调用工具：补齐 assistant 消息（OpenAI 协议要求），再逐个执行
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                self._emit("tool_call", {"name": call.name, "arguments": call.arguments})
                result = self.tools.call(call.name, call.arguments)
                self._emit("tool_result", {"name": call.name, "result": result})
                tool_trace.append(
                    {"name": call.name, "arguments": call.arguments, "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        # 超过迭代上限：给一个可读兜底，不要让对话卡死
        fallback = "（我查资料查得有点绕，你能再问一次吗？）"
        self._emit("error", "工具调用超过最大轮次")
        self._emit("speech", {"text": fallback, "tools": len(tool_trace)})
        self._emit("emotion", {"emotion": "neutral"})
        return AgentReply(text=fallback, emotion="neutral", tool_calls=tool_trace)
