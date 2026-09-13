"""
Agent 大脑测试：工具注册与调用、工具调用循环、情绪解析、长期记忆

全部使用脚本化假模型，不联网、不消耗 API 额度。
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from echo.agent.agent import Agent, guess_emotion, parse_agent_output
from echo.agent.tools import ToolRegistry, ToolSpec, build_default_tools
from echo.llm.llm_interface import LLMResponse, ToolCall
from echo.llm.mock_llm import MockToolLLM
from echo.memory.facts import FactStore


# ═════════════════ 工具注册表 ═════════════════
def test_tool_registry_specs_and_call():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo_tool",
            description="回显输入",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
            func=lambda text: f"echo:{text}",
        )
    )

    spec = registry.specs()[0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "echo_tool"
    assert registry.call("echo_tool", '{"text": "hi"}') == "echo:hi"

    # 错误都被转成可读文本，不抛异常
    assert "未注册" in registry.call("nope", "{}")
    assert "不是合法 JSON" in registry.call("echo_tool", "{oops}")
    assert "参数不匹配" in registry.call("echo_tool", '{"wrong": 1}')


def test_default_tools_time_and_memory(tmp_path):
    facts = FactStore(path=str(tmp_path / "facts.json"))
    registry = build_default_tools(facts=facts)

    assert "20" in registry.call("get_current_time", "{}")
    assert "已记住" in registry.call("remember", {"fact": "用户喜欢猫"})
    assert "猫" in registry.call("recall", {"query": "猫"})
    assert facts.all() == ["用户喜欢猫"]


def test_fact_store_dedupes_and_persists(tmp_path):
    path = str(tmp_path / "facts.json")
    store = FactStore(path=path)
    store.add("用户叫小明")
    store.add("用户叫小明")  # 重复不新增
    assert store.all() == ["用户叫小明"]

    reloaded = FactStore(path=path)
    assert reloaded.all() == ["用户叫小明"]


# ═════════════════ 情绪解析 ═════════════════
def test_parse_agent_output_json_and_fallback():
    text, emotion = parse_agent_output('{"reply": "你好呀", "emotion": "happy"}')
    assert (text, emotion) == ("你好呀", "happy")

    # 前后有多余文字也能提取 JSON；情绪大小写归一
    text, emotion = parse_agent_output('好的：{"reply": "嗯嗯", "emotion": "SHY"}')
    assert (text, emotion) == ("嗯嗯", "shy")

    # 完全不是 JSON：整段当文本，关键词兜底猜情绪
    text, emotion = parse_agent_output("哈哈，太好了！")
    assert text == "哈哈，太好了！"
    assert emotion == "happy"
    assert guess_emotion("今天真难过") == "sad"
    assert guess_emotion("这是普通的一句话") == "neutral"


# ═════════════════ Agent 工具调用循环 ═════════════════
def test_agent_tool_call_loop():
    llm = MockToolLLM(
        script=[
            LLMResponse(
                tool_calls=[ToolCall(id="call_1", name="get_current_time", arguments="{}")]
            ),
            LLMResponse(content='{"reply": "现在是下午三点", "emotion": "neutral"}'),
        ]
    )
    events = []
    agent = Agent(
        llm=llm,
        tools=build_default_tools(facts=FactStore(path=None)),
        on_event=lambda kind, payload=None: events.append(kind),
    )

    reply = asyncio.run(agent.respond("现在几点了？"))

    assert reply.text == "现在是下午三点"
    assert reply.emotion == "neutral"
    assert reply.tool_calls[0]["name"] == "get_current_time"
    assert "20" in reply.tool_calls[0]["result"]  # 工具真的执行了

    for expected in ("thinking", "tool_call", "tool_result", "speech", "emotion"):
        assert expected in events
    # 工具结果已按 OpenAI 协议回填给模型
    assert any(m.get("role") == "tool" for m in llm.seen_messages[-1])


def test_agent_stops_after_max_iterations():
    llm = MockToolLLM(
        script=[
            LLMResponse(
                tool_calls=[ToolCall(id=f"c{i}", name="get_current_time", arguments="{}")]
            )
            for i in range(5)
        ]
    )
    agent = Agent(llm=llm, tools=build_default_tools(), max_iterations=2)

    reply = asyncio.run(agent.respond("几点了"))

    assert "再问一次" in reply.text  # 有兜底，不会卡死
    assert len(reply.tool_calls) == 2  # 只跑了 max_iterations 轮
