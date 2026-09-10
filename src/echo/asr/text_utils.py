"""
SenseVoice 结果清洗工具
对照 Open-LLM-VTuber src/open_llm_vtuber/asr/fun_asr.py（2026-08-13 检索）

背景：SenseVoice 家族模型（无论走 sherpa-onnx 还是 FunASR 推理路径）的
输出文本可能带元数据标签，例如：
    '<|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验语音识别模型'
这些标签描述语言、事件、情感等，不是用户说的话，必须剥离。
真实项目在 fun_asr.py 里用 re.sub 实现，本课把它抽成独立函数以便单元测试。
"""
import re

# 标准形态 <|zh|>，以及真实项目注释里出现过的带空格形态 < | zh | >
_TAG_PATTERN = re.compile(r"<\|.*?\|>|< \|.*?\| >")


def clean_sensevoice_text(text: str) -> str:
    """剥离 SenseVoice 元数据标签（<|...|>），并去掉首尾空白。"""
    if not text:
        return text
    return _TAG_PATTERN.sub("", text).strip()
