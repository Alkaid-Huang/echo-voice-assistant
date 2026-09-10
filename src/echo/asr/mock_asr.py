"""
MockASR 假实现 —— 用于测试和开发
证明只要实现 ASRInterface，工厂就能创建它
"""
from .asr_interface import ASRInterface
from .asr_factory import register_asr


@register_asr("mock_asr")
class MockASR(ASRInterface):
    """假实现：不真听，永远返回固定文本"""

    def __init__(self, **kwargs):
        pass

    def transcribe_np(self, audio):
        return "你好世界"
