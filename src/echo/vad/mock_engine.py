"""
MockVAD 假实现 —— 用于测试和开发
证明只要实现 VADInterface，工厂就能创建它
"""
from .vad_interface import VADInterface
from .vad_factory import register_vad


@register_vad("mock_vad")
class MockVAD(VADInterface):
    def __init__(self,**kwargs):
        pass

    def detect_speech(self, audio_chunks):
        for audio_chunk in audio_chunks:
            yield audio_chunk[1]

    def process_block(self, audio_np, chunk_bytes):
        """假实现：每个块都当成一个完整语音段吐出"""
        return chunk_bytes
