"""
EdgeTTSEngine —— 微软 Edge 在线语音合成（edge-tts）

中文音色示例（2026-09-10 检索）：
- zh-CN-XiaoxiaoNeural  女声，自然
- zh-CN-YunxiNeural     男声
- zh-CN-XiaoyiNeural    女声，活泼

注意：需要联网；输出为 mp3，播放前由 echo.audio.io 解码。
"""
import asyncio
import os
import uuid

import edge_tts

from .tts_interface import TTSInterface
from .tts_factory import register_tts


@register_tts("edge_tts")
class EdgeTTSEngine(TTSInterface):
    """edge-tts 合成后端（库本身是异步 API）"""

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        output_dir: str = "outputs",
        **kwargs,
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.output_dir = output_dir

    async def async_synthesize(self, text: str, output_path=None) -> str:
        """edge-tts 原生异步，这里直接 await（比 to_thread 包装更高效）"""
        path = output_path or os.path.join(
            self.output_dir, f"tts_{uuid.uuid4().hex[:8]}.mp3"
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate, volume=self.volume
        )
        await communicate.save(path)
        return path

    def synthesize(self, text: str, output_path=None) -> str:
        """同步入口：库是异步的，这里起一个临时事件循环"""
        return asyncio.run(self.async_synthesize(text, output_path))
