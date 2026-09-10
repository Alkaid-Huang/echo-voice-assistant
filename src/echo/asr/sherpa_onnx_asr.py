"""
SherpaOnnxASR —— sherpa-onnx 后端（SenseVoice 模型）
对照 Open-LLM-VTuber src/open_llm_vtuber/asr/sherpa_onnx_asr.py（2026-08-13 检索）

真实实现里值得对齐的 3 个细节：
1. SenseVoice 是【单模型文件】：model.onnx（或 model.int8.onnx）+ tokens.txt，
   不是 encoder/decoder 双文件——字段名 model_path / tokens 也照真实配置来
2. 模型缺失时先给清晰错误；真实项目还会自动下载模型（本课做成可选进阶）
3. 识别结果可能带 SenseVoice 元数据标签（<|zh|>、<|NEUTRAL|> 等），要剥离
   （对照真实项目 fun_asr.py 的 re.sub 写法，本课抽到 text_utils.py 里）

依赖: pip install sherpa-onnx
模型: 官方发布包 sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
      [来源：sherpa-onnx 官方文档 SenseVoice 页，2026-08-13 检索]
"""
import os

import numpy as np
import sherpa_onnx

from .asr_interface import ASRInterface
from .asr_factory import register_asr
from .text_utils import clean_sensevoice_text


@register_asr("sherpa_onnx")
class SherpaOnnxASR(ASRInterface):
    """SenseVoice 离线识别器"""

    def __init__(
        self,
        model_path: str = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
        tokens: str = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
        num_threads: int = 4,
        use_itn: bool = True,
        provider: str = "cpu",  # "cpu" | "cuda"
    ):
        # 模型文件存在性检查（错误降级：缺失时给出清晰提示，而不是晦涩报错）
        missing = [p for p in (model_path, tokens) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"SenseVoice 模型文件缺失: {missing}\n"
                f"请按 sherpa-onnx 官方文档下载模型到 models/ 目录"
                f"（官方 tar.bz2 内含 model.onnx 与 tokens.txt）"
            )

        # ═══════════════════════════════════════════════════════════
        # TODO 6: 初始化 sherpa-onnx 离线识别器
        # ═══════════════════════════════════════════════════════════
        # ⚠️ 这是演化知识：sherpa-onnx 的 API 会变，禁止凭记忆写。
        # 先打开官方文档 k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html
        # 确认 OfflineRecognizer.from_sense_voice(...) 的参数名。
        # 步骤（以官方文档为准，这里只是方向）:
        #   1. self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        #        model=model_path,
        #        tokens=tokens,
        #        num_threads=num_threads,
        #        use_itn=use_itn,
        #        provider=provider,
        #      )
        #   2. self.SAMPLE_RATE = ASRInterface.SAMPLE_RATE  # 16k 约定来自接口
        # 可选进阶（对照真实源码）：仿照真实项目在 __init__ 里检查
        #   onnxruntime.get_available_providers()，provider 配了 cuda
        #   但机器没有 CUDA 时自动回退 CPU
        raise NotImplementedError("请检索官方文档后实现")

    def transcribe_np(self, audio_np: np.ndarray) -> str:
        """同步转写：audio_np 必须是 float32 单声道 16kHz（接口契约）"""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.SAMPLE_RATE, audio_np)
        self.recognizer.decode_streams([stream])
        return clean_sensevoice_text(stream.result.text)
