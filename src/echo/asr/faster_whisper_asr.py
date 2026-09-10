"""
FasterWhisperASR —— faster-whisper 后端（CTranslate2 加速）
对照 Open-LLM-VTuber src/open_llm_vtuber/asr/faster_whisper_asr.py（2026-08-13 检索）

对齐真实源码的 3 个细节：
1. 参数名：model_path / download_root / language / device / compute_type / prompt
2. prompt 会作为 initial_prompt 传入——给识别提供上下文（语言应与音频一致）
3. numpy 输入必须是 16k float32（接口契约），transcribe 不传采样率

依赖: pip install faster-whisper
"""
from typing import Optional

import numpy as np

from .asr_interface import ASRInterface
from .asr_factory import register_asr


def _load_whisper_model(model_path, download_root, device, compute_type):
    """加载 faster-whisper 模型（抽成模块级函数，便于测试替换）"""
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size_or_path=model_path,
        download_root=download_root,
        device=device,
        compute_type=compute_type,
    )


def _is_device_error(error: Exception) -> bool:
    """
    判断异常是否属于"设备不可用"类问题。

    真实案例（2026-09-10）：device="auto" 选了 CUDA，但机器缺 cuBLAS，
    报错 'Library cublas64_12.dll is not found or cannot be loaded'。
    """
    msg = str(error).lower()
    return any(
        k in msg
        for k in ("cuda", "cublas", "cudnn", "dll is not found", "gpu", "device")
    )


@register_asr("faster_whisper")
class FasterWhisperASR(ASRInterface):
    """faster-whisper 转写器"""

    def __init__(
        self,
        model_path: str = "small",
        download_root: str = "models/whisper",
        language: Optional[str] = "zh",
        device: str = "auto",  # "cpu" | "cuda" | "auto"
        compute_type: str = "int8",
        prompt: Optional[str] = None,
        fallback_to_cpu: bool = True,
    ):
        self.model_path = model_path
        self.download_root = download_root
        self.device = device
        self.compute_type = compute_type
        self.fallback_to_cpu = fallback_to_cpu
        self.model = _load_whisper_model(
            model_path, download_root, device, compute_type
        )
        self.language = language
        self.prompt = prompt

    def transcribe_np(self, audio_np: np.ndarray) -> str:
        """同步转写：audio_np 必须是 float32 单声道 16kHz（接口契约）"""
        try:
            return self._transcribe(audio_np)
        except RuntimeError as e:
            # 设备不可用（缺 CUDA 库等）→ 用 CPU 重建模型重试一次
            if self.fallback_to_cpu and self.device != "cpu" and _is_device_error(e):
                print(f"[降级] faster-whisper 在 {self.device} 上推理失败，回退 CPU：{e}")
                self.model = _load_whisper_model(
                    self.model_path, self.download_root, "cpu", "int8"
                )
                self.device = "cpu"
                return self._transcribe(audio_np)
            raise

    def _transcribe(self, audio_np: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio_np,
            beam_size=5,
            language=self.language if self.language else None,
            condition_on_previous_text=False,
            initial_prompt=self.prompt,
        )
        return "".join(segment.text for segment in segments).strip()
    
