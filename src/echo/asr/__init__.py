"""echo.asr 包 —— ASR 语音识别"""
from .asr_interface import ASRInterface
from .asr_factory import create_asr, register_asr, _ASR_REGISTRY

# Mock 后端无外部依赖，总是可用
from .mock_asr import MockASR

# 真实后端依赖第三方包；未安装时跳过注册（可选依赖，不影响 mock 测试）
try:
    from .sherpa_onnx_asr import SherpaOnnxASR
except ImportError:
    pass

try:
    from .faster_whisper_asr import FasterWhisperASR
except ImportError:
    pass

__all__ = [
    "ASRInterface",
    "create_asr",
    "register_asr",
    "MockASR",
]
