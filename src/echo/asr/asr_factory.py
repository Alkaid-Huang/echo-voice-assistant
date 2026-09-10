"""
ASR 工厂函数 —— 根据配置创建 ASR 实例
对照 Open-LLM-VTuber asr_factory.py（2026-08-13 检索）

设计说明：
1. 真实后端用延迟导入：未安装依赖时，mock 后端仍可正常使用
2. 保持注册表 + 装饰器模式（开闭原则）
"""
import importlib
from typing import Dict, Type

from .asr_interface import ASRInterface


# 注册表：后端名 → 类
_ASR_REGISTRY: Dict[str, Type[ASRInterface]] = {}

# 后端名 → (模块路径, 依赖包名)，用于延迟导入
_BACKEND_MODULES = {
    "sherpa_onnx": ("echo.asr.sherpa_onnx_asr", "sherpa-onnx"),
    "faster_whisper": ("echo.asr.faster_whisper_asr", "faster-whisper"),
}


def register_asr(name: str):
    """装饰器：注册 ASR 后端"""
    def decorator(cls):
        _ASR_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_backend_loaded(asr_type: str) -> None:
    """延迟导入真实后端：第一次用到时才 import，缺依赖抛清晰错误"""
    if asr_type not in _BACKEND_MODULES:
        return
    if asr_type in _ASR_REGISTRY:
        return
    module_path, package = _BACKEND_MODULES[asr_type]
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"ASR 后端 {asr_type} 加载失败: {e}\n"
            f"请先安装依赖: pip install {package}"
        ) from e


def create_asr(asr_type: str, **kwargs) -> ASRInterface:
    """
    工厂函数：根据类型字符串创建 ASR 实例

    参数:
        asr_type: 后端类型 ("sherpa_onnx" | "faster_whisper" | "mock_asr")
        **kwargs: 传给后端 __init__ 的参数
    返回:
        ASRInterface 实例
    """
    _ensure_backend_loaded(asr_type)
    if asr_type not in _ASR_REGISTRY:
        raise ValueError(f"未知 ASR 类型: {asr_type}，可选: {list(_ASR_REGISTRY.keys())}")
    return _ASR_REGISTRY[asr_type](**kwargs)


def create_asr_with_fallback(asr_type: str, **kwargs) -> ASRInterface:
    """创建 ASR；设备相关错误（cuda/device/gpu）时回退 CPU 重试"""
    try:
        return create_asr(asr_type, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if asr_type in ["faster_whisper", "sherpa_onnx"] and ("cuda" in msg or "device" in msg or "gpu" in msg):
            print("CUDA/device 错误，自动回退到 CPU")
            kwargs["device"] = "cpu"
            return create_asr(asr_type, **kwargs)
        raise
