"""
TTS 工厂 —— 按配置字符串创建 TTS 实例
注册表 + 装饰器 + 真实后端延迟导入，与 ASR/LLM 工厂同构。
"""
import importlib
from typing import Dict, Type

from .tts_interface import TTSInterface


_TTS_REGISTRY: Dict[str, Type[TTSInterface]] = {}

_BACKEND_MODULES = {
    "edge_tts": ("echo.tts.edge_tts_engine", "edge-tts"),
}


def register_tts(name: str):
    """装饰器：注册 TTS 后端"""
    def decorator(cls):
        _TTS_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_backend_loaded(tts_type: str) -> None:
    """延迟导入真实后端"""
    if tts_type not in _BACKEND_MODULES:
        return
    if tts_type in _TTS_REGISTRY:
        return
    module_path, package = _BACKEND_MODULES[tts_type]
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"TTS 后端 {tts_type} 加载失败: {e}\n"
            f"请先安装依赖: pip install {package}"
        ) from e


def create_tts(tts_type: str, **kwargs) -> TTSInterface:
    """工厂函数：根据类型字符串创建 TTS 实例"""
    _ensure_backend_loaded(tts_type)
    if tts_type not in _TTS_REGISTRY:
        raise ValueError(
            f"未知 TTS 类型: {tts_type}，可选: {list(_TTS_REGISTRY.keys())}"
        )
    return _TTS_REGISTRY[tts_type](**kwargs)
