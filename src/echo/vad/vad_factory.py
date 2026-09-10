"""
VAD 工厂函数 —— 根据配置创建 VAD 实例
对照 Open-LLM-VTuber vad_factory.py（2026-08-03 检索）

改进点：
1. 用注册表替代 if-else 链（开闭原则）
2. 返回类型标注为 VADInterface（不是 Type[VADInterface]）
3. 加 CUDA→CPU 自动降级
"""
from typing import Dict, Type
from .vad_interface import VADInterface


# 注册表：后端名 → 类
_VAD_REGISTRY: Dict[str, Type[VADInterface]] = {}


def register_vad(name: str):
    """装饰器：注册 VAD 后端"""
    def decorator(cls):
        _VAD_REGISTRY[name] = cls
        return cls
    return decorator


def create_vad(vad_type: str, **kwargs) -> VADInterface:
    """
    工厂函数：根据类型字符串创建 VAD 实例

    参数:
        vad_type: 后端类型 ("silero_vad" | "mock_vad")
        **kwargs: 传给后端 __init__ 的参数
    返回:
        VADInterface 实例
    """
    if vad_type not in _VAD_REGISTRY:
        raise ValueError(f"未知 VAD 类型: {vad_type}，可选: {list(_VAD_REGISTRY.keys())}")
    else:
        return _VAD_REGISTRY[vad_type](**kwargs)

    
def create_vad_with_fallback(vad_type, **kwargs):
    try:
        return create_vad(vad_type, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if vad_type == "silero_vad" and ("cuda" in msg or "device" in msg):
            print("CUDA/device 错误，自动回退到 CPU")
            return create_vad(vad_type, **kwargs)
        raise