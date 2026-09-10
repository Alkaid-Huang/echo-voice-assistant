"""
LLM 工厂 —— 按配置字符串创建 LLM 实例

与 ASR 工厂同构：注册表 + 装饰器 + 真实后端延迟导入。
加一个新后端 = 新文件 + @register_llm，不改本文件。
"""
import importlib
from typing import Dict, Type

from .llm_interface import LLMInterface


_LLM_REGISTRY: Dict[str, Type[LLMInterface]] = {}

# 后端名 → (模块路径, 依赖包名)，用于延迟导入
_BACKEND_MODULES = {
    "ollama": ("echo.llm.ollama_llm", "requests"),
    "openai_compatible": ("echo.llm.openai_compatible_llm", "requests"),
}


def register_llm(name: str):
    """装饰器：注册 LLM 后端"""
    def decorator(cls):
        _LLM_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_backend_loaded(llm_type: str) -> None:
    """延迟导入真实后端：第一次用到时才 import，缺依赖抛清晰错误"""
    if llm_type not in _BACKEND_MODULES:
        return
    if llm_type in _LLM_REGISTRY:
        return
    module_path, package = _BACKEND_MODULES[llm_type]
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"LLM 后端 {llm_type} 加载失败: {e}\n"
            f"请先安装依赖: pip install {package}"
        ) from e


def create_llm(llm_type: str, **kwargs) -> LLMInterface:
    """工厂函数：根据类型字符串创建 LLM 实例"""
    _ensure_backend_loaded(llm_type)
    if llm_type not in _LLM_REGISTRY:
        raise ValueError(
            f"未知 LLM 类型: {llm_type}，可选: {list(_LLM_REGISTRY.keys())}"
        )
    return _LLM_REGISTRY[llm_type](**kwargs)
