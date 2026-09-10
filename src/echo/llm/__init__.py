"""echo.llm 包 —— 大模型（LLM）后端"""
from .llm_interface import LLMInterface
from .llm_factory import create_llm, register_llm, _LLM_REGISTRY

# Mock 后端无外部依赖，总是可用
from .mock_llm import MockLLM

# 真实后端按需导入（可选依赖）
try:
    from .ollama_llm import OllamaLLM
except ImportError:
    pass

try:
    from .openai_compatible_llm import OpenAICompatibleLLM
except ImportError:
    pass

__all__ = [
    "LLMInterface",
    "create_llm",
    "register_llm",
    "MockLLM",
]
