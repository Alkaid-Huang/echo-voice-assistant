"""echo.vad 包 —— VAD 语音活动检测"""
from .vad_interface import VADInterface
from .state_machine import StateMachine, State
from .vad_factory import create_vad, register_vad, _VAD_REGISTRY

# 导入后端模块，触发 @register_vad 装饰器注册
# 这一步必须在 create_vad 被调用前执行，否则注册表是空的
from .silero_engine import SileroVADEngine
from .mock_engine import MockVAD

__all__ = [
    "VADInterface",
    "StateMachine",
    "State",
    "create_vad",
    "register_vad",
    "SileroVADEngine",
    "MockVAD",
]
