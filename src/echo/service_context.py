"""
ServiceContext —— 依赖注入容器
对照 Open-LLM-VTuber service_context.py（2026-08-03 检索）

改进点：
1. 统一管理 VAD / ASR / LLM / TTS 四个组件
2. init_vad / init_asr 用工厂而非直接 new
"""
from typing import Optional
from .config import EchoConfig
from .vad.vad_interface import VADInterface
from .vad.vad_factory import create_vad
from .asr.asr_interface import ASRInterface
from .asr.asr_factory import create_asr
from .llm.llm_interface import LLMInterface
from .llm.llm_factory import create_llm
from .tts.tts_interface import TTSInterface
from .tts.tts_factory import create_tts


class ServiceContext:
    """服务容器：统一管理 Echo 的所有组件"""

    def __init__(self, config: EchoConfig):
        self.config = config
        self.vad_engine: Optional[VADInterface] = None
        self.asr_engine: Optional[ASRInterface] = None
        self.llm_engine: Optional[LLMInterface] = None
        self.tts_engine: Optional[TTSInterface] = None

    def init_vad(self) -> None:
        """根据配置创建 VAD 引擎"""
        vad_type = self.config.vad_config.vad_type
        if vad_type == "none":
            self.vad_engine = None
            return
        params = self.config.vad_config.get_vad_params()
        self.vad_engine = create_vad(vad_type,**params)

    def init_asr(self) -> None:
        """根据配置创建 ASR 引擎"""
        asr_type = self.config.asr_config.asr_type
        if asr_type == "none":
            self.asr_engine = None
            return
        params = self.config.asr_config.get_asr_params()
        self.asr_engine = create_asr(asr_type, **params)

    def init_llm(self) -> None:
        """根据配置创建 LLM 引擎"""
        llm_type = self.config.llm_config.llm_type
        if llm_type == "none":
            self.llm_engine = None
            return
        params = self.config.llm_config.get_llm_params()
        self.llm_engine = create_llm(llm_type, **params)

    def init_tts(self) -> None:
        """根据配置创建 TTS 引擎"""
        tts_type = self.config.tts_config.tts_type
        if tts_type == "none":
            self.tts_engine = None
            return
        params = self.config.tts_config.get_tts_params()
        self.tts_engine = create_tts(tts_type, **params)

    def init_all(self) -> None:
        """一次性初始化全部组件（VAD → ASR → LLM → TTS）"""
        self.init_vad()
        self.init_asr()
        self.init_llm()
        self.init_tts()
