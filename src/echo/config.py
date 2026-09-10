"""
Pydantic 配置模型 —— 配置驱动 + 类型校验

VAD / ASR 部分对照 Open-LLM-VTuber（2026-08-03 / 08-13 检索）；
LLM / TTS / 应用部分由助手扩展（2026-09-10）。
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SileroVADConfig(BaseModel):
    """Silero VAD 参数配置（带校验）"""
    prob_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    db_threshold: float = Field(default=-30.0, le=0)  # 越低越灵敏（麦克风声音小就调低）
    required_hits: int = Field(default=3, gt=0)
    required_misses: int = Field(default=24, gt=0)
    smoothing_window: int = Field(default=5, gt=0)
    pre_buffer_size: int = Field(default=20, gt=0)
    window_size_samples: int = Field(default=512, gt=0)
    sample_rate: Literal[8000, 16000] = Field(default=16000)  # 与 window_size_samples 必须匹配
    db_margin: float = Field(default=6.0, ge=0)  # 说话至少要比环境底噪高多少 dB
    db_adapt_limit: float = Field(default=10.0, ge=0)  # 自适应最多抬高多少 dB


class VADConfig(BaseModel):
    """VAD 顶层配置：选择哪个后端 + 对应参数"""
    vad_type: str = Field(default="silero_vad")  # "silero_vad" | "mock_vad" | "none"
    silero: Optional[SileroVADConfig] = Field(default=None)

    def get_vad_params(self) -> dict:
        """根据 vad_type 取对应后端的参数字典"""
        if self.vad_type == "silero_vad":
            if self.silero is None:
                return SileroVADConfig().model_dump()
            return self.silero.model_dump()
        return {}


class SherpaOnnxASRConfig(BaseModel):
    """SenseVoice 后端参数（带校验，字段名对齐真实源码 config_manager/asr.py）"""
    # SenseVoice 是单模型文件：model.onnx + tokens.txt（对照真实源码 sense_voice 参数）
    model_path: str = Field(
        default="models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"
    )
    tokens: str = Field(
        default="models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"
    )
    num_threads: int = Field(default=4, gt=0)
    use_itn: bool = Field(default=True)
    provider: Literal["cpu", "cuda"] = Field(default="cpu")


class FasterWhisperASRConfig(BaseModel):
    """faster-whisper 后端参数（带校验，字段名对齐真实源码 config_manager/asr.py）"""
    model_path: str = Field(default="small")  # 模型名（如 small）或本地路径
    download_root: str = Field(default="models/whisper")
    language: Optional[str] = Field(default="zh")
    device: Literal["cpu", "cuda", "auto"] = Field(default="auto")
    compute_type: Literal["int8", "float16", "float32"] = Field(default="int8")
    prompt: Optional[str] = Field(default=None)  # initial_prompt，引导识别


class ASRConfig(BaseModel):
    """ASR 顶层配置：选择哪个后端 + 对应参数"""
    asr_type: str = Field(default="mock_asr")  # "sherpa_onnx" | "faster_whisper" | "mock_asr" | "none"
    sherpa_onnx: Optional[SherpaOnnxASRConfig] = Field(default=None)
    faster_whisper: Optional[FasterWhisperASRConfig] = Field(default=None)

    def get_asr_params(self) -> dict:
        """根据 asr_type 取对应后端的参数字典"""
        if self.asr_type == "sherpa_onnx":
            if self.sherpa_onnx is None:
                return SherpaOnnxASRConfig().model_dump()
            return self.sherpa_onnx.model_dump()
        if self.asr_type == "faster_whisper":
            if self.faster_whisper is None:
                return FasterWhisperASRConfig().model_dump()
            return self.faster_whisper.model_dump()
        return {}


# ═══════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════
class OllamaLLMConfig(BaseModel):
    """本地 Ollama 后端参数"""
    model: str = Field(default="qwen2.5:3b")
    host: str = Field(default="http://127.0.0.1:11434")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    num_predict: int = Field(default=256, gt=0)
    timeout: float = Field(default=60.0, gt=0)


class OpenAICompatibleLLMConfig(BaseModel):
    """OpenAI 兼容 HTTP 后端参数（DeepSeek / 通义 / vLLM / OpenAI）"""
    model: str = Field(default="deepseek-chat")
    base_url: str = Field(default="https://api.deepseek.com/v1")
    api_key_env: str = Field(default="ECHO_LLM_API_KEY")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, gt=0)
    timeout: float = Field(default=60.0, gt=0)


class LLMConfig(BaseModel):
    """LLM 顶层配置：后端选型 + 系统提示词 + 历史轮数"""
    llm_type: Literal["ollama", "openai_compatible", "mock_llm", "none"] = Field(
        default="mock_llm"
    )
    system_prompt: str = Field(
        default="你是 Echo，一个中文语音助手。回复口语化、简短，控制在两三句话内。"
    )
    max_history_turns: int = Field(default=6, ge=0, le=50)
    ollama: Optional[OllamaLLMConfig] = Field(default=None)
    openai_compatible: Optional[OpenAICompatibleLLMConfig] = Field(default=None)

    def get_llm_params(self) -> dict:
        """根据 llm_type 取对应后端的参数字典"""
        if self.llm_type == "ollama":
            if self.ollama is None:
                return OllamaLLMConfig().model_dump()
            return self.ollama.model_dump()
        if self.llm_type == "openai_compatible":
            if self.openai_compatible is None:
                return OpenAICompatibleLLMConfig().model_dump()
            return self.openai_compatible.model_dump()
        return {}


# ═══════════════════════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════════════════════
class EdgeTTSConfig(BaseModel):
    """edge-tts 音色参数"""
    voice: str = Field(default="zh-CN-XiaoxiaoNeural")
    rate: str = Field(default="+0%")
    volume: str = Field(default="+0%")


class TTSConfig(BaseModel):
    """TTS 顶层配置：后端选型 + 输出目录"""
    tts_type: Literal["edge_tts", "mock_tts", "none"] = Field(default="mock_tts")
    output_dir: str = Field(default="outputs")
    edge_tts: Optional[EdgeTTSConfig] = Field(default=None)

    def get_tts_params(self) -> dict:
        """根据 tts_type 取对应后端的参数字典"""
        if self.tts_type == "edge_tts":
            params = (self.edge_tts or EdgeTTSConfig()).model_dump()
            params["output_dir"] = self.output_dir
            return params
        if self.tts_type == "mock_tts":
            return {"output_dir": self.output_dir}
        return {}


# ═══════════════════════════════════════════════════════════
# 应用层
# ═══════════════════════════════════════════════════════════
class AppConfig(BaseModel):
    """应用层参数：音频链路 + 打断行为"""
    sample_rate: int = Field(default=16000, gt=0)
    block_size_samples: int = Field(default=512, gt=0)
    input_device: Optional[int] = Field(default=None)  # None = 系统默认麦克风
    barge_in: bool = Field(default=True)  # 播放中检测到人声是否打断
    min_utterance_seconds: float = Field(default=0.3, ge=0.0)
    max_utterance_seconds: float = Field(default=15.0, gt=0)
    history_file: str = Field(default="outputs/chat_history.json")


class EchoConfig(BaseModel):
    """Echo 全局配置"""
    vad_config: VADConfig = Field(default_factory=VADConfig)
    asr_config: ASRConfig = Field(default_factory=ASRConfig)
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    tts_config: TTSConfig = Field(default_factory=TTSConfig)
    app_config: AppConfig = Field(default_factory=AppConfig)
