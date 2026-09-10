"""
ASR 验收测试
运行: python -m pytest tests/test_asr.py -v

测试不依赖真实模型（sherpa-onnx / faster-whisper 未安装也能跑）：
1. 抽象基类不可实例化
2. MockASR 实现接口
3. 工厂创建 mock / 拒绝未知类型
4. 异步接口不阻塞
5. ServiceContext 依赖注入（mock / none 切换）
6. Pydantic 校验（非法 provider / compute_type 被拦）
7. conf.yaml 加载 asr_config
8. SenseVoice 标签剥离工具
9. VAD→ASR 流水线整合（mock 版端到端）
"""
import asyncio
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import ValidationError

from echo.asr.asr_interface import ASRInterface
from echo.asr.asr_factory import create_asr
from echo.asr.mock_asr import MockASR
from echo.config import (
    ASRConfig,
    EchoConfig,
    FasterWhisperASRConfig,
    SherpaOnnxASRConfig,
    VADConfig,
)
from echo.service_context import ServiceContext


# ═══════════════════════════════════════════════════════════
# 测试 1: 抽象基类不可实例化
# ═══════════════════════════════════════════════════════════
def test_abstract_cannot_instantiate():
    """ASRInterface 是抽象类，不能直接 new"""
    with pytest.raises(TypeError):
        ASRInterface()


# ═══════════════════════════════════════════════════════════
# 测试 2: MockASR 实现接口
# ═══════════════════════════════════════════════════════════
def test_mock_asr_implements_interface():
    """MockASR 能实例化，且 transcribe_np 返回固定文本"""
    engine = MockASR()
    audio = np.zeros(512, dtype=np.float32)
    text = engine.transcribe_np(audio)
    assert text == "你好世界", f"MockASR 应返回固定文本，实际 {text!r}"


# ═══════════════════════════════════════════════════════════
# 测试 3: 工厂创建 MockASR
# ═══════════════════════════════════════════════════════════
def test_factory_create_mock():
    """工厂能创建 MockASR"""
    engine = create_asr("mock_asr")
    assert isinstance(engine, ASRInterface)
    assert isinstance(engine, MockASR)


# ═══════════════════════════════════════════════════════════
# 测试 4: 工厂拒绝未知类型
# ═══════════════════════════════════════════════════════════
def test_factory_unknown_type():
    """传不认识的类型应该报 ValueError"""
    with pytest.raises(ValueError, match="未知 ASR 类型"):
        create_asr("nonexistent_asr")


# ═══════════════════════════════════════════════════════════
# 测试 5: 异步接口不阻塞（asyncio.to_thread 包装）
# ═══════════════════════════════════════════════════════════
def test_async_transcribe():
    """async_transcribe_np 返回与同步版相同的结果"""
    engine = MockASR()
    audio = np.zeros(512, dtype=np.float32)
    result = asyncio.run(engine.async_transcribe_np(audio))
    assert result == "你好世界"


# ═══════════════════════════════════════════════════════════
# 测试 6: ServiceContext 依赖注入 —— mock
# ═══════════════════════════════════════════════════════════
def test_service_context_init_asr_mock():
    """ServiceContext 能通过配置创建 ASR"""
    config = EchoConfig(asr_config=ASRConfig(asr_type="mock_asr"))
    ctx = ServiceContext(config)
    assert ctx.asr_engine is None  # 初始化前是 None

    ctx.init_asr()
    assert ctx.asr_engine is not None
    assert isinstance(ctx.asr_engine, ASRInterface)


# ═══════════════════════════════════════════════════════════
# 测试 7: ServiceContext —— none 禁用
# ═══════════════════════════════════════════════════════════
def test_service_context_init_asr_none():
    """asr_type == 'none' 时引擎保持 None"""
    config = EchoConfig(asr_config=ASRConfig(asr_type="none"))
    ctx = ServiceContext(config)
    ctx.init_asr()
    assert ctx.asr_engine is None


# ═══════════════════════════════════════════════════════════
# 测试 8: Pydantic 配置校验
# ═══════════════════════════════════════════════════════════
def test_config_validation():
    """非法参数在配置加载时就被拦住"""
    with pytest.raises(ValidationError):
        SherpaOnnxASRConfig(num_threads=0)
    with pytest.raises(ValidationError):
        SherpaOnnxASRConfig(provider="rocm")  # 真实源码允许 rocm，本课只收 cpu/cuda
    with pytest.raises(ValidationError):
        FasterWhisperASRConfig(compute_type="float8")


# ═══════════════════════════════════════════════════════════
# 测试 9: conf.yaml 加载
# ═══════════════════════════════════════════════════════════
def test_load_yaml_config():
    """从 conf.yaml 加载 ASR 配置"""
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "conf.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = EchoConfig.model_validate(raw)
    assert config.asr_config.asr_type in {"mock_asr", "faster_whisper", "sherpa_onnx"}
    assert config.asr_config.sherpa_onnx is not None
    assert config.asr_config.sherpa_onnx.model_path.endswith("model.int8.onnx")
    assert config.asr_config.sherpa_onnx.tokens.endswith("tokens.txt")
    assert config.asr_config.faster_whisper is not None
    assert config.asr_config.faster_whisper.model_path == "small"
    assert config.asr_config.faster_whisper.prompt is None


# ═══════════════════════════════════════════════════════════
# 测试 10: SenseVoice 标签剥离（对照 fun_asr.py）
# ═══════════════════════════════════════════════════════════
def test_clean_sensevoice_text():
    """SenseVoice 元数据标签（<|zh|>、<|NEUTRAL|> 等）被剥离"""
    from echo.asr.text_utils import clean_sensevoice_text

    assert (
        clean_sensevoice_text("<|zh|><|NEUTRAL|><|Speech|>欢迎大家体验语音识别")
        == "欢迎大家体验语音识别"
    )
    # 真实项目注释里出现的带空格异常形态
    assert clean_sensevoice_text("< | en | > < | EMO _ UNKNOWN | > hello") == "hello"
    assert clean_sensevoice_text("没有标签的文本") == "没有标签的文本"


# ═══════════════════════════════════════════════════════════
# 测试 11: VAD → ASR 流水线整合（mock 版端到端）
# ═══════════════════════════════════════════════════════════
def test_pipeline_vad_to_asr():
    """音频块流进 VAD 产出语音段，段交给 ASR 转成文本"""
    config = EchoConfig(
        vad_config=VADConfig(vad_type="mock_vad"),
        asr_config=ASRConfig(asr_type="mock_asr"),
    )
    ctx = ServiceContext(config)
    ctx.init_vad()
    ctx.init_asr()

    # 模拟麦克风来的音频块（mock_vad 原样吐出 chunk_bytes；4 字节 = 1 个 float32 样本）
    audio_np = np.zeros(512, dtype=np.float32)
    chunk_bytes = np.zeros(4, dtype=np.float32).tobytes()
    chunks = [(audio_np, chunk_bytes), (audio_np, chunk_bytes)]

    segments = list(ctx.vad_engine.detect_speech(chunks))
    assert len(segments) == 2, "mock_vad 应原样吐出 2 个语音段"

    # 每个语音段交给 ASR 转写
    texts = [
        ctx.asr_engine.transcribe_np(np.frombuffer(seg, dtype=np.float32))
        for seg in segments
    ]
    assert texts == ["你好世界", "你好世界"]
