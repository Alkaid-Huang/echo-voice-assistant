"""
VAD 验收测试
运行: python -m pytest tests/test_vad.py -v

测试覆盖：
1. Pydantic 配置校验（合法/非法值）
2. 工厂函数创建真实现和假实现
3. 工厂拒绝未知类型
4. MockVAD 通过接口测试
5. ServiceContext 依赖注入
6. 配置驱动切换后端
7. conf.yaml 加载
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from echo.config import SileroVADConfig, VADConfig, EchoConfig
from echo.vad.vad_interface import VADInterface
from echo.vad.vad_factory import create_vad, _VAD_REGISTRY
from echo.vad.mock_engine import MockVAD
from echo.service_context import ServiceContext


# ═══════════════════════════════════════════════════════════════
# 测试 1: Pydantic 配置校验 —— 合法值
# ═══════════════════════════════════════════════════════════════
def test_config_valid():
    """合法配置应该正常创建"""
    config = SileroVADConfig(prob_threshold=0.6, required_hits=5)
    assert config.prob_threshold == 0.6
    assert config.required_hits == 5
    assert config.db_threshold == -30.0  # 默认值（2026-09-11 从 -20 放宽，兼容灵敏度较低的麦克风）


# ═══════════════════════════════════════════════════════════════
# 测试 2: Pydantic 配置校验 —— 非法值被拦截
# ═══════════════════════════════════════════════════════════════
def test_config_invalid_prob_threshold():
    """概率超出 0~1 应该报错"""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SileroVADConfig(prob_threshold=-0.1)
    with pytest.raises(ValidationError):
        SileroVADConfig(prob_threshold=1.5)


def test_config_invalid_required_hits():
    """required_hits <= 0 应该报错"""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SileroVADConfig(required_hits=0)
    with pytest.raises(ValidationError):
        SileroVADConfig(required_hits=-3)


# ═══════════════════════════════════════════════════════════════
# 测试 3: 工厂创建 MockVAD
# ═══════════════════════════════════════════════════════════════
def test_factory_create_mock():
    """工厂能创建 MockVAD"""
    engine = create_vad("mock_vad")
    assert isinstance(engine, VADInterface)
    assert isinstance(engine, MockVAD)


# ═══════════════════════════════════════════════════════════════
# 测试 4: 工厂拒绝未知类型
# ═══════════════════════════════════════════════════════════════
def test_factory_unknown_type():
    """传不认识的类型应该报 ValueError"""
    with pytest.raises(ValueError, match="未知 VAD 类型"):
        create_vad("nonexistent_vad")


# ═══════════════════════════════════════════════════════════════
# 测试 5: MockVAD 功能测试
# ═══════════════════════════════════════════════════════════════
def test_mock_vad_detect():
    """MockVAD 应该把输入原样吐出"""
    engine = create_vad("mock_vad")
    chunks = [(b"audio_np_placeholder", b"chunk1"), (b"audio_np_placeholder", b"chunk2")]
    results = list(engine.detect_speech(chunks))
    assert results == [b"chunk1", b"chunk2"]


# ═══════════════════════════════════════════════════════════════
# 测试 6: ServiceContext 依赖注入
# ═══════════════════════════════════════════════════════════════
def test_service_context_init_vad():
    """ServiceContext 能通过配置创建 VAD"""
    config = EchoConfig(
        vad_config=VADConfig(vad_type="mock_vad")
    )
    ctx = ServiceContext(config)
    assert ctx.vad_engine is None  # 初始化前是 None

    ctx.init_vad()
    assert ctx.vad_engine is not None
    assert isinstance(ctx.vad_engine, VADInterface)


# ═══════════════════════════════════════════════════════════════
# 测试 7: 配置驱动切换后端
# ═══════════════════════════════════════════════════════════════
def test_config_driven_switch():
    """改一行配置就能切换 VAD 后端"""
    # 用 mock_vad
    config_mock = EchoConfig(vad_config=VADConfig(vad_type="mock_vad"))
    ctx_mock = ServiceContext(config_mock)
    ctx_mock.init_vad()
    assert isinstance(ctx_mock.vad_engine, MockVAD)

    # 改成 none
    config_none = EchoConfig(vad_config=VADConfig(vad_type="none"))
    ctx_none = ServiceContext(config_none)
    ctx_none.init_vad()
    assert ctx_none.vad_engine is None


# ═══════════════════════════════════════════════════════════════
# 测试 8: conf.yaml 加载
# ═══════════════════════════════════════════════════════════════
def test_load_yaml_config():
    """从 conf.yaml 加载配置"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "conf.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = EchoConfig.model_validate(raw)
    assert config.vad_config.vad_type == "silero_vad"
    assert config.vad_config.silero is not None
    assert config.vad_config.silero.prob_threshold == 0.5
