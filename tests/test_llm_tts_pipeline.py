"""
LLM / TTS / 记忆 / 对话管线测试

全部使用 mock 后端，不联网、不使用声卡，可在 CI 里跑。
"""
import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from pydantic import ValidationError

from echo.config import ASRConfig, EchoConfig, LLMConfig, TTSConfig, VADConfig
from echo.llm.llm_factory import create_llm
from echo.llm.mock_llm import MockLLM
from echo.memory.chat_history import ChatHistory
from echo.pipeline.conversation import ConversationPipeline
from echo.service_context import ServiceContext
from echo.tts.mock_tts import MockTTS
from echo.tts.tts_factory import create_tts


def _mock_config(tmp_path) -> EchoConfig:
    config = EchoConfig(
        vad_config=VADConfig(vad_type="mock_vad"),
        asr_config=ASRConfig(asr_type="mock_asr"),
        llm_config=LLMConfig(llm_type="mock_llm", max_history_turns=2),
        tts_config=TTSConfig(tts_type="mock_tts"),
    )
    config.app_config.history_file = str(tmp_path / "history.json")
    config.tts_config.output_dir = str(tmp_path / "tts")
    return config


def _fake_player_factory(played: list):
    def player(path, stop_event=None):
        played.append(path)
    return player


class _FakeMic:
    """假麦克风：按顺序吐出预置音频块，取空后返回 None"""

    def __init__(self, items):
        self.items = list(items)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get(self, timeout=0.5):
        if self.items:
            return self.items.pop(0)
        time.sleep(0.01)
        return None


# ═════════════════ 工厂与配置 ═════════════════
def test_is_device_error_detects_cublas():
    from echo.asr.faster_whisper_asr import _is_device_error

    err = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
    assert _is_device_error(err)
    assert not _is_device_error(ValueError("模型文件不存在"))


def test_faster_whisper_falls_back_to_cpu(monkeypatch):
    """真实 bug 防回归：CUDA 库缺失时，推理阶段自动回退 CPU 重试"""
    from echo.asr import faster_whisper_asr as mod

    class _FakeSegment:
        text = " 你好 "

    class _FakeModel:
        def __init__(self, device):
            self.device = device

        def transcribe(self, audio, **kwargs):
            if self.device != "cpu":
                raise RuntimeError(
                    "Library cublas64_12.dll is not found or cannot be loaded"
                )
            return [_FakeSegment()], None

    loaded_devices = []

    def fake_loader(model_path, download_root, device, compute_type):
        loaded_devices.append(device)
        return _FakeModel(device)

    monkeypatch.setattr(mod, "_load_whisper_model", fake_loader)
    engine = mod.FasterWhisperASR(model_path="small", device="auto")
    text = engine.transcribe_np(np.zeros(16000, dtype=np.float32))

    assert text == "你好"
    assert engine.device == "cpu"
    assert loaded_devices == ["auto", "cpu"]


def test_create_mock_llm():
    engine = create_llm("mock_llm")
    assert isinstance(engine, MockLLM)
    assert engine.chat([{"role": "user", "content": "hi"}])


def test_create_unknown_llm_raises():
    with pytest.raises(ValueError, match="未知 LLM 类型"):
        create_llm("not_a_llm")


def test_create_mock_tts():
    engine = create_tts("mock_tts")
    assert isinstance(engine, MockTTS)


def test_llm_config_rejects_unknown_type():
    with pytest.raises(ValidationError):
        LLMConfig(llm_type="chatgpt")


def test_tts_params_carry_output_dir(tmp_path):
    config = TTSConfig(tts_type="mock_tts", output_dir=str(tmp_path))
    assert config.get_tts_params()["output_dir"] == str(tmp_path)


# ═════════════════ TTS 与记忆 ═════════════════
def test_mock_tts_writes_wav(tmp_path):
    path = MockTTS(output_dir=str(tmp_path)).synthesize("你好呀")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 44  # 大于 WAV 头


def test_chat_history_roundtrip(tmp_path):
    path = str(tmp_path / "h.json")
    history = ChatHistory(path=path, max_messages=4)
    history.append("user", "你好")
    history.append("assistant", "你好呀")
    history.save()

    reloaded = ChatHistory(path=path, max_messages=4)
    assert reloaded.messages()[0]["content"] == "你好"
    assert len(reloaded) == 2


def test_chat_history_trims_old_messages(tmp_path):
    history = ChatHistory(path=None, max_messages=2)
    history.append("user", "1")
    history.append("assistant", "2")
    history.append("user", "3")
    assert len(history) == 2
    assert history.messages()[0]["content"] == "2"


# ═════════════════ 容器 ═════════════════
def test_service_context_init_all(tmp_path):
    ctx = ServiceContext(_mock_config(tmp_path))
    ctx.init_all()
    assert type(ctx.vad_engine).__name__ == "MockVAD"
    assert type(ctx.asr_engine).__name__ == "MockASR"
    assert type(ctx.llm_engine).__name__ == "MockLLM"
    assert type(ctx.tts_engine).__name__ == "MockTTS"


# ═════════════════ 管线 ═════════════════
def test_pipeline_text_reply(tmp_path):
    ctx = ServiceContext(_mock_config(tmp_path))
    ctx.init_all()
    played = []
    pipeline = ConversationPipeline(ctx, player=_fake_player_factory(played))

    reply = asyncio.run(pipeline.respond_text("你好"))
    assert reply == "（模拟回复）我在听，你继续说。"
    assert len(played) == 1  # TTS 产出后触发了播放
    assert len(pipeline.history) == 2  # 用户 + 助手各一条


def test_pipeline_handles_segment(tmp_path):
    ctx = ServiceContext(_mock_config(tmp_path))
    ctx.init_all()
    pipeline = ConversationPipeline(ctx, player=_fake_player_factory([]))

    audio = np.zeros(16000, dtype=np.float32)  # 1 秒，超过最短时长
    reply = asyncio.run(pipeline.handle_segment(audio.tobytes()))
    assert reply == "（模拟回复）我在听，你继续说。"


def test_pipeline_drops_too_short_segment(tmp_path):
    ctx = ServiceContext(_mock_config(tmp_path))
    ctx.init_all()
    pipeline = ConversationPipeline(ctx, player=_fake_player_factory([]))

    audio = np.zeros(1600, dtype=np.float32)  # 0.1 秒 < 0.3 秒阈值
    assert asyncio.run(pipeline.handle_segment(audio.tobytes())) is None


def test_pipeline_interrupt_stops_playback(tmp_path):
    ctx = ServiceContext(_mock_config(tmp_path))
    ctx.init_all()
    stopped = threading.Event()

    def slow_player(path, stop_event=None):
        # 播放 2 秒，除非被打断
        if stop_event is not None and stop_event.wait(2.0):
            stopped.set()

    pipeline = ConversationPipeline(ctx, player=slow_player)
    asyncio.run(pipeline.respond_text("说一句长一点的话"))
    assert pipeline._is_playing()
    pipeline.interrupt()
    assert stopped.wait(1.0), "打断后播放线程应及时退出"
    assert not pipeline._is_playing()


def test_pipeline_run_loop_end_to_end(tmp_path):
    """假麦克风喂一块音频 → VAD 出段 → ASR → LLM → TTS → 播放"""
    ctx = ServiceContext(_mock_config(tmp_path))
    ctx.init_all()
    events = []
    played = []
    chunk = np.zeros(512, dtype=np.float32)
    mic = _FakeMic([(chunk, np.zeros(16000, dtype=np.float32).tobytes())])
    pipeline = ConversationPipeline(
        ctx,
        mic=mic,
        player=_fake_player_factory(played),
        on_event=lambda kind, payload=None: events.append(kind),
    )

    async def scenario():
        task = asyncio.create_task(pipeline.run())
        for _ in range(100):
            if "llm" in events:
                break
            await asyncio.sleep(0.05)
        pipeline.stop()
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())
    assert "listening" in events
    assert "asr" in events
    assert "llm" in events
    assert "tts" in events
    assert len(played) == 1
    assert mic.stopped


# ═════════════════ .env 加载与 API 后端（全部不联网） ═════════════════
class _FakeResponse:
    """伪造 requests 响应对象"""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def test_env_loader_reads_file_and_keeps_existing(tmp_path, monkeypatch):
    from echo.env_loader import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text('ECHO_TEST_KEY="from_file"\n# 注释行\nBADLINE\n', encoding="utf-8")

    monkeypatch.delenv("ECHO_TEST_KEY", raising=False)
    load_dotenv(str(env_file))
    assert os.environ["ECHO_TEST_KEY"] == "from_file"

    # 环境变量优先：已存在时不被文件覆盖
    monkeypatch.setenv("ECHO_TEST_KEY", "from_env")
    load_dotenv(str(env_file))
    assert os.environ["ECHO_TEST_KEY"] == "from_env"


def test_openai_compatible_builds_request(monkeypatch):
    from echo.llm import openai_compatible_llm as mod

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, payload=json, headers=headers, timeout=timeout)
        return _FakeResponse(200, {"choices": [{"message": {"content": " 你好呀 "}}]})

    monkeypatch.setenv("ECHO_TEST_API_KEY", "sk-test")
    monkeypatch.setattr(mod.requests, "post", fake_post)

    llm = mod.OpenAICompatibleLLM(
        model="test-model",
        base_url="https://example.com/v1",
        api_key_env="ECHO_TEST_API_KEY",
        max_tokens=64,
    )
    reply = llm.chat([{"role": "user", "content": "hi"}])

    assert reply == "你好呀"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["max_tokens"] == 64
    assert captured["payload"]["stream"] is False
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_openai_compatible_error_hints(monkeypatch):
    from echo.llm import openai_compatible_llm as mod

    monkeypatch.setenv("ECHO_TEST_API_KEY", "sk-test")
    llm = mod.OpenAICompatibleLLM(api_key_env="ECHO_TEST_API_KEY")

    monkeypatch.setattr(
        mod.requests, "post", lambda *a, **k: _FakeResponse(401, text="unauthorized")
    )
    with pytest.raises(RuntimeError, match="API Key 无效"):
        llm.chat([{"role": "user", "content": "hi"}])

    monkeypatch.setattr(
        mod.requests, "post", lambda *a, **k: _FakeResponse(404, text="not found")
    )
    with pytest.raises(RuntimeError, match="接口不存在"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_openai_compatible_requires_api_key(monkeypatch):
    from echo.llm import openai_compatible_llm as mod

    monkeypatch.delenv("ECHO_TEST_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="未找到环境变量"):
        mod.OpenAICompatibleLLM(api_key_env="ECHO_TEST_MISSING_KEY")


# ═════════════════ 并发流水线 / 双工 / 收尾保护 ═════════════════
def test_state_machine_force_flush():
    from echo.vad.state_machine import State, StateMachine

    sm = StateMachine()
    sm.state = State.ACTIVE
    sm.bytes_buffer = b"hello"
    assert sm.force_flush() == b"hello"
    assert sm.state == State.IDLE
    assert sm.bytes_buffer == b""
    assert sm.force_flush() is None


def test_mic_drops_oldest_block_and_counts():
    from echo.audio.io import MicStream

    mic = MicStream()
    for _ in range(mic._queue.maxsize):
        mic._queue.put_nowait((np.zeros(512, dtype=np.float32), b"old"))
    before = mic.dropped_blocks

    mic._callback(np.zeros((512, 1), dtype=np.float32), 512, None, None)

    assert mic.dropped_blocks == before + 1
    _, last_bytes = list(mic._queue.queue)[-1]
    assert last_bytes != b"old"  # 保留了最新一块


class _CountingMic:
    """可注入的假麦克风：记录 start/stop 次数"""

    def __init__(self, items):
        self.items = list(items)
        self.start_count = 0
        self.stop_count = 0
        self.dropped_blocks = 0

    def start(self):
        self.start_count += 1

    def stop(self):
        self.stop_count += 1

    def get(self, timeout=0.5):
        if self.items:
            return self.items.pop(0)
        time.sleep(0.02)
        return None

    def is_stalled(self, threshold_seconds=2.0):
        return False


def test_pipeline_half_duplex_pauses_capture(tmp_path):
    """half 模式：播放期间应关闭采集（防自我对话 + 避免声卡双流争用）"""
    config = _mock_config(tmp_path)
    ctx = ServiceContext(config)
    ctx.init_all()

    started = threading.Event()
    release = threading.Event()

    def slow_player(path, stop_event=None):
        started.set()
        release.wait(2.0)

    chunk = np.zeros(512, dtype=np.float32)
    mic = _CountingMic([(chunk, np.zeros(16000, dtype=np.float32).tobytes())])
    pipeline = ConversationPipeline(ctx, mic=mic, player=slow_player)

    async def scenario():
        task = asyncio.create_task(pipeline.run())
        try:
            for _ in range(100):
                if started.is_set() and mic.stop_count >= 1:
                    break
                await asyncio.sleep(0.05)
            paused_during_playback = mic.stop_count >= 1
        finally:
            release.set()
            pipeline.stop()
            await asyncio.wait_for(task, timeout=5)
        return paused_during_playback

    assert asyncio.run(scenario())


def test_pipeline_forced_flush_on_max_duration(tmp_path):
    """单句超长时应强制切分并送进应答队列（死配置 max_utterance_seconds 生效）"""
    config = _mock_config(tmp_path)
    config.app_config.max_utterance_seconds = 0.2
    ctx = ServiceContext(config)
    ctx.init_all()

    segment = np.zeros(16000, dtype=np.float32).tobytes()

    class _AlwaysSpeakingVAD:
        def __init__(self):
            self.flushed = 0

        def process_block(self, audio_np, chunk_bytes):
            return None  # 永远不自然结束

        def is_speaking(self):
            return True

        def force_flush(self):
            self.flushed += 1
            return segment

    vad = _AlwaysSpeakingVAD()
    ctx.vad_engine = vad
    events = []
    mic = _CountingMic([(np.zeros(512, dtype=np.float32), b"") for _ in range(4)])
    pipeline = ConversationPipeline(
        ctx,
        mic=mic,
        player=lambda path, stop_event=None: None,
        on_event=lambda kind, payload=None: events.append(kind),
    )

    async def scenario():
        task = asyncio.create_task(pipeline.run())
        try:
            for _ in range(60):
                if "asr" in events:
                    break
                await asyncio.sleep(0.05)
        finally:
            pipeline.stop()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())
    assert vad.flushed >= 1
    assert "asr" in events
