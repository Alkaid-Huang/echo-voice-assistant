"""
ConversationPipeline —— VAD → ASR → LLM → TTS 的端到端对话回路（支持打断）

数据流：
    麦克风(16k) → VAD 切段 → ASR 转文字 → LLM 生成回复 → TTS 合成 → 播放

打断逻辑（barge_in）：
    播放回复期间继续读麦克风；一旦 VAD 判定用户开始说话，
    立即 set() 停止事件，播放线程退出，然后处理这句新话。

事件回调 on_event(kind, payload) 用于打印或推送前端，kind 取值：
    listening / asr / llm / tts / interrupt / error
"""
import asyncio
import threading
import time
from typing import Callable, Optional

import numpy as np

from ..audio.io import MicStream, play_file
from ..memory.chat_history import ChatHistory
from ..service_context import ServiceContext


class ConversationPipeline:
    """语音对话管线（依赖 ServiceContext 注入的四个组件）"""

    def __init__(
        self,
        ctx: ServiceContext,
        mic: Optional[object] = None,
        player: Optional[Callable] = None,
        on_event: Optional[Callable[[str, object], None]] = None,
    ):
        self.ctx = ctx
        self.config = ctx.config
        self.vad = ctx.vad_engine
        self.asr = ctx.asr_engine
        self.llm = ctx.llm_engine
        self.tts = ctx.tts_engine

        app = self.config.app_config
        self.sample_rate = app.sample_rate
        # mic / player 可注入，方便测试时不碰真实声卡
        self.mic = mic if mic is not None else MicStream(
            app.sample_rate, app.block_size_samples
        )
        self.player = player if player is not None else play_file
        self.on_event = on_event or (lambda kind, payload=None: None)

        self.history = ChatHistory(
            path=app.history_file,
            max_messages=self.config.llm_config.max_history_turns * 2,
        )
        self._stop_event = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self._running = False
        self.last_reply: Optional[str] = None

    # ═════════════════ 事件 ═════════════════
    def _emit(self, kind: str, payload=None) -> None:
        try:
            self.on_event(kind, payload)
        except Exception:
            pass  # 回调里的异常不该拖垮主管线

    # ═════════════════ 上下文组装 ═════════════════
    def build_messages(self, user_text: str) -> list[dict]:
        """系统提示词 + 最近 N 轮历史 + 本轮用户输入"""
        messages = [
            {"role": "system", "content": self.config.llm_config.system_prompt}
        ]
        messages.extend(self.history.messages())
        messages.append({"role": "user", "content": user_text})
        return messages

    # ═════════════════ 播放与打断 ═════════════════
    def _is_playing(self) -> bool:
        return self._play_thread is not None and self._play_thread.is_alive()

    def _start_playback(self, path: str) -> None:
        self._stop_event.clear()
        self._play_thread = threading.Thread(
            target=self._play_worker, args=(path,), daemon=True
        )
        self._play_thread.start()

    def _play_worker(self, path: str) -> None:
        try:
            self.player(path, self._stop_event)
        except Exception as e:  # 播放失败不应该结束整个对话
            self._emit("error", f"播放失败: {e}")

    def interrupt(self) -> None:
        """打断当前播放（幂等）"""
        if self._is_playing():
            self._stop_event.set()
            self._play_thread.join(timeout=1.0)
            self._emit("interrupt", None)

    # ═════════════════ 应答 ═════════════════
    async def respond_text(self, user_text: str) -> str:
        """文本进 → LLM → TTS → 播放，返回回复文本"""
        if self.llm is None:
            raise RuntimeError("LLM 未初始化，请先调用 ServiceContext.init_all()")
        t0 = time.perf_counter()
        reply = await self.llm.async_chat(self.build_messages(user_text))
        self._emit("llm", {"text": reply, "ms": round((time.perf_counter() - t0) * 1000)})

        self.history.append("user", user_text)
        self.history.append("assistant", reply)
        self.history.save()
        self.last_reply = reply

        if self.tts is not None:
            t1 = time.perf_counter()
            path = await self.tts.async_synthesize(reply)
            self._emit("tts", {"path": path, "ms": round((time.perf_counter() - t1) * 1000)})
            self._start_playback(path)
        return reply

    async def handle_segment(self, segment_bytes: bytes) -> Optional[str]:
        """一段语音 bytes → ASR → 应答"""
        if self.asr is None:
            raise RuntimeError("ASR 未初始化，请先调用 ServiceContext.init_all()")
        audio = np.frombuffer(segment_bytes, dtype=np.float32)
        duration = len(audio) / self.sample_rate
        if duration < self.config.app_config.min_utterance_seconds:
            return None  # 太短，多半是咳嗽/噪声

        t0 = time.perf_counter()
        user_text = (await self.asr.async_transcribe_np(audio) or "").strip()
        self._emit("asr", {"text": user_text, "ms": round((time.perf_counter() - t0) * 1000)})
        if not user_text:
            return None
        return await self.respond_text(user_text)

    # ═════════════════ 主循环 ═════════════════
    async def run(self) -> None:
        """读麦克风 → 逐块喂 VAD → 有语音段就应答（阻塞直到 stop()）"""
        if self.vad is None or self.asr is None or self.llm is None:
            raise RuntimeError("管线未初始化，请先调用 ServiceContext.init_all()")

        self._running = True
        self.mic.start()
        self._emit("listening", None)
        try:
            while self._running:
                item = await asyncio.to_thread(self.mic.get, 0.5)
                if item is None:
                    continue
                audio_np, chunk_bytes = item
                segment = self.vad.process_block(audio_np, chunk_bytes)

                # 播放回复时用户在说话 → 打断
                if (
                    self.config.app_config.barge_in
                    and self._is_playing()
                    and self.vad.is_speaking()
                ):
                    self.interrupt()

                if segment:
                    try:
                        await self.handle_segment(segment)
                    except Exception as e:
                        self._emit("error", str(e))
        finally:
            self._running = False
            self.interrupt()
            self.mic.stop()
            self.history.save()

    def stop(self) -> None:
        """请求停止主循环（下一次循环检查时退出）"""
        self._running = False
