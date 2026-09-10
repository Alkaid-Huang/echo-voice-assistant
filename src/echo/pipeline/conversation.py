"""
ConversationPipeline —— VAD → ASR → LLM → TTS 的端到端对话回路（支持打断）

数据流：
    麦克风(16k) → VAD 切段 → ASR 转文字 → LLM 生成回复 → TTS 合成 → 播放

编排模型（对照 Pipecat 的处理器并发模型改造）：
    采集/VAD 与 识别/生成/合成 是两条并发流水线，中间用队列传语音段。
    这样"思考"期间不会停止听，避免采集队列溢出丢音频。

双工模式（duplex_mode）：
    half     —— 播放时暂停采集（外放安全，默认）
    barge_in —— 播放时继续听，用户开口即打断（建议戴耳机）
    full     —— 完全并发（需要回声消除，当前仅预留）

收尾保护：
    max_utterance_seconds —— 单句超长强制切分
    audio_idle_timeout    —— 说话中音频断流时强制收尾并重启采集

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
            app.sample_rate, app.block_size_samples, device=app.input_device
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
        # 并发流水线状态
        self._segment_queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=4)
        self._mic_paused = False
        self._utterance_started_at: Optional[float] = None
        self._dropped_segments = 0
        self._restart_cooldown_until = 0.0
        self._last_stats_ts = 0.0

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

    # ═════════════════ 并发主循环 ═════════════════
    async def run(self) -> None:
        """采集/VAD 与 应答 两条流水线并发运行（阻塞直到 stop()）"""
        if self.vad is None or self.asr is None or self.llm is None:
            raise RuntimeError("管线未初始化，请先调用 ServiceContext.init_all()")

        app = self.config.app_config
        self._running = True
        self.mic.start()
        print(f"[麦克风] {getattr(self.mic, 'device_info', '未知设备')}", flush=True)
        print(f"[双工] 模式 = {app.duplex_mode}", flush=True)
        self._emit("listening", None)
        try:
            await asyncio.gather(self._capture_loop(), self._respond_loop())
        finally:
            self._running = False
            self.interrupt()
            self.mic.stop()
            self.history.save()
            print(
                f"[统计] 丢块 {getattr(self.mic, 'dropped_blocks', 0)} 个，"
                f"丢弃语音段 {self._dropped_segments} 段",
                flush=True,
            )

    async def _capture_loop(self) -> None:
        """只负责采集与 VAD：产出语音段放进队列，绝不被推理阻塞"""
        app = self.config.app_config
        while self._running:
            # half 模式：播放期间关闭采集，避免"自我对话"与声卡双流争用
            if app.duplex_mode == "half":
                if self._is_playing() and not self._mic_paused:
                    self.mic.stop()
                    self._mic_paused = True
                    print("[双工] 播放中，暂停采集（half 模式）", flush=True)
                elif not self._is_playing() and self._mic_paused:
                    self.mic.start()
                    self._mic_paused = False
                    print("[双工] 播放结束，恢复采集", flush=True)
                if self._mic_paused:
                    await asyncio.sleep(0.05)
                    continue

            item = await asyncio.to_thread(self.mic.get, 0.5)
            if item is None:
                await self._enforce_max_utterance()
                await self._handle_mic_idle()
                continue

            audio_np, chunk_bytes = item
            segment = self.vad.process_block(audio_np, chunk_bytes)

            # 仅 barge_in 模式允许插话打断
            if (
                app.duplex_mode == "barge_in"
                and self._is_playing()
                and self.vad.is_speaking()
            ):
                self.interrupt()

            # 保护一：单句超长强制切分，避免状态机长时间挂在 ACTIVE
            await self._enforce_max_utterance()

            if segment:
                self._utterance_started_at = None
                await self._push_segment(segment)

            self._maybe_report_stats()

    async def _enforce_max_utterance(self) -> None:
        """
        单句超过 max_utterance_seconds 就强制切分。

        注意：每轮循环都要检查（包括"收不到音频块"的轮次），
        否则音频断流时这段保护永远不会触发（真实 bug，由单测抓出）。
        """
        if not self.vad.is_speaking():
            self._utterance_started_at = None
            return
        now = time.monotonic()
        if self._utterance_started_at is None:
            self._utterance_started_at = now
            return
        if now - self._utterance_started_at > self.config.app_config.max_utterance_seconds:
            print("[VAD] 超过单句最长时长，强制切分", flush=True)
            await self._flush_and_push()

    async def _respond_loop(self) -> None:
        """只负责应答：从队列取语音段，串行执行 ASR → LLM → TTS → 播放"""
        while self._running:
            try:
                segment = await asyncio.wait_for(self._segment_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                await self.handle_segment(segment)
            except Exception as e:
                self._emit("error", str(e))

    async def _push_segment(self, segment: bytes) -> None:
        """语音段入队；队列满时丢最旧的一段并计数（让"漏听"可见）"""
        if self._segment_queue.full():
            try:
                self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._dropped_segments += 1
            print(
                f"[管线] 语音段积压，丢弃最旧一段（累计 {self._dropped_segments}）",
                flush=True,
            )
        await self._segment_queue.put(segment)

    async def _flush_and_push(self) -> None:
        """强制收尾当前语音段并送进应答队列"""
        self._utterance_started_at = None
        forced = self.vad.force_flush()
        if forced:
            await self._push_segment(forced)

    async def _handle_mic_idle(self) -> None:
        """
        麦克风长时间没有数据：先抢救正在说的那句话，再重启采集流。

        对应 Pipecat 的 audio_idle_timeout（SPEAKING 状态无音频就强制结束）。
        """
        app = self.config.app_config
        if not hasattr(self.mic, "is_stalled"):
            return
        if not self.mic.is_stalled(app.audio_idle_timeout):
            return
        if self.vad.is_speaking():
            print("[VAD] 采集断流，强制结束当前语音段", flush=True)
            await self._flush_and_push()
        now = time.monotonic()
        if now < self._restart_cooldown_until:
            return
        self._restart_cooldown_until = now + 5.0  # 防止反复重启
        print("[麦克风] 采集流中断，正在重启…", flush=True)
        try:
            self.mic.stop()
            self.mic.start()
            self._emit("error", "麦克风采集已重启")
        except Exception as e:
            self._emit("error", f"麦克风重启失败: {e}")

    def _maybe_report_stats(self) -> None:
        """每 30 秒报一次丢块情况，便于发现"悄悄漏听" """
        now = time.monotonic()
        if now - self._last_stats_ts < 30.0:
            return
        self._last_stats_ts = now
        dropped = getattr(self.mic, "dropped_blocks", 0)
        if dropped or self._dropped_segments:
            print(
                f"[统计] 采集丢块 {dropped} 个，丢弃语音段 {self._dropped_segments} 段",
                flush=True,
            )

    def stop(self) -> None:
        """请求停止主循环（下一次循环检查时退出）"""
        self._running = False
