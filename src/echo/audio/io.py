"""
音频输入输出 —— 麦克风采集与扬声器播放

基于 sounddevice（PortAudio）+ soundfile（libsndfile），
整条链路统一 16kHz / 单声道 / float32，与 ASRInterface 的约定一致。
"""
import os
import queue
import threading
from typing import Iterator, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
_QUEUE_MAXSIZE = 200  # 上游堵住时最多缓存 ~6.4 秒音频


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性插值重采样（够用于语音；高保真场景应换 soxr/librosa）"""
    if src_rate == dst_rate or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * dst_rate / src_rate))
    src_idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(src_idx, np.arange(len(x)), x).astype(np.float32)


def load_audio_mono(path: str, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """读取任意音频文件 → 单声道 float32 → 重采样到目标采样率"""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    return resample_linear(mono, sr, target_rate)


class MicStream:
    """16kHz 单声道麦克风采集流（可当同步迭代器用）"""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self._queue: "queue.Queue[Optional[Tuple[np.ndarray, bytes]]]" = queue.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    # ---- 内部：音频回调 ----
    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[MicStream] {status}", flush=True)
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        try:
            self._queue.put_nowait((mono, mono.tobytes()))
        except queue.Full:
            pass  # 消费不过来时直接丢，避免延迟无限增长

    # ---- 生命周期 ----
    def start(self) -> None:
        if self._running:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get(self, timeout: float = 0.5) -> Optional[Tuple[np.ndarray, bytes]]:
        """取一个音频块；超时返回 None（保持链路可中断）"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, bytes]]:
        while self._running:
            item = self.get()
            if item is not None:
                yield item

    def __enter__(self) -> "MicStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def play_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    stop_event: Optional[threading.Event] = None,
    block: int = 1024,
) -> None:
    """
    播放单声道 float32 音频。

    stop_event 被 set() 时立即停止播放——打断功能就靠它。
    """
    audio = np.asarray(audio, dtype=np.float32)
    with sd.OutputStream(
        samplerate=sample_rate, channels=1, dtype="float32"
    ) as stream:
        for start in range(0, len(audio), block):
            if stop_event is not None and stop_event.is_set():
                break
            stream.write(audio[start : start + block])


def play_file(
    path: str,
    stop_event: Optional[threading.Event] = None,
    target_rate: int = SAMPLE_RATE,
) -> None:
    """播放音频文件（自动重采样到 16k 单声道）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"音频文件不存在: {path}")
    play_audio(load_audio_mono(path, target_rate), target_rate, stop_event)
