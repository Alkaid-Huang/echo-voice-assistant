"""
Echo 演示入口 —— 语音对话 / 文本对话 / 单点测试

用法（仓库根目录、激活 venv 后）：
    python main.py --mode check                     检查配置与组件是否创建成功
    python main.py --mode llm --text "你好"          只测大模型接口（排查 API 配置最快）
    python main.py --mode text                      文本对话（不占麦克风，先跑通它）
    python main.py --mode console                   语音对话：mic → VAD → ASR → LLM → TTS
    python main.py --mode tts --text "你好，我是 Echo"
    python main.py --mode asr --wav 你的录音.wav       # 16k 单声道 wav，其它格式会自动重采样
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml

from echo.audio.io import MicStream, load_audio_mono, play_file
from echo.config import EchoConfig
from echo.env_loader import load_dotenv
from echo.pipeline.conversation import ConversationPipeline
from echo.service_context import ServiceContext


def load_config(path: str = str(ROOT / "conf.yaml")) -> EchoConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EchoConfig.model_validate(raw)


def build_context(
    config: EchoConfig, components: tuple = ("vad", "asr", "llm", "tts")
) -> ServiceContext:
    """按需初始化组件，避免不必要的模型加载（--mode tts 不该加载 ASR）"""
    ctx = ServiceContext(config)
    for name in components:
        getattr(ctx, f"init_{name}")()
    return ctx


def make_printer():
    def printer(kind: str, payload=None) -> None:
        if payload is None:
            print(f"[{kind}]", flush=True)
        elif isinstance(payload, dict):
            text = payload.get("text") or payload.get("path") or ""
            ms = payload.get("ms")
            print(f"[{kind}] {text}" + (f"（{ms}ms）" if ms is not None else ""), flush=True)
        else:
            print(f"[{kind}] {payload}", flush=True)
    return printer


async def mode_check(config: EchoConfig) -> None:
    ctx = build_context(config)
    print(f"VAD : {type(ctx.vad_engine).__name__ if ctx.vad_engine else 'None'}")
    print(f"ASR : {type(ctx.asr_engine).__name__ if ctx.asr_engine else 'None'}")
    print(f"LLM : {type(ctx.llm_engine).__name__ if ctx.llm_engine else 'None'}")
    print(f"TTS : {type(ctx.tts_engine).__name__ if ctx.tts_engine else 'None'}")


async def mode_text(config: EchoConfig) -> None:
    # 文本模式只需要 LLM + TTS，不必加载 VAD/ASR 大模型
    ctx = build_context(config, ("llm", "tts"))
    pipeline = ConversationPipeline(ctx, on_event=make_printer())
    print("文本模式：输入内容回车；Ctrl+C 退出")
    while True:
        try:
            user = await asyncio.to_thread(input, "你: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user.strip():
            continue
        reply = await pipeline.respond_text(user.strip())
        print(f"Echo: {reply}")


async def mode_llm(config: EchoConfig, text: str) -> None:
    """只验证 LLM 接口是否通畅（排查 API 配置问题最快的方式）"""
    ctx = build_context(config, ("llm",))
    messages = [
        {"role": "system", "content": config.llm_config.system_prompt},
        {"role": "user", "content": text},
    ]
    reply = await ctx.llm_engine.async_chat(messages)
    print(f"LLM 回复: {reply}")


async def mode_console(config: EchoConfig) -> None:
    ctx = build_context(config)
    pipeline = ConversationPipeline(ctx, on_event=make_printer())
    print("语音模式：直接说话即可（建议戴耳机，避免扬声器回声触发打断）；Ctrl+C 退出")
    try:
        await pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


async def mode_mic(config: EchoConfig, seconds: float) -> None:
    """
    麦克风诊断：实时打印电平(dB)与语音概率(prob)，用于判断"收不到声音"的原因。

    判断方法：
      - 安静时 dB 应低于 -50；说话时应明显上升（高于 -35 比较理想）
      - dB 长期低于 -60 → 麦克风没收到声音（设备选错/权限/静音开关）
      - dB 正常但 prob 一直很低 → 环境噪声，或阈值需要调整
    """
    import time as _time

    import numpy as np

    ctx = build_context(config, ("vad",))
    engine = ctx.vad_engine
    silero_cfg = config.vad_config.silero
    db_threshold = silero_cfg.db_threshold if silero_cfg else -30.0
    prob_threshold = silero_cfg.prob_threshold if silero_cfg else 0.5

    mic = MicStream(
        config.app_config.sample_rate,
        config.app_config.block_size_samples,
        device=config.app_config.input_device,
    )
    mic.start()
    print(f"[麦克风] {mic.device_info}")
    print(f"诊断 {seconds:.0f} 秒：请正常说几句话，中间停顿一下")
    print(f"判定阈值：prob >= {prob_threshold} 且 dB >= {db_threshold}")
    try:
        start = _time.time()
        while _time.time() - start < seconds:
            item = await asyncio.to_thread(mic.get, 0.5)
            if item is None:
                continue
            audio, _ = item
            rms = float(np.sqrt(np.mean(audio**2)))
            db = 20 * np.log10(rms + 1e-7)
            prob = engine.speech_probability(audio) if hasattr(engine, "speech_probability") else -1.0
            flag = "  <== 判定为语音" if (prob >= prob_threshold and db >= db_threshold) else ""
            print(f"dB={db:7.1f}  prob={prob:4.2f}{flag}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
    print("诊断结束。dB 长期低于 -60 说明没收到声音；说话时 dB 明显上升即可正常工作。")


async def mode_tts(config: EchoConfig, text: str) -> None:
    ctx = build_context(config, ("tts",))
    if ctx.tts_engine is None:
        raise RuntimeError("TTS 未启用")
    path = await ctx.tts_engine.async_synthesize(text)
    print(f"生成音频: {path}")
    play_file(path)


async def mode_asr(config: EchoConfig, wav: str) -> None:
    ctx = build_context(config, ("asr",))
    if ctx.asr_engine is None:
        raise RuntimeError("ASR 未启用")
    audio = load_audio_mono(wav)
    text = await ctx.asr_engine.async_transcribe_np(audio)
    print(f"识别结果: {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo 语音对话演示")
    parser.add_argument(
        "--mode",
        default="check",
        choices=["check", "text", "llm", "console", "mic", "tts", "asr"],
    )
    parser.add_argument("--config", default=str(ROOT / "conf.yaml"))
    parser.add_argument("--text", default="你好，我是 Echo，很高兴认识你。")
    parser.add_argument("--wav", default=None, help="--mode asr 时必填：音频文件路径")
    parser.add_argument("--seconds", type=float, default=10.0, help="--mode mic 诊断时长（秒）")
    parser.add_argument("--device", type=int, default=None, help="麦克风设备编号（覆盖配置）")
    args = parser.parse_args()

    # 统一以项目根目录为工作目录，避免从别处启动时找不到 conf.yaml / .env / models
    os.chdir(ROOT)
    load_dotenv(str(ROOT / ".env"))  # 已存在的环境变量优先
    config = load_config(args.config)
    if args.device is not None:
        config.app_config.input_device = args.device
    try:
        if args.mode == "check":
            asyncio.run(mode_check(config))
        elif args.mode == "text":
            asyncio.run(mode_text(config))
        elif args.mode == "llm":
            asyncio.run(mode_llm(config, args.text))
        elif args.mode == "console":
            asyncio.run(mode_console(config))
        elif args.mode == "mic":
            asyncio.run(mode_mic(config, args.seconds))
        elif args.mode == "tts":
            asyncio.run(mode_tts(config, args.text))
        elif args.mode == "asr":
            if not args.wav:
                parser.error("--mode asr 需要 --wav <音频文件路径>")
            asyncio.run(mode_asr(config, args.wav))
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C 属于正常退出，不该甩一大段 traceback
        print("\n已退出。")
    except RuntimeError as e:
        # 配置/依赖类错误直接用可读提示，不甩 traceback
        print(f"启动失败：{e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
