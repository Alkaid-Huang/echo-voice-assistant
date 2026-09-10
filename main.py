"""
Echo 演示入口 —— 语音对话 / 文本对话 / 单点测试

用法（仓库根目录、激活 venv 后）：
    python main.py --mode check                     检查配置与组件是否创建成功
    python main.py --mode text                      文本对话（不占麦克风，先跑通它）
    python main.py --mode console                   语音对话：mic → VAD → ASR → LLM → TTS
    python main.py --mode tts --text "你好，我是 Echo"
    python main.py --mode asr --wav 你的录音.wav       # 16k 单声道 wav，其它格式会自动重采样
"""
import argparse
import asyncio
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import yaml

from echo.audio.io import load_audio_mono, play_file
from echo.config import EchoConfig
from echo.pipeline.conversation import ConversationPipeline
from echo.service_context import ServiceContext


def load_config(path: str = "conf.yaml") -> EchoConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EchoConfig.model_validate(raw)


def build_context(config: EchoConfig) -> ServiceContext:
    ctx = ServiceContext(config)
    ctx.init_all()
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
    ctx = build_context(config)
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


async def mode_tts(config: EchoConfig, text: str) -> None:
    ctx = build_context(config)
    if ctx.tts_engine is None:
        raise RuntimeError("TTS 未启用")
    path = await ctx.tts_engine.async_synthesize(text)
    print(f"生成音频: {path}")
    play_file(path)


async def mode_asr(config: EchoConfig, wav: str) -> None:
    ctx = build_context(config)
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
        choices=["check", "text", "console", "tts", "asr"],
    )
    parser.add_argument("--config", default="conf.yaml")
    parser.add_argument("--text", default="你好，我是 Echo，很高兴认识你。")
    parser.add_argument("--wav", default=None, help="--mode asr 时必填：音频文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode == "check":
        asyncio.run(mode_check(config))
    elif args.mode == "text":
        asyncio.run(mode_text(config))
    elif args.mode == "console":
        asyncio.run(mode_console(config))
    elif args.mode == "tts":
        asyncio.run(mode_tts(config, args.text))
    elif args.mode == "asr":
        if not args.wav:
            parser.error("--mode asr 需要 --wav <音频文件路径>")
        asyncio.run(mode_asr(config, args.wav))


if __name__ == "__main__":
    main()
