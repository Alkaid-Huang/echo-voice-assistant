"""
单次链路延迟测量：LLM 生成 + TTS 合成

用法（项目根目录）：
    python benchmarks/benchmark_once.py --text "用一句话介绍你自己"

输出冷启动（含建连）与热启动两次 LLM 耗时，以及 TTS 合成耗时，
用于给 benchmarks/results.md 提供可复现的数字。
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml

from echo.config import EchoConfig
from echo.env_loader import load_dotenv
from echo.service_context import ServiceContext


def load_config() -> EchoConfig:
    with open(ROOT / "conf.yaml", "r", encoding="utf-8") as f:
        return EchoConfig.model_validate(yaml.safe_load(f))


async def main_async(text: str, repeat: int) -> None:
    load_dotenv(str(ROOT / ".env"))
    config = load_config()

    ctx = ServiceContext(config)
    ctx.init_llm()
    ctx.init_tts()

    messages = [
        {"role": "system", "content": config.llm_config.system_prompt},
        {"role": "user", "content": text},
    ]

    reply = ""
    for i in range(repeat):
        t0 = time.perf_counter()
        reply = await ctx.llm_engine.async_chat(messages)
        cost_ms = (time.perf_counter() - t0) * 1000
        label = "冷启动（含建连）" if i == 0 else f"热启动第 {i} 次"
        print(f"LLM {label}: {cost_ms:.0f} ms")

    print(f"回复内容: {reply}")

    if ctx.tts_engine is not None:
        t0 = time.perf_counter()
        path = await ctx.tts_engine.async_synthesize(reply)
        cost_ms = (time.perf_counter() - t0) * 1000
        size = os.path.getsize(path)
        print(f"TTS 合成: {cost_ms:.0f} ms -> {path} ({size} 字节)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="用一句话介绍你自己")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main_async(args.text, args.repeat))
