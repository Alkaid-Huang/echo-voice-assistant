# Echo 延迟基线

> 先测量、再优化。每次接入新模块都补一条记录。

## 2026-09-08 · ASR 首次真实转写

**环境**：Windows / Python 3.14.6 / faster-whisper 1.2.1 / ctranslate2 4.8.2 / CPU int8
**模型**：Systran/faster-whisper-small（缓存于 `models/whisper`）
**输入**：`recording.wav`（16kHz 单声道 16bit，5.0 秒人声）

| 指标 | 数值 |
|------|------|
| 首次模型加载 | 1.87 s |
| 5 秒音频转写 | 1.5 s |

**识别结果**：`字幕by索兰娅`

**结论**：5 秒音频 CPU 转写约 1.5 秒，单句延迟预算乐观；模型常驻后加载时间不计入链路。

## 2026-09-10 · TTS 首次真实合成

**环境**：同前 / edge-tts 7.2.8（在线）
**输入**：`你好，我是 Echo，很高兴认识你。`（约 15 字）
**产出**：`outputs/tts_*.mp3`，20,736 字节

**结论**：合成耗时约 1–2 秒（含网络往返），可作为端到端预算中的固定项。

## 待补充

## 2026-09-11 · LLM（DeepSeek API）+ TTS 真实延迟

**环境**：Windows / Python 3.14 / DeepSeek `deepseek-chat`（OpenAI 兼容 API）/ edge-tts
**输入**：`用一句话介绍你自己，要口语化`
**脚本**：`python benchmarks/benchmark_once.py --text "..." --repeat 3`（可复现）

| 指标 | 数值 |
|------|------|
| LLM 冷启动（含建连） | 859 ms |
| LLM 热启动第 1 次 | 913 ms |
| LLM 热启动第 2 次 | 588 ms |
| TTS 合成（约 25 字回复） | 2724 ms |
| TTS 产出 | `outputs/tts_*.mp3`，31,824 字节 |

**结论**：

1. LLM 整句生成约 0.6–0.9s，比预期快；
2. **TTS 合成 2.7s 是当前最大单项开销**，优化优先级应调整为「TTS 分句流式 > LLM 流式」；
3. 端到端预算（VAD 收尾约 0.8s + ASR 约 1.5s + LLM 约 0.7s + TTS 约 2.7s）已超过 2s 目标，
   必须靠流式（先出声、边合成边播）才能达标。

## 待补充

- 端到端（说话结束 → 听到回复）总延迟的实测分解：VAD 收尾 + ASR + LLM + TTS + 播放启动
- 打断响应（用户开口 → 播放停止）实测值
- 内存占用基线
