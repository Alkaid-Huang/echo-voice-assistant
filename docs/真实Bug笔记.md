# 真实 Bug 笔记

> 记录开发过程中真实踩到的坑（不是预埋题目）。
> 格式：现象 → 假设 → 证据 → 根因 → 修复 → 防回归。

---

## Bug #1：device="auto" 选了 CUDA，但机器缺 cuBLAS，推理时崩溃

**发现时间**：2026-09-10

**现象**

`python main.py --mode asr --wav recording.wav` 在模型加载阶段正常，
推理到一半抛异常退出：

```
File "...\faster_whisper\transcribe.py", line 1400, in encode
    return self.model.encode(features, to_cpu=to_cpu)
RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
```

**假设**

1. 模型文件损坏？—— 但之前用同一份模型在 CPU 上转写成功过。
2. 配置写错了？—— `conf.yaml` 里是 `device: "auto"`，不是 `cpu`。
3. 更可能：`auto` 探测到本机有 NVIDIA 显卡 → 选了 CUDA，
   但 CTranslate2 需要 cuBLAS 12 / cuDNN 运行库，本机没装。

**证据**

- 之前显式传 `device="cpu"` 的两次转写都成功（5 秒音频 1.5 秒）。
- 报错发生在 `model.encode(...)`（真正推理时），而不是加载时——
  说明模型能建出来，只有 CUDA 算子跑不起来。
- 报错信息直接指向 `cublas64_12.dll`，即缺 CUDA 运行库。

**根因**

`device="auto"` 的语义是"有可用 GPU 就用 GPU"，但它只检查硬件，
不检查 CUDA 运行库是否齐全；缺库时错误要等到第一次推理才爆出来。
项目原本的降级逻辑只包住了"创建模型"这一步，包不住"推理"这一步。

**修复**

1. `FasterWhisperASR` 增加**运行期**降级：转写时捕获设备类 `RuntimeError`，
   用 CPU + int8 重建模型并重试一次，同时打印降级提示。
2. `conf.yaml` 默认改成 `device: "cpu"`（稳定优先），
   并在注释里说明 GPU 需要装 cuBLAS/cuDNN。

**防回归**

新增单测 `test_faster_whisper_falls_back_to_cpu`：伪造一个"第一次调用抛
cublas 错误、CPU 上正常返回"的模型，断言引擎最终用 CPU 出结果。

**面试可讲点**

"自动选设备"这类"智能默认值"最容易藏坑：它的失败点不在初始化，而在第一次
真正计算时。处理这类问题要把异常边界放在**能用得上回退的地方**，
而不是放在创建对象的地方。

---

## Bug #2：silero-vad 要求 Tensor，代码传了 numpy，VAD 一推理就崩

**发现时间**：2026-09-11（首次真机语音对话）

**现象**

`run.bat --mode console` 启动即失败：

```
启动失败：forward() Expected a value of type 'Tensor' for argument 'x'
but instead found type 'ndarray'. Position: 1
```

**为什么之前没暴露**

- 单元测试用的是 `MockVAD`，不碰真实模型；
- `--mode check` 只**加载**模型（构造 `SileroVADEngine`），不做一次推理；
- 直到第一次真的对着麦克风说话，VAD 才开始逐帧调用模型。

**根因（两个问题叠在一起）**

1. `silero_vad` 的模型是 TorchScript 模型，入参必须是 `torch.Tensor`，
   而 `process_block` 直接传了 numpy 数组（不同版本对 numpy 宽容度不同，所以"换个环境就崩"）；
2. 修好类型后暴露出更深的一层：**模型第二个参数是采样率**（只接受 8000/16000），
   而代码传的是"每帧采样点数 512"，于是报
   `ValueError: Supported sampling rates: [8000, 16000]`。

这两个错误从课程早期的 VAD 实现里就存在，因为从来没跑过一次真实推理而一直隐藏。

**修复**

1. `SileroVADEngine.process_block` 统一把 numpy 转成 `torch.Tensor`（float32、连续内存），
   对已经是 Tensor 的输入保持兼容；
2. 引擎新增 `sample_rate` 参数并校验"采样率 ↔ 帧长"必须匹配（16k→512、8k→256），
   把配置错误提前到启动阶段，而不是等到推理时报晦涩的模型错误；
3. 配置里新增 `vad_config.silero.sample_rate`，并写入 `conf.yaml`。

**防回归**

新增 `tests/test_silero_vad.py`：

1. 单帧 numpy 静音调用 `process_block` 不抛异常，且不产出语音段；
2. `detect_speech` 生成器对多帧静音跑通。

**面试可讲点**

"加载成功 ≠ 能推理"。凡是第三方推理库，都要有一次**真实前向调用**的测试；
否则 mock 测试和"只构造对象"的冒烟测试都会给出虚假的安全感。
