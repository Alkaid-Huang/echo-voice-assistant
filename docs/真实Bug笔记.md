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
