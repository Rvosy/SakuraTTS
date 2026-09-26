# DirectML GPT：固定形状与显卡 KV 缓存

在 Radeon 780M、ORT 1.24.4 上，固定 Decode 形状并将 KV 缓存留在显卡后，短句固定历史的第二轮 16 步解码耗时为 **139.56 ms，平均每步 8.72 ms**。这次只测 GPT，不包含声学、文本前端或完整请求，也没有自然采样。原始记录、输入和图哈希见[紧凑证据](../experiments/data/2026-09-27-directml-gpt-static.json)。

本次探针使用 FP16 Transformer 图与 KV，CPU embedding 和采样接口仍为 FP32；缓存容量显式设为 512。完整输入与参考沿用已保存的真实日语短句。加载两个 Session 约 1.97 秒，首轮 Prefill 为 651.28 ms、16 步 Decode 为 187.45 ms；第二轮分别为 83.10 ms、139.56 ms。两轮所有 logits 有限且逐位相同。测试只有两轮，启用了 profile，未控制桌面负载、温度和功耗，不用于估计稳定延迟。

## 改变了什么

原动态 GPT 把各层有效 KV 前缀作为 CPU 输入，每一步交给 DirectML。它没有在 Python 中复制完整历史，但 ORT 仍需要把这些输入传到显卡。历史长度变化也使图中的 Shape / Gather 保留运行时工作。

静态图固定各层历史输入为 `[capacity + 1, heads, head_dim]`。第一个槽永久遮罩，尚未使用的槽初始化为零并遮罩；当前 token 的注意力另占一个放行槽。ScatterND 将本步 K/V 写入指定位置，完整输出绑定到显卡上的另一组缓存。每步交换两组缓冲区，不假定输入与输出可以原地别名。图仍会在显卡内读写完整缓存，不能称为零复制。

Prefill 使用独立的 DirectML 动态 Session。它的 KV 先返回 CPU，再一次性上传到固定显卡缓存。Decode 每步只上传 hidden、mask 和写入位置，返回 logits；48 份历史 KV 不逐步回到 CPU。两套 Transformer Session 和两组 KV 增加了常驻内存，本次没有量化其工作集或 WDDM 占用。

本机 profile 中，32 次 Decode 各执行一个 DirectML 融合节点，没有 CPU 节点；缓存 OrtValue 的设备均为 `dml`。固定形状和缓存驻留同时发生变化，本实验没有隔离两者各自的收益。[ORT DirectML 文档](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#performance-tuning)说明了已知形状对常量折叠、数据传输和图优化的影响。

## 首轮 NaN 与修复

初版只在 Prefill 后绑定一次 CPU hidden、mask、write_index，Decode 时修改原 NumPy 数组。首轮 Prefill 有限，但所有 Decode logits 都是 NaN；第二轮则有限。逐元素回读确认 GPU KV 上传正确，改用 ORT 自动分配的 CPU logits 输出也未消除问题。

原因是绑定输入时已产生设备副本，随后修改 CPU 数组不会刷新该副本。初始 mask 全为负无穷，第二轮偶然沿用上一轮的有效 mask。[ORT 1.24.4 的 IOBinding 实现](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.24.4/onnxruntime/core/session/IOBinding.cc)在 `BindInput` 中调用 `CopyOneInputAcrossDevices`。

修复后，每步更新三个小输入，再重新绑定；KV 继续绑定显卡 OrtValue。独立新进程的第一、第二轮均有限，逐步结果完全相同。没有采用预热跳过首轮错误。初版失败及两次排查结果保存在 `.cache/directml-gpt-static-fp16-*`，紧凑证据记录对应哈希；初版汇总中的 `logits_finite` 只代表最后一轮，不能用它掩盖首轮失败。当前工具逐轮记录并汇总全部结果。

静态与动态 DirectML FP16 在相同输入、相同 16 步历史下的最大 logits 绝对差为 0.0546875，RMSE 为 0.019225，17 个位置没有 argmax 分歧，Prefill 结果相同。固定形状会改变融合与归约方式；该检查不足以保证自然采样序列、语音内容或听感相同。

## 实现与边界

正式实现为 [`StaticDirectMLGPT`](../../src/sakuratts/backends/directml/static_gpt.py)，图生成器为 [`export_gpt_directml`](../../src/sakuratts/_internal/conversion/export_gpt_directml.py)。研究驱动复用这两个实现，不保留另一套解码逻辑。

```powershell
python -m sakuratts._internal.conversion.export_gpt_directml --gpt models/navi-cpu-amd/gpt --precision fp16 --capacity 2048
```

输出到新的 `gpt/directml-fp16-cap2048/`；FP32 可单独导出。加载器核对原 GPT 包、所选动态 ONNX sidecar、静态图和声明的容量，配置不匹配会报错，不截断输入。默认容量为 2048。本页的 512 容量短探针不能代表默认容量的耗时，也放不下已保存的长句完整历史。

5 项局部测试通过：真实小型 CPU ONNX 图对照独立 FP64 计算，覆盖 FP32 / FP16 首次请求、重复请求、容量、失败清理、文件身份及保留异常 traceback 时的 Session 回收。设备绑定由会在绑定时复制输入的替身模拟，专门覆盖这次首轮错误。完整请求、默认容量、资源释放与试听结果由后续整链记录负责；本页不作音质验收结论。
