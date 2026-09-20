# Windows WDDM 显存归因与长句后保留量

日期：2026-09-20。两次自然生成探针均使用 GPT FP16、baseline attention、已通过工程筛查的声学 FP16 包，按“短句 → 长句 → 短句 → 卸载 → 关闭”采样。主要发现是：长句后保留的 Dedicated Usage 增量主要出现在声学 worker；GPT 的固定 KV 也有余量，但不是当前最大的回收机会。

这份报告汇总本轮两次实机探针，整理报告时没有重复运行 GPU。[汇总 JSON](data/2026-09-20-windows-wddm-memory.json) 保存逐角色 / LUID 的原始字节值、缺失实例、适配器行、进程 RSS、CuPy pool、PCM 哈希和执行源码哈希。

## 计量边界

原始结果分别为 `outputs/windows-wddm-memory/both-fp16-worker/result.json` 和 `both-fp16-shared/result.json`，SHA256 为：

- 独立 worker：`ce307fe1c26808fc944dadfed1c39663a26a63acaf413a0df579f690c327458c`。
- 共享进程：`8d76748ada5e9344c52025c609743f8c92ccdfd531c0e18e1d26ebd3da673da2`。

两轮配置哈希相同，均完成，运行期间记录的源码哈希没有变化。后续清理代码即使改变，也不能用当前源码哈希替代这两份结果内的 `source_sha256`。只读查询 `Win32_OperatingSystem` 确认本机为 Microsoft Windows 11 专业工作站版，版本 `10.0.22631`、build `22631`；这个 OS 查询在探针后进行，不是原探针内置字段。

探针在每个边界等待 150 ms 后读取 PDH，再读取全卡 `nvidia-smi`。它测到的是边界状态，不能替代请求期间的峰值采样，也不用于推理速度比较。

读取的三个进程计数器分别为 `Dedicated Usage`、`Shared Usage`、`Total Committed`。保持三者各自含义，不把 committed 当作实际驻留显存。[Microsoft 原文](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)明确说明，进程值包含私有和跨进程共享分配，跨 PID 相加会重复计数；“跨进程共享分配”也不等于 GPU 使用系统内存的 Shared Usage。因此本报告不提供 PID 字节总和，也不将这些值称为进程独占 VRAM。

[KB4490156](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/gpu-process-memory-counters-report-wrong-value)描述 Windows 10 version 1709 及以后版本中 GPU Process Memory 可能错误显示持续增长的已知问题，页面适用范围仍写 Windows 10。本机是 Windows 11 build 22631，不能直接宣称它受该问题影响，也不能反过来把这篇 KB 当作本机计数器绝对准确的保证。出现异常时应结合全卡数据、分配器及 WPR/WPA 核对。

原记录包含三个适配器 LUID，不能按行序号映射成 CUDA ordinal。本次相关执行进程的有效行均位于 `0x00000000_0x0001df70`、physical adapter 0；其他适配器行在 JSON 中分别保留，没有跨 LUID 求和。

## 每个角色的边界结果

独立运行的角色为主进程 PID 46108、前端 PID 5456、声学 PID 40204。共享运行的主进程为 PID 41440、前端 PID 45048，没有独立声学 PID。模型加载前均无匹配的 GPU 进程实例；加载后独立方案有两个有效进程实例，共享方案有一个，均属于上述 LUID。前端没有匹配实例，这表示缺失，不是测得零字节。

下表只列各角色的 Dedicated Usage，单位 MiB：

| 边界 | 独立方案 GPT 主进程 | 独立方案声学 worker | 共享主进程 |
| --- | ---: | ---: | ---: |
| 模型加载后 | 263.53 | 241.52 | 401.53 |
| 首个短句后 | 377.53 | 331.52 | 609.54 |
| 长句后 | 431.53 | 871.52 | 1201.54 |
| 再次短句后 | 431.53 | 871.52 | 1201.54 |
| 显式卸载后 | 101.53 | 进程退出，实例缺失 | 105.53 |
| 引擎关闭后 | 101.53 | 进程退出，实例缺失 | 105.53 |

这是三个进程各自的计数，不应将前两列相加后与第三列比较。可直接比较同一个 PID 的变化：声学 worker 从短句后到长句后增加 540 MiB，并在下一次短句后保留；GPT 主进程同期增加 54 MiB。这里尚未区分声学权重、图激活、cuDNN 工作区和 ORT arena 的各项分配，不能把 540 MiB 全部认定成某一个算子。

独立方案主进程的 Shared Usage 在模型加载至关闭后一直为 82 MiB，声学 worker 活跃时为 76 MiB。共享主进程从加载后的 82 MiB 增至请求后的 84 MiB，卸载后回到 82 MiB。独立方案卸载后声学 PID 消失，关闭后前端也退出；共享方案仅在关闭时退出前端。两边仍活着的主进程都有计数，因此模型卸载不等于 CUDA context 退出。

主存也按角色分别保留：

| 边界 | 独立 GPT 主进程 RSS MiB | 独立声学 worker RSS MiB | 共享主进程 RSS MiB |
| --- | ---: | ---: | ---: |
| 首个短句后 | 916.76 | 1107.48 | 1455.14 |
| 长句后 | 919.19 | 1144.75 | 1493.12 |
| 卸载后 | 918.11 | 已退出 | 1472.29 |
| 关闭后 | 790.84 | 已退出 | 1344.98 |

CuPy pool 在两轮卸载后都回到 used=0、total=0，而主进程仍保留主存与 GPU 计数。剩余部分不在这个 CuPy pool 中；本轮没有继续定位原生库和驱动分配，不把它直接称为泄漏。共享方案的主存回收边界与独立 worker 不同，和[此前资源对照](2026-09-20-windows-ort-abi-comparison.md)的观察一致。

三对请求记录的 PCM SHA256 全部相同：短句 79 个语义 token、110,720 个 PCM 样本；长句 629 个语义 token、814,720 个 PCM 样本；最后短句回到相同哈希。参数、参考身份、采样与语义 token、停止原因和输出形状也一致。探针没有另存 WAV，这里比较的是记录的哈希，没有重新进行波形或音质分析。

## KV 和工作区是否分配过多

已核对与探针记录哈希相同的 `cuda_gpt.py`。默认 capacity=2048，24 层、16 头、head dimension=32，FP16 K/V 实际固定分配为：

`2 × 24 × 16 × 2048 × 32 × 2 bytes = 96 MiB`。

固定 Decode 工作区只有 15,364 bytes，状态数组另有 12 bytes，不是主要显存占用。它没有为每个 token 保留一份完整 attention scores；baseline 内核使用的 shared memory 也不能按全模型常驻张量计算。

两轮 CuPy live allocation 在首个短句后均为 259,989,504 bytes，长句和后续短句不再增加；pool total 从 277,232,640 增至 330,963,456 bytes。长句后的 pool free 为 70,973,952 bytes，约 67.69 MiB，反映 Prefill 等临时分配留下的缓存。候选可以只在请求结束或空闲时释放 pool 的空闲块，保留 K/V 和 Graph；实际可归还量仍须测量，不能把全部 free 数字直接当作确定的 WDDM 降幅。

KV 确有余量，但节省规模有限：capacity 从 2048 改成 1536，FP16 只少 24 MiB；512 对短句可少 72 MiB，但当前长句使用约 1158 个槽位，1024 已不够。不能为省显存截断文本或缩短生成；若要自动缩容，需要在安全边界增长 / 分桶并重新捕获 Graph。对于当前目标，先设置显式小容量做同请求验证，比立即引入分页 KV 简单。

## 建议先实现的候选

优先做一个显式的 ORT arena 回收实验。[ORT 1.19.2 对应提交的头文件](https://github.com/microsoft/onnxruntime/blob/ffceed9d44/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h)已定义 `memory.enable_memory_arena_shrinkage`，支持值 `gpu:0`。Python 路径可通过 `RunOptions.add_run_config_entry()` 将它传给选定的声学 `session.run()`。

[ORT 官方说明](https://onnxruntime.ai/docs/get-started/with-c.html#features)指出，arena 默认不收缩，这个选项在指定 Run 结束时检查并归还空闲区域，正是为偶发大动态请求抬高后续常驻内存而提供。对应版本的 [BFCArena 实现](https://github.com/microsoft/onnxruntime/blob/ffceed9d44/onnxruntime/core/framework/bfc_arena.cc)还明确：当前使用的 `kSameAsRequested` 会把全部空闲 allocation region 纳入回收候选，包括第一个区域；区域内仍有活跃 chunk 时不会释放。

它首先解决长句后的保留量，不保证降低长句执行中的真实峰值。应单独记录新选项并重新跑 FP16 工程筛查，再比较“短 → 长 → 短”的同 PID 计数、主存、重复性和下一次请求耗时。若真正峰值仍高，下一步再按声学阶段定位激活 / 工作区，验证有限感受野 vocoder 分块。现有运行包中约 829 MiB 的重复 DLL 属于磁盘体积问题，不能算作这项显存优化的收益。
