# ADR 0007：按 GPT-SoVITS 职责组织核心代码

状态：已接受。日期：2026-10-03。更新 [ADR 0002](0002-product-layout.md) 的源码布局，保留 [ADR 0004](0004-composable-components.md) 的设备与发行边界。

## 原因

完整兼容 GPT-SoVITS 需要能直接定位上游行为在本项目中的实现。此前 `engine.py` 同时承载公共 Engine、服务配置、转换缓存与 Inference，`frontend/text_frontend.py` 混合切分、语言检测与特征组合，`_internal/` 混放算法和进程设施，跨文件比对和扩展容易遗漏调用方。

上游参考固定为 [`48b1a016`](https://github.com/RVC-Boss/GPT-SoVITS/tree/48b1a0169a28582a8984402f82cf438d3bfa6aca)。Genie 的角色调用方式可作为使用体验参考，但本项目的核心边界按 GPT-SoVITS 的配置、权重、参考和推理行为组织。

## 决定

核心 Python 包从 `src/sakuratts/` 移到根目录 `sakuratts/`，与根目录服务入口相邻。保留包名和已公开 Python / CLI 入口，更新 setuptools、源码包和整合包构建路径。

`TTS_infer_pack` 承载配置、模型与请求编排。其 `TTS.py` 管理 Inference 的激活、切换和请求；`TextPreprocessor.py` 组合目标文本及特征；`text_segmentation_method.py` 放固定上游切分规则。语言实现归入 `text`，语言分段单列 `LangSegmenter.py`。语义生成与采样归入 `AR`，共用声学与资源格式归入 `module`。

不同设备的计算实现保留在 `backends`；工作进程、IPC、取消、休眠和安装位置归入 `runtime`；转换与准备归入 `prepare`。CPU 与 DirectML 的共用装配放在 `backends/ort.py`，两个设备分别提供自己的引擎入口。声学进程客户端与 worker 同放 `runtime`；环境与资源检查集中在 `diagnostics`，CLI 只负责参数与操作分发。公共 Engine 的完整请求测量放在包顶层 `benchmark.py`。转换缓存的身份计算和发布由 `prepare/cache.py` 负责，Inference 只使用准备结果。现行文件职责与调用链由[架构说明](../architecture.md)维护。

目录对齐不要求复制上游全局状态、启动时的重依赖导入或训练环境。独立解释器、按需导入、单请求所有权、模型切换恢复及资源回收继续满足[推理契约](../specs/inference-contract.md)。新增模型或语言直接扩展对应职责，不通过空目录或尚无消费者的插件层表示支持。

## 兼容与代价

模型描述、资源目录、HTTP 字段及默认值、采样和数值实现保持现有行为。`Engine`、`Model`、`Audio`、转换函数、NVIDIA 入口和 CLI 保留。原 `sakuratts.engine.Inference` 与配置读取函数继续转发到唯一实现；内部导入路径随代码迁移，不保留整套重复目录。开发安装需重新执行 `pip install -e .` 更新包位置。

文件迁移会改变转换脚本的来源哈希，依赖这些哈希的缓存可能重新准备一次；已有模型及参考资源仍可直接加载，不删除用户缓存。冻结实验报告和原始数据保留历史身份，研究工具的可执行导入更新到现行目录。

此决定完成代码组织调整，不表示完整兼容已达成。旧版 API、原版 Python 接口、Gradio、模型家族、语言及训练工具联动的缺口统一记录在[兼容矩阵](../specs/compatibility-matrix.md)。
