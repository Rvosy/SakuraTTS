# 英文前端来源

`frontend/english.py` 的分词后发音规则来自 GPT-SoVITS `text/english.py`，数字、时间、单位规范化来自 Cosmo-klara 的 `text/en_normalization/expend.py`。原项目版权归 RVC-Boss 等贡献者所有，遵循 [MIT 许可证](GPT-SoVITS-LICENSE.txt)。保留原规则，资源加载改为显式读取离线包。

英文 OOV 预测的 `sigmoid`、`grucell`、`gru`、`encode`、`predict` 来自 Kyubyong Park 与 Jongseok Kim 的 [g2p-en 2.1.0](https://github.com/Kyubyong/g2p)，遵循 [Apache-2.0](Apache-2.0.txt)。计算顺序保留，模型参数改从准备包读取；不包含原模块的下载和全局初始化路径。

资源准备工具记录上游文件 SHA-256 和导出环境版本。词典、名字词典、热词、POS 模型及预测权重由用户指定的可信准备环境导出；推理端只读取 JSON 和不含对象数组的 NPZ。CMUdict 的版权与使用条件随资源包保留在 `CMUdict-README.txt`。NLTK、wordsegment、inflect 的代码和词频资源由各自的安装包提供，其许可证随依赖分发。
