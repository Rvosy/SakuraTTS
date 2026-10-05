---
tasks:
- speaker-verification
model_type:
- ERes2NetV2
domain:
- audio
frameworks:
- pytorch
backbone:
- ERes2NetV2
license: Apache License 2.0
language:
- cn
tags:
- ERes2NetV2
- 中文模型
widgets:
  - task: speaker-verification
    model_revision: v1.0.1
    inputs:
      - type: audio
        name: input
        title: 音频
    extendsParameters:
      thr: 0.400
    examples:
      - name: 1
        title: 示例1
        inputs:
          - name: enroll
            data: git://examples/speaker1_a_cn_16k.wav
          - name: input
            data: git://examples/speaker1_b_cn_16k.wav
      - name: 2
        title: 示例2
        inputs:
          - name: enroll
            data: git://examples/speaker1_a_cn_16k.wav
          - name: input
            data: git://examples/speaker2_a_cn_16k.wav
    inferencespec:
      cpu: 8 #CPU数量
      memory: 1024
---

## ERes2NetV2_w24s4ep4 说话人识别模型
ERes2NetV2模型是在ERes2Net的基础上，通过模型剪枝来减小冗余结构并增强短时说话人识别准确率，该模型在原版ERes2NetV2基础上通过调整模型参数进一步提高实验性能。
### 模型简述
ERes2NetV2局部融合如下图黄色部分所示，使用Attentianal feature fusion阶梯式融合各分组特征来增强局部信息连接，获取更细粒度特征；全局融合如下图绿色部分所示，通过自底向上的全局特征融合来增强说话人信息。在ERes2Net的基础上，增加channel维度进一步增强特征鲁棒性，同时去除stage1,2,3之间的特征融合模块降低结构冗余，最后通过expand-squeeze-expand结构来进一步提升实验性能。

<div align=center>
<img src="images/ERes2NetV2.png" width="700" />
</div>

更详细的信息见
- github项目地址：[3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-eres2netv2)

### 训练数据
本模型使用大型中文说话人数据集进行训练，包含约200k个说话人，可以对16k采样率的中文音频进行识别。

### 模型效果评估
在CN-Celeb中文测试集的EER评测结果对比：
| Model | #Spks trained | CN-Celeb Test |
|:-----:|:------:|:------:|
|ResNet34|~3k|6.97%|
|ECAPA-TDNN|~3k|7.45%|
|CAM++|~200k|4.32%|
|ERes2NetV2w24s4ep4|~200k|2.31%|

### 在线体验
在页面右侧，可以在“在线体验”栏内看到我们预先准备好的示例音频，点击播放按钮可以试听，点击“执行测试”按钮，会在下方“测试结果”栏中显示相似度得分(范围为[-1,1])和是否判断为同一个人。如果您想要测试自己的音频，可点“更换音频”按钮，选择上传或录制一段音频，完成后点击执行测试，识别内容将会在测试结果栏中显示。

### 在Notebook中体验
```python
from modelscope.pipelines import pipeline
sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common',
    model_revision='v1.0.1'
)
speaker1_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker1_a_cn_16k.wav'
speaker1_b_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker1_b_cn_16k.wav'
speaker2_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker2_a_cn_16k.wav'
# 相同说话人语音
result = sv_pipeline([speaker1_a_wav, speaker1_b_wav])
print(result)
# 不同说话人语音
result = sv_pipeline([speaker1_a_wav, speaker2_a_wav])
print(result)
# 可以自定义得分阈值来进行识别
result = sv_pipeline([speaker1_a_wav, speaker2_a_wav], thr=0.400)
print(result)
```

### 训练和测试自己的ERes2NetV2模型
本项目已在[3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker)开源了训练、测试和推理代码，使用者可按下面方式下载安装使用：
``` sh
git clone https://github.com/alibaba-damo-academy/3D-Speaker.git && cd 3D-Speaker
conda create -n 3D-Speaker python=3.8
conda activate 3D-Speaker
pip install -r requirements.txt
```

运行ERes2NetV2在VoxCeleb集上的训练脚本
``` sh
cd egs/voxceleb/sv-eres2netv2
# 需要在run.sh中提前配置训练使用的GPU信息，默认是8卡
bash run.sh
```

### 使用本预训练模型快速提取embedding
``` sh
pip install modelscope
cd 3D-Speaker
# 配置模型名称并指定wav路径，wav路径可以是单个wav，也可以包含多条wav路径的list文件
model_id=iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
# 提取embedding
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```


### 3D-Speaker 开发者社区钉钉群
<div align=left>
<img src="images/ding.png" width="300" />
</div>
