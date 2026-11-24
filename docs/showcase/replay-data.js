const recordedTake = {
  "video": null,
  "duration": 78.0,
  "records": [
    {
      "at": 0.0,
      "raw": "[22:40:13] [TTS] [info] SakuraTTS 已休眠 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 event=tts.engine.sleeping gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] requested_backend=cuda runtime_mode=managed",
      "cue": "sleeping"
    },
    {
      "at": 0.248,
      "raw": "2026-11-13 22:40:13,248 INFO [uvicorn.access] [请求 -] 127.0.0.1:50531 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 1.275,
      "raw": "2026-11-13 22:40:14,275 INFO [uvicorn.access] [请求 -] 127.0.0.1:50532 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 2.3,
      "raw": "2026-11-13 22:40:15,300 INFO [uvicorn.access] [请求 -] 127.0.0.1:50536 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 3.327,
      "raw": "2026-11-13 22:40:16,327 INFO [uvicorn.access] [请求 -] 127.0.0.1:50537 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 4.0,
      "raw": "[22:40:17] [TTS] [info] SakuraTTS 正在提前加载模型 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] requested_backend=cuda runtime_mode=managed",
      "cue": "wake"
    },
    {
      "at": 4.0,
      "raw": "[22:40:17] [TTS] [info] SakuraTTS 正在准备默认参考音频 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] reference_audio=VO08_0011.ogg requested_backend=cuda runtime_mode=managed"
    },
    {
      "at": 4.0,
      "raw": "[22:40:17] [PLUGIN] [info] 正在等待模型回复 │ op=chat-b478d720-dc23-40b7-9a98-99492089fb9f plugin=sakura.assistant.default channel=API event=api.request.started 模型=gpt-6.1-sol model_call=1 operation_id=chat-b478d720-dc23-40b7-9a98-99492089fb9f purpose=agent_step trace_id=3",
      "cue": "llm-start"
    },
    {
      "at": 4.353,
      "raw": "2026-11-13 22:40:17,353 INFO [uvicorn.access] [请求 -] 127.0.0.1:50541 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 4.719,
      "raw": "2026-11-13 22:40:17,719 INFO [uvicorn.access] [请求 -] 127.0.0.1:50544 - \"POST /runtime/wake HTTP/1.1\" 202"
    },
    {
      "at": 4.902,
      "raw": "2026-11-13 22:40:17,902 INFO [sakuratts.engine] [请求 -] 加载 GPT / SoVITS…"
    },
    {
      "at": 4.903,
      "raw": "2026-11-13 22:40:17,903 DEBUG [sakuratts.engine] [请求 -] Loading GPT and SoVITS weights: [本机数据目录]\\tts\\sakura\\runtime-data\\cache\\models\\080162a5a91d7a53887e082e5e293b69ffe6281e0e261cc58d110317dd0d19d0\\model.json"
    },
    {
      "at": 5.0,
      "raw": "[22:40:18] [TTS] [info] SakuraTTS 正在加载模型 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 event=tts.engine.waking gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] requested_backend=cuda runtime_mode=managed",
      "cue": "loading"
    },
    {
      "at": 5.381,
      "raw": "2026-11-13 22:40:18,381 INFO [uvicorn.access] [请求 -] 127.0.0.1:50550 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 6.409,
      "raw": "2026-11-13 22:40:19,409 INFO [uvicorn.access] [请求 -] 127.0.0.1:50556 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 7.426,
      "raw": "2026-11-13 22:40:20,426 INFO [uvicorn.access] [请求 -] 127.0.0.1:50557 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 8.449,
      "raw": "2026-11-13 22:40:21,449 INFO [uvicorn.access] [请求 -] 127.0.0.1:50558 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 9.467,
      "raw": "2026-11-13 22:40:22,467 INFO [uvicorn.access] [请求 -] 127.0.0.1:50559 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 10.0,
      "raw": "[22:40:23] [TTS] [info] SakuraTTS 模型与默认参考音频已就绪 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 elapsed_ms=5437.0 gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] requested_backend=cuda runtime_mode=managed"
    },
    {
      "at": 10.0,
      "raw": "[22:40:23] [TTS] [info] SakuraTTS 操作结束 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 elapsed_ms=5437 event=tts.operation.finished gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti outcome=success profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] requested_backend=cuda runtime_mode=managed stage=preload"
    },
    {
      "at": 10.0,
      "raw": "[22:40:23] [TTS] [info] SakuraTTS 模型已加载 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 event=tts.engine.awake gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] requested_backend=cuda runtime_mode=managed",
      "cue": "awake"
    },
    {
      "at": 10.121,
      "raw": "2026-11-13 22:40:23,121 DEBUG [sakuratts.engine] [请求 -] Model weights loaded | N.A.V.I._e10_s1310 | cuda | ja, en"
    },
    {
      "at": 10.128,
      "raw": "2026-11-13 22:40:23,128 DEBUG [sakuratts.TTS_infer_pack.reference] [请求 -] 参考音频条件 | 命中磁盘缓存 | [本机数据目录]\\tts\\sakura\\runtime-data\\cache\\references\\1c06fca6ba1690ed6883d3cc6afd3fba35129c124a88c7e053ccf50ce95311a6"
    },
    {
      "at": 10.13,
      "raw": "2026-11-13 22:40:23,130 DEBUG [sakuratts.TTS_infer_pack.reference] [请求 -] 参考音频条件 | 命中内存缓存 | SHA256=82bd7a2f2221c779ed35c093b348700458b6caecf0ad5f67385bfaec1d2eefa5"
    },
    {
      "at": 10.13,
      "raw": "2026-11-13 22:40:23,130 INFO [uvicorn.access] [请求 -] 127.0.0.1:50545 - \"GET /set_refer_audio?refer_audio_path=%5C%5C%3F%5CC%3A%5CUsers%5CRvosy%5CAppData%5CLocal%5CSakura+Development%5Ccharacters%5CN.A.V.I%5Cvoice%5Crefs%5Ctone_refs%5CVO08_0011.ogg HTTP/1.1\" 200"
    },
    {
      "at": 10.484,
      "raw": "2026-11-13 22:40:23,484 INFO [uvicorn.access] [请求 -] 127.0.0.1:50563 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 11.502,
      "raw": "2026-11-13 22:40:24,502 INFO [uvicorn.access] [请求 -] 127.0.0.1:50564 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 12.0,
      "raw": "[22:40:25] [PLUGIN] [info] 回复已生成 │ op=chat-b478d720-dc23-40b7-9a98-99492089fb9f plugin=sakura.assistant.default channel=AgentRuntime event=reply.processing.finished model_call=1 operation_id=chat-b478d720-dc23-40b7-9a98-99492089fb9f parse_status=ready segment_count=1 trace_id=3 turn_elapsed_ms=8115",
      "cue": "llm-end"
    },
    {
      "at": 12.0,
      "raw": "[22:40:25] [TTS] [info] SakuraTTS 开始合成 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti language=ja profile=fp16 provider_id=sakura.tts.sakuratts queue_ms=0.0 redacted=[REDACTED] reference_audio=VO08_0011.ogg requested_backend=cuda runtime_mode=managed text_chars=58"
    },
    {
      "at": 12.0,
      "raw": "[22:40:25] [TTS] [info] 开始合成语音 │ op=chat-b478d720-dc23-40b7-9a98-99492089fb9f segment_index=0 segments=1",
      "cue": "job-start"
    },
    {
      "at": 12.528,
      "raw": "2026-11-13 22:40:25,528 INFO [uvicorn.access] [请求 -] 127.0.0.1:50565 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 12.936,
      "raw": "2026-11-13 22:40:25,936 INFO [sakuratts.engine] [请求 0003] 请求 #0003  ja · 58 字 · seed=3496123598",
      "cue": "synth-start",
      "request": "0003"
    },
    {
      "at": 12.936,
      "raw": "2026-11-13 22:40:25,936 DEBUG [sakuratts.engine] [请求 0003] 请求参数: {'text': 'シム。当たっていてよかったです。お褒めの言葉も、ありがたく受け取ります。真紅は、お気に入りのキャラクターなんですか？', 'text_lang': 'ja', 'ref_audio_path': '[本机数据目录]\\\\characters\\\\N.A.V.I\\\\voice\\\\refs\\\\tone_refs\\\\VO08_0011.ogg', 'aux_ref_audio_paths': None, 'prompt_lang': 'ja', 'prompt_text': '私は“学園”の開発したナビゲーションシステム。通称Ｎ．ａ．ｖ．ｉ．です', 'top_k': 15, 'top_p': 1, 'temperature': 1, 'text_split_method': 'cut5', 'batch_size': 1, 'batch_threshold': 0.75, 'split_bucket': True, 'speed_factor': 1, 'fragment_interval': 0.3, 'seed': 3496123598, 'media_type': 'wav', 'streaming_mode': False, 'parallel_infer': False, 'repetition_penalty': 1.35, 'sample_steps': 32, 'super_sampling': False, 'overlap_length': 2, 'min_chunk_length': 16}"
    },
    {
      "at": 12.936,
      "raw": "2026-11-13 22:40:25,936 DEBUG [sakuratts.engine] [请求 0003] 当前 GPT 权重 | [本机数据目录]\\characters\\N.A.V.I\\voice\\models\\N.A.V.I.-e15.ckpt | SHA256=250400bfebe8547ef9167dc0225239eb2cac57e8648afbaa0834ac3ff95d4fcf"
    },
    {
      "at": 12.937,
      "raw": "2026-11-13 22:40:25,937 DEBUG [sakuratts.engine] [请求 0003] 当前 SOVITS 权重 | [本机数据目录]\\characters\\N.A.V.I\\voice\\models\\N.A.V.I._e10_s1310.pth | SHA256=d12ec3ec4d6c859f4a24dac3b2ade01d722e1df9a366c291b99c26b888e22b81"
    },
    {
      "at": 12.937,
      "raw": "2026-11-13 22:40:25,937 DEBUG [sakuratts.engine] [请求 0003] 参考音频 | [本机数据目录]\\characters\\N.A.V.I\\voice\\refs\\tone_refs\\VO08_0011.ogg | 语言=ja | 转写='私は“学園”の開発したナビゲーションシステム。通称Ｎ．ａ．ｖ．ｉ．です'"
    },
    {
      "at": 12.937,
      "raw": "2026-11-13 22:40:25,937 DEBUG [sakuratts.TTS_infer_pack.reference] [请求 0003] 参考音频条件 | 命中内存缓存 | SHA256=82bd7a2f2221c779ed35c093b348700458b6caecf0ad5f67385bfaec1d2eefa5"
    },
    {
      "at": 13.149,
      "raw": "2026-11-13 22:40:26,149 INFO [sakuratts.engine] [请求 0003] 参考    VO08_0011.ogg · 已准备"
    },
    {
      "at": 13.149,
      "raw": "2026-11-13 22:40:26,149 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,149  INFO [sakuratts.engine]: 参考    VO08_0011.ogg · 已准备"
    },
    {
      "at": 13.149,
      "raw": "2026-11-13 22:40:26,149 DEBUG [sakuratts.engine] [请求 0003] 参考条件就绪 | 0.212 s | 84 个音素 | 157 个参考语义 Token"
    },
    {
      "at": 13.15,
      "raw": "2026-11-13 22:40:26,150 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,149  DEBUG [sakuratts.engine]: 参考条件就绪 | 0.212 s | 84 个音素 | 157 个参考语义 Token"
    },
    {
      "at": 13.15,
      "raw": "2026-11-13 22:40:26,150 DEBUG [sakuratts.inference] [请求 0003] 推理参数 | seed=3496123598 | ja | cut5 | top_k=15 | temperature=1 | repetition_penalty=1.35"
    },
    {
      "at": 13.15,
      "raw": "2026-11-13 22:40:26,150 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,149  DEBUG [sakuratts.inference]: 推理参数 | seed=3496123598 | ja | cut5 | top_k=15 | temperature=1 | repetition_penalty=1.35"
    },
    {
      "at": 13.15,
      "raw": "2026-11-13 22:40:26,150 DEBUG [sakuratts.inference] [请求 0003] 切分文本并提取文本特征 | 输入: 'シム。当たっていてよかったです。お褒めの言葉も、ありがたく受け取ります。真紅は、お気に入りのキャラクターなんですか？'"
    },
    {
      "at": 13.151,
      "raw": "2026-11-13 22:40:26,151 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,149  DEBUG [sakuratts.inference]: 切分文本并提取文本特征 | 输入: 'シム。当たっていてよかったです。お褒めの言葉も、ありがたく受け取ります。真紅は、お気に入りのキャラクターなんですか？'"
    },
    {
      "at": 13.151,
      "raw": "2026-11-13 22:40:26,151 INFO [sakuratts.inference] [请求 0003] 文本: シム。当たっていてよかったです。お褒めの言葉も、ありがたく受け取ります。真紅は、お気に入りのキャラクターなんですか？"
    },
    {
      "at": 13.151,
      "raw": "2026-11-13 22:40:26,151 INFO [sakuratts.inference] [请求 0003] 提取文本Bert特征"
    },
    {
      "at": 13.151,
      "raw": "2026-11-13 22:40:26,151 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,149  INFO [sakuratts.inference]: 文本: シム。当たっていてよかったです。お褒めの言葉も、ありがたく受け取ります。真紅は、お気に入りのキャラクターなんですか？"
    },
    {
      "at": 13.152,
      "raw": "2026-11-13 22:40:26,152 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,149  INFO [sakuratts.inference]: 提取文本Bert特征"
    },
    {
      "at": 13.161,
      "raw": "2026-11-13 22:40:26,161 INFO [sakuratts.inference] [请求 0003] 4 段 · 119 phones · 0.013 s"
    },
    {
      "at": 13.163,
      "raw": "2026-11-13 22:40:26,163 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,161  INFO [sakuratts.inference]: 4 段 · 119 phones · 0.013 s"
    },
    {
      "at": 13.163,
      "raw": "2026-11-13 22:40:26,163 DEBUG [sakuratts.inference] [请求 0003] 分句 2/4 | 22 个音素 | 前端处理后的文本: 'お褒めの言葉も、。'"
    },
    {
      "at": 13.163,
      "raw": "2026-11-13 22:40:26,163 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,161  DEBUG [sakuratts.inference]: 分句 2/4 | 22 个音素 | 前端处理后的文本: 'お褒めの言葉も、。'"
    },
    {
      "at": 13.163,
      "raw": "2026-11-13 22:40:26,163 INFO [sakuratts.inference] [请求 0003] 片段 [2/4]"
    },
    {
      "at": 13.163,
      "raw": "2026-11-13 22:40:26,163 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,163  INFO [sakuratts.inference]: 片段 [2/4]"
    },
    {
      "at": 13.164,
      "raw": "2026-11-13 22:40:26,164 INFO [sakuratts.inference] [请求 0003] 预测语义Token"
    },
    {
      "at": 13.164,
      "raw": "2026-11-13 22:40:26,164 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,163  INFO [sakuratts.inference]: 预测语义Token"
    },
    {
      "at": 13.164,
      "raw": "2026-11-13 22:40:26,164 DEBUG [sakuratts.inference] [请求 0003] 开始语义预测，最多 1500 步"
    },
    {
      "at": 13.164,
      "raw": "2026-11-13 22:40:26,164 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,163  DEBUG [sakuratts.inference]: 开始语义预测，最多 1500 步"
    },
    {
      "at": 13.371,
      "raw": "2026-11-13 22:40:26,371 DEBUG [sakuratts.inference] [请求 0003] 语义预测结束 | 采样 37 次 | 0.209 s | 177.0 it/s | 停止原因: argmax EOS, 采样 EOS"
    },
    {
      "at": 13.371,
      "raw": "2026-11-13 22:40:26,371 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,371  DEBUG [sakuratts.inference]: 语义预测结束 | 采样 37 次 | 0.209 s | 177.0 it/s | 停止原因: argmax EOS, 采样 EOS"
    },
    {
      "at": 13.372,
      "raw": "2026-11-13 22:40:26,372 INFO [sakuratts.inference] [请求 0003] GPT     37 it · 177.0 it/s · 0.209 s"
    },
    {
      "at": 13.372,
      "raw": "2026-11-13 22:40:26,372 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,371  INFO [sakuratts.inference]: GPT     37 it · 177.0 it/s · 0.209 s"
    },
    {
      "at": 13.372,
      "raw": "2026-11-13 22:40:26,372 INFO [sakuratts.inference] [请求 0003] T2S Decoding EOS [157 -> 194]"
    },
    {
      "at": 13.373,
      "raw": "2026-11-13 22:40:26,373 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,371  INFO [sakuratts.inference]: T2S Decoding EOS [157 -> 194]"
    },
    {
      "at": 13.373,
      "raw": "2026-11-13 22:40:26,373 INFO [sakuratts.inference] [请求 0003] 合成音频"
    },
    {
      "at": 13.373,
      "raw": "2026-11-13 22:40:26,373 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,372  INFO [sakuratts.inference]: 合成音频"
    },
    {
      "at": 13.373,
      "raw": "2026-11-13 22:40:26,373 DEBUG [sakuratts.inference] [请求 0003] 合成音频 | 分句 2/4 | 36 个有效语义 Token"
    },
    {
      "at": 13.373,
      "raw": "2026-11-13 22:40:26,373 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,373  DEBUG [sakuratts.inference]: 合成音频 | 分句 2/4 | 36 个有效语义 Token"
    },
    {
      "at": 13.548,
      "raw": "2026-11-13 22:40:26,548 INFO [uvicorn.access] [请求 -] 127.0.0.1:50569 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 13.783,
      "raw": "2026-11-13 22:40:26,783 DEBUG [sakuratts.inference] [请求 0003] 分句音频完成 | 0.409 s | 音频 1.74 s | 采样率 32000 Hz"
    },
    {
      "at": 13.783,
      "raw": "2026-11-13 22:40:26,783 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,782  DEBUG [sakuratts.inference]: 分句音频完成 | 0.409 s | 音频 1.74 s | 采样率 32000 Hz"
    },
    {
      "at": 13.783,
      "raw": "2026-11-13 22:40:26,783 INFO [sakuratts.inference] [请求 0003] SoVITS  0.409 s"
    },
    {
      "at": 13.784,
      "raw": "2026-11-13 22:40:26,784 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,782  INFO [sakuratts.inference]: SoVITS  0.409 s"
    },
    {
      "at": 13.784,
      "raw": "2026-11-13 22:40:26,784 DEBUG [sakuratts.inference] [请求 0003] 分句 3/4 | 23 个音素 | 前端处理后的文本: 'ありがたく受け取ります。'"
    },
    {
      "at": 13.785,
      "raw": "2026-11-13 22:40:26,785 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,783  DEBUG [sakuratts.inference]: 分句 3/4 | 23 个音素 | 前端处理后的文本: 'ありがたく受け取ります。'"
    },
    {
      "at": 13.785,
      "raw": "2026-11-13 22:40:26,785 INFO [sakuratts.inference] [请求 0003] 片段 [3/4]"
    },
    {
      "at": 13.785,
      "raw": "2026-11-13 22:40:26,785 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,783  INFO [sakuratts.inference]: 片段 [3/4]"
    },
    {
      "at": 13.786,
      "raw": "2026-11-13 22:40:26,786 INFO [sakuratts.inference] [请求 0003] 预测语义Token"
    },
    {
      "at": 13.786,
      "raw": "2026-11-13 22:40:26,786 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,783  INFO [sakuratts.inference]: 预测语义Token"
    },
    {
      "at": 13.786,
      "raw": "2026-11-13 22:40:26,786 DEBUG [sakuratts.inference] [请求 0003] 开始语义预测，最多 1500 步"
    },
    {
      "at": 13.786,
      "raw": "2026-11-13 22:40:26,786 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,783  DEBUG [sakuratts.inference]: 开始语义预测，最多 1500 步"
    },
    {
      "at": 13.876,
      "raw": "2026-11-13 22:40:26,876 DEBUG [sakuratts.inference] [请求 0003] 语义预测结束 | 采样 42 次 | 0.092 s | 455.4 it/s | 停止原因: argmax EOS, 采样 EOS"
    },
    {
      "at": 13.876,
      "raw": "2026-11-13 22:40:26,876 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,875  DEBUG [sakuratts.inference]: 语义预测结束 | 采样 42 次 | 0.092 s | 455.4 it/s | 停止原因: argmax EOS, 采样 EOS"
    },
    {
      "at": 13.876,
      "raw": "2026-11-13 22:40:26,876 INFO [sakuratts.inference] [请求 0003] GPT     42 it · 455.4 it/s · 0.092 s"
    },
    {
      "at": 13.876,
      "raw": "2026-11-13 22:40:26,876 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,875  INFO [sakuratts.inference]: GPT     42 it · 455.4 it/s · 0.092 s"
    },
    {
      "at": 13.876,
      "raw": "2026-11-13 22:40:26,876 INFO [sakuratts.inference] [请求 0003] T2S Decoding EOS [157 -> 199]"
    },
    {
      "at": 13.876,
      "raw": "2026-11-13 22:40:26,876 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,875  INFO [sakuratts.inference]: T2S Decoding EOS [157 -> 199]"
    },
    {
      "at": 13.878,
      "raw": "2026-11-13 22:40:26,878 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,878  INFO [sakuratts.inference]: 合成音频"
    },
    {
      "at": 13.878,
      "raw": "2026-11-13 22:40:26,878 INFO [sakuratts.inference] [请求 0003] 合成音频"
    },
    {
      "at": 13.878,
      "raw": "2026-11-13 22:40:26,878 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,878  DEBUG [sakuratts.inference]: 合成音频 | 分句 3/4 | 41 个有效语义 Token"
    },
    {
      "at": 13.879,
      "raw": "2026-11-13 22:40:26,879 DEBUG [sakuratts.inference] [请求 0003] 合成音频 | 分句 3/4 | 41 个有效语义 Token"
    },
    {
      "at": 13.917,
      "raw": "2026-11-13 22:40:26,917 DEBUG [sakuratts.inference] [请求 0003] 分句音频完成 | 0.039 s | 音频 1.94 s | 采样率 32000 Hz"
    },
    {
      "at": 13.918,
      "raw": "2026-11-13 22:40:26,918 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,917  DEBUG [sakuratts.inference]: 分句音频完成 | 0.039 s | 音频 1.94 s | 采样率 32000 Hz"
    },
    {
      "at": 13.918,
      "raw": "2026-11-13 22:40:26,918 INFO [sakuratts.inference] [请求 0003] SoVITS  0.039 s"
    },
    {
      "at": 13.918,
      "raw": "2026-11-13 22:40:26,918 DEBUG [sakuratts.inference] [请求 0003] 分句 1/4 | 31 个音素 | 前端处理后的文本: 'シム。当たっていてよかったです。'"
    },
    {
      "at": 13.918,
      "raw": "2026-11-13 22:40:26,918 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,917  INFO [sakuratts.inference]: SoVITS  0.039 s"
    },
    {
      "at": 13.918,
      "raw": "2026-11-13 22:40:26,918 INFO [sakuratts.inference] [请求 0003] 片段 [1/4]"
    },
    {
      "at": 13.919,
      "raw": "2026-11-13 22:40:26,919 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,917  DEBUG [sakuratts.inference]: 分句 1/4 | 31 个音素 | 前端处理后的文本: 'シム。当たっていてよかったです。'"
    },
    {
      "at": 13.919,
      "raw": "2026-11-13 22:40:26,919 INFO [sakuratts.inference] [请求 0003] 预测语义Token"
    },
    {
      "at": 13.92,
      "raw": "2026-11-13 22:40:26,920 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,917  INFO [sakuratts.inference]: 片段 [1/4]"
    },
    {
      "at": 13.92,
      "raw": "2026-11-13 22:40:26,920 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,917  INFO [sakuratts.inference]: 预测语义Token"
    },
    {
      "at": 13.92,
      "raw": "2026-11-13 22:40:26,920 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:26,917  DEBUG [sakuratts.inference]: 开始语义预测，最多 1500 步"
    },
    {
      "at": 13.92,
      "raw": "2026-11-13 22:40:26,920 DEBUG [sakuratts.inference] [请求 0003] 开始语义预测，最多 1500 步"
    },
    {
      "at": 14.0,
      "raw": "[22:40:27] [TTS] [info] SakuraTTS 合成耗时 │ plugin=sakura.tts.sakuratts audio_seconds=10.96 backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 bytes=701484 engine_total_ms=1405.73 frontend_ms=12.83 gpt_ms=635.83 gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti inference_ms=1400.45 profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] reference_cache=memory reference_ms=212.47 request_ms=1438.0 requested_backend=cuda rtf=0.131 runtime_mode=managed sovits_ms=534.26 synthesis_status=completed"
    },
    {
      "at": 14.0,
      "raw": "[22:40:27] [TTS] [info] SakuraTTS 操作结束 │ plugin=sakura.tts.sakuratts audio_seconds=10.96 backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 bytes=701484 elapsed_ms=1438 engine_total_ms=1405.73 event=tts.operation.finished frontend_ms=12.83 gpt_ms=635.83 gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti inference_ms=1400.45 language=ja outcome=success profile=fp16 provider_id=sakura.tts.sakuratts queue_ms=0.0 redacted=[REDACTED] reference_cache=memory reference_ms=212.47 request_ms=1438.0 requested_backend=cuda rtf=0.131 runtime_mode=managed sovits_ms=534.26 stage=synthesis synthesis_status=completed text_chars=58"
    },
    {
      "at": 14.0,
      "raw": "[22:40:27] [TTS] [info] 语音记录已保存 │ op=chat-b478d720-dc23-40b7-9a98-99492089fb9f provider=sakura.tts.sakuratts segment_index=0 segments=1 recording_id=6a3fd2aaf1cf4926b7f0579f3640eea7 bytes=701484"
    },
    {
      "at": 14.0,
      "raw": "[22:40:27] [TTS] [info] 语音合成完成 │ op=chat-b478d720-dc23-40b7-9a98-99492089fb9f provider=sakura.tts.sakuratts segment_index=0 segments=1 recording_id=6a3fd2aaf1cf4926b7f0579f3640eea7 bytes=701484 elapsed_ms=1562ms",
      "cue": "job-end"
    },
    {
      "at": 14.0,
      "raw": "[22:40:27] [TTS] [info] 开始播放语音 │ segment_index=0 segments=1 recording_id=6a3fd2aaf1cf4926b7f0579f3640eea7 status=started code=null",
      "cue": "play-start"
    },
    {
      "at": 14.08,
      "raw": "2026-11-13 22:40:27,080 DEBUG [sakuratts.inference] [请求 0003] 语义预测结束 | 采样 89 次 | 0.162 s | 548.0 it/s | 停止原因: 采样 EOS"
    },
    {
      "at": 14.08,
      "raw": "2026-11-13 22:40:27,080 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,080  DEBUG [sakuratts.inference]: 语义预测结束 | 采样 89 次 | 0.162 s | 548.0 it/s | 停止原因: 采样 EOS"
    },
    {
      "at": 14.081,
      "raw": "2026-11-13 22:40:27,081 INFO [sakuratts.inference] [请求 0003] GPT     89 it · 548.0 it/s · 0.162 s"
    },
    {
      "at": 14.081,
      "raw": "2026-11-13 22:40:27,081 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,080  INFO [sakuratts.inference]: GPT     89 it · 548.0 it/s · 0.162 s"
    },
    {
      "at": 14.081,
      "raw": "2026-11-13 22:40:27,081 INFO [sakuratts.inference] [请求 0003] T2S Decoding EOS [157 -> 246]"
    },
    {
      "at": 14.081,
      "raw": "2026-11-13 22:40:27,081 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,080  INFO [sakuratts.inference]: T2S Decoding EOS [157 -> 246]"
    },
    {
      "at": 14.082,
      "raw": "2026-11-13 22:40:27,082 INFO [sakuratts.inference] [请求 0003] 合成音频"
    },
    {
      "at": 14.082,
      "raw": "2026-11-13 22:40:27,082 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,081  INFO [sakuratts.inference]: 合成音频"
    },
    {
      "at": 14.082,
      "raw": "2026-11-13 22:40:27,082 DEBUG [sakuratts.inference] [请求 0003] 合成音频 | 分句 1/4 | 88 个有效语义 Token"
    },
    {
      "at": 14.082,
      "raw": "2026-11-13 22:40:27,082 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,082  DEBUG [sakuratts.inference]: 合成音频 | 分句 1/4 | 88 个有效语义 Token"
    },
    {
      "at": 14.119,
      "raw": "2026-11-13 22:40:27,119 DEBUG [sakuratts.inference] [请求 0003] 分句音频完成 | 0.036 s | 音频 3.82 s | 采样率 32000 Hz"
    },
    {
      "at": 14.119,
      "raw": "2026-11-13 22:40:27,119 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,119  DEBUG [sakuratts.inference]: 分句音频完成 | 0.036 s | 音频 3.82 s | 采样率 32000 Hz"
    },
    {
      "at": 14.119,
      "raw": "2026-11-13 22:40:27,119 INFO [sakuratts.inference] [请求 0003] SoVITS  0.036 s"
    },
    {
      "at": 14.119,
      "raw": "2026-11-13 22:40:27,119 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,119  INFO [sakuratts.inference]: SoVITS  0.036 s"
    },
    {
      "at": 14.119,
      "raw": "2026-11-13 22:40:27,119 DEBUG [sakuratts.inference] [请求 0003] 分句 4/4 | 43 个音素 | 前端处理后的文本: '真紅は、お気に入りのキャラクターなんですか？'"
    },
    {
      "at": 14.12,
      "raw": "2026-11-13 22:40:27,120 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,119  DEBUG [sakuratts.inference]: 分句 4/4 | 43 个音素 | 前端处理后的文本: '真紅は、お気に入りのキャラクターなんですか？'"
    },
    {
      "at": 14.12,
      "raw": "2026-11-13 22:40:27,120 INFO [sakuratts.inference] [请求 0003] 片段 [4/4]"
    },
    {
      "at": 14.121,
      "raw": "2026-11-13 22:40:27,121 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,119  INFO [sakuratts.inference]: 片段 [4/4]"
    },
    {
      "at": 14.121,
      "raw": "2026-11-13 22:40:27,121 INFO [sakuratts.inference] [请求 0003] 预测语义Token"
    },
    {
      "at": 14.121,
      "raw": "2026-11-13 22:40:27,121 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,119  INFO [sakuratts.inference]: 预测语义Token"
    },
    {
      "at": 14.122,
      "raw": "2026-11-13 22:40:27,122 DEBUG [sakuratts.inference] [请求 0003] 开始语义预测，最多 1500 步"
    },
    {
      "at": 14.122,
      "raw": "2026-11-13 22:40:27,122 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,119  DEBUG [sakuratts.inference]: 开始语义预测，最多 1500 步"
    },
    {
      "at": 14.284,
      "raw": "2026-11-13 22:40:27,284 DEBUG [sakuratts.inference] [请求 0003] 语义预测结束 | 采样 80 次 | 0.165 s | 485.7 it/s | 停止原因: argmax EOS"
    },
    {
      "at": 14.284,
      "raw": "2026-11-13 22:40:27,284 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,284  DEBUG [sakuratts.inference]: 语义预测结束 | 采样 80 次 | 0.165 s | 485.7 it/s | 停止原因: argmax EOS"
    },
    {
      "at": 14.285,
      "raw": "2026-11-13 22:40:27,285 INFO [sakuratts.inference] [请求 0003] GPT     80 it · 485.7 it/s · 0.165 s"
    },
    {
      "at": 14.285,
      "raw": "2026-11-13 22:40:27,285 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,284  INFO [sakuratts.inference]: GPT     80 it · 485.7 it/s · 0.165 s"
    },
    {
      "at": 14.285,
      "raw": "2026-11-13 22:40:27,285 INFO [sakuratts.inference] [请求 0003] T2S Decoding EOS [157 -> 237]"
    },
    {
      "at": 14.285,
      "raw": "2026-11-13 22:40:27,285 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,284  INFO [sakuratts.inference]: T2S Decoding EOS [157 -> 237]"
    },
    {
      "at": 14.286,
      "raw": "2026-11-13 22:40:27,286 INFO [sakuratts.inference] [请求 0003] 合成音频"
    },
    {
      "at": 14.286,
      "raw": "2026-11-13 22:40:27,286 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,285  INFO [sakuratts.inference]: 合成音频"
    },
    {
      "at": 14.286,
      "raw": "2026-11-13 22:40:27,286 DEBUG [sakuratts.inference] [请求 0003] 合成音频 | 分句 4/4 | 79 个有效语义 Token"
    },
    {
      "at": 14.286,
      "raw": "2026-11-13 22:40:27,286 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,286  DEBUG [sakuratts.inference]: 合成音频 | 分句 4/4 | 79 个有效语义 Token"
    },
    {
      "at": 14.336,
      "raw": "2026-11-13 22:40:27,336 DEBUG [sakuratts.inference] [请求 0003] 分句音频完成 | 0.050 s | 音频 3.46 s | 采样率 32000 Hz"
    },
    {
      "at": 14.337,
      "raw": "2026-11-13 22:40:27,337 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,336  DEBUG [sakuratts.inference]: 分句音频完成 | 0.050 s | 音频 3.46 s | 采样率 32000 Hz"
    },
    {
      "at": 14.337,
      "raw": "2026-11-13 22:40:27,337 INFO [sakuratts.inference] [请求 0003] SoVITS  0.050 s"
    },
    {
      "at": 14.337,
      "raw": "2026-11-13 22:40:27,337 DEBUG [sakuratts.engine] [请求 -] Inference worker: 2026-11-13 22:40:27,336  INFO [sakuratts.inference]: SoVITS  0.050 s"
    },
    {
      "at": 14.341,
      "raw": "2026-11-13 22:40:27,341 INFO [sakuratts.server] [请求 0003] 完成 #0003  音频 10.96 s · 总耗时 1.400 s · RTF 0.128",
      "cue": "synth-end",
      "request": "0003"
    },
    {
      "at": 14.341,
      "raw": "2026-11-13 22:40:27,341 DEBUG [sakuratts.server] [请求 0003] 参考 0.212 s · 文本 0.013 s · GPT 0.636 s · SoVITS 0.534 s"
    },
    {
      "at": 14.341,
      "raw": "2026-11-13 22:40:27,341 DEBUG [sakuratts.server] [请求 0003] 合成报告: {'status': 'completed', 'text': 'シム。当たっていてよかったです。お褒めの言葉も、ありがたく受け取ります。真紅は、お気に入りのキャラクターなんですか？', 'reference': '82bd7a2f2221c779ed35c093b348700458b6caecf0ad5f67385bfaec1d2eefa5', 'reference_identity': {'gpt_checkpoint_sha256': '250400bfebe8547ef9167dc0225239eb2cac57e8648afbaa0834ac3ff95d4fcf', 'sovits_checkpoint_sha256': 'd12ec3ec4d6c859f4a24dac3b2ade01d722e1df9a366c291b99c26b888e22b81', 'audio_sha256': '82bd7a2f2221c779ed35c093b348700458b6caecf0ad5f67385bfaec1d2eefa5', 'official_commit': 'source-sha256:83a849f2d0accc8a9d51e7ad1cb7475fa135eb7952de8dfae377ea95e5d85c09', 'reference_text': '私は“学園”の開発したナビゲーションシステム。通称Ｎ．ａ．ｖ．ｉ．です', 'reference_language': 'ja'}, 'parameters': {'seed': 3496123598, 'rng': 'numpy.default_rng (not Torch seed-equivalent)', 'language': 'ja', 'split_method': 'cut5', 'split_bucket': True, 'top_k': 15, 'top_p': 1, 'temperature': 1, 'repetition_penalty': 1.35, 'early_stop_num': 2700, 'noise_scale': 0.5, 'speed': 1.0, 'fragment_interval': 0.3}, 'policy': 'release-state', 'capacity': 2048, 'backend': 'cuda', 'gpt_device': 'cuda', 'acoustic_device': 'cuda', 'cuda_graph': True, 'gpt_attention': 'baseline', 'gpt_attention_chunk_size': 256, 'request_ms': 1400.4493999818806, 'frontend_ms': 12.829999992391095, 'sample_rate': 32000, 'audio_seconds': 9.76, 'pcm_seconds': 10.96, 'rtf': 0.12168763319617729, 'fragments': [{'index': 0, 'normalized_text': 'シム。当たっていてよかったです。', 'phones': [251, 160, 323, 225, 254, 3, 96, 322, 252, 96, 126, 252, 129, 86, 160, 322, 252, 129, 318, 229, 323, 222, 96, 126, 252, 96, 127, 129, 250, 254, 3], 'sampled_tokens': [290, 885, 566, 235, 600, 603, 397, 577, 886, 638, 27, 844, 593, 185, 837, 318, 14, 570, 505, 234, 280, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 280, 280, 271, 530, 958, 346, 604, 562, 267, 627, 99, 16, 8, 281, 201, 583, 485, 103, 528, 978, 423, 198, 620, 334, 259, 202, 982, 581, 899, 845, 638, 145, 839, 850, 598, 320, 132, 545, 910, 142, 17, 98, 29, 170, 53, 1024], 'semantic_tokens': [290, 885, 566, 235, 600, 603, 397, 577, 886, 638, 27, 844, 593, 185, 837, 318, 14, 570, 505, 234, 280, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 486, 280, 280, 271, 530, 958, 346, 604, 562, 267, 627, 99, 16, 8, 281, 201, 583, 485, 103, 528, 978, 423, 198, 620, 334, 259, 202, 982, 581, 899, 845, 638, 145, 839, 850, 598, 320, 132, 545, 910, 142, 17, 98, 29, 170, 53], 'stop_reasons': ['sample_eos'], 'returned_index': 88, 'waveform_samples': 112640, 'pcm_samples': 122240, 'timings': {'frontend_seconds': 0.0, 'condition_seconds': 5.729999975301325e-05, 'semantic_seconds': 0.16427219999604858, 'acoustic_seconds': 0.035892599989892915, 'output_copy_pcm_seconds': 0.0010066000104416162, 'prepared_inference_seconds': 0.20122869999613613, 'compute_seconds': 0.20122869999613613}, 'acoustic_transport': {'roundtrip_ms': 35.51600000355393, 'worker_compute_ms': 34.82379999999985, 'transport_and_scheduling_ms': 0.6922000035540776, 'upload_bytes': 142264, 'download_bytes': 450560, 'worker_acoustic': {'experiment': 'chunked-vocoder', 'chunk_frames': 256, 'chunks': 1, 'latent_dtype': 'float16', 'latent_frames': 176, 'latent_d2h_bytes': 67584, 'latent_inputs_h2d_bytes': 142268, 'vocoder_inputs_h2d_bytes': 71680, 'vocoder_outputs_d2h_bytes': 450560, 'byte_scope': 'Logical host tensor sizes; excludes driver copies, workspaces and transfer profiling', 'latent_ms': 19.569300000000567, 'decode_ms': 34.74969999999988, 'acoustic_session_policy': 'resident', 'stage_intervals': [{'stage': 'acoustic.latent.run', 'start_unix_s': 1794580827.0826163, 'end_unix_s': 1794580827.1026275, 'duration_ms': 19.561000000000384, 'pid': 90744}, {'stage': 'acoustic.vocoder.run', 'start_unix_s': 1794580827.1031313, 'end_unix_s': 1794580827.1170976, 'duration_ms': 14.765900000000443, 'pid': 90744, 'chunk_index': 0}], 'plans': [{'input_start': 0, 'input_end': 176, 'core_start_frame': 0, 'core_end_frame': 176, 'core_sample_start': 0, 'core_sample_end': 112640, 'crop_start': 0, 'crop_end': 112640, 'samples_per_frame': 640, 'left_halo_frames': 0, 'right_halo_frames': 0}], 'pcm': 'Normalized once by the unchanged engine after full waveform reconstruction'}, 'ipc_scope': 'FP32/int64 inputs and complete FP32 waveform; split latent stays in worker'}}, {'index': 1, 'normalized_text': 'お褒めの言葉も、。', 'phones': [229, 322, 158, 229, 225, 129, 323, 227, 229, 86, 222, 229, 322, 252, 229, 122, 96, 323, 225, 229, 1, 3], 'sampled_tokens': [105, 271, 505, 221, 551, 101, 117, 469, 701, 730, 354, 362, 852, 661, 334, 156, 11, 185, 460, 321, 421, 202, 519, 26, 92, 812, 569, 746, 334, 656, 451, 451, 14, 609, 140, 540, 1024], 'semantic_tokens': [105, 271, 505, 221, 551, 101, 117, 469, 701, 730, 354, 362, 852, 661, 334, 156, 11, 185, 460, 321, 421, 202, 519, 26, 92, 812, 569, 746, 334, 656, 451, 451, 14, 609, 140, 540], 'stop_reasons': ['argmax_eos', 'sample_eos'], 'returned_index': 36, 'waveform_samples': 46080, 'pcm_samples': 55680, 'timings': {'frontend_seconds': 0.0, 'condition_seconds': 0.0001126999850384891, 'semantic_seconds': 0.21043909998843446, 'acoustic_seconds': 0.40890480001689866, 'output_copy_pcm_seconds': 0.0006549999816343188, 'prepared_inference_seconds': 0.6201115999720059, 'compute_seconds': 0.6201115999720059}, 'acoustic_transport': {'roundtrip_ms': 408.7168999831192, 'worker_compute_ms': 408.19160000000034, 'transport_and_scheduling_ms': 0.5252999831188845, 'upload_bytes': 61904, 'download_bytes': 184320, 'worker_acoustic': {'experiment': 'chunked-vocoder', 'chunk_frames': 256, 'chunks': 1, 'latent_dtype': 'float16', 'latent_frames': 72, 'latent_d2h_bytes': 27648, 'latent_inputs_h2d_bytes': 61908, 'vocoder_inputs_h2d_bytes': 31744, 'vocoder_outputs_d2h_bytes': 184320, 'byte_scope': 'Logical host tensor sizes; excludes driver copies, workspaces and transfer profiling', 'latent_ms': 375.2621999999999, 'decode_ms': 408.1070000000002, 'acoustic_session_policy': 'resident', 'stage_intervals': [{'stage': 'acoustic.latent.run', 'start_unix_s': 1794580826.3736792, 'end_unix_s': 1794580826.7480512, 'duration_ms': 375.25669999999957, 'pid': 90744}, {'stage': 'acoustic.vocoder.run', 'start_unix_s': 1794580826.7495599, 'end_unix_s': 1794580826.7820292, 'duration_ms': 32.00320000000012, 'pid': 90744, 'chunk_index': 0}], 'plans': [{'input_start': 0, 'input_end': 72, 'core_start_frame': 0, 'core_end_frame': 72, 'core_sample_start': 0, 'core_sample_end': 46080, 'crop_start': 0, 'crop_end': 46080, 'samples_per_frame': 640, 'left_halo_frames': 0, 'right_halo_frames': 0}], 'pcm': 'Normalized once by the unchanged engine after full waveform reconstruction'}, 'ipc_scope': 'FP32/int64 inputs and complete FP32 waveform; split latent stays in worker'}}, {'index': 2, 'normalized_text': 'ありがたく受け取ります。', 'phones': [96, 322, 248, 160, 156, 96, 252, 96, 222, 254, 254, 222, 129, 252, 229, 248, 160, 225, 96, 323, 250, 254, 3], 'sampled_tokens': [280, 105, 234, 98, 156, 1007, 973, 129, 736, 38, 97, 232, 430, 202, 876, 350, 768, 136, 604, 382, 397, 456, 539, 777, 253, 461, 7, 900, 186, 145, 544, 132, 921, 500, 965, 965, 65, 17, 98, 29, 505, 1024], 'semantic_tokens': [280, 105, 234, 98, 156, 1007, 973, 129, 736, 38, 97, 232, 430, 202, 876, 350, 768, 136, 604, 382, 397, 456, 539, 777, 253, 461, 7, 900, 186, 145, 544, 132, 921, 500, 965, 965, 65, 17, 98, 29, 505], 'stop_reasons': ['argmax_eos', 'sample_eos'], 'returned_index': 41, 'waveform_samples': 52480, 'pcm_samples': 62080, 'timings': {'frontend_seconds': 0.0, 'condition_seconds': 0.00016999998479150236, 'semantic_seconds': 0.09457040001871064, 'acoustic_seconds': 0.039059000002453104, 'output_copy_pcm_seconds': 0.00011710001854225993, 'prepared_inference_seconds': 0.1339165000244975, 'compute_seconds': 0.1339165000244975}, 'acoustic_transport': {'roundtrip_ms': 38.85549999540672, 'worker_compute_ms': 38.35619999999995, 'transport_and_scheduling_ms': 0.4992999954067656, 'upload_bytes': 69632, 'download_bytes': 209920, 'worker_acoustic': {'experiment': 'chunked-vocoder', 'chunk_frames': 256, 'chunks': 1, 'latent_dtype': 'float16', 'latent_frames': 82, 'latent_d2h_bytes': 31488, 'latent_inputs_h2d_bytes': 69636, 'vocoder_inputs_h2d_bytes': 35584, 'vocoder_outputs_d2h_bytes': 209920, 'byte_scope': 'Logical host tensor sizes; excludes driver copies, workspaces and transfer profiling', 'latent_ms': 22.211899999999396, 'decode_ms': 38.27929999999924, 'acoustic_session_policy': 'resident', 'stage_intervals': [{'stage': 'acoustic.latent.run', 'start_unix_s': 1794580826.8790567, 'end_unix_s': 1794580826.901268, 'duration_ms': 22.20349999999982, 'pid': 90744}, {'stage': 'acoustic.vocoder.run', 'start_unix_s': 1794580826.901268, 'end_unix_s': 1794580826.9173107, 'duration_ms': 15.588699999999456, 'pid': 90744, 'chunk_index': 0}], 'plans': [{'input_start': 0, 'input_end': 82, 'core_start_frame': 0, 'core_end_frame': 82, 'core_sample_start': 0, 'core_sample_end': 52480, 'crop_start': 0, 'crop_end': 52480, 'samples_per_frame': 640, 'left_halo_frames': 0, 'right_halo_frames': 0}], 'pcm': 'Normalized once by the unchanged engine after full waveform reconstruction'}, 'ipc_scope': 'FP32/int64 inputs and complete FP32 waveform; split latent stays in worker'}}, {'index': 3, 'normalized_text': '真紅は、お気に入りのキャラクターなんですか？', 'phones': [251, 160, 323, 64, 222, 254, 316, 96, 1, 229, 322, 222, 160, 227, 160, 160, 248, 160, 227, 229, 86, 223, 96, 322, 248, 96, 222, 254, 252, 96, 96, 227, 96, 86, 64, 323, 127, 129, 250, 254, 222, 96, 4], 'sampled_tokens': [105, 13, 566, 235, 600, 246, 470, 464, 414, 760, 749, 450, 189, 203, 265, 478, 718, 638, 16, 936, 581, 185, 365, 451, 14, 609, 140, 505, 149, 535, 481, 193, 334, 80, 506, 557, 313, 615, 510, 446, 759, 228, 615, 309, 455, 119, 578, 311, 491, 387, 88, 998, 198, 114, 839, 16, 131, 378, 332, 393, 268, 387, 222, 789, 501, 183, 559, 132, 696, 750, 993, 168, 699, 99, 263, 623, 848, 98, 540, 693], 'semantic_tokens': [105, 13, 566, 235, 600, 246, 470, 464, 414, 760, 749, 450, 189, 203, 265, 478, 718, 638, 16, 936, 581, 185, 365, 451, 14, 609, 140, 505, 149, 535, 481, 193, 334, 80, 506, 557, 313, 615, 510, 446, 759, 228, 615, 309, 455, 119, 578, 311, 491, 387, 88, 998, 198, 114, 839, 16, 131, 378, 332, 393, 268, 387, 222, 789, 501, 183, 559, 132, 696, 750, 993, 168, 699, 99, 263, 623, 848, 98, 540], 'stop_reasons': ['argmax_eos'], 'returned_index': 79, 'waveform_samples': 101120, 'pcm_samples': 110720, 'timings': {'frontend_seconds': 0.0, 'condition_seconds': 4.4199987314641476e-05, 'semantic_seconds': 0.16655190000892617, 'acoustic_seconds': 0.05040269999881275, 'output_copy_pcm_seconds': 0.00012160002370364964, 'prepared_inference_seconds': 0.2171204000187572, 'compute_seconds': 0.2171204000187572}, 'acoustic_transport': {'roundtrip_ms': 50.12999998871237, 'worker_compute_ms': 49.59910000000001, 'transport_and_scheduling_ms': 0.5308999887123633, 'upload_bytes': 128464, 'download_bytes': 404480, 'worker_acoustic': {'experiment': 'chunked-vocoder', 'chunk_frames': 256, 'chunks': 1, 'latent_dtype': 'float16', 'latent_frames': 158, 'latent_d2h_bytes': 60672, 'latent_inputs_h2d_bytes': 128468, 'vocoder_inputs_h2d_bytes': 64768, 'vocoder_outputs_d2h_bytes': 404480, 'byte_scope': 'Logical host tensor sizes; excludes driver copies, workspaces and transfer profiling', 'latent_ms': 28.121699999999805, 'decode_ms': 49.52999999999985, 'acoustic_session_policy': 'resident', 'stage_intervals': [{'stage': 'acoustic.latent.run', 'start_unix_s': 1794580827.2864497, 'end_unix_s': 1794580827.314727, 'duration_ms': 28.110200000000418, 'pid': 90744}, {'stage': 'acoustic.vocoder.run', 'start_unix_s': 1794580827.315733, 'end_unix_s': 1794580827.3363173, 'duration_ms': 20.69720000000075, 'pid': 90744, 'chunk_index': 0}], 'plans': [{'input_start': 0, 'input_end': 158, 'core_start_frame': 0, 'core_end_frame': 158, 'core_sample_start': 0, 'core_sample_end': 101120, 'crop_start': 0, 'crop_end': 101120, 'samples_per_frame': 640, 'left_halo_frames': 0, 'right_halo_frames': 0}], 'pcm': 'Normalized once by the unchanged engine after full waveform reconstruction'}, 'ipc_scope': 'FP32/int64 inputs and complete FP32 waveform; split latent stays in worker'}}], 'execution_order': [1, 2, 0, 3], 'timing_scope': 'Original text through complete PCM, including missing model loads, transfers and sampling; file output separate', 'precision': 'GPT FP16, acoustic FP16; FP16 paths are experimental; public logits and acoustic I/O remain FP32', 'gpt_precision': 'fp16', 'acoustic_precision': 'fp16', 'acoustic_arena_shrink': True, 'acoustic_chunk_frames': 256, 'acoustic_session_policy': 'resident', 'gpt_prefill_query_chunk_size': 0, 'frontend_profile': {'implementation': 'pyopenjtalk-classic', 'version': '0.3.4', 'module_directory': 'classic-python', 'main_dictionary': 'classic-python/pyopenjtalk/open_jtalk_dic_utf_8-1.11'}, 'random_inputs': 'fresh', 'quality': {'human_listening': 'not_run', 'asr': 'not_run'}, 'profile': 'fp16', 'reference_ms': 212.46989999781363, 'reference_cache': 'memory', 'native_inference_ms': 1187.6712999946903}"
    },
    {
      "at": 14.341,
      "raw": "2026-11-13 22:40:27,341 INFO [uvicorn.access] [请求 -] 127.0.0.1:50568 - \"POST /tts HTTP/1.1\" 200"
    },
    {
      "at": 14.575,
      "raw": "2026-11-13 22:40:27,575 INFO [uvicorn.access] [请求 -] 127.0.0.1:50570 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 15.601,
      "raw": "2026-11-13 22:40:28,601 INFO [uvicorn.access] [请求 -] 127.0.0.1:50573 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 16.628,
      "raw": "2026-11-13 22:40:29,628 INFO [uvicorn.access] [请求 -] 127.0.0.1:56431 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 17.646,
      "raw": "2026-11-13 22:40:30,646 INFO [uvicorn.access] [请求 -] 127.0.0.1:56432 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 18.672,
      "raw": "2026-11-13 22:40:31,672 INFO [uvicorn.access] [请求 -] 127.0.0.1:56433 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 19.692,
      "raw": "2026-11-13 22:40:32,692 INFO [uvicorn.access] [请求 -] 127.0.0.1:56435 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 20.707,
      "raw": "2026-11-13 22:40:33,707 INFO [uvicorn.access] [请求 -] 127.0.0.1:56436 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 21.738,
      "raw": "2026-11-13 22:40:34,738 INFO [uvicorn.access] [请求 -] 127.0.0.1:56437 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 22.765,
      "raw": "2026-11-13 22:40:35,765 INFO [uvicorn.access] [请求 -] 127.0.0.1:56438 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 23.78,
      "raw": "2026-11-13 22:40:36,780 INFO [uvicorn.access] [请求 -] 127.0.0.1:56439 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 24.808,
      "raw": "2026-11-13 22:40:37,808 INFO [uvicorn.access] [请求 -] 127.0.0.1:56440 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 25.0,
      "raw": "[22:40:38] [TTS] [info] 语音播放完成 │ segment_index=0 segments=1 recording_id=6a3fd2aaf1cf4926b7f0579f3640eea7 status=finished code=null",
      "cue": "play-end"
    },
    {
      "at": 25.839,
      "raw": "2026-11-13 22:40:38,839 INFO [uvicorn.access] [请求 -] 127.0.0.1:56441 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 26.855,
      "raw": "2026-11-13 22:40:39,855 INFO [uvicorn.access] [请求 -] 127.0.0.1:56444 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 27.882,
      "raw": "2026-11-13 22:40:40,882 INFO [uvicorn.access] [请求 -] 127.0.0.1:56448 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 28.908,
      "raw": "2026-11-13 22:40:41,908 INFO [uvicorn.access] [请求 -] 127.0.0.1:56449 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 29.933,
      "raw": "2026-11-13 22:40:42,933 INFO [uvicorn.access] [请求 -] 127.0.0.1:56452 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 30.949,
      "raw": "2026-11-13 22:40:43,949 INFO [uvicorn.access] [请求 -] 127.0.0.1:56453 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 31.965,
      "raw": "2026-11-13 22:40:44,965 INFO [uvicorn.access] [请求 -] 127.0.0.1:56454 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 32.982,
      "raw": "2026-11-13 22:40:45,982 INFO [uvicorn.access] [请求 -] 127.0.0.1:56458 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 34.009,
      "raw": "2026-11-13 22:40:47,009 INFO [uvicorn.access] [请求 -] 127.0.0.1:56459 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 35.025,
      "raw": "2026-11-13 22:40:48,025 INFO [uvicorn.access] [请求 -] 127.0.0.1:56460 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 36.052,
      "raw": "2026-11-13 22:40:49,052 INFO [uvicorn.access] [请求 -] 127.0.0.1:56461 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 37.079,
      "raw": "2026-11-13 22:40:50,079 INFO [uvicorn.access] [请求 -] 127.0.0.1:56462 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 38.107,
      "raw": "2026-11-13 22:40:51,107 INFO [uvicorn.access] [请求 -] 127.0.0.1:56463 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 39.136,
      "raw": "2026-11-13 22:40:52,136 INFO [uvicorn.access] [请求 -] 127.0.0.1:56469 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 40.162,
      "raw": "2026-11-13 22:40:53,162 INFO [uvicorn.access] [请求 -] 127.0.0.1:56470 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 41.187,
      "raw": "2026-11-13 22:40:54,187 INFO [uvicorn.access] [请求 -] 127.0.0.1:56472 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 42.203,
      "raw": "2026-11-13 22:40:55,203 INFO [uvicorn.access] [请求 -] 127.0.0.1:56473 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 43.221,
      "raw": "2026-11-13 22:40:56,221 INFO [uvicorn.access] [请求 -] 127.0.0.1:56474 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 44.249,
      "raw": "2026-11-13 22:40:57,249 INFO [uvicorn.access] [请求 -] 127.0.0.1:56475 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 45.268,
      "raw": "2026-11-13 22:40:58,268 INFO [uvicorn.access] [请求 -] 127.0.0.1:56476 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 46.287,
      "raw": "2026-11-13 22:40:59,287 INFO [uvicorn.access] [请求 -] 127.0.0.1:60458 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 47.305,
      "raw": "2026-11-13 22:41:00,305 INFO [uvicorn.access] [请求 -] 127.0.0.1:51598 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 48.321,
      "raw": "2026-11-13 22:41:01,321 INFO [uvicorn.access] [请求 -] 127.0.0.1:51599 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 49.347,
      "raw": "2026-11-13 22:41:02,347 INFO [uvicorn.access] [请求 -] 127.0.0.1:51616 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 50.364,
      "raw": "2026-11-13 22:41:03,364 INFO [uvicorn.access] [请求 -] 127.0.0.1:51620 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 51.379,
      "raw": "2026-11-13 22:41:04,379 INFO [uvicorn.access] [请求 -] 127.0.0.1:51621 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 52.399,
      "raw": "2026-11-13 22:41:05,399 INFO [uvicorn.access] [请求 -] 127.0.0.1:51622 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 53.418,
      "raw": "2026-11-13 22:41:06,418 INFO [uvicorn.access] [请求 -] 127.0.0.1:52006 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 54.435,
      "raw": "2026-11-13 22:41:07,435 INFO [uvicorn.access] [请求 -] 127.0.0.1:52009 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 55.454,
      "raw": "2026-11-13 22:41:08,454 INFO [uvicorn.access] [请求 -] 127.0.0.1:52010 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 56.484,
      "raw": "2026-11-13 22:41:09,484 INFO [uvicorn.access] [请求 -] 127.0.0.1:52011 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 57.503,
      "raw": "2026-11-13 22:41:10,503 INFO [uvicorn.access] [请求 -] 127.0.0.1:52012 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 58.521,
      "raw": "2026-11-13 22:41:11,521 INFO [uvicorn.access] [请求 -] 127.0.0.1:52013 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 59.538,
      "raw": "2026-11-13 22:41:12,538 INFO [uvicorn.access] [请求 -] 127.0.0.1:52014 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 60.555,
      "raw": "2026-11-13 22:41:13,555 INFO [uvicorn.access] [请求 -] 127.0.0.1:52015 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 61.583,
      "raw": "2026-11-13 22:41:14,583 INFO [uvicorn.access] [请求 -] 127.0.0.1:52016 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 62.619,
      "raw": "2026-11-13 22:41:15,619 INFO [uvicorn.access] [请求 -] 127.0.0.1:52017 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 63.639,
      "raw": "2026-11-13 22:41:16,639 INFO [uvicorn.access] [请求 -] 127.0.0.1:52018 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 64.658,
      "raw": "2026-11-13 22:41:17,658 INFO [uvicorn.access] [请求 -] 127.0.0.1:52020 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 65.677,
      "raw": "2026-11-13 22:41:18,677 INFO [uvicorn.access] [请求 -] 127.0.0.1:52021 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 66.693,
      "raw": "2026-11-13 22:41:19,693 INFO [uvicorn.access] [请求 -] 127.0.0.1:52022 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 67.711,
      "raw": "2026-11-13 22:41:20,711 INFO [uvicorn.access] [请求 -] 127.0.0.1:52023 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 68.726,
      "raw": "2026-11-13 22:41:21,726 INFO [uvicorn.access] [请求 -] 127.0.0.1:52024 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 69.743,
      "raw": "2026-11-13 22:41:22,743 INFO [uvicorn.access] [请求 -] 127.0.0.1:52030 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 70.76,
      "raw": "2026-11-13 22:41:23,760 INFO [uvicorn.access] [请求 -] 127.0.0.1:52031 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 71.789,
      "raw": "2026-11-13 22:41:24,789 INFO [uvicorn.access] [请求 -] 127.0.0.1:52035 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 72.81,
      "raw": "2026-11-13 22:41:25,810 INFO [uvicorn.access] [请求 -] 127.0.0.1:52036 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 73.83,
      "raw": "2026-11-13 22:41:26,830 INFO [uvicorn.access] [请求 -] 127.0.0.1:52038 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 74.0,
      "raw": "[22:41:27] [TTS] [info] SakuraTTS 已休眠 │ plugin=sakura.tts.sakuratts backend=cuda bundle_source_commit=8fdc37f08962033956fbb65ea279687a0549aeb7 bundle_version=0.1.0a1-local-8fdc37f-20261005 event=tts.engine.sleeping gpu_driver=610.62 gpu_name=NVIDIA GeForce RTX 5070 Ti profile=fp16 provider_id=sakura.tts.sakuratts redacted=[REDACTED] requested_backend=cuda runtime_mode=managed",
      "cue": "sleeping"
    },
    {
      "at": 74.846,
      "raw": "2026-11-13 22:41:27,846 INFO [uvicorn.access] [请求 -] 127.0.0.1:52042 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 75.867,
      "raw": "2026-11-13 22:41:28,867 INFO [uvicorn.access] [请求 -] 127.0.0.1:52043 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 76.885,
      "raw": "2026-11-13 22:41:29,885 INFO [uvicorn.access] [请求 -] 127.0.0.1:52044 - \"GET /runtime HTTP/1.1\" 200"
    },
    {
      "at": 77.914,
      "raw": "2026-11-13 22:41:30,914 INFO [uvicorn.access] [请求 -] 127.0.0.1:52045 - \"GET /runtime HTTP/1.1\" 200"
    }
  ]
};
