"""Run with: python examples/basic.py MODEL_DIRECTORY OUTPUT.wav"""

import sys
from sakuratts import Engine

with Engine.load(sys.argv[1]) as engine:
    audio = engine.synthesize("こんにちは。今日はどんな一日でしたか。")
    audio.save(sys.argv[2])
    print(audio.report["status"])
