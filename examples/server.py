"""Run with: python examples/server.py TTS_CONFIG_PATH"""

import sys
from sakuratts import start_server

if __name__ == "__main__":
    start_server(tts_config=sys.argv[1])
