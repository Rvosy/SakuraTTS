"""Build an offline terminal and lifecycle replay from one recorded take."""

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
from urllib.parse import quote


SERVICE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\s+\w+\s+(.*)")
DESKTOP = re.compile(r"^\[(\d\d:\d\d:\d\d)\] \[[^\]]+\] (?:\[[^\]]+\] )?(.*)")
ENGINE_REQUEST = re.compile(r"^\[sakuratts\.(engine|server)\] \[请求 (\d+)\] (.*)")
CUES = {
    "正在等待模型回复": "llm-start", "回复已生成": "llm-end",
    "发送模型请求": "llm-start", "收到模型回复": "llm-end", "模型请求成功": "llm-end",
    "模型请求已完成": "llm-end", "模型请求失败": "llm-end",
    "SakuraTTS 正在提前加载模型": "wake", "SakuraTTS 正在加载模型": "loading",
    "SakuraTTS 模型已加载": "awake", "SakuraTTS 正在释放模型": "releasing",
    "SakuraTTS 已休眠": "sleeping", "开始合成语音": "job-start", "语音合成完成": "job-end",
    "开始播放语音": "play-start", "开始播放音频": "play-start",
    "语音播放完成": "play-end", "音频播放完成": "play-end",
    "语音播放已停止": "play-end", "语音播放失败": "play-end",
}


def read_records(path, start, duration):
    stamp = None
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            raw = line.rstrip("\r\n")
            cue = request = None
            service = SERVICE.match(raw)
            desktop = DESKTOP.match(raw)
            if service:
                stamp = datetime.strptime(service[1], "%Y-%m-%d %H:%M:%S,%f")
                event = ENGINE_REQUEST.match(service[2])
                if event:
                    request = event[2]
                    if event[3].startswith("请求 #"):
                        cue = "synth-start"
                    elif event[3].startswith(("完成 #", "达到长度上限 #", "取消 #")):
                        cue = "synth-end"
            elif desktop:
                clock = datetime.strptime(desktop[1], "%H:%M:%S").time()
                stamp = datetime.combine(start.date(), clock)
                if stamp + timedelta(seconds=1) <= start:
                    stamp += timedelta(days=1)
                cue = CUES.get(desktop[2].partition(" │ ")[0].strip())
            if stamp is None:
                continue
            seconds = (stamp - start).total_seconds()
            if 0 <= seconds <= duration:
                row = dict(at=seconds, raw=raw)
                if cue:
                    row["cue"] = cue
                    if request:
                        row["request"] = request
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=datetime.fromisoformat, help="local time of the first video frame")
    parser.add_argument("--seconds", required=True, type=float, help="length of the uncut take")
    parser.add_argument("--video", type=Path, help="matching MP4 or WebM")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("replay-data.js"))
    parser.add_argument("logs", nargs="+", type=Path, help="log extracts from the recording")
    args = parser.parse_args()
    if args.video and not args.video.is_file():
        parser.error("Video file does not exist")
    rows = sorted((row for path in args.logs for row in read_records(path, args.start, args.seconds)), key=lambda r: r["at"])
    if not rows:
        parser.error("No log lines in the selected time range")
    video = quote(Path(os.path.relpath(args.video, args.output.parent)).as_posix()) if args.video else None
    take = dict(video=video, duration=args.seconds, records=rows)
    args.output.write_text("const recordedTake = " + json.dumps(take, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


if __name__ == "__main__":
    main()
