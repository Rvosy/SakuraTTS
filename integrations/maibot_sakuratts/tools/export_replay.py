"""Export only anonymized dialogue and counts; model configs and raw corpus stay remote."""

from collections import Counter
from html import escape
import json
from pathlib import Path
import re
import sys


def main(directory):
    work = Path(directory).resolve()
    if work.name != "sakura-voice-candidate-20261026":
        raise ValueError("Expected isolated voice replay directory")
    corpus = json.loads((work / "private/corpus.json").read_text())
    result_files = sorted((work / "private/results").glob("*.json"))
    data = [json.loads(p.read_text()) for p in result_files if p.name.startswith(("baseline-", "candidate-"))]
    diagnostics = [json.loads(p.read_text()) for p in result_files if p.name.startswith("auto-probe-")]
    expected = sum(len(case["steps"]) for case in corpus["cases"])
    if {d["condition"] for d in data} != {"baseline", "candidate"} or len(data) != 2:
        raise ValueError("Expected one natural replay per arm")
    if any(len(d["records"]) != expected for d in data):
        raise ValueError("Natural replay is incomplete; inspect private results before exporting")
    names = {}
    people = {}
    for case in corpus["cases"]:
        for row in case["seed"] + [r for step in case["steps"] for r in step["inputs"]]:
            if row["user_id"] != case["bot_id"]:
                label = people.setdefault(row["user_id"], f"参与者{len(people) + 1}")
                if row["name"]:
                    names[row["name"]] = label
    if alias := names.get("朱雀院红叶") or names.get("红叶"):
        names["红叶"] = names["紅葉"] = alias

    def clean(value):
        for name in sorted(names, key=len, reverse=True):
            value = value.replace(name, names[name])
        return re.sub(r"(?<!\d)\d{5,12}(?!\d)", "[编号]", value)

    rows = {}
    counts = []
    exported = []
    diagnostic_rows = []
    for arm in data + diagnostics:
        diagnostic = bool(arm.get("forced_voice_auto"))
        label = "auto-probe" if diagnostic else arm["condition"]
        records = arm["records"]
        counts.append({
            "arm": label, "records": len(records),
            "errors": sum("error" in r for r in records),
            "extension_selected": sum(bool(r.get("extension_selected")) for r in records),
            "voice_messages": sum(len(r.get("voice_ja", [])) for r in records),
            "models": dict(Counter(r.get("reply_model", "unknown") for r in records)),
        })
        for row in records:
            safe = {k: row.get(k) for k in ["case", "turn", "extension_selected", "selected_options", "seconds"]}
            safe.update(arm=label, reply=clean(row.get("reply", "")),
                        voice_ja=[clean(v) for v in row.get("voice_ja", [])],
                        raw_reply=clean(row.get("raw_reply", "")), error=clean(row.get("error", "")))
            if diagnostic:
                diagnostic_rows.append(safe)
            else:
                rows[(arm["condition"], row["case"], row["turn"])] = safe
            exported.append(safe)
    output = work / "export"
    output.mkdir(exist_ok=True)
    failures = []
    for attempt in ["format-conflict", "system-format-conflict", "format-replacement"]:
        previous = work / "private" / attempt / "auto-probe-candidate-gpt-6-luna-reply.json"
        if previous.exists():
            failures.extend({"attempt": attempt, "case": r["case"], "raw_reply": clean(r.get("raw_reply", r.get("reply", ""))),
                             "error": clean(r.get("error", ""))} for r in json.loads(previous.read_text())["records"])
    (output / "replay-results.json").write_text(json.dumps({"counts": counts, "records": exported,
                                                            "prior_format_diagnostic": failures}, ensure_ascii=False, indent=2))
    style = "body{font-family:system-ui,sans-serif;max-width:1400px;margin:36px auto;padding:0 24px;line-height:1.7;color:#243247}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #d3dce7;padding:14px;vertical-align:top;text-align:left}th{background:#eef3f8}td{width:33%}pre{white-space:pre-wrap}small{color:#607085}.voice{background:#edf6f1;padding:10px}"
    parts = [f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>日语语音回复扩展回放</title><style>{style}</style><body>',
             "<h1>日语语音回复扩展：0.2.0 候选回放</h1>",
             "<p>2026-10-26 · 角色卡与交接候选冻结 · 未部署 · 无 QQ 发送</p>",
             "<p>两组均使用角色交接候选 1.2.0、相同角色字段与原作资料。基线仅文字；候选注册可选的日语回复扩展。Planner 使用 gpt-6.1-sol，Replyer 固定为已配置的 gpt-6-luna-reply。每组含 12 个历史回应点和 4 个独立构造的测试场景。</p>",
             "<p>运行真实 Planner、Replyer、插件运行器与 Host 回复扩展控制器；音频来自容器内 HTTP 替身。测试没有运行真实语音模型、音质评估、完整聊天调度或平台发送。历史续接为固定用户消息，每组只采样一次，结果不代表稳定触发率。</p>",
             "<p>最终候选配套最小 Host 补丁，将回复 Hook 提供的 response_schema 交给已有模型接口。角色卡、交接插件和回复模板均与基线一致；只有选中扩展的请求使用 JSON Schema。完整 generate_reply_with_context 的参数传递另以模型替身验证。</p>",
             "<p>各分支将自身生成的可见文字带入后续上下文；日语语音的历史文本通过 Host 的消息转换器提取。最多保留 80 条消息，朋友故事最后一轮前有超过该窗口的用户消息，不能作为四轮连续记忆证明。结构异常和模型请求失败照实保留。</p>",
             "<h2>计数</h2><pre>" + escape(json.dumps(counts, ensure_ascii=False, indent=2)) + "</pre>"]
    review = work / "private/review.txt"
    if review.exists():
        parts.append("<h2>复核结论</h2><pre>" + escape(review.read_text()) + "</pre>")
    for case in corpus["cases"]:
        parts.append("<h2>" + escape(case["id"]) + ("（构造测试）" if case.get("synthetic") else "（历史片段）") + "</h2>")
        parts.append("<table><tr><th>本轮用户输入</th><th>文字基线</th><th>语音候选</th></tr>")
        for turn, step in enumerate(case["steps"], 1):
            inputs = "\n".join(people.get(r["user_id"], "Navi") + "：" + clean(r["text"]) for r in step["inputs"])
            input_cell = "<pre>" + escape(inputs) + "</pre>"
            if len(step["inputs"]) > 8:
                input_cell = f'<details><summary>本轮 {len(step["inputs"])} 条用户消息，展开查看</summary>' + input_cell + "</details>"
            cells = [input_cell]
            for arm in ["baseline", "candidate"]:
                row = rows.get((arm, case["id"], turn))
                if row is None:
                    cells.append("本轮未完成")
                    continue
                cell = "<pre>" + escape(row["reply"] or row["error"]) + "</pre>"
                for voice in row["voice_ja"]:
                    cell += '<p class="voice">日语台词：' + escape(voice) + "</p>"
                cell += "<small>扩展选择：" + escape(str(row["selected_options"])) + "</small>"
                if row["raw_reply"]:
                    cell += "<details><summary>Replyer 原始输出</summary><pre>" + escape(row["raw_reply"]) + "</pre></details>"
                cells.append(cell)
            parts.append("<tr>" + "".join("<td>" + cell + "</td>" for cell in cells) + "</tr>")
        parts.append("</table>")
    if diagnostic_rows:
        parts.append("<h2>auto 模式单独诊断</h2><p>以下场景由测试脚本强制选择 auto 扩展，再由真实 Replyer 决定台词或 null。它们用于检查 Replyer 的选择和输出格式，不计入 Planner 自发选择语音的结果。</p>")
        for row in diagnostic_rows:
            parts.append("<h3>" + escape(row["case"]) + "</h3><pre>" + escape(row["raw_reply"] or row["error"]) + "</pre>")
    if failures:
        parts.append("<h2>修复前的格式失败</h2><p>仅在 extra_prompt 中要求 JSON、向 system 追加协议、替换 Host 的普通文字格式要求，三种尝试各有三次诊断失败，均被发送边界拒绝。最终候选改用模型接口的 JSON Schema 约束，删除了文案替换逻辑，并重新运行候选组和 auto 诊断。这些旧结果不计入最终候选通过数。</p>")
        for row in failures:
            parts.append("<details><summary>" + escape(row["attempt"] + " / " + row["case"]) + "</summary><pre>" + escape(row["raw_reply"] + "\n" + row["error"]) + "</pre></details>")
    parts.append("</body></html>")
    (output / "replay-review.html").write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
