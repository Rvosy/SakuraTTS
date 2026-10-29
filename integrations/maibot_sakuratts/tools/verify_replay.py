"""Audit handoff evidence, frozen character inputs, and production state without exporting secrets."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib


def main(directory):
    work = Path(directory).resolve()
    if work.name != "sakura-voice-candidate-20261026":
        raise ValueError("Expected isolated voice replay directory")
    frozen = json.loads((work / "private/frozen-inputs.json").read_text())
    source = Path(frozen["source"])
    baseline, candidate = work / "baseline", work / "candidate"
    runtime_files = [p for p in frozen["files"] if p.startswith(("plugins/", "data/"))]
    assert all((baseline / p).read_bytes() == (candidate / p).read_bytes() for p in runtime_files)
    role_fields = ["personality", "behavior_style", "reply_style"]
    configs = [tomllib.loads((root / "config/bot_config.toml").read_text()) for root in [baseline, candidate, source]]
    assert all(c["personality"][k] == configs[0]["personality"][k] for c in configs for k in role_fields)
    upstream_changes = [p for p in runtime_files if (source / p).read_bytes() != (candidate / p).read_bytes()]

    before = json.loads((work / "private/production-before.json").read_text())
    states = json.loads(subprocess.check_output(["docker", "inspect", "maibot-core", "maibot-snowluma"], text=True))
    starts = "".join(s["Name"] + " " + s["State"]["StartedAt"] + "\n" for s in states)
    assert starts == before["started"] and all(s["State"]["Running"] for s in states)

    audits = []
    for filename in sorted((work / "private/results").glob("*.json")):
        data = json.loads(filename.read_text())
        evidence_count = 0
        for row in data["records"]:
            assert "error" not in row, (filename.name, row["case"], row.get("error"))
            reference = json.loads(row["reply_args"]["reply_reference"])
            assert set(reference) == {"version", "context_id", "source_ids", "response_act", "scene_ids", "evidence"}
            sources = {}
            for item in row["planner_request"]:
                if item["item_type"] in ["UserMessageItem", "AssistantMessageItem"]:
                    text = "\n".join(p["text"] for p in item["parts"] if p["type"] == "text")
                    match = re.match(r'^<message\s+msg_id="([^"]+)"', text)
                    if match:
                        sources["message:" + match[1]] = text
                elif item["item_type"] == "FunctionCallOutputItem":
                    sources["tool:" + item["call_id"]] = json.dumps({k:item[k] for k in ["tool_name", "success", "output"]}, ensure_ascii=False)
            assert reference["source_ids"] == [e["id"] for e in reference["evidence"]]
            for evidence in reference["evidence"]:
                assert sources[evidence["id"]] == evidence["text"]
                evidence_count += 1
            if row["extension_selected"]:
                envelope = json.loads(row["raw_reply"])
                assert row["reply"] == envelope["text"] and row["voice_ja"] == ([envelope["voice_ja"]] if envelope["voice_ja"] else [])
                assert row["extra_prompt"].count("本次回复启用了日语语音扩展。") == 1
                assert row["response_schema"]["strict"] is True
        audits.append({"file": filename.name, "replies": len(data["records"]), "verbatim_evidence_items": evidence_count})
    host_log = (work / "private/final-host-validation.log").read_text()
    host_checks = ["REAL_HOST_EXTENSION_BOUNDARY_OK", "REAL_REPLYER_RESPONSE_SCHEMA_OK", "REAL_PLUGIN_RUNNER_VALIDATION_OK"]
    assert all(check in host_log for check in host_checks)
    result = {
        "role_fields_unchanged": role_fields,
        "frozen_runtime_files_per_arm": len(runtime_files),
        "upstream_changes": upstream_changes,
        "production_containers_running_without_restart": [s["Name"] for s in states],
        "host_checks": host_checks,
        "replays": audits,
    }
    (work / "export").mkdir(exist_ok=True)
    (work / "export/validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
