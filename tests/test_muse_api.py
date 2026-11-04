"""Muse Spark configuration and OpenAI-compatible tool-call adapter."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import muse_api as M  # noqa: E402


def clear_env(monkeypatch):
    for key in M.CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_name_muse_spark_and_request_key(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setattr(M, "_file_config", lambda path=None: {})
    s = M.status()
    assert s["model"] == "muse-spark-1.3"
    assert s["endpoint"] == "https://api.meta.ai/v1"
    assert not s["configured"] and s["needs_api_key"]
    code, out = M.plan({"message": "change the width"})
    assert code == 503 and out["needs_api_key"]


def test_local_config_file_is_parsed_without_shell_execution(tmp_path):
    p = tmp_path / ".env.muse"
    p.write_text("# local only\nMODEL_API_KEY='secret'\nMUSE_MODEL=muse-spark-1.3\nIGNORED=x\n")
    assert M._file_config(str(p)) == {"MODEL_API_KEY": "secret", "MUSE_MODEL": "muse-spark-1.3"}


def test_plan_sends_context_and_parses_registered_tool_call(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setattr(M, "_file_config", lambda path=None: {"MODEL_API_KEY": "test-key"})
    seen = {}

    class Reply:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"model": "muse-spark-1.3", "choices": [{"message": {"content": "",
                "tool_calls": [{"function": {"name": "set_controls", "arguments": '{"changes":[{"key":"lig.length_target_mm","value":34}]}'}}]}}]}).encode()

    def fake(req, timeout):
        seen["url"], seen["auth"], seen["body"], seen["timeout"] = req.full_url, req.headers["Authorization"], json.loads(req.data), timeout
        return Reply()

    monkeypatch.setattr(M.urllib.request, "urlopen", fake)
    body = {"message": "Use the patient-provided ACL length", "context": {"patient_details": "Reviewed ACL length: 34 mm"},
            "tools": [{"type": "function", "function": {"name": "set_controls"}}]}
    code, out = M.plan(body)
    assert code == 200 and out["calls"][0]["name"] == "set_controls"
    assert seen["url"] == "https://api.meta.ai/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert "Reviewed ACL length: 34 mm" in seen["body"]["messages"][1]["content"]
