"""Muse Spark configuration and OpenAI-compatible tool-call adapter."""
import json
import os
import socket
import sys
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import muse_api as M  # noqa: E402


def control_tool():
    return {"type": "function", "function": {
        "name": "set_controls",
        "parameters": {"type": "object", "properties": {
            "changes": {"type": "array", "items": {"type": "object", "properties": {
                "key": {"type": "string", "enum": ["lig.length_target_mm", "sheath.wall_mm"]},
                "value": {}}, "required": ["key", "value"]}}
        }, "required": ["changes"]}
    }}


class Reply:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode()


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
    assert s["provider"] == "Meta Model API" and s["tool_calling"]


def test_local_config_file_is_parsed_without_shell_execution(tmp_path):
    p = tmp_path / ".env.muse"
    p.write_text("# local only\nMODEL_API_KEY='secret'\nMUSE_MODEL=muse-spark-1.3\nIGNORED=x\n")
    assert M._file_config(str(p)) == {"MODEL_API_KEY": "secret", "MUSE_MODEL": "muse-spark-1.3"}


def test_plan_sends_context_and_parses_registered_tool_call(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setattr(M, "_file_config", lambda path=None: {"MODEL_API_KEY": "test-key"})
    seen = {}

    def fake(req, timeout):
        seen["url"], seen["auth"], seen["body"], seen["timeout"] = req.full_url, req.headers["Authorization"], json.loads(req.data), timeout
        return Reply({"model": "muse-spark-1.3", "choices": [{"message": {"content": "",
            "tool_calls": [{"function": {"name": "set_controls", "arguments": '{"changes":[{"key":"lig.length_target_mm","value":34}]}'}}]}}]})

    monkeypatch.setattr(M.urllib.request, "urlopen", fake)
    body = {"message": "Use the patient-provided ACL length", "context": {"patient_details": "Reviewed ACL length: 34 mm"},
            "tools": [control_tool()]}
    code, out = M.plan(body)
    assert code == 200 and out["calls"][0]["name"] == "set_controls"
    assert seen["url"] == "https://api.meta.ai/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert "Reviewed ACL length: 34 mm" in seen["body"]["messages"][1]["content"]


def test_respond_is_grounded_in_actual_results_and_has_no_tools(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setattr(M, "_file_config", lambda path=None: {"MODEL_API_KEY": "test-key"})
    seen = {}

    def fake(req, timeout):
        seen["body"] = json.loads(req.data)
        return Reply({"model": "muse-spark-1.3", "choices": [{"message": {"content": "Wall changed to 0.3 mm; validation passed."}}]})

    monkeypatch.setattr(M.urllib.request, "urlopen", fake)
    code, out = M.respond({"message": "set wall", "context": {"revision": 4},
                           "results": [{"status": "applied", "changes": [{"to": 0.3}], "validation": "all fit checks pass"}]})
    assert code == 200 and "0.3 mm" in out["text"]
    assert "tools" not in seen["body"]
    assert "revision" in seen["body"]["messages"][1]["content"]


def configured(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setattr(M, "_file_config", lambda path=None: {"MODEL_API_KEY": "sentinel-secret"})


def plan_with(monkeypatch, payload, tools=None):
    configured(monkeypatch)
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda req, timeout: Reply(payload))
    return M.plan({"message": "test", "tools": tools if tools is not None else [control_tool()]})


def test_status_and_errors_never_expose_secret_or_authorization(monkeypatch):
    configured(monkeypatch)
    status = json.dumps(M.status())
    assert "sentinel-secret" not in status and "Authorization" not in status
    err = urllib.error.HTTPError("https://api.meta.ai/v1/chat/completions", 401, "no", {}, None)
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(err))
    code, out = M.plan({"message": "x", "tools": []})
    blob = json.dumps(out)
    assert code == 401 and out["category"] == "authentication_failed"
    assert "sentinel-secret" not in blob and "Authorization" not in blob


def test_text_only_and_tool_plus_text_are_valid(monkeypatch):
    code, out = plan_with(monkeypatch, {"choices": [{"message": {"content": "I need a clarification."}}]})
    assert code == 200 and out["calls"] == [] and out["text"]
    code, out = plan_with(monkeypatch, {"choices": [{"message": {"content": "Applying the requested value.", "tool_calls": [
        {"function": {"name": "set_controls", "arguments": '{"changes":[{"key":"sheath.wall_mm","value":0.3}]}'}}]}}]})
    assert code == 200 and out["calls"][0]["args"]["changes"][0]["value"] == 0.3 and out["text"]


def test_multiple_and_duplicate_tool_calls(monkeypatch):
    tc = {"function": {"name": "set_controls", "arguments": '{"changes":[{"key":"sheath.wall_mm","value":0.3}]}'}}
    code, out = plan_with(monkeypatch, {"choices": [{"message": {"tool_calls": [tc, tc]}}]})
    assert code == 200 and len(out["calls"]) == 1


def test_malformed_unknown_and_conflicting_tools_are_rejected(monkeypatch):
    code, out = plan_with(monkeypatch, {"choices": [{"message": {"tool_calls": [{"function": {"name": "set_controls", "arguments": "{bad"}}]}}]})
    assert code == 502 and out["category"] == "malformed_tool_call"
    code, out = plan_with(monkeypatch, {"choices": [{"message": {"tool_calls": [{"function": {"name": "run_shell", "arguments": "{}"}}]}}]})
    assert code == 502 and out["category"] == "invalid_tool_call"
    args = '{"changes":[{"key":"sheath.wall_mm","value":0.3},{"key":"sheath.wall_mm","value":0.4}]}'
    code, out = plan_with(monkeypatch, {"choices": [{"message": {"tool_calls": [{"function": {"name": "set_controls", "arguments": args}}]}}]})
    assert code == 502 and "conflicting" in out["error"]


def test_missing_choices_invalid_json_and_empty_grounded_response(monkeypatch):
    code, out = plan_with(monkeypatch, {})
    assert code == 502 and out["category"] == "invalid_response"
    configured(monkeypatch)
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda req, timeout: Reply(b"not json"))
    code, out = M.plan({"message": "x", "tools": []})
    assert code == 502 and out["category"] == "invalid_response"
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda req, timeout: Reply({"choices": [{"message": {"content": ""}}]}))
    code, out = M.respond({"message": "x", "context": {}, "results": []})
    assert code == 502 and out["category"] == "invalid_response"


def test_rate_limit_retries_but_auth_does_not(monkeypatch):
    configured(monkeypatch); calls = []
    def limited(*a, **k):
        calls.append(1)
        if len(calls) < 3: raise urllib.error.HTTPError("x", 429, "limited", {}, None)
        return Reply({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(M.urllib.request, "urlopen", limited); monkeypatch.setattr(M.time, "sleep", lambda n: None)
    code, out = M.plan({"message": "x", "tools": []})
    assert code == 200 and len(calls) == 3
    calls.clear()
    def denied(*a, **k): calls.append(1); raise urllib.error.HTTPError("x", 403, "denied", {}, None)
    monkeypatch.setattr(M.urllib.request, "urlopen", denied)
    code, out = M.plan({"message": "x", "tools": []})
    assert code == 403 and len(calls) == 1 and out["category"] == "authorization_failed"


def test_provider_500_retries_and_is_sanitized(monkeypatch):
    configured(monkeypatch); calls = []
    def broken(*a, **k):
        calls.append(1)
        if len(calls) < 3: raise urllib.error.HTTPError("x", 500, "secret provider body", {}, None)
        return Reply({"choices": [{"message": {"content": "recovered"}}]})
    monkeypatch.setattr(M.urllib.request, "urlopen", broken); monkeypatch.setattr(M.time, "sleep", lambda n: None)
    code, out = M.plan({"message": "x", "tools": []})
    assert code == 200 and len(calls) == 3 and "secret provider body" not in json.dumps(out)


def test_timeout_and_network_failure_are_structured(monkeypatch):
    configured(monkeypatch); monkeypatch.setattr(M.time, "sleep", lambda n: None)
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(socket.timeout()))
    code, out = M.plan({"message": "x", "tools": []})
    assert code == 504 and out["category"] == "timeout" and out["retryable"]
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    code, out = M.plan({"message": "x", "tools": []})
    assert code == 502 and out["category"] == "network_unavailable"


def test_second_pass_rejects_tool_calls(monkeypatch):
    configured(monkeypatch)
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda req, timeout: Reply({"choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "set_controls", "arguments": "{}"}}]}}]}))
    code, out = M.respond({"message": "x", "context": {}, "results": []})
    assert code == 502 and out["category"] == "invalid_response"


def test_empty_plan_is_rejected(monkeypatch):
    code, out = plan_with(monkeypatch, {"choices": [{"message": {"content": "", "tool_calls": []}}]})
    assert code == 502 and out["category"] == "invalid_response" and out["retryable"]


def test_credential_file_is_ignored_and_state_store_does_not_include_env(monkeypatch, tmp_path):
    ignored = (open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read().splitlines())
    assert ".env.muse" in ignored
    import server_v2
    monkeypatch.setattr(server_v2, "STORE", str(tmp_path))
    monkeypatch.setenv("MODEL_API_KEY", "sentinel-secret")
    sid = server_v2.save_state({"case": {"synthetic": True}, "params": {"sheath.wall_mm": 0.3}, "provenance": {}, "history": []})
    blob = (tmp_path / (sid + ".json")).read_text()
    assert "sentinel-secret" not in blob and "MODEL_API_KEY" not in blob and "Authorization" not in blob
