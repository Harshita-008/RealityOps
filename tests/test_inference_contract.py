from inference import (
    _heuristic_action,
    _safe_json_parse,
    _sanitize_action,
    action_to_text,
    log_end,
    log_start,
    log_step,
)


def test_safe_json_parse_accepts_direct_json_object() -> None:
    parsed = _safe_json_parse('{"type":"wait"}')
    assert parsed == {"type": "wait"}


def test_safe_json_parse_extracts_embedded_json() -> None:
    parsed = _safe_json_parse("noise before {\"type\":\"probe\"} noise after")
    assert parsed == {"type": "probe"}


def test_safe_json_parse_rejects_non_object_json() -> None:
    assert _safe_json_parse("[1,2,3]") is None


def test_sanitize_action_rejects_unknown_type() -> None:
    action = _sanitize_action({"type": "hack_prod"})
    assert action == {"type": "wait"}


def test_sanitize_action_drops_non_dict_payload() -> None:
    action = _sanitize_action({"type": "update_belief", "payload": "bad"})
    assert action == {"type": "update_belief"}


def test_action_to_text_sorts_payload_keys() -> None:
    text = action_to_text({"type": "commit_fix", "payload": {"z": 1, "a": 2}})
    assert text == "commit_fix(a=2,z=1)"


def test_log_line_contracts(capsys) -> None:
    log_start(task="false_alarm", env="realityops_arena", model="model-x")
    log_step(step=2, action="check_logs", reward=0.5, done=False, error=None)
    log_end(success=True, steps=3, score=0.8123, rewards=[0.1, 0.2, 0.3])

    lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0] == "[START] task=false_alarm env=realityops_arena model=model-x"
    assert lines[1] == "[STEP] step=2 action=check_logs reward=0.50 done=false error=null"
    assert lines[2] == "[END] success=true steps=3 score=0.812 rewards=0.10,0.20,0.30"


def test_heuristic_false_alarm_first_step_prefers_metrics() -> None:
    obs = {
        "logs": ["all clear"],
        "metrics": {"latency": 180.0, "error_rate": 0.02},
    }
    action = _heuristic_action("false_alarm", obs, step=1, history=[], info={})
    assert action["type"] == "check_metrics"


def test_heuristic_commit_fix_uses_fix_payload_when_confirmation_needed() -> None:
    obs = {
        "logs": ["cross-az packet loss above threshold"],
        "metrics": {"latency": 480.0, "error_rate": 0.22},
    }
    history = [
        {"type": "check_logs"},
        {"type": "check_metrics"},
        {"type": "probe"},
        {"type": "update_belief"},
        {"type": "safe_mitigation"},
    ]
    action = _heuristic_action("revenue_tradeoff", obs, step=6, history=history, info={"requires_fix_confirmation": True})
    assert action["type"] == "commit_fix"
    assert action["payload"]["fix"] == "reroute_traffic"


def test_heuristic_multi_incident_transitions_to_commit_fix() -> None:
    obs = {
        "logs": ["cross-az packet loss above threshold", "db pool exhausted for payments-writer"],
        "metrics": {"latency": 420.0, "error_rate": 0.24},
    }
    history = [
        {"type": "check_logs"},
        {"type": "check_metrics"},
        {"type": "probe"},
        {"type": "update_belief"},
        {"type": "update_belief"},
        {"type": "update_belief"},
        {"type": "safe_mitigation"},
    ]

    action = _heuristic_action("multi_incident", obs, step=8, history=history, info={"requires_fix_confirmation": True})
    assert action["type"] == "commit_fix"
    assert action["payload"]["fix"] in {"reroute_traffic", "increase_pool"}
