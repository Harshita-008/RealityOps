from __future__ import annotations

from typing import Dict, Tuple


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _belief_alignment(state: Dict) -> float:
    return _clamp01(state["beliefs"].get(state["active_world"], 0.0))


def _evidence_coverage(state: Dict) -> float:
    checks = state["investigations"]
    covered = sum(1 for value in checks.values() if value)
    return covered / float(len(checks))


def _revenue_control(state: Dict) -> float:
    max_loss = state["task_spec"]["revenue_loss_per_step"] * state["task_spec"]["max_steps"]
    return _clamp01(1.0 - (state["revenue_loss"] / float(max_loss)))


def _efficiency(state: Dict) -> float:
    return _clamp01(1.0 - (state["steps"] / float(state["task_spec"]["max_steps"])))


def _anti_gaming(state: Dict) -> float:
    repeat_penalty = min(0.25, state.get("repeat_actions", 0) * 0.05)
    premature_penalty = min(0.30, state.get("premature_fix_count", 0) * 0.10)
    invalid_penalty = min(0.30, state.get("invalid_fix_count", 0) * 0.10)
    wait_penalty = min(0.20, max(0, state.get("wait_count", 0) - 2) * 0.04)
    return _clamp01(1.0 - repeat_penalty - premature_penalty - invalid_penalty - wait_penalty)


def _score_false_alarm(state: Dict) -> Tuple[float, Dict[str, float]]:
    no_risky_change = 0.0 if state["risky_used"] else 1.0
    no_unnecessary_fix = 1.0 if state["applied_fix"] in (None, "no_fix") else 0.0
    evidence = _evidence_coverage(state)
    efficiency = _efficiency(state)
    anti_gaming = _anti_gaming(state)

    score = (
        0.40 * no_unnecessary_fix
        + 0.25 * no_risky_change
        + 0.20 * evidence
        + 0.10 * efficiency
        + 0.05 * anti_gaming
    )
    return _clamp01(score), {
        "no_unnecessary_fix": no_unnecessary_fix,
        "safe_ops": no_risky_change,
        "evidence": evidence,
        "efficiency": efficiency,
        "anti_gaming": anti_gaming,
    }


def _score_ambiguous_root(state: Dict, worlds: Dict) -> Tuple[float, Dict[str, float]]:
    correct_fix = 1.0 if state["applied_fix"] == worlds[state["active_world"]]["fix"] else 0.0
    evidence = _evidence_coverage(state)
    belief = _belief_alignment(state)
    efficiency = _efficiency(state)
    anti_gaming = _anti_gaming(state)

    score = 0.48 * correct_fix + 0.20 * evidence + 0.17 * belief + 0.10 * efficiency + 0.05 * anti_gaming
    return _clamp01(score), {
        "correct_fix": correct_fix,
        "evidence": evidence,
        "belief_alignment": belief,
        "efficiency": efficiency,
        "anti_gaming": anti_gaming,
    }


def _score_revenue_tradeoff(state: Dict, worlds: Dict) -> Tuple[float, Dict[str, float]]:
    correct_fix = 1.0 if state["applied_fix"] == worlds[state["active_world"]]["fix"] else 0.0
    mitigation_before_fix = 0.0
    if state["mitigation_step"] is not None:
        if state["fix_step"] is None or state["mitigation_step"] <= state["fix_step"]:
            mitigation_before_fix = 1.0
    revenue_control = _revenue_control(state)
    belief = _belief_alignment(state)
    anti_gaming = _anti_gaming(state)

    score = (
        0.42 * correct_fix
        + 0.25 * mitigation_before_fix
        + 0.20 * revenue_control
        + 0.08 * belief
        + 0.05 * anti_gaming
    )
    if state["risky_used"]:
        score -= 0.15

    return _clamp01(score), {
        "correct_fix": correct_fix,
        "mitigation_before_fix": mitigation_before_fix,
        "revenue_control": revenue_control,
        "belief_alignment": belief,
        "anti_gaming": anti_gaming,
    }


def _score_cascading_failure(state: Dict, worlds: Dict) -> Tuple[float, Dict[str, float]]:
    correct_fix = 1.0 if state["applied_fix"] == worlds[state["active_world"]]["fix"] else 0.0
    belief_updates = sum(1 for item in state["action_history"] if item["type"] == "update_belief")
    coordination = _clamp01(belief_updates / 2.0)
    evidence = _evidence_coverage(state)
    revenue_control = _revenue_control(state)
    anti_gaming = _anti_gaming(state)

    score = (
        0.37 * correct_fix
        + 0.25 * coordination
        + 0.20 * evidence
        + 0.13 * revenue_control
        + 0.05 * anti_gaming
    )
    if state["risky_used"]:
        score -= 0.10

    return _clamp01(score), {
        "correct_fix": correct_fix,
        "coordination": coordination,
        "evidence": evidence,
        "revenue_control": revenue_control,
        "anti_gaming": anti_gaming,
    }


def grade(state: Dict, worlds: Dict) -> Dict[str, object]:
    task_name = state["task_name"]
    if task_name == "false_alarm":
        score, components = _score_false_alarm(state)
    elif task_name == "ambiguous_root":
        score, components = _score_ambiguous_root(state, worlds)
    elif task_name == "revenue_tradeoff":
        score, components = _score_revenue_tradeoff(state, worlds)
    elif task_name == "cascading_failure":
        score, components = _score_cascading_failure(state, worlds)
    else:
        score, components = 0.0, {"unsupported_task": 0.0}

    return {
        "task": task_name,
        "score": _clamp01(score),
        "components": {key: round(value, 4) for key, value in components.items()},
    }