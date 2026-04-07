from env.core import RealityOpsEnv
from env.models import Action


def test_reset_defaults_to_supported_task() -> None:
    env = RealityOpsEnv()

    obs = env.reset(task_name="not_a_task")
    assert env.state["task_name"] == "ambiguous_root"
    assert obs.step == 0


def test_false_alarm_can_finish_without_fix() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="false_alarm")

    env.step(Action(type="check_metrics"))
    env.step(Action(type="check_logs"))
    result = env.step(Action(type="wait"))

    assert result["done"] is True
    assert result["info"]["task"] == "false_alarm"


def test_hard_task_needs_confidence_before_done() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="revenue_tradeoff")

    # Premature fix should not finish the episode and should require confirmation.
    premature = env.step(Action(type="commit_fix", payload={"fix": "reroute_traffic"}))
    assert premature["done"] is False
    assert premature["info"]["requires_fix_confirmation"] is True

    # Gather required confidence gates and confirm.
    env.step(Action(type="check_logs"))
    env.step(Action(type="check_metrics"))
    env.step(Action(type="probe"))
    env.step(
        Action(
            type="update_belief",
            payload={
                "network_partition": 0.6,
                "db_overload": 0.2,
                "cache_bug": 0.1,
                "auth_expiry": 0.05,
                "no_incident": 0.05,
            },
        )
    )
    env.step(Action(type="safe_mitigation"))
    final = env.step(Action(type="commit_fix", payload={"fix": "reroute_traffic"}))

    assert final["done"] is True


def test_step_after_done_returns_terminal_message() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="false_alarm")
    env.step(Action(type="check_metrics"))
    env.step(Action(type="check_logs"))
    env.step(Action(type="wait"))

    after_done = env.step(Action(type="probe"))
    assert after_done["done"] is True
    assert "already complete" in after_done["info"]["message"].lower()


def test_repetition_penalty_appears_for_repeated_actions() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="ambiguous_root")

    env.step(Action(type="check_logs"))
    repeated = env.step(Action(type="check_logs"))

    assert "repetition_penalty" in repeated["info"]["reward_components"]
    assert repeated["info"]["reward_components"]["repetition_penalty"] < 0


def test_invalid_fix_penalty_and_counter_are_recorded() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="ambiguous_root")

    result = env.step(Action(type="commit_fix", payload={"fix": "drop_database"}))

    assert result["info"]["reward_components"].get("invalid_fix_penalty", 0.0) < 0
    assert env.state["invalid_fix_count"] == 1


def test_risky_hotfix_sets_risk_and_confirmation_flags() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="revenue_tradeoff")

    result = env.step(Action(type="risky_hotfix", payload={"fix": "reroute_traffic"}))

    assert env.state["risky_used"] is True
    assert env.state["requires_fix_confirmation"] is True
    assert result["done"] is False


def test_cascading_failure_requires_two_belief_updates() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="cascading_failure")

    env.step(Action(type="check_logs"))
    env.step(Action(type="check_metrics"))
    env.step(Action(type="probe"))

    env.step(Action(type="update_belief", payload={"auth_expiry": 0.7, "network_partition": 0.2, "db_overload": 0.1}))
    first_fix_attempt = env.step(Action(type="commit_fix", payload={"fix": "refresh_token"}))
    assert first_fix_attempt["done"] is False
    assert first_fix_attempt["info"]["requires_fix_confirmation"] is True

    second_update = env.step(
        Action(type="update_belief", payload={"auth_expiry": 0.8, "network_partition": 0.1, "db_overload": 0.1})
    )
    assert second_update["done"] is True


def test_state_view_contains_structured_score_payload() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="false_alarm")
    view = env.state_view()

    assert "score" in view
    assert view["score"]["task"] == "false_alarm"
    assert isinstance(view["score"]["components"], dict)
