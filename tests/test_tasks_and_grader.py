from env.core import RealityOpsEnv
from env.grader import grade
from env.models import Action
from env.tasks import TASK_SPECS, default_beliefs, normalize_beliefs, task_names
from env.worlds import WORLDS


def test_task_catalog_has_four_or_more_tasks() -> None:
    names = task_names()
    assert len(names) >= 4
    assert set(names) == set(TASK_SPECS.keys())


def test_task_specs_have_required_keys() -> None:
    required = {
        "difficulty",
        "description",
        "alert",
        "candidate_worlds",
        "ground_truth_world",
        "max_steps",
        "revenue_loss_per_step",
        "required_evidence",
        "required_belief_updates",
        "require_mitigation_before_fix",
    }
    for task, spec in TASK_SPECS.items():
        assert required.issubset(set(spec.keys())), task


def test_default_beliefs_are_normalized() -> None:
    for task in task_names():
        beliefs = default_beliefs(task)
        total = sum(beliefs.values())
        assert abs(total - 1.0) < 1e-9
        assert all(value >= 0.0 for value in beliefs.values())


def test_normalize_beliefs_falls_back_when_all_zero() -> None:
    allowed = {"db_overload": 0.7, "cache_bug": 0.3}
    normalized = normalize_beliefs({"db_overload": 0.0, "cache_bug": 0.0}, allowed)

    assert abs(sum(normalized.values()) - 1.0) < 1e-9
    assert normalized["db_overload"] > normalized["cache_bug"]


def test_normalize_beliefs_ignores_unknown_and_clamps_negative() -> None:
    allowed = {"db_overload": 0.5, "cache_bug": 0.5}
    normalized = normalize_beliefs(
        {"db_overload": -2.0, "cache_bug": 1.0, "unknown_world": 999.0},
        allowed,
    )
    assert set(normalized.keys()) == set(allowed.keys())
    assert normalized["db_overload"] == 0.0
    assert normalized["cache_bug"] == 1.0


def test_each_world_has_fix_and_metrics() -> None:
    required_fixes = {
        "increase_pool",
        "flush_cache",
        "refresh_token",
        "reroute_traffic",
        "block_ip",
        "scale_up",
        "no_fix",
    }
    for world_name, spec in WORLDS.items():
        assert "logs" in spec and len(spec["logs"]) >= 1, world_name
        assert "metrics" in spec and isinstance(spec["metrics"], dict), world_name
        assert spec["fix"] in required_fixes, world_name


def test_grade_output_stays_in_range_for_all_tasks() -> None:
    env = RealityOpsEnv()

    for task in task_names():
        env.reset(task_name=task)
        score = grade(env.state, WORLDS)
        assert score["task"] == task
        assert 0.0 <= score["score"] <= 1.0
        assert isinstance(score["components"], dict)
        assert len(score["components"]) > 0


def test_anti_gaming_component_drops_after_action_spam() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="ambiguous_root")

    baseline = grade(env.state, WORLDS)["components"]["anti_gaming"]
    for _ in range(5):
        env.step(Action(type="wait"))

    noisy = grade(env.state, WORLDS)["components"]["anti_gaming"]
    assert noisy < baseline


def test_grade_is_deterministic_for_same_state_snapshot() -> None:
    env = RealityOpsEnv()
    env.reset(task_name="revenue_tradeoff")

    first = grade(env.state, WORLDS)
    second = grade(env.state, WORLDS)
    assert first == second
