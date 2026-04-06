from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

TaskName = Literal[
    "false_alarm",
    "ambiguous_root",
    "revenue_tradeoff",
    "cascading_failure",
]

class Observation(BaseModel):
    alerts: List[str]
    logs: List[str]
    metrics: Dict[str, float]
    slack: List[str]
    revenue_loss: float
    step: int

class Action(BaseModel):
    type: Literal[
        "probe",
        "check_logs",
        "check_metrics",
        "update_belief",
        "commit_fix",
        "safe_mitigation",
        "risky_hotfix",
        "wait"
    ]
    payload: Optional[Dict] = None


class Reward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task: Optional[TaskName] = None


class ResetResponse(BaseModel):
    observation: Observation
    done: bool
    task: TaskName


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]