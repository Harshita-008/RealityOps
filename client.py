from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from models import Action, ResetResponse, StepResponse


class RealityOpsClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task: Optional[str] = None) -> ResetResponse:
        payload: Dict[str, Any] = {}
        if task:
            payload["task"] = task
        response = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        response.raise_for_status()
        return ResetResponse(**response.json())

    def step(self, action: Action) -> StepResponse:
        response = requests.post(f"{self.base_url}/step", json=action.dict(), timeout=30)
        response.raise_for_status()
        return StepResponse(**response.json())

    def state(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()
