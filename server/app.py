from typing import Optional

from fastapi import FastAPI
import uvicorn
from env.core import RealityOpsEnv
from env.models import Action, ResetRequest, ResetResponse, StepResponse

app = FastAPI()
env = RealityOpsEnv()

@app.post("/reset", response_model=ResetResponse)
def reset(payload: Optional[ResetRequest] = None):
    selected_task = payload.task if payload else None
    obs = env.reset(task_name=selected_task)
    return ResetResponse(observation=obs, done=False, task=env.state["task_name"])

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    result = env.step(action)
    return StepResponse(
        observation=result["observation"],
        reward=result["reward"],
        done=result["done"],
        info=result["info"],
    )

@app.get("/state")
def state():
    return env.state_view()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()