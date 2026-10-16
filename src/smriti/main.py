from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import AwareDatetime, BaseModel, Field

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


turns = {}
next_id = 1


class TurnIn(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=4000)
    sent_at: AwareDatetime


@app.post("/v1/users/{user_id}/turns")
def create_turn(user_id: str, turn: TurnIn, response: Response):
    global next_id

    user_turns = turns.setdefault(user_id, {})

    if turn.external_id in user_turns:
        existing = user_turns[turn.external_id]

        if (
            existing["role"] == turn.role
            and existing["text"] == turn.text
            and existing["sent_at"] == turn.sent_at
        ):
            response.status_code = 200
            return existing

        raise HTTPException(
            status_code=409, detail="Conflicting turn with same external_id"
        )

    stored_turn = {
        "id": next_id,
        "external_id": turn.external_id,
        "role": turn.role,
        "text": turn.text,
        "sent_at": turn.sent_at,
    }

    user_turns[turn.external_id] = stored_turn
    next_id += 1

    response.status_code = 201
    return stored_turn


@app.get("/v1/users/{user_id}/turns")
def get_turns(user_id: str):
    user_turns = turns.get(user_id, {})

    return sorted(user_turns.values(), key=lambda turn: (turn["sent_at"], turn["id"]))
