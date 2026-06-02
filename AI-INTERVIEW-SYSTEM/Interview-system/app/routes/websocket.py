from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect
)

import uuid

from app.agents.interviewer_agent import (
    process_interview,
    start_interview
)

from app.services.memory_service import (
    memory_store
)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    print("WebSocket Connected")

    session_id = str(uuid.uuid4())

    try:

        while True:

            data = (
                await websocket.receive_json()
            )

            action = data.get(
                "action"
            )

          

            if action == "start":

                result = (
                    start_interview()
                )

                memory_store.add_message(
                    session_id,
                    "AI",
                    result["next_question"]
                )

                await websocket.send_json(
                    result
                )

          

            elif action == "answer":

                user_answer = data.get(
                    "message",
                    ""
                )

                memory_store.add_message(
                    session_id,
                    "Candidate",
                    user_answer
                )

                history = "\n".join(
                    memory_store.get_history(
                        session_id
                    )
                )

                result = process_interview(
                    answer=user_answer,
                    history=history
                )

                memory_store.add_message(
                    session_id,
                    "AI",
                    result["next_question"]
                )

                await websocket.send_json(
                    result
                )

    except WebSocketDisconnect:

        print(
            f"Disconnected: {session_id}"
        )