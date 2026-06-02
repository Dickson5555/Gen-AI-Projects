from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.websocket import router

app = FastAPI(
    title="AI Voice Interviewer"
)

app.include_router(router)

app.mount(
    "/",
    StaticFiles(directory="app/static", html=True),
    name="static"
)