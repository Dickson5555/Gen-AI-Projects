from pydantic import BaseModel


class InterviewResponse(BaseModel):
    reaction: str
    feedback: str
    score: int
    next_question: str