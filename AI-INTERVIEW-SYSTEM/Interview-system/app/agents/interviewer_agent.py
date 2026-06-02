import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.prompts.interview_prompt import (
    SYSTEM_PROMPT
)

from app.schemas.response_schema import (
    InterviewResponse
)

load_dotenv()

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.7
)

prompt = ChatPromptTemplate.from_template(
    SYSTEM_PROMPT
)

chain = prompt | llm


def process_interview(
    answer,
    history
):

    response = chain.invoke({
        "history": history,
        "answer": answer
    })

    raw = response.content

    try:

        parsed = json.loads(raw)

        validated = (
            InterviewResponse(**parsed)
        )

        return validated.dict()

    except Exception:

        return {
            "reaction":
            "I couldn't understand fully.",

            "feedback":
            "Please explain again.",

            "score": 0,

            "next_question":
            "Can you explain machine learning?"
        }


def start_interview():

    return {
        "reaction":
        "Welcome to the interview.",

        "feedback":
        "Relax and take your time.",

        "score": 0,

        "next_question":
        "Tell me the difference between supervised and unsupervised learning."
    }