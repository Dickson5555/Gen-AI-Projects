import pandas as pd
from langchain_groq import ChatGroq
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name= "llama-3.1-8b-instant",
    #"llama-3.3-70b-versatile",
    #"llama-3.1-8b-instant",
    temperature=0
)

def create_agent(df):

    analyst_prompt = """
You are expert a Data Analyst,PowerBI Specialist,Machine Learning engineer.

- Use pandas to analyze the dataframe 'df'
- Give clear and short answers
- Suggest ML models only when asked
- Provide Python code when necessary
- Do not overthink
- Keep responses concise
"""

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=45,
        early_stopping_method="force",
        prefix=analyst_prompt
    )

    return agent


def ask_agent(agent, question):
    return agent.invoke({"input":question})