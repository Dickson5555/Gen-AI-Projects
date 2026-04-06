import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not HF_TOKEN:
    st.error(" Hugging Face token not found")
    st.stop()

if not GROQ_API_KEY:
    st.error("Groq API key not found")
    st.stop()


os.environ["HF_TOKEN"] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)


st.title("Conversational RAG with CSV/Excel Upload")
st.write("Upload CSV or Excel files and chat with your data")

session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}


uploaded_files = st.file_uploader(
    "Upload CSV or Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:

    documents = []

    for file in uploaded_files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        for _, row in df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(Document(page_content=text))

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # Contextual retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and the latest user question, reformulate it into a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for answering questions using the data.\n"
         "Use the retrieved context to answer.\n"
         "If unknown, say you don't know.\n"
         "Max 3 sentences.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Memory
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    
    user_input = st.text_input("Ask a question about your data:")

    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        st.write("Assistant:", response["answer"])

else:
    st.info(" Upload at least one file to begin")