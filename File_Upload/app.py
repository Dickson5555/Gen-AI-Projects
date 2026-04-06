import pandas as pd
import streamlit as st
from agent import create_agent, ask_agent

st.set_page_config(
    page_title="Chat with Your Data",
    layout="wide"
)

st.title("📊 Chat with Your Data")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None


# File Upload

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file:
    try:
        # Detect file type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")

        # Show preview
        st.dataframe(df.head())

        # Create agent and store it
        st.session_state.agent = create_agent(df)

    except Exception as e:
        st.error(f"Error loading file: {e}")


# Sidebar: Chat History

with st.sidebar:
    st.header("💬 Chat History")

    if len(st.session_state.messages) == 0:
        st.caption("No conversations yet")
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**🧑 You:** {msg['content'][:60]}")

    st.divider()
    if st.button(" Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Chat Window

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat Input

user_input = st.chat_input("Ask a question about your data...")

if user_input:
    if st.session_state.agent is None:
        st.warning("Please upload a dataset first.")
    else:
        # Save user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response
        response = ask_agent(st.session_state.agent, user_input)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        with st.chat_message("assistant"):
            st.markdown(response)
