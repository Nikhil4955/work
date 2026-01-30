from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader

# ------------------ LLM ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ------------------ UI ------------------
st.title("AskBuddy AI Chatbot")
st.markdown("Ask questions from uploaded data or general knowledge")

# ------------------ Session State ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader(
    "Upload a document (PDF / TXT)",
    type=["pdf", "txt"]
)

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")

    st.session_state.uploaded_text = text
    st.success("File uploaded and processed successfully!")

# ------------------ Show Chat History ------------------
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# ------------------ Chat Input ------------------
query = st.chat_input("Ask your question...")

if query:
    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    st.chat_message("user").markdown(query)

    # ------------------ Build Context ------------------
    history = ""
    for msg in st.session_state.messages:
        history += f"{msg['role'].capitalize()}: {msg['content']}\n"

    context = f"""
You are an AI assistant.
Answer the question based on the uploaded document and chat history.

--- Uploaded Document ---
{st.session_state.uploaded_text}

--- Chat History ---
{history}

--- Question ---
{query}
"""

    # ------------------ LLM Call ------------------
    response = llm.invoke(context)

    # Save AI response
    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )

    st.chat_message("assistant").markdown(response.content)
