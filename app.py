# csv_chatbot_streamlit_multi.py

import os
import time
import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
# Import both lLLms
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="CSV Chatbot", layout="centered")
st.title("📊 Chat with your CSV using LangChain")

# Step 1: Choose Provider
provider = st.sidebar.selectbox("🤖 Select Model Provider", ["DeepSeek", "Gemini"])

# Step 2: API Key
if provider == "DeepSeek":
    api_key = st.sidebar.text_input("🔑DEEKSEEK-API-KEY", type="password")
elif provider == "Gemini":
    api_key = st.sidebar.text_input("🔑GEMINI-API-KEY", type="password")
# Step 3: Upload CSV
uploaded_file = st.sidebar.file_uploader("📁 Upload a CSV file", type=["csv"])

# Step 4: Proceed if keys and file are present
if api_key and uploaded_file:
    with open("temp_uploaded.csv", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Step 5: Initialize LLM
    if provider == "DeepSeek":
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model="deepseek-chat",
            temperature=0,
            openai_api_base="https://api.deepseek.com/v1"  # DeepSeek API endpoint
        )
    else:
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Step 6: Create Agent
    agent = create_csv_agent(
        llm,
        "temp_uploaded.csv",
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True  
    )

    # Step 7: Query input
    query = st.text_input("💬 Ask something about your CSV:")

    if query:
        with st.spinner("🧠 Thinking..."):
            start = time.time()
            answer = agent.run(query)
            end = time.time()
            st.success("✅ Answer:")
            st.write(answer)
            st.caption(f"⏱️ Took {round(end - start, 2)} seconds")

else:
    st.info("Please enter your API key and upload a CSV file to begin.")
