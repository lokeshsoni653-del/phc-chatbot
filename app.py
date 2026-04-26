import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub

# --- 1. SETUP ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

@st.cache_resource
def load_ai_agent():
    # Load and split the local data (Using utf-8 for stability)
    loader = TextLoader("phc_bootcamp_data.txt", encoding="utf-8")
    docs = loader.load()
    
    # Optimized chunking to save tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    splits = text_splitter.split_documents(docs)
    
    # Using the latest stable embedding model
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    )
    
    # K=2 prevents sending too much data and triggering Quota errors
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    retriever_tool = create_retriever_tool(
        retriever,
        "phc_knowledge_base",
        "Use this for any questions about the Pakistan Hindu Council. Be concise."
    )

    # Limit search results to save tokens
    search_tool = TavilySearchResults(k=2)
    tools = [retriever_tool, search_tool]

    # STABLE 2026 MODEL: Bypasses the experimental "thought_signature" bug
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # max_iterations stops the bot from looping infinitely
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

agent_executor = load_ai_agent()

# --- 3. BUILD THE USER INTERFACE ---
st.title("🇵🇰 PHC Digital Assistant")
st.markdown("I can check official records or the web to assist you.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_input := st.chat_input("What is your question?"):
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show AI response
    with st.chat_message("assistant"):
        try:
            # The agent decides which tool to use automatically
            response = agent_executor.invoke({"input": user_input})
            answer = response["output"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            # If Google API fails (Quota, Key, Server down), show the exact reason
            st.error(f"Actual Error: {str(e)}")
