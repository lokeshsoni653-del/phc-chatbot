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
    # Optimization: Use utf-8 encoding to support Sindhi/Urdu characters in your data
    loader = TextLoader("phc_bootcamp_data.txt", encoding="utf-8")
    docs = loader.load()
    
    # Optimization: Increase chunk size to 800 but keep overlap low. 
    # This gives the AI "bigger pictures" with fewer total API calls.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    
    # Optimization: Reduce k to 2. This sends enough info to answer but saves 33% of your token quota.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    retriever_tool = create_retriever_tool(
        retriever,
        "phc_knowledge_base",
        "Use this for any questions about the Pakistan Hindu Council. Be concise."
    )

    # Optimization: Reduce search results to 2 to prevent "ResourceExhausted" during web lookups.
    search_tool = TavilySearchResults(k=2)
    tools = [retriever_tool, search_tool]

    # Note: Using gemini-1.5-flash (the most stable high-speed model in 2026)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Optimization: Added 'max_iterations'. This prevents the agent from getting stuck 
    # in a loop and draining your API quota.
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

agent_executor = load_ai_agent()

# --- 3. BUILD THE USER INTERFACE ---
st.title("🇵🇰 PHC Digital Assistant")
st.markdown("I can check PHC records or the web to assist you.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is your question?"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        try:
            response = agent_executor.invoke({"input": user_input})
            answer = response["output"]
            st.markdown(answer)
            # Fix: Only store the text 'answer', not the whole response object
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error("Google's servers are a bit busy. Please wait 30 seconds and try again.")
