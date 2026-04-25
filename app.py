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

# API keys are pulled from Streamlit Secrets

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]



# --- 2. INITIALIZE AI AGENT ---

@st.cache_resource

def load_ai_agent():

    # Load and split the local data

    loader = TextLoader("phc_bootcamp_data.txt")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    splits = text_splitter.split_documents(docs)

    

    # Create the vectorstore

    vectorstore = Chroma.from_documents(

        documents=splits, 

        embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    )

    # Only send the top 3 most relevant pieces of information instead of 10
   retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    

    # Define Tool 1: The local PHC Knowledge Base

    retriever_tool = create_retriever_tool(

        retriever,

        "phc_knowledge_base",

        "Use this for any questions about the Pakistan Hindu Council or the Youth Digital Empowerment Bootcamp. This is your primary source."

    )



    # Define Tool 2: Web Search

    search_tool = TavilySearchResults(k=3)

    

    tools = [retriever_tool, search_tool]



    # Initialize the LLM (Gemini 2.5 Flash)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    

    # Get a standard prompt template for tool-calling agents

    prompt = hub.pull("hwchase17/openai-functions-agent")

    

    # Construct the Agent

    agent = create_tool_calling_agent(llm, tools, prompt)

    

    # Create the Executor

    return AgentExecutor(agent=agent, tools=tools, verbose=True)



agent_executor = load_ai_agent()



# --- 3. BUILD THE USER INTERFACE ---

st.title("🇵🇰 PHC Digital Assistant (Hybrid Search)")

st.markdown("I can now check official records or the web to assist you better.")



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

        # The agent decides which tool to use automatically

        response = agent_executor.invoke({"input": user_input})

        st.markdown(response["output"])

    st.session_state.messages.append({"role": "assistant", "content": response["output"]})

    st.session_state.messages.append({"role": "assistant", "content": response})
