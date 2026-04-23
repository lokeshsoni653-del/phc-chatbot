import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETUP ---
# Replace with your copied key!
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- 2. INITIALIZE AI (Cached so it only loads once) ---
@st.cache_resource
def load_ai_brain():
    loader = TextLoader("phc_bootcamp_data.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    system_prompt = (
        "You are the official digital assistant for the Pakistan Hindu Council. "
        "Use the provided context to answer the user's question accurately. "
        "If the answer is not in the context, politely state that you do not have that information "
        "and direct them to visit www.phc.org.pk. Maintain a polite, optimistic, and professional tone. "
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = load_ai_brain()

# --- 3. BUILD THE USER INTERFACE ---
st.title("🇵🇰 PHC Digital Assistant")
st.markdown("Welcome! Ask me anything about the Youth Digital Empowerment Bootcamp.")

# Create a memory to store the chat history on the screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input at the bottom of the screen
if prompt := st.chat_input("What is your question?"):
    
    # Show what the user typed
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the AI's response and show it
    with st.chat_message("assistant"):
        response = rag_chain.invoke(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})