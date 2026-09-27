import streamlit as st

# Set up the webpage
st.set_page_config(page_title="PhysiMate", page_icon="⚛️")
st.title("⚛️ PhysiMate")
st.subheader("Your 12th CBSE Physics Assistant")

# A simple chat message to test the UI
with st.chat_message("assistant"):
    st.write("Hello! I am ready to help you with physics. What would you like to learn today?")
import streamlit as st

st.set_page_config(page_title="PhysiMate", page_icon="⚛️")
st.title("⚛️ PhysiMate")
st.subheader("Your 12th CBSE Physics Assistant")

# Initialize conversation history in session memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to help you with physics. What would you like to learn today?"}
    ]

# Render all past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Add the interactive text input box pinned to the bottom
user_query = st.chat_input("Ask a physics question...")

if user_query:
    # 1. Display and save user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # 2. Display placeholder response
    bot_reply = f"Thinking about: '{user_query}'..."
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration & Environment Setup
st.set_page_config(page_title="PhysiMate", page_icon="⚛️")
load_dotenv() 

st.title("⚛️ PhysiMate: 12th CBSE Physics")
st.subheader("Your AI-Powered Physics Assistant")

# 2. Build the RAG Knowledge Base
@st.cache_resource
def setup_rag():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore.as_retriever()

try:
    retriever = setup_rag()
except Exception as e:
    st.error("Error loading PDFs. Please make sure your 'data' folder exists and contains PDF files.")
    st.stop()

# 3. Define the CBSE Physics Tutor Persona
system_prompt = (
    "You are an expert CBSE Class 12 Physics tutor. "
    "Use the following retrieved context from the NCERT textbook to answer the student's question. "
    "If the answer is not in the context, say 'I cannot find this in the standard 12th Physics syllabus.' "
    "Explain concepts clearly, provide step-by-step mathematical derivations if asked, and use standard NCERT notation.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Connect to Gemini AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Manage Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to help you with physics. What would you like to learn today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a physics question (e.g., Explain Gauss's Law)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching textbook and thinking..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("An error occurred. Make sure your API key is correct in the .env file.")