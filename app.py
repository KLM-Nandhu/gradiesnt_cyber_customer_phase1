import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from pinecone import Pinecone
from PyPDF2 import PdfReader
import os
from langsmith import Client
from dotenv import load_dotenv
import uuid

# [Previous code remains unchanged]

# Streamlit UI
st.title("Gradient Cyber Q&A System")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# Main content area
main_content = st.container()

# Sidebar
with st.sidebar:
    st.title("Gradient Cyber")
    
    st.header("PDF Uploader")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        st.write("File uploaded successfully!")
        if st.button("Process and Upsert to Pinecone"):
            with st.spinner("Processing PDF and upserting to Pinecone..."):
                num_chunks = process_and_upsert_pdf(uploaded_file)
                st.success(f"Processed and upserted {num_chunks} chunks to Pinecone.")
    
    st.header("Conversation History")
    if st.button("Toggle Conversation History"):
        st.session_state.show_history = not st.session_state.show_history
    
    if st.button("Clear History"):
        st.session_state.messages = []
        if "doc_ids" in st.session_state:
            st.session_state.doc_ids = []
        st.success("Conversation history and document references cleared.")
    
    if st.session_state.show_history:
        for message in st.session_state.messages:
            st.write(f"**{message['role'].capitalize()}:** {message['content']}")

# Display conversation in main content area
with main_content:
    for message in st.session_state.messages:
        st.write(f"**{message['role'].capitalize()}:** {message['content']}")

# Question input at the bottom
st.write("---")  # Separator
query = st.text_input("Ask a question about the uploaded documents:")
if st.button("Send"):
    if query:
        st.session_state.messages.append({"role": "human", "content": query})
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Adjust k as needed
        
        if "doc_ids" in st.session_state and st.session_state.doc_ids:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "filter": {"doc_id": {"$in": st.session_state.doc_ids}},
                    "k": 3
                }
            )
        
        relevant_docs = get_relevant_documents(query, retriever)
        
        if not relevant_docs:
            response = "I'm sorry, but I don't have enough information in the database to answer that question."
            st.warning(response)
        else:
            response = generate_response(query, relevant_docs)
            st.write(f"**Assistant:** {response}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()  # Rerun the app to update the main content area

st.write("Note: Make sure you have set up your Pinecone index and OpenAI API key correctly.")
