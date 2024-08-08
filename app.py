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

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "gradient_cyber_customer_bot"

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Ensure all required API keys are present
if not all([OPENAI_API_KEY, PINECONE_API_KEY, LANGCHAIN_API_KEY]):
    st.error("Missing one or more required API keys. Please check your .env file or Streamlit secrets.")
    st.stop()

# Initialize Pinecone and OpenAI
INDEX_NAME = "gradientcyber"

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "gradientcyber"

# Initialize LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index_name=index_name, embedding=embeddings)

# Initialize LangSmith client
client = Client(api_key=LANGCHAIN_API_KEY)

# Initialize LangChain tracer and callback manager
tracer = LangChainTracer(project_name="gradient_cyber_customer_bot", client=client)
callback_manager = CallbackManager([tracer])
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4o",
    temperature=0,
    callback_manager=callback_manager
)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

def process_and_upsert_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    doc_id = str(uuid.uuid4())
    metadatas = [{"source": pdf_file.name, "doc_id": doc_id, "chunk_id": f"chunk_{i}"} for i, _ in enumerate(chunks)]
    
    vectorstore.add_texts(chunks, metadatas=metadatas)
    
    if "doc_ids" not in st.session_state:
        st.session_state.doc_ids = []
    st.session_state.doc_ids.append(doc_id)
    
    return len(chunks)

def get_relevant_documents(query, retriever):
    docs = retriever.get_relevant_documents(query)
    return docs

def generate_response(query, relevant_docs):
    if not relevant_docs:
        return None, None
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""
    Based solely on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

    Context: {context}

    Question: {query}

    Answer:
    """
    
    response = llm.predict(prompt)
    source = relevant_docs[0].metadata if relevant_docs else None
    return response, source

# Streamlit UI
st.title("Gradient Cyber")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False

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
    
    st.header("Controls")
    if st.button("Conversation History"):
        st.session_state.show_history = not st.session_state.show_history
    
    if st.button("Clear History"):
        st.session_state.messages = []
        if "doc_ids" in st.session_state:
            st.session_state.doc_ids = []
        st.success("Conversation history and document references cleared.")
    
    if st.session_state.show_history:
        st.subheader("Conversation History")
        history_text = "\n\n".join([f"Q: {msg['content']}\nA: {msg['response']}" for msg in st.session_state.messages])
        st.text_area("Full Conversation", value=history_text, height=300)

# Main content area
for message in st.session_state.messages:
    with st.chat_message("human"):
        st.markdown(message['content'])
    with st.chat_message("assistant"):
        st.markdown(message['response'])
        if message.get('source'):
            st.info(f"Source: {message['source']['source']}, Chunk ID: {message['source']['chunk_id']}")

query = st.chat_input("Ask a question about the uploaded documents:")
if query:
    with st.chat_message("human"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
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
            message_placeholder.warning(response)
            source = None
        else:
            response, source = generate_response(query, relevant_docs)
            message_placeholder.markdown(response)
            if source:
                st.info(f"Source: {source['source']}, Chunk ID: {source['chunk_id']}")
    
    st.session_state.messages.append({"content": query, "response": response, "source": source})

st.write("Note: Make sure you have set up your Pinecone index and OpenAI API key correctly.")
