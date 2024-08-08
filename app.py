import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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
    metadatas = [{"source": pdf_file.name, "doc_id": doc_id, "chunk_index": i} for i, _ in enumerate(chunks)]
    
    vectorstore.add_texts(chunks, metadatas=metadatas)
    
    if "doc_ids" not in st.session_state:
        st.session_state.doc_ids = []
    st.session_state.doc_ids.append(doc_id)
    
    return len(chunks)

# Custom CSS for chat layout
st.markdown("""
    <style>
        .chat-box {
            display: flex;
            align-items: flex-start;
            margin: 10px 0;
        }
        .chat-box.user {
            justify-content: flex-start;
        }
        .chat-box.assistant {
            justify-content: flex-end;
        }
        .chat-message {
            max-width: 70%;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
        }
        .chat-message.user {
            background-color: #e6f7ff;
            text-align: left;
        }
        .chat-message.assistant {
            background-color: #f0f0f0;
            text-align: right;
        }
        .title {
            color: #003366;
            text-align: center;
            margin-bottom: 20px;
        }
        .conversation-history {
            background-color: #f8f8f8;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

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
    if st.button("Toggle Conversation History"):
        st.session_state.show_history = not st.session_state.show_history
    
    if st.button("Clear History"):
        st.session_state.messages = []
        if "doc_ids" in st.session_state:
            st.session_state.doc_ids = []
        st.success("Conversation history and document references cleared.")
    
    if st.session_state.show_history:
        st.subheader("Conversation History")
        st.markdown('<div class="conversation-history">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")
        st.markdown('</div>', unsafe_allow_html=True)

# Main content area
st.markdown('<h1 class="title">Gradient Cyber Q&A System</h1>', unsafe_allow_html=True)

# Response Box (will be populated when there's a response)
response_container = st.container()

query = st.chat_input("Ask a question about the uploaded documents:")
if query:
    st.session_state.messages.append({"role": "human", "content": query})
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever()
    if "doc_ids" in st.session_state and st.session_state.doc_ids:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "filter": {"doc_id": {"$in": st.session_state.doc_ids}}
            }
        )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        callback_manager=callback_manager
    )
    result = qa_chain({"question": query, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
    full_response = result["answer"]
    
    # Get the metadata of the matched chunk
    matched_metadata = result['source_documents'][0].metadata
    chunk_info = f"Source: {matched_metadata['source']}, Chunk Index: {matched_metadata['chunk_index']}"

    st.session_state.messages.append({"role": "assistant", "content": full_response, "chunk_info": chunk_info})

    # Display response in the center box
    with response_container:
        for message in st.session_state.messages:
            if message["role"] == "human":
                st.markdown('<div class="chat-box user">', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message user">👤 {message["content"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="chat-box assistant">', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message assistant">🤖 {message["content"]}<br><small>{message["chunk_info"]}</small></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
