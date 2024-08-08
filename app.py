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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "gradientcyber"
    index = pc.Index(index_name)
    logger.info("Successfully connected to Pinecone")
except Exception as e:
    logger.error(f"Error connecting to Pinecone: {str(e)}")
    st.error(f"Error connecting to Pinecone: {str(e)}")
    st.stop()

# Initialize LangChain components
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index_name=index_name, embedding=embeddings)
    logger.info("Successfully initialized LangChain components")
except Exception as e:
    logger.error(f"Error initializing LangChain components: {str(e)}")
    st.error(f"Error initializing LangChain components: {str(e)}")
    st.stop()

# Initialize LangSmith client
try:
    client = Client(api_key=LANGCHAIN_API_KEY)
    tracer = LangChainTracer(project_name="gradient_cyber_customer_bot", client=client)
    callback_manager = CallbackManager([tracer])
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
        temperature=0,
        callback_manager=callback_manager
    )
    logger.info("Successfully initialized LangSmith client and LLM")
except Exception as e:
    logger.error(f"Error initializing LangSmith client or LLM: {str(e)}")
    st.error(f"Error initializing LangSmith client or LLM: {str(e)}")
    st.stop()

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def process_and_upsert_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    if not text:
        return 0
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        doc_id = str(uuid.uuid4())
        metadatas = [{"source": pdf_file.name, "doc_id": doc_id, "chunk_id": f"chunk_{i}"} for i, _ in enumerate(chunks)]
        
        vectorstore.add_texts(chunks, metadatas=metadatas)
        
        if "doc_ids" not in st.session_state:
            st.session_state.doc_ids = []
        st.session_state.doc_ids.append(doc_id)
        
        return len(chunks)
    except Exception as e:
        logger.error(f"Error processing and upserting PDF: {str(e)}")
        st.error(f"Error processing and upserting PDF: {str(e)}")
        return 0

def get_relevant_documents(query, retriever):
    try:
        docs = retriever.get_relevant_documents(query)
        return docs
    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {str(e)}")
        st.error(f"Error retrieving relevant documents: {str(e)}")
        return []

def generate_response(query, relevant_docs):
    if not relevant_docs:
        return "I don't have any relevant information in my database to answer this question. Please try asking about topics related to the documents you've uploaded.", None
    
    try:
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
        Based solely on the following context, answer the question. If the answer cannot be found in the context, explain why you can't answer the question based on the available information.

        Context: {context}

        Question: {query}

        Answer:
        """
        
        response = llm.predict(prompt)
        source = relevant_docs[0].metadata if relevant_docs else None
        return response, source
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"An error occurred while generating the response: {str(e)}", None

# Streamlit UI
st.title("Gradient Cyber Q&A System")

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
                if num_chunks > 0:
                    st.success(f"Processed and upserted {num_chunks} chunks to Pinecone.")
                else:
                    st.error("Failed to process and upsert PDF.")
    
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
        response, source = generate_response(query, relevant_docs)
        
        message_placeholder.markdown(response)
        if source:
            st.info(f"Source: {source['source']}, Chunk ID: {source['chunk_id']}")
    
    st.session_state.messages.append({"content": query, "response": response, "source": source})

st.write("Note: Make sure you have set up your Pinecone index and OpenAI API key correctly.")
