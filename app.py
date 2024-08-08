import streamlit as st
from langchain.vectorstores import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Ensure all required API keys are present
if not all([OPENAI_API_KEY, PINECONE_API_KEY]):
    st.error("Missing one or more required API keys. Please check your .env file or Streamlit secrets.")
    st.stop()

# Initialize Pinecone
INDEX_NAME = "gradientcyber"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4",
    temperature=0
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
        .chat-message {
            padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
        }
        .chat-message.user {
            background-color: #2b313e
        }
        .chat-message.bot {
            background-color: #475063
        }
        .chat-message .avatar {
          width: 20%;
        }
        .chat-message .avatar img {
          max-width: 78px;
          max-height: 78px;
          border-radius: 50%;
          object-fit: cover;
        }
        .chat-message .message {
          width: 80%;
          padding: 0 1.5rem;
          color: #fff;
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
                try:
                    num_chunks = process_and_upsert_pdf(uploaded_file)
                    st.success(f"Processed and upserted {num_chunks} chunks to Pinecone.")
                except Exception as e:
                    st.error(f"An error occurred while processing the PDF: {str(e)}")
    
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
        for message in st.session_state.messages:
            st.write(f"{message['role'].capitalize()}: {message['content']}")

# Main content area
st.title("Gradient Cyber Q&A System")

# Chat interface
for message in st.session_state.messages:
    if message["role"] == "human":
        st.markdown(f'<div class="chat-message user"><div class="avatar">👤</div><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">{message["content"]}<br><small>{message.get("chunk_info", "")}</small></div></div>', unsafe_allow_html=True)

# User input
query = st.text_input("Ask a question about the uploaded documents:")
if query:
    st.session_state.messages.append({"role": "human", "content": query})
    
    try:
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
            memory=memory
        )
        result = qa_chain({"question": query, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
        full_response = result["answer"]
        
        # Get the metadata of the matched chunk
        matched_metadata = result['source_documents'][0].metadata
        chunk_info = f"Source: {matched_metadata['source']}, Chunk Index: {matched_metadata['chunk_index']}"

        st.session_state.messages.append({"role": "assistant", "content": full_response, "chunk_info": chunk_info})
        
        # Display the new message
        st.markdown(f'<div class="chat-message bot"><div class="avatar">🤖</div><div class="message">{full_response}<br><small>{chunk_info}</small></div></div>', unsafe_allow_html=True)
    except AttributeError as ae:
        st.error(f"AttributeError occurred: {str(ae)}")
    except KeyError as ke:
        st.error(f"KeyError occurred: {str(ke)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        st.error("If the error persists, please check your API keys and ensure all required libraries are up to date.")

# Rerun the app to update the chat interface
st.experimental_rerun()
