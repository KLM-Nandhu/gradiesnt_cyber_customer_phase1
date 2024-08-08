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
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv()

# Set environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "gradient_cyber_customer_bot")

# Initialize Pinecone and OpenAI
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("Missing API keys. Please set PINECONE_API_KEY and OPENAI_API_KEY in your .env file.")
    st.stop()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "gradientcyber"

def ensure_pinecone_index(index_name, dimension=1536):
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine'
        )
        print(f"Created new Pinecone index: {index_name}")
    else:
        print(f"Pinecone index {index_name} already exists")

# Ensure Pinecone index exists
ensure_pinecone_index(index_name)

# Initialize LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index_name=index_name, embedding=embeddings)

# Initialize LangSmith client
client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

# Initialize LangChain tracer and callback manager
tracer = LangChainTracer(project_name="gradient_cyber_customer_bot", client=client)
callback_manager = CallbackManager([tracer])

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4",
    temperature=0,
    callback_manager=callback_manager
)

@st.cache_data
def get_embedding(_openai_api_key, text):
    embeddings = OpenAIEmbeddings(openai_api_key=_openai_api_key)
    return embeddings.embed_query(text)

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
    
    vectorstore.add_texts(chunks, metadatas=[{"source": pdf_file.name} for _ in chunks])
    return len(chunks)

# Streamlit UI
st.sidebar.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #008080;
    }
    </style>
    <p class="big-font">Gradient Cyber</p>
    """, unsafe_allow_html=True)

st.sidebar.title("PDF Uploader")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.sidebar.write("File uploaded successfully!")
    
    if st.sidebar.button("Process and Upsert to Pinecone"):
        with st.sidebar.spinner("Processing PDF and upserting to Pinecone..."):
            try:
                num_chunks = process_and_upsert_pdf(uploaded_file)
                st.sidebar.success(f"Processed and upserted {num_chunks} chunks to Pinecone.")
            except Exception as e:
                st.sidebar.error(f"An error occurred while processing the PDF: {str(e)}")

st.title("Gradient Cyber Q&A System")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Q&A section
query = st.chat_input("Ask a question about the uploaded documents:")

if query:
    st.session_state.messages.append({"role": "human", "content": query})
    with st.chat_message("human"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Create a ConversationalRetrievalChain
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                callback_manager=callback_manager
            )

            # Generate the response
            result = qa_chain({"question": query, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
            full_response = result["answer"]

            message_placeholder.markdown(full_response)
        except Exception as e:
            error_message = f"An error occurred while generating the response: {str(e)}"
            message_placeholder.error(error_message)
            full_response = error_message
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.title("Conversation History")
if st.sidebar.button("Clear History"):
    st.session_state.messages = []

st.sidebar.markdown("---")
st.sidebar.markdown("v1.0.0 | [GitHub](https://github.com/yourusername/gradient-cyber-qa)")

st.markdown("---")
st.markdown("Need help? Check out the [documentation](https://your-docs-link.com) or [report an issue](https://github.com/yourusername/gradient-cyber-qa/issues).")

st.write("Note: Make sure you have set up your Pinecone index and OpenAI API key correctly.")
