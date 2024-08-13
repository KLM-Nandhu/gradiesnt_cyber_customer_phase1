import streamlit as st
import io
import time
import os
from dotenv import load_dotenv

# Pinecone import for serverless database
import pinecone  # Note: Using serverless Pinecone database

from PyPDF2 import PdfReader
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Set LangChain environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "gradient_cyber_bot"

# Set page configuration
st.set_page_config(layout="wide", page_title="Gradient Cyber Bot", page_icon="🤖")

# Custom CSS for improved UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f4f8;
        color: #1e1e1e;
    }
    .reportview-container {
        background-color: #f0f4f8;
    }
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 6rem;
        margin: auto;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stChatMessage:hover {
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stChatMessage.user {
        background-color: #e6f3ff;
        border-left: 5px solid #2196F3;
    }
    .stChatMessage .content p {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    .stTextInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    .stTextInput > div {
        display: flex;
        justify-content: space-between;
        max-width: 900px;
        margin: auto;
    }
    .stTextInput input {
        flex-grow: 1;
        margin-right: 1rem;
        border-radius: 25px;
        border: 2px solid #2196F3;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stTextInput input:focus {
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.3);
        outline: none;
    }
    .stButton button {
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .answer-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    .answer-card h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .source-list {
        margin-top: 1rem;
        padding-left: 1.5rem;
    }
    .source-list li {
        margin-bottom: 0.5rem;
        color: #546E7A;
    }
    #scroll-to-bottom {
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 50px;
        height: 50px;
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 50%;
        font-size: 24px;
        display: none;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        z-index: 9999;
    }
    #scroll-to-bottom:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlit app title
st.title("🤖 Gradient Cyber Bot (Serverless Pinecone Edition)")

# Initialize Pinecone and OpenAI with environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = "gradientcyber"

# Initialize clients
try:
    # Explicitly mention serverless specification
    pinecone.init(api_key=PINECONE_API_KEY)
    st.write("Initializing serverless Pinecone database...")
    
    # Initialize Pinecone index with serverless configuration
    index = pinecone.Index(
        PINECONE_INDEX_NAME,
        host="https://gradientcyber-z1co3qo.svc.aped-4627-b74a.pinecone.io"
    )
    st.write(f"Successfully connected to serverless Pinecone index: {PINECONE_INDEX_NAME}")
    
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Error initializing serverless Pinecone client: {str(e)}")
    st.stop()

# Initialize LangChain components
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 30}),
        return_source_documents=True,
    )
except Exception as e:
    st.error(f"Error initializing LangChain components: {str(e)}")
    st.stop()

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def create_chunks(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def upsert_to_pinecone(chunks, pdf_name):
    batch_size = 50
    total_chunks = len(chunks)
    progress_bar = st.sidebar.progress(0)
    chunk_counter = 0
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        ids = [f"{pdf_name}_{j}" for j in range(i, i+len(batch))]
        
        try:
            embeddings_batch = embeddings.embed_documents(batch)
            
            to_upsert = [
                (id, embedding, {"text": chunk, "source": pdf_name})
                for id, embedding, chunk in zip(ids, embeddings_batch, batch)
            ]
            
            index.upsert(vectors=to_upsert)
            
            chunk_counter += len(batch)
            progress = min(1.0, chunk_counter / total_chunks)
            progress_bar.progress(progress)
            
            st.sidebar.text(f"Processed {chunk_counter}/{total_chunks} chunks for {pdf_name}")
            
        except Exception as e:
            st.sidebar.error(f"Error during upsert to serverless Pinecone: {str(e)}")
            st.sidebar.error(f"Failed at chunk {i} for {pdf_name}")
            return False
        
        time.sleep(1)
    return True

def format_answer(answer, sources):
    prompt = f"""
    Format the following answer in an attractive and easy-to-read manner. 
    Use markdown formatting to highlight key points, create sections if applicable, 
    and ensure the information is presented clearly and engagingly.
    At the end, list the sources used.

    Answer: {answer}

    Sources: {', '.join(sources)}
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error formatting answer: {str(e)}")
        return answer

def answer_question(question):
    try:
        st_callback = StreamlitCallbackHandler(st.container())
        result = qa_chain({"query": question}, callbacks=[st_callback])
        answer = result['result']
        sources = list(set([doc.metadata['source'] for doc in result['source_documents']]))
        
        # Format the answer
        formatted_answer = format_answer(answer, sources)
        
        return formatted_answer
    except Exception as e:
        st.error(f"Error in answer_question: {str(e)}")
        return f"I encountered an error while trying to answer your question. Error: {str(e)}"

def format_conversation_history(history):
    prompt = f"""
    Summarize and format the following conversation history in an engaging and easy-to-read manner.
    Highlight key points, questions asked, and main takeaways from the answers.
    Use markdown formatting to make it visually appealing.

    Conversation History:
    {history}
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error formatting conversation history: {str(e)}")
        return history

# Sidebar for file upload and buttons
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) uploaded successfully!")
        
        if st.button("Process and Upsert to Serverless Pinecone"):
            overall_progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"Processing {uploaded_file.name}", expanded=True):
                    st.write(f"Extracting text from {uploaded_file.name}...")
                    pdf_text = extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
                    
                    if pdf_text:
                        st.write("Creating chunks...")
                        chunks = create_chunks(pdf_text)
                        
                        st.write(f"Upserting {len(chunks)} chunks to serverless Pinecone...")
                        success = upsert_to_pinecone(chunks, uploaded_file.name)
                        
                        if success:
                            st.success(f"Successfully processed {uploaded_file.name}")
                        else:
                            st.error(f"Failed to process {uploaded_file.name}")
                    else:
                        st.error(f"Failed to extract text from {uploaded_file.name}")
                
                overall_progress.progress((i + 1) / len(uploaded_files))
            
            st.success("Finished processing all PDF files to serverless Pinecone!")
    
    st.header("Chat Options")
    if st.button("View Conversation History"):
        if st.session_state['chat_history']:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['chat_history']])
            formatted_history = format_conversation_history(history_text)
            st.markdown(formatted_history)
        else:
            st.write("No conversation history yet.")
    
    if st.button("Reset Chat"):
        st.session_state['chat_history'] = []
        st.rerun()

# Scroll to bottom button and JavaScript
st.markdown(
    """
    <button id="scroll-to-bottom" onclick="scrollToBottom()">⬇</button>
    
    <script>
    function scrollToBottom() {
        window.scrollTo(0, document.body.scrollHeight);
    }

    function toggleScrollButton() {
        var scrollButton = document.getElementById('scroll-to-bottom');
        if ((window.innerHeight + window.pageYOffset) < document.body.offsetHeight - 100) {
            scrollButton.style.display = 'flex';
        } else {
            scrollButton.style.display = 'none';
        }
    }

    // Initial call to set button visibility
    toggleScrollButton();

    // Add scroll event listener
    window.addEventListener('scroll', toggleScrollButton);
// Add resize event listener to handle window size changes
    window.addEventListener('resize', toggleScrollButton);

    // MutationObserver to watch for changes in the DOM
    var observer = new MutationObserver(function(mutations) {
        toggleScrollButton();
    });

    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """,
    unsafe_allow_html=True
)

# Display chat messages
for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask a question about the documents in the serverless Pinecone database:")

if question:
    # Add user message to chat history
    st.session_state['chat_history'].append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching serverless Pinecone database for an answer..."):
            answer = answer_question(question)
        
        message_placeholder.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
        
        # Add assistant message to chat history
        st.session_state['chat_history'].append({"role": "assistant", "content": answer})

# Error handling and logging
if st.session_state.get("show_error", False):
    st.error(st.session_state.error_message)
    st.session_state.show_error = False

# You can add any additional error handling or logging here if needed

if __name__ == "__main__":
    st.write("Gradient Cyber Bot is ready to assist you with the serverless Pinecone database!")
