import streamlit as st
from pinecone import Pinecone
from PyPDF2 import PdfReader
from openai import OpenAI
import io
import time
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import os

# Set LangChain environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "grdient_cyber_bot"

# Set page configuration
st.set_page_config(layout="wide", page_title="Gradient Cyber Bot", page_icon="🤖")

# Custom CSS (keeping your existing styles)
st.markdown(
    """
    <style>
    ... (your existing CSS styles)
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlit app title
st.title("🤖 Gradient Cyber Bot")

# Initialize Pinecone and OpenAI with hardcoded values
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
INDEX_NAME = "gradientcyber"

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Try to initialize the index, handle potential errors
try:
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.sidebar.error(f"Error connecting to index: {str(e)}")
    st.stop()

# Initialize LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

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
            st.sidebar.error(f"Error during upsert: {str(e)}")
            st.sidebar.error(f"Failed at chunk {i} for {pdf_name}")
            return False
        
        time.sleep(1)
    return True

def question_to_pinecone_query(question, top_k=10):
    # Convert question to embedding
    question_embedding = embeddings.embed_query(question)
    
    # Query Pinecone
    query_response = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Process and return results
    results = []
    for match in query_response.matches:
        results.append({
            'id': match.id,
            'score': match.score,
            'metadata': match.metadata
        })
    
    return results

def format_answer(answer, sources):
    prompt = f"""
    Format the following answer in an attractive and easy-to-read manner. 
    Use markdown formatting to highlight key points, create sections if applicable, 
    and ensure the information is presented clearly and engagingly.
    At the end, list the sources used.

    Answer: {answer}

    Sources: {', '.join(sources)}
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

def answer_question(question):
    try:
        # Get relevant documents from Pinecone
        relevant_docs = question_to_pinecone_query(question)
        
        # Prepare context from relevant documents
        context = "\n\n".join([doc['metadata']['text'] for doc in relevant_docs])
        
        # Prepare prompt for GPT-4o
        prompt = f"""
        Based on the following context, answer the question: {question}

        Context:
        {context}

        Please provide a comprehensive and accurate answer. If the information is not available in the context, please say so.
        """
        
        # Get answer from GPT-4o
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = response.choices[0].message.content
        
        # Get sources
        sources = list(set([doc['metadata']['source'] for doc in relevant_docs]))
        
        # Format the answer
        formatted_answer = format_answer(answer, sources)
        
        return formatted_answer
    except Exception as e:
        return f"Error: {str(e)}"

def format_conversation_history(history):
    prompt = f"""
    Summarize and format the following conversation history in an engaging and easy-to-read manner.
    Highlight key points, questions asked, and main takeaways from the answers.
    Use markdown formatting to make it visually appealing.

    Conversation History:
    {history}
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# Sidebar for file upload and buttons
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) uploaded successfully!")
        
        if st.button("Process and Upsert to Pinecone"):
            overall_progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"Processing {uploaded_file.name}", expanded=True):
                    st.write(f"Extracting text from {uploaded_file.name}...")
                    pdf_text = extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
                    
                    st.write("Creating chunks...")
                    chunks = create_chunks(pdf_text)
                    
                    st.write(f"Upserting {len(chunks)} chunks to Pinecone...")
                    success = upsert_to_pinecone(chunks, uploaded_file.name)
                    
                    if success:
                        st.success(f"Successfully processed {uploaded_file.name}")
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                
                overall_progress.progress((i + 1) / len(uploaded_files))
            
            st.success("Finished processing all PDF files!")
    
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

# Scroll to bottom button and JavaScript (keeping your existing code)
st.markdown(
    """
    <button id="scroll-to-bottom" onclick="scrollToBottom()">⬇</button>
    
    <script>
    ... (your existing JavaScript)
    </script>
    """,
    unsafe_allow_html=True
)

# Display chat messages
for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask a question about the uploaded documents:")

if question:
    # Add user message to chat history
    st.session_state['chat_history'].append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching for an answer..."):
            answer = answer_question(question)
        
        message_placeholder.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
        
        # Add assistant message to chat history
        st.session_state['chat_history'].append({"role": "assistant", "content": answer})
