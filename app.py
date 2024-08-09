import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from pinecone import Pinecone
from PyPDF2 import PdfReader
import os
from langsmith import Client
from dotenv import load_dotenv
import uuid
import openai

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
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index_name=index_name, embedding=embeddings)
except Exception as e:
    st.error(f"Error initializing LangChain components: {str(e)}")
    st.stop()

# Initialize LangSmith client
client = Client(api_key=LANGCHAIN_API_KEY)

# Initialize LangChain tracer and callback manager
tracer = LangChainTracer(project_name="gradient_cyber_customer_bot", client=client)
callback_manager = CallbackManager([tracer])

try:
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
        temperature=0,
        callback_manager=callback_manager
    )
except Exception as e:
    st.error(f"Error initializing ChatOpenAI: {str(e)}")
    st.stop()

# ... [rest of the code remains the same until the query processing part]

if query:
    st.session_state.messages.append({"role": "human", "content": query})
    with st.chat_message("human"):
        st.markdown(query)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
            if "doc_ids" in st.session_state and st.session_state.doc_ids:
                retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "filter": {"doc_id": {"$in": st.session_state.doc_ids}},
                        "k": 50
                    }
                )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                callback_manager=callback_manager,
                return_source_documents=True,
                verbose=True
            )
            result = qa_chain({"question": query, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
            full_response = result["answer"]
            
            # Add information about source documents
            source_docs = result['source_documents']
            if source_docs:
                full_response += "\n\nSources:\n"
                for i, doc in enumerate(source_docs):
                    full_response += f"{i+1}. {doc.metadata.get('source', 'Unknown source')}\n"
                    full_response += f"   Content: {doc.page_content[:200]}...\n\n"  # Display a snippet of the content
        except openai.BadRequestError as e:
            full_response = f"Error: {str(e)}. Please try rephrasing your question or check your API key."
        except Exception as e:
            full_response = f"An error occurred: {str(e)}"
        
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
