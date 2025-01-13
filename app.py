import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import re
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityAdvisor:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY in your .env file.")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
    
    def process_query(self, query: str):
        """Extract name and clean query from timestamp format"""
        pattern = r'^([^,]+),\s*(?:[^,]+,\s*\d+\s+\w+\s+\d+\s+[\d:]+\s+\w+)\s*\n(.+)$'
        match = re.match(pattern, query.strip(), re.DOTALL)
        
        if match:
            name, content = match.groups()
            return name.strip(), content.strip()
        return None, query.strip()
    
    def generate_response(self, sitrep: str, query: str):
        """Generate response based on sitrep analysis"""
        name, cleaned_query = self.process_query(query)
        greeting = f"Hey {name}" if name else "Hey"

        if not cleaned_query or cleaned_query.lower().startswith(('thank', 'ok', 'got it')):
            return f"{greeting}, thank you for your message. - Gradient Cyber Team!"

        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an experienced cyber security analyst handling the role from a Security operations center perspective. 
                When I provide a message, it contains the summary of a "sitrep" which is a situational report of a particular 
                security incident or event. The goal is to first analyze this sitrep. It will be followed always with a 
                "query" from a user. Your goal will be to understand the sitrep and then focus on answering the query based 
                on your role as an experience cyber security analyst. The concept is to ensure that the response is brief as it 
                primarily is provided as part of a web interface or email.
                Always start with "{greeting}" and end with "We hope this answers your question. Thank you! Gradient Cyber Team!"
                """
            ),
            HumanMessagePromptTemplate.from_template(
                """Sitrep: {sitrep}
                Query: {query}"""
            )
        ])

        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        return chain.run(greeting=greeting, sitrep=sitrep, query=cleaned_query)

def main():
    st.set_page_config(page_title="Security Advisor", layout="wide")
    st.title("Security Advisory System")
    
    try:
        advisor = SecurityAdvisor()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sitrep = st.text_area("Situation Report (Sitrep)", height=450)
        
        with col2:
            query = st.text_area("Query", 
                                placeholder="Example:\nRyan O'Neill, Mon, 06 Jan 2025 20:35:46 GMT\nWhat does this alert mean?",
                                height=200)
        
        if st.button("Generate Response", type="primary"):
            if not sitrep or not query:
                st.error("Please provide both sitrep and query.")
                return
                
            with st.spinner("Generating response..."):
                with get_openai_callback() as cb:
                    response = advisor.generate_response(sitrep, query)
                    st.markdown("### Response:")
                    st.markdown(response)
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
