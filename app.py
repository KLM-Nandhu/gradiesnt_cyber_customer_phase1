import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import re
import logging
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityAdvisor:
    def __init__(self):
        # Get credentials from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.db_connection = os.getenv("NEON_DATABASE_URL")
        
        if not self.openai_api_key or not self.db_connection:
            raise ValueError("Missing required environment variables. Please check .env file.")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
        self.engine = create_engine(self.db_connection)

    def upload_excel_to_db(self, excel_file):
        """Upload Excel file data to Neon database"""
        try:
            # Read the Excel file
            df = pd.read_excel(excel_file)
            
            # Display Excel data information
            st.write("Excel Data Preview:")
            st.write(df.head())
            st.write("Columns in Excel:", df.columns.tolist())
            st.write("Number of rows:", len(df))
            
            # Clean column names - remove spaces and special characters
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
            
            # Create the table if it doesn't exist
            df.to_sql('clients', self.engine, if_exists='replace', index=False)
            
            # Verify the upload
            verify_query = "SELECT * FROM clients LIMIT 5"
            verify_df = pd.read_sql(verify_query, self.engine)
            st.write("Database Data Preview:")
            st.write(verify_df.head())
            
            return True, "Data uploaded successfully! Please check the preview above."
        except Exception as e:
            logger.error(f"Upload Error: {str(e)}")
            return False, f"Error uploading data: {str(e)}"

    def get_organization_summary(self, org_data: str) -> str:
        """Generate a brief summary of organization using LLM"""
        try:
            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """Given the following organization information, provide a brief 1-2 line summary focusing on their security profile and services:
                    {org_info}
                    Keep it concise and professional, focusing only on key security aspects."""
                )
            ])
            
            chain = LLMChain(llm=self.llm, prompt=summary_prompt)
            return chain.run(org_info=org_data)
        except Exception as e:
            logger.error(f"Summary Error: {str(e)}")
            return None

    def get_organization_info(self, org_name: str) -> tuple:
        """Fetch organization information from database and return both raw data and summary"""
        try:
            # Get all data from the table
            all_data_query = "SELECT * FROM clients"
            df = pd.read_sql(all_data_query, self.engine)
            
            if df.empty:
                return None, None
            
            # Search across all columns
            for column in df.columns:
                matches = df[df[column].astype(str).str.contains(org_name, case=False, na=False)]
                if not matches.empty:
                    # Get raw data and create formatted string
                    matched_row = matches.iloc[0]
                    raw_data = " | ".join([f"{col}: {matched_row[col]}" for col in matches.columns 
                                       if pd.notna(matched_row[col]) and str(matched_row[col]).strip()])
                    
                    # Generate LLM summary
                    summary = self.get_organization_summary(raw_data)
                    return raw_data, summary
            
            return None, None

        except Exception as e:
            logger.error(f"Search Error: {str(e)}")
            return None, None

    def process_query(self, query: str):
        """Extract name and clean query from timestamp format"""
        pattern = r'^([^,]+),\s*(?:[^,]+,\s*\d+\s+\w+\s+\d+\s+[\d:]+\s+\w+)\s*\n(.+)$'
        match = re.match(pattern, query.strip(), re.DOTALL)
        
        if match:
            name, content = match.groups()
            return name.strip(), content.strip()
        return None, query.strip()

    def generate_response(self, sitrep: str, query: str, org_info: str = None):
        """Generate response based on sitrep analysis and organization info"""
        name, cleaned_query = self.process_query(query)
        greeting = f"Hey {name}" if name else "Hey"
        
        if not cleaned_query or cleaned_query.lower().startswith(('thank', 'ok', 'got it')):
            return f"{greeting}, thank you for your message. - Gradient Cyber Team!"

        # Include organization info in the system message if available
        org_context = f"\nOrganization Context: {org_info}" if org_info else ""
        
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                f"""You are an experienced cyber security analyst handling the role from a Security operations center perspective. 
                {org_context}
                When I provide a message, it contains the summary of a "sitrep" which is a situational report of a particular 
                security incident or event. The goal is to first analyze this sitrep. It will be followed always with a 
                "query" from a user. Your goal will be to understand the sitrep and then focus on answering the query based 
                on your role as an experience cyber security analyst. The concept is to ensure that the response is brief as it 
                primarily is provided as part of a web interface or email.
                Always start with "{{greeting}}" and end with "We hope this answers your question. Thank you! Gradient Cyber Team!"
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
    
    advisor = SecurityAdvisor()
    
    # Add tabs for Query Response and Upload Data
    tab1, tab2 = st.tabs(["Query Response", "Upload Data"])
    
    with tab1:
        # Organization name input
        st.subheader("Organization Search")
        org_name = st.text_input("Organization Name (Optional)")
        org_info = None
        org_summary = None
        
        if org_name:
            org_info, org_summary = advisor.get_organization_info(org_name)
            if org_info:
                col1, col2 = st.columns(2)
                with col1:
                    st.info("### Organization Data\n" + org_info)
                with col2:
                    st.success("### Organization Summary\n" + (org_summary if org_summary else "Summary not available"))
            else:
                st.warning("No organization information found in database.")
        
        # Query response section
        st.subheader("Query Section")
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
            
            # Show organization context if available
            if org_summary:
                st.info(f"### Organization Context:\n{org_summary}")
            
            # Generate and show response
            with st.spinner("Generating response..."):
                with get_openai_callback() as cb:
                    response = advisor.generate_response(sitrep, query, org_summary)
                    st.markdown("### Response:")
                    st.markdown(response)
    
    with tab2:
        st.header("Upload Excel Data")
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            if st.button("Upload to Database"):
                with st.spinner("Uploading data..."):
                    success, message = advisor.upload_excel_to_db(uploaded_file)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

if __name__ == "__main__":
    main()
