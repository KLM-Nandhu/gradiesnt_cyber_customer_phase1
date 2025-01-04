import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings 
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import openai
from typing import Dict, Optional, List
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SitrepAnalyzer:
   def __init__(self):
       # Only use environment variable for API key
       self.openai_api_key = os.getenv("OPENAI_API_KEY")
       if not self.openai_api_key:
           raise ValueError("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY.")
       openai.api_key = self.openai_api_key
       self.llm = ChatOpenAI(
           model_name="gpt-4o-mini",
           temperature=0.1,
           openai_api_key=self.openai_api_key
       )
   def identify_phase(self, query: str) -> str:
        """Identify which phase the query belongs to"""
        phase_prompt = f"""
        Analyze the following query and determine if it belongs to Phase 2 or Phase 3:

        Phase 2 Characteristics:
        - Requests for filtering or excluding specific traffic
        - Whitelisting requests
        - Traffic suppression requests
        - Alert filtering configuration
        - Deals with expected or routine traffic handling

        Phase 3 Characteristics:
        - Requests for specific data analysis
        - Investigation of anomalies
        - Pattern analysis requests
        - Traffic investigation
        - System-specific insights
        - Log analysis requests

        Query: {query}

        Return ONLY "Phase 2" or "Phase 3" based on the characteristics above.
        If unclear, default to "Phase 3".
        """

        try:
            response = self.llm.predict(phase_prompt).strip()
            return response if response in ["Phase 2", "Phase 3"] else "Phase 3"
        except Exception as e:
            logger.error(f"Error in phase identification: {str(e)}")
            return "Phase 3"    
   def extract_client_metadata(self, query: str) -> Dict[str, str]:
    """
    Extract client metadata (name, timestamp) and clean query content
    """
    metadata_prompt = """
    Given this message, extract metadata and content.
    Rules:
    1. IF the message starts with a name followed by timestamp, extract them
    2. IF the message is just a question/comment, treat entire text as content
    3. Remove any metadata from content
if
    4. Check the content, if the content is just a information just say the greeting message like thank you...etc make by own llm
else

    Input: "{query}"

    Return exactly in this format (include all fields):
    {{"name": "extracted name or null",
    "timestamp": "extracted timestamp or null",
    "content": "cleaned message content"}}

    Examples:
    Input: "Wade Jones, Tue, 29 Oct 2024 15:34:26 GMT\nNot sure I understand what you are trying to tell me?"
    Output: {{"name": "Wade Jones", "timestamp": "Tue, 29 Oct 2024 15:34:26 GMT", "content": "Not sure I understand what you are trying to tell me?"}}

    Input: "Not sure I understand what you are trying to tell me?"
    Output: {{"name": null, "timestamp": null, "content": "Not sure I understand what you are trying to tell me?"}}
    """
    
    try:
        response = self.llm.predict(metadata_prompt.format(query=query))
        metadata = json.loads(response)
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {
            "name": None,
            "timestamp": None,
            "content": query
        }

   def is_general_query(self, query: str) -> bool:
    """Determine if a query is general or specific using LLM"""
    # Extract only the content part of the query
    metadata = self.extract_client_metadata(query)
    query_content = metadata["content"]
    
    query_analysis_prompt = f"""
    Analyze if this query is general or specific to customer logs/systems:
    Query: {query_content}

    Guide:
    - General queries ask about understanding alerts, security concepts, or general procedures
    - Specific queries reference customer data, specific systems, or require log analysis
    
    Examples:
    General: "What does this alert mean?", "Not sure I understand what you are trying to tell me?"
    Specific: "Why did we see this traffic spike yesterday?"

    Return only 'general' or 'specific'.
    """

    try:
        response = self.llm.predict(query_analysis_prompt)
        return response.strip().lower() == "general"
    except Exception as e:
        logger.error(f"Error in query classification: {str(e)}")
        return False

   def is_acknowledgment(self, query: str) -> bool:
        """Determine if query is an acknowledgment rather than a question"""
        ack_prompt = f"""
        Analyze if this message is an acknowledgment/statement or a question:
        Message: {query}
        
        Examples of acknowledgments/statements:
        - "I received the documents"
        - "This traffic is expected"
        - "Thank you for the information"
        - "This is from our normal operations"
        
        Examples of questions:
        - "Why did this happen?"
        - "What does this mean?"
        - "Should I be concerned?"
        
        Return only 'acknowledgment' or 'question'.
        """
        
        try:
            response = self.llm.predict(ack_prompt).strip().lower()
            return response == 'acknowledgment'
        except Exception as e:
            logger.error(f"Error in acknowledgment detection: {str(e)}")
            return False

   def generate_json_path_filter(self, sitrep_data: Dict) -> Optional[Dict]:
       """Generate JSON path filters based on sitrep data"""
       try:
           filter_prompt = f"""
           Create a JSON path filter based on this security alert:
           Alert Summary: {sitrep_data.get('alert_summary', '')}
           Customer Query: {sitrep_data.get('feedback', '')}

           Generate a JSON filter that would help process similar alerts.
           Include:
           1. Key paths to monitor
           2. Conditions to match
           3. Thresholds or patterns to detect

           Return only valid JSON without explanation.
           """
           
           filter_response = self.llm.predict(filter_prompt)
           
           try:
               filter_data = json.loads(filter_response)
               filter_data["metadata"] = {
                   "generated_for": sitrep_data.get("alert_type", "unknown"),
                   "query_type": "general" if self.is_general_query(sitrep_data.get("feedback", "")) else "specific"
               }
               return filter_data
               
           except json.JSONDecodeError:
               logger.error("Failed to parse JSON filter response")
               return None
               
       except Exception as e:
           logger.error(f"Error generating JSON path filter: {str(e)}")
           return None

   def analyze_sitrep(self, alert_summary: str, client_query: Optional[str] = None) -> Dict:
        try:
            with get_openai_callback() as cb:
                metadata = self.extract_client_metadata(client_query) if client_query else {
                    "name": None,
                    "timestamp": None,
                    "content": ""
                }
                
                is_general = True if not client_query else self.is_general_query(metadata["content"])
                
                result = {
                    "is_general_query": is_general,
                    "requires_manual_review": not is_general,
                }
                
                # Add phase identification for manual review cases
                if not is_general and client_query:
                    result["phase"] = self.identify_phase(metadata["content"])
                
                if client_query:
                    json_filter = self.generate_json_path_filter({
                        "alert_summary": alert_summary,
                        "feedback": metadata["content"]
                    })
                    if json_filter:
                        result["json_filter"] = json_filter
                
                if is_general:
                    analysis_result = self.generate_analysis(
                        alert_summary,
                        client_query,
                        is_general
                    )
                    result["analysis"] = analysis_result.get("analysis")
                
                # Add token usage information to result
                result["token_usage"] = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": f"${cb.total_cost:.4f}"
                }
                
                return result
        
        except Exception as e:
            logger.error(f"Error in analyze_sitrep: {str(e)}")
            return {"error": str(e)}

   def generate_analysis(self, alert_summary: str, client_query: Optional[str], 
                         is_general: bool) -> Dict:
        """Generate analysis based on query type"""
        metadata = self.extract_client_metadata(client_query) if client_query else {
            "name": None,
            "timestamp": None,
            "content": client_query or ""
        }
        
        # Check if it's an acknowledgment
        if self.is_acknowledgment(metadata["content"]):
            greeting = f"Hey {metadata['name']}" if metadata['name'] else "Hey"
            response = f"{greeting}, thank you for letting us know. We've noted your response. - Gradient Cyber Team !"
        else:
            # Original analysis logic for questions
            system_prompt = SystemMessagePromptTemplate.from_template(
                """You are a senior security analyst providing clear, accurate and concise responses.
                Rules:
                1. Start with "{greeting}"
                2. State the current security context and its implication
                3. Add one clear recommendation that should tell by "we" not "I"
                4. Use exactly 3-5 sentences maximum
                5. End with "We hope this answers your question. Thank you! Gradient Cyber Team !"
                
                Query Type: {query_type}"""
            )
            
            human_template = """
            Alert Summary: {alert_summary}
            Client Query: {query}
            
            Provide a clear, concise explanation of:
            1. What the alert means
            2. Why it matters
            3. What action is recommended (if any)
            
            Follow the exact format described in the system prompt.
            """
            
            greeting = f"Hey {metadata['name']}" if metadata['name'] else "Hey"
            
            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_template])
            chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            
            response = chain.run(
                greeting=greeting,
                query_type="General" if is_general else "Specific",
                alert_summary=alert_summary,
                query=metadata["content"]
            )

        return {
            "analysis": response.strip(),
            "is_general_query": is_general,
            "requires_manual_review": not is_general
        }

def main():
   st.set_page_config(page_title="Sitrep Analyzer", layout="wide")
   
   st.markdown("""
       <style>
       .json-box {
           background-color: #f8f9fa;
           padding: 15px;
           border-radius: 8px;
           border-left: 5px solid #28a745;
           font-family: monospace;
           white-space: pre-wrap;
       }
       .automation-box {
           background-color: #f8f9fa;
           padding: 15px;
           border-radius: 8px;
           border-left: 5px solid #17a2b8;
       }
       .manual-review-box {
           background-color: #fff3cd;
           padding: 15px;
           border-radius: 8px;
           border-left: 5px solid #ffc107;
       }
       .token-box {
           background-color: #f8f9fa;
           padding: 10px;
           border-radius: 4px;
           margin-top: 20px;
           font-size: 0.8em;
       }
       .token-title {
           font-size: 0.9em;
           color: #666;
           margin-bottom: 8px;
       }
       .token-metric {
           font-size: 0.8em !important;
       }
       </style>
   """, unsafe_allow_html=True)
   
   analyzer = SitrepAnalyzer()
   
   st.title("Sitrep Analysis System")
   
   col1, col2 = st.columns([2, 1])
   
   with col1:
       st.subheader("Alert Summary")
       alert_summary = st.text_area(
           "Paste your security alert details here",
           height=300
       )
       
   with col2:
       st.subheader("Client Query")
       client_query = st.text_area(
           "Enter client questions or feedback",
           height=150
       )
       
       st.subheader("JSON Display Options")
       show_filter_json = st.checkbox("Show Generated Filters", value=True)
   
   if st.button("Analyze Alert", type="primary"):
       if not alert_summary:
           st.error("Please enter an alert summary to analyze.")
           return
       
       with st.spinner("Analyzing security alert..."):
           result = analyzer.analyze_sitrep(alert_summary, client_query)
           
           if "error" in result:
               st.error(result["error"])
           else:
               if result.get("requires_manual_review"):
                   st.markdown("""
                       <div class="manual-review-box">
                       <h4>👥 Manual Review Required</h4>
                       <p>This query requires specific analysis of customer logs or systems.</p>
                       </div>
                   """, unsafe_allow_html=True)
                   
                   if "phase" in result:
                       st.markdown(f"""
                           <div class="manual-review-box">
                           <h4>Query Classification</h4>
                           <p>{result["phase"]}: {
                               "Request for traffic filtering/exclusion" if result["phase"] == "Phase 2"
                               else "Request for specific data analysis/insights"
                           }</p>
                           </div>
                       """, unsafe_allow_html=True)
               else:
                   st.markdown("""
                       <div class="automation-box">
                       <h4>🤖 Automated Processing</h4>
                       <p>Phase 1: This query has been identified as a general inquiry and can be handled automatically using standardized responses.</p>
                       </div>
                   """, unsafe_allow_html=True)
               
               if "analysis" in result:
                   st.subheader("Analysis")
                   st.markdown(result["analysis"])
               
               if show_filter_json and "json_filter" in result:
                   st.subheader("Generated JSON Filter")
                   st.json(result["json_filter"])
               
               # Token Analysis at the bottom with smaller font
               if "token_usage" in result:
                   st.markdown("<div class='token-box'>", unsafe_allow_html=True)
                   st.markdown("<p class='token-title'>📊 Token Analysis</p>", unsafe_allow_html=True)
                   
                   cols = st.columns(4)
                   with cols[0]:
                       st.markdown(f"<div class='token-metric'>Total Tokens<br><b>{result['token_usage']['total_tokens']:,}</b></div>", unsafe_allow_html=True)
                   with cols[1]:
                       st.markdown(f"<div class='token-metric'>Prompt Tokens<br><b>{result['token_usage']['prompt_tokens']:,}</b></div>", unsafe_allow_html=True)
                   with cols[2]:
                       st.markdown(f"<div class='token-metric'>Completion Tokens<br><b>{result['token_usage']['completion_tokens']:,}</b></div>", unsafe_allow_html=True)
                   with cols[3]:
                       st.markdown(f"<div class='token-metric'>Cost<br><b>{result['token_usage']['total_cost']}</b></div>", unsafe_allow_html=True)
                   
                   st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
