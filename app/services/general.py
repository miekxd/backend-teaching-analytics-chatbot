from typing import List, Dict, Any, Optional, AsyncGenerator 
import json

# Basic LangChain imports - only the essentials
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler

from app.core.config import settings
from app.db.supabase import get_supabase_client

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token"""
        self.tokens.append(token)

class EnhancedTeachingAssistant:
    """
    Enhanced LangChain-based teaching assistant with lesson context
    
    Features:
    1. Access to lesson summaries for context
    2. LLM-based flag detection via JSON output
    3. Singapore Teaching Practice framework awareness
    """
    
    def __init__(self):
        # Azure OpenAI connection
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            temperature=0.7,
            max_tokens=500
        )
        
        # Supabase client for accessing file summaries
        self.supabase = get_supabase_client()
        
        # Enhanced system prompt for teaching context with LLM-based flagging
        self.system_prompt = """You are a supportive AI teaching assistant for Singapore educators, designed to help teachers reflect on their lessons and improve their practice.

<role>
You are a friendly, constructive teaching coach specializing in the Singapore Teaching Practice framework. Your goal is to provide meaningful feedback that helps teachers grow professionally and route complex questions to detailed analysis when needed.
</role>

<primary_task>
Analyze user questions to determine if they require detailed lesson transcript analysis or can be answered using provided lesson summaries. Based on this analysis, either provide a direct response or respond with semantic search queries for detailed analysis.
</primary_task>

<flagging_mechanism>
<flag_criteria>
Set flag_specific score based on question type:

HIGH SPECIFICITY (0.7-1.0) - Requires detailed transcript search:
- Specific time references ("first 15 minutes", "between 10:30-11:00", "last quarter")
- Evidence-based requests ("show me examples", "find instances when", "quote what I said")
- Moment-by-moment analysis ("walk me through", "step-by-step breakdown")
- Specific utterance searches ("when did I mention", "how many times did I ask")
- Detailed interaction analysis ("student responses to my questions")

LOW SPECIFICITY (0.0-0.6) - Can use lesson summaries:
- General greetings and social conversation
- Overall lesson evaluation ("how did my lesson go?", "what's your feedback?")
- Broad performance questions ("did I engage students well?")
- General teaching advice ("how can I improve?")
- Summary-level content questions ("what topics did we cover?")
- High-level reflection prompts
</flag_criteria>

<confidence_guidelines>
- 0.9-1.0: Explicitly asks for specific examples, quotes, or time-based analysis
- 0.7-0.8: Implies need for detailed evidence but not explicitly stated
- 0.4-0.6: Borderline cases that could benefit from either approach
- 0.1-0.3: Clearly answerable from summaries but might benefit from examples
- 0.0: Definitely doesn't need transcript analysis
</confidence_guidelines>
</flagging_mechanism>

<singapore_teaching_framework>
Reference these EXACT WORDING of these areas when providing feedback or generating search queries:
- 1.1 Establishing Interaction and Rapport
- 1.2 Setting and Maintaining Rules and Routine
- 3.1 Activating Prior Knowledge
- 3.2 Motivating Learners for Learning Engagement
- 3.3 Using Questions to Deepen Learning
- 3.4 Facilitating Collaborative Learning
- 3.5 Concluding the Lesson
- 4.1 Checking for Understanding and Providing Feedback
</singapore_teaching_framework>

<response_guidelines>
<general_responses>
- Always be supportive and constructive
- Focus on actionable insights for improvement
- Use evidence from lesson summaries when available
- Frame feedback positively while addressing areas for growth
- Connect observations to Singapore Teaching Practice standards
</general_responses>

<search_query_responses>
- Include search queries directly in your message response
- Format queries as a simple numbered or bulleted list
- Ensure queries directly address the user's specific question
- Make queries likely to retrieve relevant transcript chunks
</search_query_responses>
</response_guidelines>

<output_format>
ALWAYS respond with JSON containing only:
- flag_specific: float (0.0 to 1.0 confidence score)
- message: string (your response - helpful feedback formatted with markdown OR search queries)

For flag_specific ≤ 0.6: Include your helpful response using lesson summaries in the message
For responses, use markdown formatting:
- Use **bold** for important numbers, percentages, and key terms
- Use ### for section headings
- Use • for bullet points
- Use `code` for specific teaching area codes (e.g., `1.2`, `4.1`)
- Use > for important quotes or highlights

For flag_specific > 0.6: Include 2-4 search queries in the message
</output_format>"""

        # Prompt template for enhanced responses
        self.enhanced_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """<lesson_context>
{lesson_summaries}
</lesson_context>

<user_question>
{user_message}
</user_question>
""")
])
        # Output parser
        self.output_parser = StrOutputParser()
        
        # Chain for enhanced responses
        self.enhanced_chain = self.enhanced_prompt | self.llm | self.output_parser
    
    def _get_file_summaries(self, file_ids: List[int]) -> str:
        """Get lesson summaries for the specified files"""
        summary_sections = []
        
        for fid in file_ids:
            try:
                file_info = self.supabase.table("files").select("stored_filename, data_summary").eq("file_id", fid).single().execute()
                if file_info.data:
                    filename = file_info.data.get("stored_filename", f"File {fid}")
                    summary = file_info.data.get("data_summary", "No summary available.")
                    summary_sections.append(f"File: {filename}\nLesson Summary: {summary}\n")
            except Exception as e:
                summary_sections.append(f"File {fid}: Error retrieving summary - {str(e)}\n")
        
        return "\n".join(summary_sections) if summary_sections else "No lesson summaries available."
    
    def _build_message_history(self, conversation_history: List[Dict[str, str]], current_message: str, lesson_summaries: str) -> List:
        """Build proper message history for LangChain with lesson context"""
        messages = [
            SystemMessage(content=self.system_prompt)
        ]
        
        # Add conversation history (limit to last 8 messages to save tokens)
        for msg in conversation_history[-8:]:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        
        # Add current message with lesson context
        current_with_context = f"""<lesson_context>
{lesson_summaries}
</lesson_context>

<user_question>
{current_message}
</user_question>
"""
        messages.append(HumanMessage(content=current_with_context))
        
        return messages
    
    async def get_response(self, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Get enhanced response with lesson context and LLM-based flag detection
        
        Returns:
            Dict with "flag_specific" and "response" fields only
        """
        try:
            # Get lesson summaries
            lesson_summaries = self._get_file_summaries(file_ids)
            
            # Generate response with LLM handling all flag logic
            if conversation_history:
                # Use conversation history
                messages = self._build_message_history(conversation_history, user_message, lesson_summaries)
                response = await self.llm.ainvoke(messages)
                response_text = response.content
            else:
                # Use enhanced chain
                response_text = await self.enhanced_chain.ainvoke({
                    "lesson_summaries": lesson_summaries,
                    "user_message": user_message
                })
            
            # Parse the JSON response from LLM
            try:
                llm_response = json.loads(response_text)
                flag_specific = llm_response.get("flag_specific", 0.0)
                response_content = llm_response.get("message", response_text)
                
                return {
                    "flag_specific": flag_specific,
                    "response": response_content
                }
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "flag_specific": 0.0,
                    "response": response_text
                }
            
        except Exception as e:
            return {
                "flag_specific": 0.0,
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}"
            }
        
    async def get_response_stream(self, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """
        Stream enhanced response with lesson context
        
        Yields:
            Text chunks as they're generated
        """
        try:
            # Get lesson summaries
            lesson_summaries = self._get_file_summaries(file_ids)
            
            # Configure LLM for streaming
            streaming_llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                temperature=0.7,
                max_tokens=800,
                streaming=True
            )
            
            # Generate response with streaming
            if conversation_history:
                # Use conversation history
                messages = self._build_message_history(conversation_history, user_message, lesson_summaries)
                
                # Stream the response
                async for chunk in streaming_llm.astream(messages):
                    if chunk.content:
                        yield chunk.content
            else:
                # Use enhanced chain with streaming
                enhanced_streaming_chain = self.enhanced_prompt | streaming_llm | self.output_parser
                
                async for chunk in enhanced_streaming_chain.astream({
                    "lesson_summaries": lesson_summaries,
                    "user_message": user_message
                }):
                    if chunk:
                        yield chunk
            
        except Exception as e:
            yield f"I apologize, but I encountered an error while processing your request: {str(e)}"

# Create instances
general_assistant = EnhancedTeachingAssistant()