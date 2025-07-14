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
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_GENERAL,
            temperature=0.2,
            max_tokens=500
        )
        
        # Supabase client for accessing file summaries
        self.supabase = get_supabase_client()
        
        # Enhanced system prompt for teaching context with LLM-based flagging
        self.system_prompt = """

<role>
You are an intent analyzer and response generator for a teaching assistant application. 
You have 2 tasks:
1. Analyze user's question to determine if it requires a RAG of transcript data to answer (flag_specific > 0.5)
2. Generate a helpful response based on the user's question and available lesson summaries (flag_specific ≤ 0.5) 
### CRITICAL: ALWAYS OBEY THE OUTPUT FORMAT JSON
</role>


<flagging_mechanism>
<flag_criteria>
Set flag_specific score based on question type AND conversation context:

HIGH SPECIFICITY (0.6-1.0) - Requires detailed transcript search:
- Specific time references ("first 15 minutes", "between 10:30-11:00", "last quarter")
- Evidence-based requests ("show me examples", "find instances when", "quote what I said")
- Moment-by-moment analysis ("walk me through", "step-by-step breakdown")
- Specific utterance searches ("when did I mention", "how many times did I ask")
- Detailed interaction analysis ("student responses to my questions")
- Follow-up questions requesting specifics after general responses ("can you give me specific examples?", "show me when this happened")
- Requests for more detail on previously discussed topics ("tell me more about that", "can you elaborate on the questioning part?")

LOW SPECIFICITY (0.0-0.5) - Can use lesson summaries:
- General greetings and social conversation
- Overall lesson evaluation ("how did my lesson go?", "what's your feedback?")
- Broad performance questions ("did I engage students well?")
- General teaching advice ("how can I improve?")
- Summary-level content questions ("what topics did we cover?")
- High-level reflection prompts
- Initial broad questions that haven't been explored yet
</flag_criteria>

<conversation_history_considerations>
<context_escalation>
When conversation history shows:
- User previously received general feedback and now asks for specifics → INCREASE flag_specific by 0.2-0.3
- User asks follow-up questions like "can you show me examples?" → SET flag_specific to 0.8+
- User references previous general response and wants details → INCREASE flag_specific to 0.7+
- User asks "tell me more" or "elaborate" after general feedback → INCREASE flag_specific by 0.2-0.4
</context_escalation>

</conversation_history_considerations>
</flagging_mechanism>

<singapore_teaching_framework>
Reference these EXACT WORDING of these areas (INCLUDING THE NUMBERS) when providing feedback or generating search queries:
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
- Consider if user might want more specific examples and gently suggest they can ask for them
</general_responses>

<search_query_responses>
- Include search queries directly in your message response
- Format queries as a simple numbered or bulleted list
- Ensure queries directly address the user's specific question
- Make queries likely to retrieve relevant transcript chunks
- Reference conversation history when relevant ("Following up on your question about...")
</search_query_responses>
</response_guidelines>

<output_format>
- flag_specific: float (0.0 to 1.0 confidence score, adjusted for conversation history)
- response: string (your response - helpful feedback formatted with markdown OR search queries)

For flag_specific ≤ 0.5: Include your helpful response using lesson summaries in the message
For responses, use markdown formatting.

For flag_specific > 0.5: Include 2-4 search queries in the message that consider conversation context
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
        for msg in conversation_history[-3:]:
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
                print(f"LLM Response text: {response_text}")
                llm_response = json.loads(response_text)
                flag_specific = llm_response.get("flag_specific", 0.0)
                response_content = llm_response.get("response", response_text)
                
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