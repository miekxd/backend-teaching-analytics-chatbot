from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional, Union, AsyncGenerator 
from app.models.chat import EnhancedChatRequest, EnhancedChatResponse, EnhancedChatErrorResponse
import json
import re

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
    Enhanced LangChain-based teaching assistant with lesson context and RAG routing
    
    Features:
    1. Access to lesson summaries for context
    2. Intelligent detection of questions needing detailed analysis
    3. Flag-based routing to suggest RAG usage
    4. Singapore Teaching Practice framework awareness
    """
    
    def __init__(self):
        # Azure OpenAI connection
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            temperature=0.7,
            max_tokens=800  # Increased for more detailed responses
        )
        
        # Supabase client for accessing file summaries
        self.supabase = get_supabase_client()
        
        # Enhanced system prompt for teaching context
        self.system_prompt = """You are a friendly AI teaching assistant for Singapore educators, with access to lesson summaries.

Your role:
1. Help teachers reflect on their lessons using provided lesson summaries
2. Answer general questions about teaching performance and lesson content
3. Provide supportive feedback aligned with Singapore Teaching Practice framework
4. Detect when questions need detailed transcript analysis and guide users appropriately

Teaching Areas (Singapore Teaching Practice):
- 1.1 Establishing Interaction and Rapport
- 1.2 Setting and Maintaining Rules and Routine  
- 3.1 Activating Prior Knowledge
- 3.2 Motivating Learners for Learning Engagement
- 3.3 Using Questions to Deepen Learning
- 3.4 Facilitating Collaborative Learning
- 3.5 Concluding the Lesson
- 4.1 Checking for Understanding and Providing Feedback

Flag Detection Rules:
- Flag for detailed search (flag_specific: 1) ONLY when users ask for:
  * Specific time ranges ("first 15 minutes", "last 10 minutes", "minute 5 to 20")
  * Specific examples with evidence ("show me examples", "find when I...", "what did I say about...")
  * Detailed moment-by-moment analysis
  * Specific utterance searches

- Keep general (flag_specific: 0) for:
  * Greetings and conversational messages
  * Overall lesson feedback ("how did I do?", "how was my lesson?")
  * General questions answerable from summary
  * Broad teaching advice

When uncertain, provide helpful guidance and suggest specific questions they could ask for detailed analysis.

Always be supportive, constructive, and focused on helping teachers improve their practice."""

        # Prompt template for enhanced responses
        self.enhanced_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Lesson Context:
{lesson_summaries}

User Question: {user_message}

Please respond helpfully. If the question needs detailed transcript analysis, explain what detailed search could provide and suggest specific example questions.""")
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
    
    def _detect_specific_query(self, message: str) -> Dict[str, Any]:
        """
        Detect if message requires detailed RAG search
        
        Returns:
            Dict with flag_specific, confidence, and reasoning
        """
        msg_lower = message.lower().strip()
        
        # High confidence patterns - definitely need RAG
        time_specific_patterns = [
            r'(first|last|initial|final|opening|ending) \d+ minute',
            r'minute \d+ to \d+',
            r'between minute \d+ and \d+',
            r'from minute \d+',
            r'at \d+:\d+',
            r'in the (first|last) \d+ minute'
        ]
        
        detailed_search_patterns = [
            r'show me (examples|instances)',
            r'find (when|where|how) (i|students)',
            r'what did i (say|do) (about|when|during)',
            r'give me examples of',
            r'search for',
            r'look for'
        ]
        
        # Check for high confidence RAG patterns
        for pattern in time_specific_patterns + detailed_search_patterns:
            if re.search(pattern, msg_lower):
                return {
                    "flag_specific": 1,
                    "confidence": 0.9,
                    "reason": "time_specific_or_detailed_search_required",
                    "pattern_matched": pattern
                }
        
        # Medium confidence patterns - might need RAG
        medium_patterns = [
            r'(techniques|strategies|methods) (i|did)',
            r'how (often|many times)',
            r'specific (examples|instances)',
            r'teaching area \d\.\d'
        ]
        
        for pattern in medium_patterns:
            if re.search(pattern, msg_lower):
                return {
                    "flag_specific": 0.5,
                    "confidence": 0.6,
                    "reason": "might_benefit_from_detailed_analysis",
                    "pattern_matched": pattern
                }
        
        # Greetings and general patterns - definitely general
        general_patterns = [
            r'^(hi|hello|hey|good morning)',
            r'how (are you|did i do)',
            r'(overall|general) (feedback|thoughts)',
            r'how was (my|the) lesson',
            r'thank you',
            r'thanks'
        ]
        
        for pattern in general_patterns:
            if re.search(pattern, msg_lower):
                return {
                    "flag_specific": 0,
                    "confidence": 0.9,
                    "reason": "general_conversation_or_summary_sufficient",
                    "pattern_matched": pattern
                }
        
        # Default: general response
        return {
            "flag_specific": 0,
            "confidence": 0.7,
            "reason": "default_general_response",
            "pattern_matched": None
        }
    
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
        current_with_context = f"""Lesson Context:
{lesson_summaries}

User Question: {current_message}"""
        
        messages.append(HumanMessage(content=current_with_context))
        
        return messages
    
    async def get_enhanced_response(self, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None) -> Union[EnhancedChatResponse, EnhancedChatErrorResponse]:
        """
        Get enhanced response with lesson context and RAG routing detection
        
        Returns:
            EnhancedChatResponse or EnhancedChatErrorResponse
        """
        try:
            # Get lesson summaries
            lesson_summaries = self._get_file_summaries(file_ids)
            
            # Detect if question needs specific analysis
            query_analysis = self._detect_specific_query(user_message)
            
            # Generate response
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
            
            # Build enhanced response
            return EnhancedChatResponse(
                response=response_text,
                method="enhanced_teaching_assistant",
                flag_specific=query_analysis["flag_specific"],
                confidence=query_analysis["confidence"],
                routing_reason=query_analysis["reason"],
                files_analyzed=len(file_ids),
                has_lesson_context=bool(lesson_summaries and "No lesson summaries available" not in lesson_summaries),
                suggested_detailed_queries=[
                    "Show me examples of my questioning techniques",
                    "Analyze my teaching in the first 15 minutes", 
                    "Find moments where students seemed confused",
                    "How did I establish rapport in the opening?"
                ] if query_analysis["flag_specific"] >= 0.5 else []
            )
            
        except Exception as e:
            return EnhancedChatErrorResponse(
                response=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                method="enhanced_teaching_assistant",
                flag_specific=0,
                confidence=0.0,
                routing_reason="error_occurred",
                error=str(e)
            )
        
    async def get_enhanced_response_stream(self, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """
        Stream enhanced response with lesson context
        
        Yields:
            JSON strings containing response chunks and metadata
        """
        try:
            # Get lesson summaries
            lesson_summaries = self._get_file_summaries(file_ids)
            
            # Detect if question needs specific analysis
            query_analysis = self._detect_specific_query(user_message)
            
            # Send initial metadata
            initial_data = {
                "type": "metadata",
                "flag_specific": query_analysis["flag_specific"],
                "confidence": query_analysis["confidence"],
                "routing_reason": query_analysis["reason"],
                "files_analyzed": len(file_ids),
                "has_lesson_context": bool(lesson_summaries and "No lesson summaries available" not in lesson_summaries)
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            # Setup streaming callback
            streaming_handler = StreamingCallbackHandler()
            
            # Configure LLM for streaming
            streaming_llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                temperature=0.7,
                max_tokens=800,
                streaming=True,  # Enable streaming
                callbacks=[streaming_handler]
            )
            
            # Generate response with streaming
            if conversation_history:
                # Use conversation history
                messages = self._build_message_history(conversation_history, user_message, lesson_summaries)
                
                # Stream the response
                async for chunk in streaming_llm.astream(messages):
                    if chunk.content:
                        chunk_data = {
                            "type": "content",
                            "content": chunk.content
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
            else:
                # Use enhanced chain with streaming
                enhanced_streaming_chain = self.enhanced_prompt | streaming_llm | self.output_parser
                
                async for chunk in enhanced_streaming_chain.astream({
                    "lesson_summaries": lesson_summaries,
                    "user_message": user_message
                }):
                    if chunk:
                        chunk_data = {
                            "type": "content", 
                            "content": chunk
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send completion metadata
            completion_data = {
                "type": "complete",
                "method": "enhanced_teaching_assistant_streaming",
                "suggested_detailed_queries": [
                    "Show me examples of my questioning techniques",
                    "Analyze my teaching in the first 15 minutes", 
                    "Find moments where students seemed confused",
                    "How did I establish rapport in the opening?"
                ] if query_analysis["flag_specific"] >= 0.5 else []
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e),
                "method": "enhanced_teaching_assistant_streaming"
            }
            yield f"data: {json.dumps(error_data)}\n\n"

# Create instances
enhanced_assistant = EnhancedTeachingAssistant()

# Create the FastAPI router
router = APIRouter()

@router.post("/enhanced", response_model=EnhancedChatResponse)
async def enhanced_chat_endpoint(request: EnhancedChatRequest) -> Union[EnhancedChatResponse, JSONResponse]:
    """
    Enhanced teaching assistant endpoint with lesson context and RAG routing
    
    Features:
    - Access to lesson summaries for context
    - Intelligent detection of questions needing detailed analysis  
    - Flag-based routing suggestions for RAG
    - Singapore Teaching Practice framework awareness
    
    Example requests:
    - "How did my lesson go?" → General response with lesson summary
    - "Show me the first 15 minutes" → Flags for RAG routing
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    if not file_ids:
        return JSONResponse({"error": "file_ids are required for lesson context"}, status_code=400)
    
    try:
        result = await enhanced_assistant.get_enhanced_response(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=request.conversation_history
        )
        
        return result
    
    except Exception as e:
        return JSONResponse({"error": f"Enhanced chat error: {str(e)}"}, status_code=500)

@router.post("/enhanced/stream")
async def enhanced_chat_streaming_endpoint(request: EnhancedChatRequest):
    """
    Enhanced teaching assistant streaming endpoint
    
    Returns Server-Sent Events (SSE) stream with:
    - Real-time response chunks as they're generated
    - Metadata about routing decisions
    - Completion signals
    
    Response format: text/event-stream
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    if not file_ids:
        return JSONResponse({"error": "file_ids are required for lesson context"}, status_code=400)
    
    try:
        return StreamingResponse(
            enhanced_assistant.get_enhanced_response_stream(
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=request.conversation_history
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    except Exception as e:
        return JSONResponse({"error": f"Streaming error: {str(e)}"}, status_code=500)