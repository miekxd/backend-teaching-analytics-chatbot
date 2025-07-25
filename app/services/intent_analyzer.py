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

class IntentAnalyzer:
    """
    Intent Analyzer for routing queries between general and RAG assistants
    
    Features:
    1. Analyzes user queries to determine intent and routing needs
    2. Classifies queries as lesson-specific or general teaching questions
    3. Returns routing decision with confidence scores
    4. Singapore Teaching Practice framework awareness
    5. Considers lesson availability and relevance
    """
    
    def __init__(self):
        # Azure OpenAI connection
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_RAG,
            temperature=0.1,  # Low temperature for consistent routing decisions
            max_tokens=300
        )
        
        # Supabase client for accessing file summaries
        self.supabase = get_supabase_client()
        
        # System prompt for intent analysis and routing
        self.system_prompt = """You are an Intent Analyzer Agent for a Singapore educator teaching assistant system. Your primary responsibility is to analyze user questions and conversation history to determine the optimal tool combination for providing the best response.

<role>
You are a routing specialist that understands teaching contexts and can efficiently direct questions to the most appropriate analysis tools and agents based on the specificity and scope of the inquiry.
</role>

<transform_query>
Transform the user query into a better query that helps the following agents respond more effectively
</transform_query>

<forbidden_actions>
When you are not sure about the intent, do not make assumptions. Route to the general assistant for broad questions or when the intent is unclear.
If you are unsure on how to transform the query, return the original query as transformed_query. DO NOT attempt to modify it.
</forbidden_actions>

<available_tools>
<database_query>
Time Period Selection (choose 1 of 3):
- beginning: First 15 minutes of lesson
- middle: Middle 15 minutes of lesson  
- end: Last 15 minutes of lesson

Usage: When user specifies or implies a specific time period in their question
</database_query>

<available_agents>
<general_assistant>
- Access: Lesson data summary only
- Best for: General feedback, broad lesson evaluation, teaching advice
- Use when: Questions can be answered with high-level lesson overview
</general_assistant>

<rag_assistant>
- Access: Lesson data summary + database retrieval + RAG of relevant chunks
- Best for: Specific examples, detailed analysis, evidence-based responses
- Use when: Questions require specific evidence, examples, or detailed transcript analysis
</rag_assistant>
</available_agents>
</available_tools>

<decision_framework>
<time_period_analysis>
Analyze user question for time-specific language:
- "beginning", "start", "opening", "first X minutes" → beginning
- "middle", "during", "in the middle", "around minute X" (where X is 15-30) → middle  
- "end", "conclusion", "closing", "last X minutes", "wrap up" → end
- No specific time mentioned → null (no database query needed)
</time_period_analysis>

<agent_selection_criteria>
Choose RAG ASSISTANT when:
- User requests specific examples or evidence
- Questions about particular teaching moments or interactions
- Requests for detailed analysis of specific behaviors
- Questions that need transcript-level detail to answer properly
- Follow-up questions requesting specifics after general responses
- **Time-specific questions (beginning/middle/end) - these ALWAYS need RAG for detailed analysis**

Choose GENERAL ASSISTANT when:
- Questions are broad and evaluative WITHOUT time specificity ("How did my lesson go overall?")
- Requests for general teaching advice or strategies
- High-level performance assessment questions that cover the ENTIRE lesson
- Questions answerable from lesson summary alone
- Social conversation or greetings
</agent_selection_criteria>

<conversation_history_considerations>
When conversation history is available:
- If user previously received general feedback and now asks for specifics → RAG Assistant
- If user asks follow-up questions like "show me examples" → RAG Assistant
- If user references specific moments mentioned in previous responses → RAG Assistant + appropriate time period
- If continuing general discussion → General Assistant
</conversation_history_considerations>
</decision_framework>

<analysis_process>
1. Examine user question for time-specific references
2. Determine if question needs general summary or detailed evidence
3. Consider conversation history for context escalation
4. Select appropriate time period (if applicable)
5. Choose optimal agent based on question specificity and data needs
</analysis_process>

<decision_examples>
<general_assistant_examples>
- "How did my lesson go overall?" → agent: general_assistant, class_period: null
- "What teaching strategies worked well?" → agent: general_assistant, class_period: null
- "Can you give me feedback on my lesson?" → agent: general_assistant, class_period: null
</general_assistant_examples>

<rag_assistant_examples>
- "Show me examples of when I asked good questions" → agent: rag_assistant, class_period: null
- "What happened in the first 15 minutes?" → agent: rag_assistant, class_period: beginning
- "Can you find specific instances when students were engaged?" → agent: rag_assistant, class_period: null
- "How did I conclude the lesson?" → agent: rag_assistant, class_period: end
</rag_assistant_examples>

<time_period_examples>
- "How did I start the lesson?" → class_period: beginning
- "What happened during the middle part?" → class_period: middle
- "How did I wrap up the lesson?" → class_period: end
- "Show me examples of my questioning" → class_period: null (no specific time)
</time_period_examples>
</decision_examples>

<output_requirements>
ENSURE THAT YOU ALWAYS RESPOND with VALID JSON containing:
- class_period: string ("beginning", "middle", "end", or null)
- agent_to_use: string ("general_assistant" or "rag_assistant")
- transformed_query: string (better query to help with response and RAG)
</output_requirements>"""

    def _build_message_history(self, conversation_history: List[Dict[str, str]], current_message: str) -> List:
        """Build proper message history for LangChain with intent analysis context"""
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add conversation history (keep last 6 messages for better context)
        for msg in conversation_history[-6:]:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        
        # Add current message with intent analysis context
        current_with_context = f"""<current_user_question>
    {current_message}
    </current_user_question>"""
        
        messages.append(HumanMessage(content=current_with_context))
        return messages

    async def analyze_intent(self, user_message: str, file_ids: List[int] = None, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze user intent and determine routing decision
        
        Args:
            user_message: The user's query
            file_ids: Available lesson file IDs
            conversation_history: Previous conversation context
            
        Returns:
            Dict with routing decision and analysis details
        """
        try:
            messages = self._build_message_history(
                conversation_history or [],
                user_message
            )
            
            # Get LLM and analyze intent
            llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_RAG,
                temperature=0.1,
                max_tokens=300
            )
            
            response = await llm.ainvoke(messages)
            response_text = response.content
            
            # Parse JSON response
            try:
                intent_analysis = json.loads(response_text.strip())
                
                # Validate required fields for new format
                required_fields = ["class_period", "agent_to_use", "transformed_query"]
                for field in required_fields:
                    if field not in intent_analysis:
                        intent_analysis[field] = self._get_default_value(field)
                
                # Ensure agent_to_use is valid
                if intent_analysis["agent_to_use"] not in ["general_assistant", "rag_assistant"]:
                    intent_analysis["agent_to_use"] = "general_assistant"
                
                # Ensure class_period is valid
                if intent_analysis["class_period"] not in ["beginning", "middle", "end", None]:
                    intent_analysis["class_period"] = None
                
                return intent_analysis
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response text: {response_text}")
                # Fallback analysis
                return self._fallback_analysis(user_message, file_ids)
                
        except Exception as e:
            print(f"Intent analysis error: {e}")
            return self._fallback_analysis(user_message, file_ids)
    
    def _get_default_value(self, field: str) -> Any:
        """Get default values for missing fields"""
        defaults = {
            "class_period": None,
            "agent_to_use": "general_assistant",
            "transformed_query": None
        }
        return defaults.get(field, None)
    
    def _fallback_analysis(self, user_message: str, file_ids: List[int] = None) -> Dict[str, Any]:
        """Fallback intent analysis using keyword matching"""
        message_lower = user_message.lower()
        
        # Time period keywords
        time_keywords = {
            "beginning": ["beginning", "start", "opening", "first"],
            "middle": ["middle", "during", "in the middle"],
            "end": ["end", "conclusion", "closing", "last", "wrap up"]
        }
        
        # Detect time period
        class_period = None
        for period, keywords in time_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                class_period = period
                break
        
        # Lesson-specific keywords
        lesson_keywords = [
            "show me examples", "specific instances", "what happened", "find examples",
            "detailed analysis", "transcript", "recording", "particular moments"
        ]
        
        # General teaching keywords  
        general_keywords = [
            "how did", "overall", "general feedback", "teaching strategies",
            "broad evaluation", "advice"
        ]
        
        # Determine agent
        lesson_score = sum(1 for keyword in lesson_keywords if keyword in message_lower)
        general_score = sum(1 for keyword in general_keywords if keyword in message_lower)
        
        agent_to_use = "rag_assistant" if lesson_score > general_score else "general_assistant"
        
        return {
            "class_period": class_period,
            "agent_to_use": agent_to_use,
            "transformed_query": user_message  # Use original query as fallback
        }
    
    async def route_query(self, user_message: str, file_ids: List[int] = None, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Route query and return the structured response
        
        Returns:
            Dict with class_period, agent_to_use, and transformed_query
        """
        return await self.analyze_intent(user_message, file_ids, conversation_history)
    
    def get_agent_type(self, analysis: Dict[str, Any]) -> str:
        """
        Extract agent type from analysis
        
        Returns:
            "general" or "rag" for backwards compatibility
        """
        agent = analysis.get("agent_to_use", "general_assistant")
        return "rag" if agent == "rag_assistant" else "general"

# Create instance
intent_analyzer = IntentAnalyzer()