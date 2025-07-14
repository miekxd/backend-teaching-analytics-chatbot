from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Basic LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.utils.time import time_to_seconds
from openai import AzureOpenAI

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token"""
        self.tokens.append(token)

class UnifiedTeachingAssistant:
    """
    Unified teaching assistant that always retrieves context but conditionally uses it
    
    Strategy: Always Retrieve, Conditionally Use
    - Always fetches lesson context and chunks for any query
    - Intelligently determines when to use RAG vs direct conversation
    - Seamlessly blends evidence-based and conversational responses
    
    Features:
    1. Query classification (lesson-specific vs general)
    2. Always retrieves available lesson context
    3. Conditional RAG application based on query relevance
    4. Maintains conversation context throughout
    5. Singapore Teaching Practice framework integration
    6. Streaming support for all response types
    """
    
    def __init__(self):
        # Azure OpenAI connection for LangChain
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_RAG,
            temperature=0.3,
            max_tokens=800
        )
        
        # OpenAI client for embeddings
        self.openai_client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        # Supabase client for accessing chunks
        self.supabase = get_supabase_client()
        
        # System prompt for unified responses
        self.system_prompt = """You are a versatile AI teaching coach for Singapore educators, capable of both detailed lesson analysis and general educational conversation.

<role>
You are an expert in the Singapore Teaching Practice framework who can:
1. Provide evidence-based analysis when lesson data is available and relevant
2. Engage in general educational conversations and coaching
3. Seamlessly blend lesson-specific insights with broader teaching expertise
4. Adapt your response style based on the query type and available context
</role>

<response_modes>
<lesson_analysis_mode>
Use when: Query relates to specific lesson content, teaching moments, or classroom interactions
Approach:
- Prioritize specific evidence from lesson chunks with timestamps [MM:SS]
- Quote directly from transcripts when highlighting behaviors
- Connect observations to Singapore Teaching Practice framework
- Provide concrete, evidence-based feedback and suggestions
</lesson_analysis_mode>

<conversational_mode>
Use when: Query is general teaching advice, pedagogical concepts, or broad educational topics
Approach:
- Draw from general teaching expertise and best practices
- Reference Singapore Teaching Practice framework principles
- Provide practical strategies and examples
- Maintain supportive, coaching tone
- Use lesson context as supporting examples if relevant
</conversational_mode>

<blended_mode>
Use when: Query could benefit from both lesson evidence and general expertise
Approach:
- Start with any relevant lesson evidence if available
- Expand with general teaching principles and strategies
- Connect specific examples to broader educational concepts
- Provide both immediate and long-term guidance
</blended_mode>
</response_modes>

<singapore_teaching_framework>
<teaching_areas>
- 1.1 Establishing Interaction and Rapport: Building positive relationships and connections
- 1.2 Setting and Maintaining Rules and Routine: Clear expectations and classroom management
- 3.1 Activating Prior Knowledge: Connecting to existing knowledge and experiences
- 3.2 Motivating Learners for Learning Engagement: Inspiring active participation
- 3.3 Using Questions to Deepen Learning: Strategic questioning for critical thinking
- 3.4 Facilitating Collaborative Learning: Effective student-to-student interactions
- 3.5 Concluding the Lesson: Summarizing and providing closure
- 4.1 Checking for Understanding and Providing Feedback: Assessment and constructive feedback
</teaching_areas>
</singapore_teaching_framework>

<adaptive_response_guidelines>
<context_assessment>
1. Evaluate the query type and intent
2. Determine relevance of available lesson data
3. Choose appropriate response mode (analysis/conversational/blended)
4. Decide whether to lead with evidence or expertise
</context_assessment>

<evidence_integration>
- When lesson data is relevant: Use specific timestamps and quotes
- When lesson data is supportive: Reference as examples
- When lesson data is irrelevant: Focus on general expertise
- Always be transparent about the basis for your response
</evidence_integration>

<conversation_continuity>
- Maintain context from previous exchanges
- Build on established themes and concerns
- Reference earlier points when relevant
- Keep the coaching relationship consistent
</conversation_continuity>
</adaptive_response_guidelines>

<output_requirements>
- Be helpful and relevant regardless of query type
- Use markdown formatting for clear structure
- Reference Singapore Teaching Practice areas when applicable
- Provide actionable insights and strategies
- Maintain warm, supportive coaching tone
- Clearly indicate when using lesson evidence vs general expertise
</output_requirements>"""

        # Prompt template for unified responses
        self.unified_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """<available_context>
<lesson_overview>
{lesson_summary}
</lesson_overview>

<lesson_chunks>
{context_chunks}
</lesson_chunks>

<query_analysis>
Query Type: {query_type}
Lesson Relevance: {lesson_relevance}
Recommended Mode: {response_mode}
</query_analysis>
</available_context>

<teacher_query>
{question}
</teacher_query>

<response_instructions>
Based on the query analysis above:

1. **If lesson_relevance is HIGH**: Lead with specific evidence from chunks, use timestamps and quotes
2. **If lesson_relevance is MEDIUM**: Blend lesson examples with general expertise  
3. **If lesson_relevance is LOW**: Focus on general teaching expertise, use lesson context as supporting examples if helpful

Adapt your response style to:
- Query type: {query_type}
- Available context: {"Rich lesson data" if context_chunks != "No detailed transcript chunks available." else "Limited lesson data"}
- Recommended approach: {response_mode}

Always maintain a supportive, expert coaching tone and reference Singapore Teaching Practice areas when relevant.
</response_instructions>""")
        ])
        
        # Output parser
        self.output_parser = StrOutputParser()
        
        # Chain for unified responses
        self.unified_chain = self.unified_prompt | self.llm | self.output_parser
    
    def _analyze_query(self, question: str) -> Dict[str, str]:
        """
        Analyze the query to determine type, lesson relevance, and response mode
        Returns classification for conditional RAG usage
        """
        q_lower = question.lower()
        
        # Keywords that indicate lesson-specific queries
        lesson_specific_keywords = [
            'in this lesson', 'during the lesson', 'my lesson', 'the class', 'my students',
            'at minute', 'first', 'last', 'beginning', 'end', 'transcript', 'recording',
            'what did i', 'how did i', 'when i said', 'student response', 'classroom interaction'
        ]
        
        # Keywords that indicate time-based queries
        time_keywords = [
            'minute', 'first', 'last', 'beginning', 'start', 'end', 'opening', 'closing',
            'from minute', 'to minute', 'during', 'throughout'
        ]
        
        # Keywords that indicate general teaching queries
        general_keywords = [
            'how to', 'what is', 'best practice', 'strategy', 'technique', 'method',
            'generally', 'in general', 'advice', 'suggestion', 'recommend', 'should i'
        ]
        
        # Determine query type
        query_type = "general"
        if any(keyword in q_lower for keyword in lesson_specific_keywords):
            query_type = "lesson_specific"
        elif any(keyword in q_lower for keyword in time_keywords):
            query_type = "time_based"
        
        # Determine lesson relevance
        lesson_relevance = "LOW"
        if query_type in ["lesson_specific", "time_based"]:
            lesson_relevance = "HIGH"
        elif any(keyword in q_lower for keyword in ['classroom', 'student', 'teaching']):
            lesson_relevance = "MEDIUM"
        
        # Determine response mode
        response_mode = "conversational"
        if lesson_relevance == "HIGH":
            response_mode = "lesson_analysis"
        elif lesson_relevance == "MEDIUM":
            response_mode = "blended"
        
        return {
            "query_type": query_type,
            "lesson_relevance": lesson_relevance,
            "response_mode": response_mode
        }
    
    def _get_chunks_from_supabase(self, file_id: int) -> List[Dict[str, Any]]:
        """Get chunks from Supabase for a specific file"""
        try:
            result = self.supabase.table("chunks").select("*").eq("file_id", file_id).order("sequence_order").execute()
            if not result.data:
                return []
            return result.data
        except Exception as e:
            print(f"❌ Error loading chunks from Supabase: {e}")
            return []
    
    def _get_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Extract text from chunk"""
        if 'chunk_text' in chunk and chunk['chunk_text']:
            return chunk['chunk_text']
        if 'utterances' in chunk and chunk['utterances']:
            utterances = chunk['utterances']
            if isinstance(utterances, list):
                texts = []
                for utterance in utterances:
                    if isinstance(utterance, dict) and 'text' in utterance:
                        texts.append(str(utterance['text']))
                return " ".join(texts)
        return ""
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return []
    
    def _parse_embedding(self, embedding_data) -> List[float]:
        """Parse embedding data from various formats"""
        if not embedding_data:
            return []
        try:
            if isinstance(embedding_data, list):
                return [float(x) for x in embedding_data]
            if isinstance(embedding_data, str):
                if embedding_data.startswith("np.str_("):
                    start = embedding_data.find("'[")
                    end = embedding_data.rfind("]'")
                    if start != -1 and end != -1:
                        embedding_data = embedding_data[start+1:end+1]
                import ast
                try:
                    parsed = ast.literal_eval(embedding_data)
                    if isinstance(parsed, list):
                        return [float(x) for x in parsed]
                except:
                    import json
                    try:
                        parsed = json.loads(embedding_data)
                        if isinstance(parsed, list):
                            return [float(x) for x in parsed]
                    except:
                        pass
            return []
        except Exception as e:
            print(f"⚠️ Error parsing embedding: {e}")
            return []
    
    def _apply_time_filters(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Apply time-based filtering to chunks"""
        q_lower = question.lower()
        
        # First X minutes
        match_first = re.search(r'(first|initial|opening) (\d+) minute', q_lower)
        if match_first:
            minutes = int(match_first.group(2))
            return [c for c in chunks if time_to_seconds(c.get('start_time', '00:00')) < minutes * 60]
        
        # Last X minutes  
        match_last = re.search(r'(last|final|ending) (\d+) minute', q_lower)
        if match_last:
            minutes = int(match_last.group(2))
            # Calculate cutoff based on max end time
            max_time = max([time_to_seconds(c.get('end_time', '00:00')) for c in chunks if c.get('end_time')], default=0)
            start_cutoff = max_time - (minutes * 60)
            return [c for c in chunks if time_to_seconds(c.get('start_time', '00:00')) >= start_cutoff]
        
        # Range: from minute X to Y
        match_range = re.search(r'from minute (\d+) to (\d+)', q_lower)
        if match_range:
            start_min = int(match_range.group(1))
            end_min = int(match_range.group(2))
            return [c for c in chunks if start_min * 60 <= time_to_seconds(c.get('start_time', '00:00')) < end_min * 60]
        
        return chunks
    
    def _semantic_search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic search on chunks and return top chunks"""
        if not chunks:
            return []
            
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return chunks[:top_k]  # Fallback to first chunks if embedding fails
        
        similarities = []
        for chunk in chunks:
            # Try to get existing embedding or generate new one
            chunk_embedding = None
            if 'embedding' in chunk and chunk['embedding']:
                chunk_embedding = self._parse_embedding(chunk['embedding'])
            
            # Generate embedding if not available
            if not chunk_embedding:
                chunk_text = self._get_chunk_text(chunk)
                if chunk_text:
                    chunk_embedding = self._get_embedding(chunk_text)
                    time.sleep(0.1)  # Rate limiting
            
            if chunk_embedding:
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                similarities.append((similarity, chunk))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]
    
    def _format_chunks_for_context(self, chunks: List[Dict[str, Any]], relevance: str) -> str:
        """Format chunks for context based on relevance level"""
        if not chunks:
            return "No detailed transcript chunks available."
        
        # Adjust detail level based on relevance
        max_chunks = 5 if relevance == "HIGH" else 3 if relevance == "MEDIUM" else 1
        chunks_to_use = chunks[:max_chunks]
        
        context_parts = []
        for chunk in chunks_to_use:
            chunk_info = [
                f"Time: {chunk.get('start_time', 'Unknown')} - {chunk.get('end_time', 'Unknown')}",
                f"Teaching Areas: {', '.join(chunk.get('teaching_areas', []))}",
                f"Content: {self._get_chunk_text(chunk)}"
            ]
            
            # Add detailed utterances only for high relevance queries
            if relevance == "HIGH" and 'utterances' in chunk and chunk['utterances']:
                chunk_info.append("Detailed Utterances:")
                for i, utterance in enumerate(chunk['utterances'], 1):
                    if isinstance(utterance, dict):
                        timestamp = utterance.get('timestamp', '')
                        text = utterance.get('text', '')
                        area = utterance.get('area', '')
                        chunk_info.append(f"  {i}. [{timestamp}] {text} (Area: {area})")
            
            context_parts.append("\n".join(chunk_info))
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
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
    
    async def get_response(self, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None, top_k: int = 5) -> str:
        """
        Get unified response - always retrieves context, conditionally uses RAG
        Returns simple text response
        """
        try:
            # Step 1: Always analyze the query
            query_analysis = self._analyze_query(user_message)
            
            # Step 2: Always retrieve lesson context
            lesson_summaries = self._get_file_summaries(file_ids)
            
            # Step 3: Always get chunks but adjust processing based on relevance
            all_chunks = []
            for fid in file_ids:
                chunks = self._get_chunks_from_supabase(fid)
                all_chunks.extend(chunks)
            
            # Step 4: Process chunks based on query relevance
            context_chunks = "No detailed transcript chunks available."
            if all_chunks and query_analysis["lesson_relevance"] in ["HIGH", "MEDIUM"]:
                # Apply time filtering for lesson-specific queries
                if query_analysis["query_type"] in ["lesson_specific", "time_based"]:
                    filtered_chunks = self._apply_time_filters(user_message, all_chunks)
                else:
                    filtered_chunks = all_chunks
                
                if filtered_chunks:
                    # Perform semantic search with relevance-adjusted parameters
                    search_k = top_k if query_analysis["lesson_relevance"] == "HIGH" else min(3, top_k)
                    top_chunks = self._semantic_search(user_message, filtered_chunks, top_k=search_k)
                    if top_chunks:
                        context_chunks = self._format_chunks_for_context(top_chunks, query_analysis["lesson_relevance"])
            
            # Step 5: Generate response with unified approach
            if conversation_history:
                # Build message history manually for conversation context
                messages = [SystemMessage(content=self.system_prompt)]
                
                # Add conversation history (limit to last 6 messages)
                for msg in conversation_history[-3:]:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))
                
                # Add current message with full context and analysis
                current_with_context = f"""<available_context>
<lesson_overview>
{lesson_summaries}
</lesson_overview>

<lesson_chunks>
{context_chunks}
</lesson_chunks>

<query_analysis>
Query Type: {query_analysis["query_type"]}
Lesson Relevance: {query_analysis["lesson_relevance"]}
Recommended Mode: {query_analysis["response_mode"]}
</query_analysis>
</available_context>

<teacher_query>
{user_message}
</teacher_query>

<response_instructions>
Based on the query analysis above:

1. **If lesson_relevance is HIGH**: Lead with specific evidence from chunks, use timestamps and quotes
2. **If lesson_relevance is MEDIUM**: Blend lesson examples with general expertise  
3. **If lesson_relevance is LOW**: Focus on general teaching expertise, use lesson context as supporting examples if helpful

Adapt your response style to:
- Query type: {query_analysis["query_type"]}
- Available context: {"Rich lesson data" if context_chunks != "No detailed transcript chunks available." else "Limited lesson data"}
- Recommended approach: {query_analysis["response_mode"]}

Always maintain a supportive, expert coaching tone and reference Singapore Teaching Practice areas when relevant.
</response_instructions>"""
                
                messages.append(HumanMessage(content=current_with_context))
                response = await self.llm.ainvoke(messages)
                return response.content
            else:
                # Use unified chain
                response_text = await self.unified_chain.ainvoke({
                    "lesson_summary": lesson_summaries,
                    "context_chunks": context_chunks,
                    "query_type": query_analysis["query_type"],
                    "lesson_relevance": query_analysis["lesson_relevance"],
                    "response_mode": query_analysis["response_mode"],
                    "question": user_message
                })
                return response_text
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    async def get_response_stream(self, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None, top_k: int = 5) -> AsyncGenerator[str, None]:
        """
        Stream unified response - always retrieves context, conditionally uses RAG
        Yields text chunks as they're generated
        """
        try:
            # Step 1: Always analyze the query
            query_analysis = self._analyze_query(user_message)
            
            # Step 2: Always retrieve lesson context
            lesson_summaries = self._get_file_summaries(file_ids)
            
            # Step 3: Always get chunks but adjust processing based on relevance
            all_chunks = []
            for fid in file_ids:
                chunks = self._get_chunks_from_supabase(fid)
                all_chunks.extend(chunks)
            
            # Step 4: Process chunks based on query relevance
            context_chunks = "No detailed transcript chunks available."
            if all_chunks and query_analysis["lesson_relevance"] in ["HIGH", "MEDIUM"]:
                # Apply time filtering for lesson-specific queries
                if query_analysis["query_type"] in ["lesson_specific", "time_based"]:
                    filtered_chunks = self._apply_time_filters(user_message, all_chunks)
                else:
                    filtered_chunks = all_chunks
                
                if filtered_chunks:
                    # Perform semantic search with relevance-adjusted parameters
                    search_k = top_k if query_analysis["lesson_relevance"] == "HIGH" else min(3, top_k)
                    top_chunks = self._semantic_search(user_message, filtered_chunks, top_k=search_k)
                    if top_chunks:
                        context_chunks = self._format_chunks_for_context(top_chunks, query_analysis["lesson_relevance"])
            
            # Setup streaming LLM
            streaming_llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_RAG,
                temperature=0.3,
                max_tokens=800,
                streaming=True
            )
            
            # Step 5: Generate streaming response with unified approach
            if conversation_history:
                # Build message history manually for conversation context
                messages = [SystemMessage(content=self.system_prompt)]
                
                # Add conversation history (limit to last 6 messages)
                for msg in conversation_history[-3:]:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))
                
                # Add current message with full context and analysis
                current_with_context = f"""<available_context>
<lesson_overview>
{lesson_summaries}
</lesson_overview>

<lesson_chunks>
{context_chunks}
</lesson_chunks>

<query_analysis>
Query Type: {query_analysis["query_type"]}
Lesson Relevance: {query_analysis["lesson_relevance"]}
Recommended Mode: {query_analysis["response_mode"]}
</query_analysis>
</available_context>

<teacher_query>
{user_message}
</teacher_query>

<response_instructions>
Based on the query analysis above:

1. **If lesson_relevance is HIGH**: Lead with specific evidence from chunks, use timestamps and quotes
2. **If lesson_relevance is MEDIUM**: Blend lesson examples with general expertise  
3. **If lesson_relevance is LOW**: Focus on general teaching expertise, use lesson context as supporting examples if helpful

Adapt your response style to:
- Query type: {query_analysis["query_type"]}
- Available context: {"Rich lesson data" if context_chunks != "No detailed transcript chunks available." else "Limited lesson data"}
- Recommended approach: {query_analysis["response_mode"]}

Always maintain a supportive, expert coaching tone and reference Singapore Teaching Practice areas when relevant.
</response_instructions>"""
                
                messages.append(HumanMessage(content=current_with_context))
                
                # Stream the response
                async for chunk in streaming_llm.astream(messages):
                    if chunk.content:
                        yield chunk.content
            else:
                # Use streaming chain
                streaming_chain = self.unified_prompt | streaming_llm | self.output_parser
                
                async for chunk in streaming_chain.astream({
                    "lesson_summary": lesson_summaries,
                    "context_chunks": context_chunks,
                    "query_type": query_analysis["query_type"],
                    "lesson_relevance": query_analysis["lesson_relevance"],
                    "response_mode": query_analysis["response_mode"],
                    "question": user_message
                }):
                    if chunk:
                        yield chunk
            
        except Exception as e:
            yield f"I apologize, but I encountered an error while processing your request: {str(e)}"

# Create instance
unified_assistant = UnifiedTeachingAssistant()