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

class RAGTeachingAssistant:
    """
    RAG-based teaching assistant with detailed transcript analysis and streaming support
    
    Features:
    1. Semantic search across lesson chunks
    2. Time-based filtering (first X minutes, last X minutes, ranges)
    3. Detailed utterance analysis with timestamps
    4. Singapore Teaching Practice framework alignment
    5. Streaming text responses
    """
    
    def __init__(self):
        # Azure OpenAI connection for LangChain
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            temperature=0.7,
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
        
        # System prompt for RAG responses
        self.system_prompt = """You are a specialized AI teaching coach for Singapore educators, designed to provide detailed, evidence-based analysis of classroom lesson transcripts.

<role>
You are an expert in the Singapore Teaching Practice framework with deep knowledge of effective classroom instruction. Your role is to help teachers reflect on their lessons using concrete evidence when available, and educational expertise when evidence is limited.
</role>

<data_context>
You may receive:
- Detailed lesson chunks with timestamps and utterances (preferred for specific analysis)
- Limited or no chunk data (requiring inference and general guidance)
- Lesson summaries and contextual information
</data_context>

<singapore_teaching_framework>
<teaching_areas>
- 1.1 Establishing Interaction and Rapport: Building positive relationships and connections between teacher-students and among students to create a safe, caring learning environment
- 1.2 Setting and Maintaining Rules and Routine: Establishing clear expectations, procedures, and consistent classroom management practices
- 3.1 Activating Prior Knowledge: Connecting new learning to students' existing knowledge and experiences
- 3.2 Motivating Learners for Learning Engagement: Inspiring and encouraging students to actively participate and invest in their learning
- 3.3 Using Questions to Deepen Learning: Employing strategic questioning techniques to promote critical thinking and deeper understanding
- 3.4 Facilitating Collaborative Learning: Organizing and guiding effective student-to-student interactions and group work
- 3.5 Concluding the Lesson: Summarizing key learning points and providing closure to the lesson
- 4.1 Checking for Understanding and Providing Feedback: Assessing student comprehension and giving timely, constructive feedback to support learning
</teaching_areas>
</singapore_teaching_framework>

<response_approach>
<when_chunks_available>
- ALWAYS cite specific utterances with exact timestamps [MM:SS]
- Quote directly when highlighting specific teacher or student language
- Reference observable behaviors and interactions from the transcript
- Connect evidence to relevant Singapore Teaching Practice areas
- Focus on what is explicitly demonstrated in the data
</when_chunks_available>

<when_chunks_limited_or_unavailable>
- Draw from lesson summary and any available contextual information
- Make reasonable educational inferences based on Singapore Teaching Practice framework
- Provide general guidance relevant to the teacher's question
- Use phrases like "Based on typical classroom situations..." or "Generally speaking..."
- Offer practical strategies and examples even without specific evidence
- Acknowledge when you're providing general guidance vs. specific evidence
</when_chunks_limited_or_unavailable>
</response_approach>

<analysis_requirements>
<evidence_standards>
- With chunks: Always cite timestamps and specific utterances
- Without chunks: Use available context and educational expertise
- Clearly indicate the basis for your response (evidence vs. inference)
- Connect observations or suggestions to Singapore Teaching Practice areas
- Maintain helpful, constructive tone regardless of data availability
</evidence_standards>

<response_structure>
1. Assess what information is available in the provided data
2. If chunks available: Provide evidence-based analysis with timestamps
3. If chunks unavailable: Offer informed guidance based on context and expertise
4. Connect to relevant Singapore Teaching Practice areas
5. Provide actionable feedback appropriate to the information available
</response_structure>
</analysis_requirements>

<response_guidelines>
<evidence_citation>
- With detailed chunks: "At [MM:SS], you said '[exact quote]'"
- With limited data: "Based on the lesson context..." or "Typically in this situation..."
- Always be transparent about the basis for your response
</evidence_citation>

<inference_guidance>
- Use educational best practices when specific evidence isn't available
- Draw reasonable conclusions from lesson summaries and context
- Provide practical strategies relevant to the teacher's question
- Reference common classroom scenarios and effective teaching approaches
- Maintain focus on Singapore Teaching Practice framework
</inference_guidance>

<feedback_principles>
- Be helpful regardless of data limitations
- Provide specific, actionable advice when possible
- Use general teaching expertise when specific evidence is lacking
- Balance recognition of what's working with growth opportunities
- Maintain supportive, professional coaching tone
</feedback_principles>
</response_guidelines>

<output_requirements>
- Reference Singapore Teaching Practice areas by code when relevant
- Provide helpful guidance whether based on evidence or expertise
- Be transparent about the basis for your response
- Focus on actionable insights for teacher development
- Maintain constructive, supportive tone
</output_requirements>"""

        # Prompt template for RAG responses
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """<lesson_overview>
{lesson_summary}
</lesson_overview>

<lesson_chunks>
{context}
</lesson_chunks>

<teacher_question>
{question}
</teacher_question>

<analysis_instructions>
1. Examine what data is available (detailed chunks, summary only, or limited information)
2. If detailed chunks are provided:
   - Cite specific utterances and behaviors with exact timestamps
   - Provide evidence-based analysis of Singapore Teaching Practice areas
3. If chunks are limited or unavailable:
   - Use the lesson summary and any available context
   - Draw from teaching expertise and Singapore Teaching Practice framework
   - Provide practical guidance and strategies relevant to the question
   - Be transparent that you're providing general guidance rather than specific evidence
4. Focus on being helpful and actionable regardless of data availability
5. Connect your response to relevant Singapore Teaching Practice areas
</analysis_instructions>""")
        ])
        
        # Output parser
        self.output_parser = StrOutputParser()
        
        # Chain for RAG responses
        self.rag_chain = self.rag_prompt | self.llm | self.output_parser
    
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
    
    def _format_chunks_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for context in prompt"""
        if not chunks:
            return "No detailed transcript chunks available."
            
        context_parts = []
        for chunk in chunks:
            chunk_info = [
                f"Time: {chunk.get('start_time', 'Unknown')} - {chunk.get('end_time', 'Unknown')}",
                f"Teaching Areas: {', '.join(chunk.get('teaching_areas', []))}",
                f"Content: {self._get_chunk_text(chunk)}"
            ]
            
            # Add detailed utterances if available
            if 'utterances' in chunk and chunk['utterances']:
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
    
    async def get_response(self, semantic_query: str, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None, top_k: int = 5) -> str:
        """
        Get RAG response with detailed transcript analysis
        Returns simple text response
        """
        try:
            # Get lesson summaries (always available)
            lesson_summaries = self._get_file_summaries(file_ids)
            
            # Try to get chunks - don't fail if unavailable
            all_chunks = []
            for fid in file_ids:
                chunks = self._get_chunks_from_supabase(fid)
                all_chunks.extend(chunks)
            
            # Process chunks if available
            context = "No detailed transcript chunks available."
            if all_chunks:
                # Apply time filtering
                filtered_chunks = self._apply_time_filters(user_message, all_chunks)
                
                if filtered_chunks:
                    # Perform semantic search
                    top_chunks = self._semantic_search(semantic_query, filtered_chunks, top_k=top_k)
                    if top_chunks:
                        context = self._format_chunks_for_context(top_chunks)
            
            # Generate response using available data
            if conversation_history:
                # Build message history manually for conversation context
                messages = [SystemMessage(content=self.system_prompt)]
                
                # Add conversation history (limit to last 6 messages)
                for msg in conversation_history[-6:]:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))
                
                # Add current message with context
                current_with_context = f"""<lesson_overview>
{lesson_summaries}
</lesson_overview>

<lesson_chunks>
{context}
</lesson_chunks>

<teacher_question>
{user_message}
</teacher_question>

<analysis_instructions>
1. Examine what data is available (detailed chunks, summary only, or limited information)
2. If detailed chunks are provided:
   - Cite specific utterances and behaviors with exact timestamps
   - Provide evidence-based analysis of Singapore Teaching Practice areas
3. If chunks are limited or unavailable:
   - Use the lesson summary and any available context
   - Draw from teaching expertise and Singapore Teaching Practice framework
   - Provide practical guidance and strategies relevant to the question
   - Be transparent that you're providing general guidance rather than specific evidence
4. Focus on being helpful and actionable regardless of data availability
5. Connect your response to relevant Singapore Teaching Practice areas
</analysis_instructions>"""
                
                messages.append(HumanMessage(content=current_with_context))
                response = await self.llm.ainvoke(messages)
                return response.content
            else:
                # Use RAG chain
                response_text = await self.rag_chain.ainvoke({
                    "lesson_summary": lesson_summaries,
                    "context": context,
                    "question": user_message
                })
                return response_text
            
        except Exception as e:
            return f"I apologize, but I encountered an error while analyzing the lesson: {str(e)}"
    
    async def get_response_stream(self, semantic_query: str, user_message: str, file_ids: List[int], conversation_history: List[Dict[str, str]] = None, top_k: int = 5) -> AsyncGenerator[str, None]:
        """
        Stream RAG response with detailed transcript analysis
        Yields text chunks as they're generated
        """
        try:
            # Get lesson summaries (always available)
            lesson_summaries = self._get_file_summaries(file_ids)
            
            # Try to get chunks - don't fail if unavailable
            all_chunks = []
            for fid in file_ids:
                chunks = self._get_chunks_from_supabase(fid)
                all_chunks.extend(chunks)
            
            # Process chunks if available
            context = "No detailed transcript chunks available."
            if all_chunks:
                # Apply time filtering
                filtered_chunks = self._apply_time_filters(user_message, all_chunks)
                
                if filtered_chunks:
                    # Perform semantic search
                    top_chunks = self._semantic_search(semantic_query, filtered_chunks, top_k=top_k)
                    if top_chunks:
                        context = self._format_chunks_for_context(top_chunks)
            
            # Setup streaming LLM
            streaming_llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                temperature=0.7,
                max_tokens=800,
                streaming=True
            )
            
            # Generate streaming response
            if conversation_history:
                # Build message history manually for conversation context
                messages = [SystemMessage(content=self.system_prompt)]
                
                # Add conversation history (limit to last 6 messages)
                for msg in conversation_history[-6:]:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    
                    if role == "user" and content:
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant" and content:
                        messages.append(AIMessage(content=content))
                
                # Add current message with context
                current_with_context = f"""<lesson_overview>
{lesson_summaries}
</lesson_overview>

<lesson_chunks>
{context}
</lesson_chunks>

<teacher_question>
{user_message}
</teacher_question>

<analysis_instructions>
1. Examine what data is available (detailed chunks, summary only, or limited information)
2. If detailed chunks are provided:
   - Cite specific utterances and behaviors with exact timestamps
   - Provide evidence-based analysis of Singapore Teaching Practice areas
3. If chunks are limited or unavailable:
   - Use the lesson summary and any available context
   - Draw from teaching expertise and Singapore Teaching Practice framework
   - Provide practical guidance and strategies relevant to the question
   - Be transparent that you're providing general guidance rather than specific evidence
4. Focus on being helpful and actionable regardless of data availability
5. Connect your response to relevant Singapore Teaching Practice areas
</analysis_instructions>"""
                
                messages.append(HumanMessage(content=current_with_context))
                
                # Stream the response
                async for chunk in streaming_llm.astream(messages):
                    if chunk.content:
                        yield chunk.content
            else:
                # Use streaming chain
                streaming_chain = self.rag_prompt | streaming_llm | self.output_parser
                
                async for chunk in streaming_chain.astream({
                    "lesson_summary": lesson_summaries,
                    "context": context,
                    "question": user_message
                }):
                    if chunk:
                        yield chunk
            
        except Exception as e:
            yield f"I apologize, but I encountered an error while analyzing the lesson: {str(e)}"

# Create instance
rag_assistant = RAGTeachingAssistant()
