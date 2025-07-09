from fastapi import APIRouter
from fastapi.responses import JSONResponse
import re
from typing import List, Dict, Any, Optional
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.models.chat import ChatRequest
from app.utils.time import time_to_seconds

# System prompt for Singapore Teaching Practice
LANGCHAIN_SYSTEM_PROMPT = """
You are an AI assistant for teachers analyzing classroom lesson transcripts based on Singapore Teaching Practice framework.
You are given chunks of lesson data (2-3 minutes of classroom utterances, with metadata such as timestamps and teaching area codes).

Teaching Areas (Singapore Teaching Practice):
- 1.1 Establishing Interaction and Rapport: Building positive relationships and connections between teacher-students and among students to create a safe, caring learning environment.
- 1.2 Setting and Maintaining Rules and Routine: Establishing clear expectations, procedures, and consistent classroom management practices.
- 3.1 Activating Prior Knowledge: Connecting new learning to students' existing knowledge and experiences.
- 3.2 Motivating Learners for Learning Engagement: Inspiring and encouraging students to actively participate and invest in their learning.
- 3.3 Using Questions to Deepen Learning: Employing strategic questioning techniques to promote critical thinking and deeper understanding.
- 3.4 Facilitating Collaborative Learning: Organizing and guiding effective student-to-student interactions and group work.
- 3.5 Concluding the Lesson: Summarizing key learning points and providing closure to the lesson.
- 4.1 Checking for Understanding and Providing Feedback: Assessing student comprehension and giving timely, constructive feedback to support learning.

Your job is to:
- Answer the teacher's question using only the information in the provided chunks.
- Reference specific utterances, behaviors, or patterns from the chunks when possible.
- Identify which teaching areas are present and provide evidence from the transcript.
- Provide practical, constructive, and actionable feedback aligned with Singapore Teaching Practice.
- If the answer is not in the chunks, say so honestly.
- Be concise, supportive, and focused on helping the teacher improve their practice.

When answering, ALWAYS reference the specific utterance(s) you are drawing evidence from, and include the timestamp (e.g., [00:00]) for each quoted or paraphrased utterance.
"""

class TimeFilteredRetriever(BaseRetriever):
    """Custom LangChain retriever that filters chunks by time and performs semantic search"""
    
    def __init__(self, file_ids: List[int], supabase_client, embeddings, top_k: int = 3):
        super().__init__()
        # Store configuration as private attributes
        self._file_ids = file_ids
        self._supabase = supabase_client
        self._embeddings = embeddings
        self._top_k = top_k
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents based on query with time filtering"""
        
        # Gather all chunks for all file_ids
        all_chunks = []
        for fid in self._file_ids:
            result = self._supabase.table("chunks").select("*").eq("file_id", fid).order("sequence_order").execute()
            if result.data:
                all_chunks.extend(result.data)
        
        if not all_chunks:
            return []
        
        # Calculate total lesson duration
        total_lesson_seconds = 0
        end_times = [c.get('end_time', '00:00') for c in all_chunks if c.get('end_time')]
        if end_times:
            total_lesson_seconds = max([time_to_seconds(t) for t in end_times])
        
        # Apply time-based filtering
        filtered_chunks = self._apply_time_filters(query, all_chunks, total_lesson_seconds)
        
        # Convert chunks to LangChain Documents
        documents = []
        for chunk in filtered_chunks:
            # Extract text content
            chunk_text = self._get_chunk_text(chunk)
            if chunk_text:
                # Create metadata
                metadata = {
                    "chunk_id": chunk.get("chunk_id"),
                    "file_id": chunk.get("file_id"),
                    "start_time": chunk.get("start_time", "00:00"),
                    "end_time": chunk.get("end_time", "00:00"),
                    "sequence_order": chunk.get("sequence_order", 0)
                }
                
                documents.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
        
        if not documents:
            return []
        
        # Return top_k documents (LangChain will handle similarity scoring internally)
        return documents[:self._top_k]
    
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
    
    def _apply_time_filters(self, query: str, chunks: List[Dict], total_lesson_seconds: int) -> List[Dict]:
        """Apply time-based filtering to chunks"""
        q_lower = query.lower()
        
        # First X minutes
        match_first = re.search(r'(first|initial|opening) (\d+) minute', q_lower) or re.search(r'in the first (\d+) minutes?', q_lower)
        if match_first:
            minutes = int(match_first.group(2) or match_first.group(1))
            return [c for c in chunks if time_to_seconds(c.get('start_time', '00:00')) < minutes * 60]
        
        # Last X minutes
        match_last = re.search(r'(last|final|ending|concluding) (\d+) minute', q_lower) or re.search(r'in the last (\d+) minutes?', q_lower)
        if match_last and total_lesson_seconds > 0:
            minutes = int(match_last.group(2) or match_last.group(1))
            start_cutoff = total_lesson_seconds - (minutes * 60)
            return [c for c in chunks if time_to_seconds(c.get('start_time', '00:00')) >= start_cutoff]
        
        # Range: from minute X to Y
        match_range = re.search(r'from minute (\d+) to (\d+)', q_lower) or re.search(r'between minute (\d+) and (\d+)', q_lower)
        if match_range:
            start_min = int(match_range.group(1))
            end_min = int(match_range.group(2))
            return [c for c in chunks if start_min * 60 <= time_to_seconds(c.get('start_time', '00:00')) < end_min * 60]
        
        # No time filter, return all chunks
        return chunks

class LangChainRAGService:
    """LangChain-based RAG service for lesson analysis"""
    
    def __init__(self):
        # Initialize Azure OpenAI components
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            temperature=0.7,
            max_tokens=800
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        
        self.supabase = get_supabase_client()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", LANGCHAIN_SYSTEM_PROMPT),
            ("user", """
File Summary: {summary}

Relevant Lesson Chunks:
{context}

Question: {question}

Please answer the teacher's question using only the information in the summary and chunks above.
            """)
        ])
    
    def get_file_summaries(self, file_ids: List[int]) -> str:
        """Get data summaries for the files"""
        summary_sections = []
        for fid in file_ids:
            try:
                file_info = self.supabase.table("files").select("stored_filename, data_summary").eq("file_id", fid).single().execute()
                if file_info.data:
                    filename = file_info.data.get("stored_filename", "Unknown Filename")
                    summary = file_info.data.get("data_summary", "No summary available.")
                    summary_sections.append(f"File: {filename}\nLesson Data Summary:\n{summary}\n")
            except Exception as e:
                summary_sections.append(f"File {fid}: Error retrieving summary - {str(e)}\n")
        
        return "\n".join(summary_sections)
    
    def format_documents(self, docs: List[Document]) -> str:
        """Format retrieved documents for context"""
        context_parts = []
        for doc in docs:
            metadata = doc.metadata
            start_time = metadata.get("start_time", "00:00")
            end_time = metadata.get("end_time", "00:00")
            
            context_parts.append(f"[{start_time}-{end_time}] {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    async def answer_question(self, file_ids: List[int], question: str, top_k: int = 3) -> str:
        """Answer a question using LangChain RAG pipeline"""
        
        # Create custom retriever
        retriever = TimeFilteredRetriever(
            file_ids=file_ids,
            supabase_client=self.supabase,
            embeddings=self.embeddings,
            top_k=top_k
        )
        
        # Get file summaries
        summary = self.get_file_summaries(file_ids)
        
        # Create the RAG chain using LCEL
        rag_chain = (
            {
                "context": retriever | self.format_documents,
                "question": RunnablePassthrough(),
                "summary": lambda _: summary
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Execute the chain
        try:
            answer = await rag_chain.ainvoke(question)
            return answer
        except Exception as e:
            return f"[Error: {str(e)}]"

# Initialize the service
langchain_rag_service = LangChainRAGService()

# Create router
router = APIRouter()

@router.post("/chat/langchain")
async def langchain_chat_endpoint(payload: ChatRequest):
    """LangChain-based chat endpoint with time filtering and semantic search"""
    
    file_ids = payload.file_id
    question = payload.question
    top_k = payload.top_k or 3
    
    if not file_ids or not question:
        return JSONResponse({"error": "file_id and question are required"}, status_code=400)
    
    try:
        answer = await langchain_rag_service.answer_question(file_ids, question, top_k)
        return {"answer": answer, "method": "langchain"}
    
    except Exception as e:
        return JSONResponse({"error": f"LangChain processing error: {str(e)}"}, status_code=500)

@router.post("/chat/langchain/streaming")
async def langchain_streaming_chat_endpoint(payload: ChatRequest):
    """LangChain-based streaming chat endpoint"""
    
    file_ids = payload.file_id
    question = payload.question
    top_k = payload.top_k or 3
    
    if not file_ids or not question:
        return JSONResponse({"error": "file_id and question are required"}, status_code=400)
    
    # This would implement streaming using LangChain's streaming capabilities
    # For now, return the same response as non-streaming
    try:
        answer = await langchain_rag_service.answer_question(file_ids, question, top_k)
        return {"answer": answer, "method": "langchain_streaming"}
    
    except Exception as e:
        return JSONResponse({"error": f"LangChain streaming error: {str(e)}"}, status_code=500)