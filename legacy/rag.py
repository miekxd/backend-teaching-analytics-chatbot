import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from app.db.supabase import get_supabase_client
from app.core.config import settings

SYSTEM_PROMPT = """
You are an AI assistant for teachers analyzing classroom lesson transcripts based on Singapore Teaching Practice framework.
You are given a chunk of lesson data (2-3 minutes of classroom utterances, with metadata such as timestamps and teaching area codes).

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
- Answer the teacher's question using only the information in the provided chunk(s).
- Reference specific utterances, behaviors, or patterns from the chunk(s) when possible.
- Identify which teaching areas are present and provide evidence from the transcript.
- Provide practical, constructive, and actionable feedback aligned with Singapore Teaching Practice.
- If the answer is not in the chunk(s), say so honestly.
- Be concise, supportive, and focused on helping the teacher improve their practice.

When answering, ALWAYS reference the specific utterance(s) you are drawing evidence from, and include the timestamp (e.g., [00:00]) for each quoted or paraphrased utterance. If you summarize, still cite the relevant timestamps. Do not omit timestamps when providing evidence or examples.
"""

class SupabaseRAG:
    def __init__(self, file_id: Optional[int] = None):
        self.supabase = get_supabase_client()
        self.file_id = file_id
        self.openai_client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,  # type: ignore
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        print("✅ Initialized Supabase RAG system with streaming")

    def get_chunks_from_supabase(self, file_id: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            target_file_id = file_id or self.file_id
            if not target_file_id:
                print("❌ No file_id specified")
                return []
            result = self.supabase.table("chunks").select("*").eq("file_id", target_file_id).order("sequence_order").execute()
            if not result.data:
                print(f"❌ No chunks found for file_id {target_file_id}")
                return []
            print(f"✅ Loaded {len(result.data)} chunks from Supabase")
            return result.data
        except Exception as e:
            print(f"❌ Error loading chunks from Supabase: {e}")
            return []

    def get_chunk_text(self, chunk: Dict[str, Any]) -> str:
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

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.openai_client.embeddings.create(
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return []

    def parse_embedding(self, embedding_data) -> List[float]:
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
            print(f"⚠️  Error parsing embedding: {e}")
            return []

    def semantic_search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[tuple]:
        print(f"🔍 Semantic search for: '{query}'")
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        chunk_embeddings = []
        valid_chunks = []
        for chunk in chunks:
            if 'embedding' in chunk and chunk['embedding']:
                parsed_embedding = self.parse_embedding(chunk['embedding'])
                if parsed_embedding:
                    chunk_embeddings.append(parsed_embedding)
                    valid_chunks.append(chunk)
            else:
                chunk_text = self.get_chunk_text(chunk)
                if chunk_text:
                    chunk_embedding = self.get_embedding(chunk_text)
                    if chunk_embedding:
                        chunk_embeddings.append(chunk_embedding)
                        valid_chunks.append(chunk)
                    time.sleep(0.1)
        if not chunk_embeddings:
            print("❌ No valid embeddings found")
            return []
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((similarities[idx], valid_chunks[idx]))
        return results

    def format_chunks_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for chunk in chunks:
            chunk_info = [
                f"Chunk: {chunk.get('chunk_id', 'Unknown')}",
                f"Time: {chunk.get('start_time', 'Unknown')} - {chunk.get('end_time', 'Unknown')}",
                f"Teaching Areas: {', '.join(chunk.get('teaching_areas', []))}",
                f"Dominant Area: {chunk.get('dominant_area', 'Unknown')}",
                f"Content: {self.get_chunk_text(chunk)}",
                ""
            ]
            if 'utterances' in chunk and chunk['utterances']:
                chunk_info.append("Detailed Utterances:")
                for i, utterance in enumerate(chunk['utterances'], 1):
                    if isinstance(utterance, dict):
                        utterance_text = utterance.get('text', '')
                        timestamp = utterance.get('timestamp', '')
                        area = utterance.get('area', '')
                        chunk_info.append(f"  {i}. [{timestamp}] {utterance_text} (Area: {area})")
            context_parts.append("\n".join(chunk_info))
            context_parts.append("---")
        return "\n".join(context_parts)

    def get_data_summary(self, file_id: Optional[int] = None) -> Optional[str]:
        try:
            target_file_id = file_id or self.file_id
            if not target_file_id:
                print("❌ No file_id specified for data summary")
                return None
            result = self.supabase.table("files").select("data_summary").eq("file_id", target_file_id).single().execute()
            if not result.data or "data_summary" not in result.data:
                print(f"❌ No data_summary found for file_id {target_file_id}")
                return None
            print(f"✅ Loaded data_summary for file_id {target_file_id}")
            return result.data["data_summary"]
        except Exception as e:
            print(f"❌ Error loading data_summary from Supabase: {e}")
            return None

    def answer_question(self, question: str) -> str:
        """
        Hybrid approach: Uses data_summary for summary/statistics questions, and chunk retrieval for detailed/contextual questions.
        Combines both for ambiguous or broad questions.
        """
        summary_keywords = [
            "how many", "total", "overall", "summary", "percentage", "most", "least", "statistics", "insights"
        ]
        use_summary = any(kw in question.lower() for kw in summary_keywords)

        answer_parts = []

        if use_summary:
            summary = self.get_data_summary()
            if summary:
                answer_parts.append("**Lesson Summary:**\n" + summary)
            else:
                answer_parts.append("No summary data available for this file.")

        # Always try to retrieve relevant chunks for detailed/contextual answers
        chunks = self.get_chunks_from_supabase()
        if chunks:
            # Use your existing semantic search or chunk analysis here
            # For demonstration, let's just add a placeholder
            answer_parts.append("**Relevant Examples from Transcript:**\n[chunk-based answer here]")
        else:
            answer_parts.append("No transcript chunks available for this file.")

        return "\n\n".join(answer_parts)
