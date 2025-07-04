from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
import re
from app.services.rag import SupabaseRAG, SYSTEM_PROMPT
from app.core.config import settings
from app.models.chat import ChatRequest
from app.utils.time import time_to_seconds

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    file_ids = payload.file_id
    question = payload.question
    top_k = payload.top_k or 3
    if not file_ids or not question:
        return JSONResponse({"error": "file_id and question are required"}, status_code=400)

    rag = SupabaseRAG()
    # Gather all chunks for all file_ids
    all_chunks = []
    for fid in file_ids:
        chunks = rag.get_chunks_from_supabase(fid)
        all_chunks.extend(chunks)
    if not all_chunks:
        return JSONResponse({"error": "No chunks found for the given file_id(s)"}, status_code=404)

    # Dynamic time filter: if question asks for 'first X minutes', filter chunks
    match = re.search(r'first (\d+) minute', question.lower())
    filtered_chunks = all_chunks
    if match:
        minutes = int(match.group(1))
        filtered_chunks = [c for c in all_chunks if time_to_seconds(c.get('start_time', '00:00')) < minutes * 60]
        if not filtered_chunks:
            return JSONResponse({"error": f"No chunks found for the first {minutes} minutes."}, status_code=404)

    # Semantic search
    results = rag.semantic_search(question, filtered_chunks, top_k=top_k)
    if not results:
        return JSONResponse({"error": "No relevant chunks found for your question."}, status_code=404)

    top_chunks = [chunk for score, chunk in results]
    context = rag.format_chunks_for_context(top_chunks)
    user_prompt = f"Here is the relevant lesson chunk data:\n\n{context}\n\nQuestion: {question}\n\nPlease answer the teacher's question using only the information in the chunk(s) above."

    # Instead of streaming, get the full answer and return as JSON
    try:
        response = rag.openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            stream=False
        )
        answer = response.choices[0].message.content if response.choices else "No answer generated."
    except Exception as e:
        answer = f"[Error: {str(e)}]"

    return {"answer": answer}
