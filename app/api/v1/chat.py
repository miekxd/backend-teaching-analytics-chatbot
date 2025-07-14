from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
import re
from legacy.rag import SupabaseRAG, SYSTEM_PROMPT
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

    # Enhanced time filter: support 'first X minutes', 'last X minutes', and ranges
    total_lesson_seconds = 0
    if all_chunks:
        # Estimate total lesson duration from the latest end_time
        end_times = [c.get('end_time', '00:00') for c in all_chunks if c.get('end_time')]
        if end_times:
            total_lesson_seconds = max([time_to_seconds(t) for t in end_times])

    filtered_chunks = all_chunks
    q_lower = question.lower()
    # First X minutes
    match_first = re.search(r'(first|initial|opening) (\d+) minute', q_lower) or re.search(r'in the first (\d+) minutes?', q_lower)
    if match_first:
        minutes = int(match_first.group(2) or match_first.group(1))
        filtered_chunks = [c for c in all_chunks if time_to_seconds(c.get('start_time', '00:00')) < minutes * 60]
        if not filtered_chunks:
            return JSONResponse({"error": f"No chunks found for the first {minutes} minutes."}, status_code=404)
    # Last X minutes
    match_last = re.search(r'(last|final|ending|concluding) (\d+) minute', q_lower) or re.search(r'in the last (\d+) minutes?', q_lower)
    if match_last and total_lesson_seconds > 0:
        minutes = int(match_last.group(2) or match_last.group(1))
        start_cutoff = total_lesson_seconds - (minutes * 60)
        filtered_chunks = [c for c in all_chunks if time_to_seconds(c.get('start_time', '00:00')) >= start_cutoff]
        if not filtered_chunks:
            return JSONResponse({"error": f"No chunks found for the last {minutes} minutes."}, status_code=404)
    # Range: from minute X to Y
    match_range = re.search(r'from minute (\d+) to (\d+)', q_lower) or re.search(r'between minute (\d+) and (\d+)', q_lower)
    if match_range:
        start_min = int(match_range.group(1))
        end_min = int(match_range.group(2))
        filtered_chunks = [c for c in all_chunks if start_min * 60 <= time_to_seconds(c.get('start_time', '00:00')) < end_min * 60]
        if not filtered_chunks:
            return JSONResponse({"error": f"No chunks found between minute {start_min} and {end_min}."}, status_code=404)

    # Semantic search
    results = rag.semantic_search(question, filtered_chunks, top_k=top_k)
    if not results:
        return JSONResponse({"error": "No relevant chunks found for your question."}, status_code=404)

    top_chunks = [chunk for score, chunk in results]
    context = rag.format_chunks_for_context(top_chunks)

    # Build a summary section for each file
    summary_sections = []
    for fid in file_ids:
        file_info = rag.supabase.table("files").select("stored_filename, data_summary").eq("file_id", fid).single().execute()
        if file_info.data:
            filename = file_info.data.get("stored_filename", "Unknown Filename")
            summary = file_info.data.get("data_summary", "No summary available.")
            summary_sections.append(f"File: {filename}\nLesson Data Summary:\n{summary}\n")
    summary_text = "\n".join(summary_sections)

    user_prompt = (
        f"{summary_text}"
        f"Here is the relevant lesson chunk data:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Please answer the teacher's question using only the information in the summary and chunk(s) above."
    )

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

@router.post("/chat/hybrid")
async def chat_hybrid_endpoint(payload: ChatRequest):
    file_ids = payload.file_id
    question = payload.question
    if not file_ids or not question:
        return JSONResponse({"error": "file_id and question are required"}, status_code=400)

    # For now, use only the first file_id for summary and chunk retrieval
    rag = SupabaseRAG(file_id=file_ids[0])
    answer = rag.answer_question(question)
    return JSONResponse({"answer": answer})
