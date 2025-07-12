import asyncio
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional, Union, AsyncGenerator 
import json
from app.models.chat import (UnifiedResponse, UnifiedChatRequest)
from app.services.rag_specific import rag_assistant
from app.services.general import general_assistant

router = APIRouter()

@router.post("/unified_chat", response_model=UnifiedResponse)
async def unified_chat_endpoint(request: UnifiedChatRequest):
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
        general_response = await general_assistant.get_response(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=request.conversation_history
        )
        
        if general_response['flag_specific'] > 0.5:
            # If flagged for RAG, route to RAG assistant
            rag_response = await rag_assistant.get_response(
                semantic_query=general_response['response'],
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=request.conversation_history
            )
            return UnifiedResponse(response=rag_response)
        
        return UnifiedResponse(response=general_response["response"])
    
    except Exception as e:
        return JSONResponse({"error": f"Enhanced chat error: {str(e)}"}, status_code=500)

async def unified_streaming_generator(
    user_message: str, 
    file_ids: List[int], 
    conversation_history: List[Dict[str, str]] = None
) -> AsyncGenerator[str, None]:
    """
    Unified streaming generator with markdown-preserving streaming for general responses
    """
    try:
        # First, get the flag decision from general assistant (non-streaming)
        general_response = await general_assistant.get_response(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=conversation_history
        )
        
        flag_specific = general_response.get('flag_specific', 0.0)
        
        if flag_specific > 0.5:
            # High specificity - use real RAG streaming
            async for chunk in rag_assistant.get_response_stream(
                semantic_query=general_response['response'],
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=conversation_history
            ):
                yield chunk  # Real streaming chunks
        else:
            # Low specificity - preserve markdown while streaming
            response_text = general_response.get("response", "")
            
            # Debug: Ensure we're working with string content
            if isinstance(response_text, dict):
                response_text = str(response_text)
            
            # Stream by sentences or logical chunks instead of words to preserve formatting
            import re
            
            # Split by sentences while preserving markdown structure
            sentences = re.split(r'(?<=[.!?])\s+', response_text)
            
            delay = 0.1  # Slightly longer delay for sentence-based streaming
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():  # Skip empty sentences
                    # Add back the space that was removed by split
                    if i < len(sentences) - 1:
                        sentence += " "
                    
                    yield sentence
                    
                    # Add delay except for the last sentence
                    if i < len(sentences) - 1:
                        await asyncio.sleep(delay)
        
    except Exception as e:
        yield f"**Error:** {str(e)}"

@router.post("/unified_chat_streaming")
async def unified_chat_streaming_endpoint(request: UnifiedChatRequest):
    """
    Enhanced teaching assistant streaming endpoint with simulated word-by-word streaming
    
    Returns plain text stream with word-by-word simulation for general responses
    and real streaming for RAG responses
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    if not file_ids:
        return JSONResponse({"error": "file_ids are required for lesson context"}, status_code=400)
    
    try:
        return StreamingResponse(
            unified_streaming_generator(
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=request.conversation_history
            ),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    except Exception as e:
        return JSONResponse({"error": f"Streaming error: {str(e)}"}, status_code=500)

async def unified_streaming_generator_with_metadata(
    user_message: str, 
    file_ids: List[int], 
    conversation_history: List[Dict[str, str]] = None
) -> AsyncGenerator[str, None]:
    """
    Alternative: Streaming generator that includes flag_specific metadata
    """
    try:
        # First, get the flag decision from general assistant
        general_response = await general_assistant.get_response(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=conversation_history
        )
        
        flag_specific = general_response.get('flag_specific', 0.0)
        
        # Send metadata first
        metadata = {
            "type": "metadata",
            "flag_specific": flag_specific,
            "using_rag": flag_specific > 0.5
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        if flag_specific > 0.5:
            # High specificity - real RAG streaming
            async for chunk in rag_assistant.get_response_stream(
                semantic_query=general_response['response'],  # FIXED: Changed from 'message' to 'response'
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=conversation_history
            ):
                content_data = {
                    "type": "content",
                    "chunk": chunk
                }
                yield f"data: {json.dumps(content_data)}\n\n"
        else:
            # Low specificity - simulated streaming
            response_text = general_response.get("response", "")  # FIXED: Changed from 'message' to 'response'
            
            # Debug: Ensure we're working with string content
            if isinstance(response_text, dict):
                response_text = str(response_text)
                
            words = response_text.split()
            
            chunk_size = 4
            delay = 0.08
            
            for i in range(0, len(words), chunk_size):
                word_chunk = words[i:i + chunk_size]
                chunk_text = " ".join(word_chunk)
                
                if i + chunk_size < len(words):
                    chunk_text += " "
                
                content_data = {
                    "type": "content",
                    "chunk": chunk_text
                }
                yield f"data: {json.dumps(content_data)}\n\n"
                
                if i + chunk_size < len(words):
                    await asyncio.sleep(delay)
        
        # Send completion with flag_specific
        completion = {
            "type": "complete",
            "flag_specific": flag_specific
        }
        yield f"data: {json.dumps(completion)}\n\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "error": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/unified_chat_streaming_with_metadata")
async def unified_chat_streaming_with_metadata_endpoint(request: UnifiedChatRequest):
    """
    Alternative endpoint that includes flag_specific metadata for conversation history building
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    if not file_ids:
        return JSONResponse({"error": "file_ids are required for lesson context"}, status_code=400)
    
    try:
        return StreamingResponse(
            unified_streaming_generator_with_metadata(
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=request.conversation_history
            ),
            media_type="text/event-stream",  # SSE format for metadata
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    except Exception as e:
        return JSONResponse({"error": f"Streaming error: {str(e)}"}, status_code=500)