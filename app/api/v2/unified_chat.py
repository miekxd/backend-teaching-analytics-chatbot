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
    Unified streaming generator that handles both general and RAG responses
    """
    try:
        # First, get the flag decision from general assistant (non-streaming)
        general_response = await general_assistant.get_response(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=conversation_history
        )
        
        flag_specific = general_response.get('flag_specific', 0.0)
        
        # Send routing metadata
        routing_data = {
            "type": "routing",
            "flag_specific": flag_specific,
            "using_rag": flag_specific > 0.5
        }
        yield f"data: {json.dumps(routing_data)}\n\n"
        
        if flag_specific > 0.5:
            # High specificity - use RAG streaming
            info_data = {
                "type": "info", 
                "message": "Analyzing detailed transcript..."
            }
            yield f"data: {json.dumps(info_data)}\n\n"
            
            # Stream RAG response
            async for chunk in rag_assistant.get_response_stream(
                semantic_query=general_response['response'],
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=conversation_history
            ):
                chunk_data = {
                    "type": "content",
                    "content": chunk,
                    "source": "rag"
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        else:
            # Low specificity - send the complete general response
            info_data = {
                "type": "info", 
                "message": "Using lesson summary..."
            }
            yield f"data: {json.dumps(info_data)}\n\n"
            
            # Send the complete response
            chunk_data = {
                "type": "content",
                "content": general_response["response"],
                "source": "general"
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Send completion signal
        completion_data = {
            "type": "complete",
            "flag_specific": flag_specific,
            "source": "rag" if flag_specific > 0.5 else "general"
        }
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "error": str(e),
            "source": "unified_streaming"
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@router.post("/unified_chat_streaming")
async def unified_chat_streaming_endpoint(request: UnifiedChatRequest):
    """
    Enhanced teaching assistant streaming endpoint with intelligent routing
    
    Returns Server-Sent Events (SSE) stream with:
    - Routing decisions based on flag_specific
    - Real-time response chunks from either general or RAG assistant
    - Metadata about which service is being used
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
            unified_streaming_generator(
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