import asyncio
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional, Union, AsyncGenerator 
import json
from app.models.chat import (UnifiedResponse, UnifiedChatRequest)
from app.services.rag_specific import rag_assistant
from app.services.general import general_assistant
from app.services.intent_analyzer import intent_analyzer
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/unified_chat", response_model=UnifiedResponse)
async def unified_chat_endpoint(request: UnifiedChatRequest):
    """
    Unified teaching assistant endpoint with intelligent routing
    
    Features:
    - Intent analysis for smart routing between general and RAG assistants
    - Access to lesson summaries and detailed transcripts
    - Class period filtering (beginning/middle/end)
    - Singapore Teaching Practice framework awareness
    
    Example requests:
    - "How did my lesson go?" → General assistant with lesson summary
    - "Show me examples from the beginning of the lesson" → RAG assistant with class period filtering
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    if not file_ids:
        return JSONResponse({"error": "file_ids are required for lesson context"}, status_code=400)
    
    try:
        # Step 1: Analyze intent to determine routing
        intent_analysis = await intent_analyzer.analyze_intent(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=request.conversation_history
        )
        
        agent_to_use = intent_analysis.get("agent_to_use", "general_assistant")
        class_period = intent_analysis.get("class_period")
        transformed_query = intent_analysis.get("transformed_query", user_message)
        
        # Step 2: Route to appropriate assistant
        if agent_to_use == "rag_assistant":
            # Use RAG assistant with class period filtering
            response = await rag_assistant.get_response(
                semantic_query=transformed_query,
                user_message=user_message,
                file_ids=file_ids,
                class_period=class_period,
                conversation_history=request.conversation_history
            )
        else:
            # Use general assistant
            response = await general_assistant.get_response(
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=request.conversation_history
            )
        
        return UnifiedResponse(response=response)
    
    except Exception as e:
        logger.error(f"Unified chat error: {str(e)}")
        return JSONResponse({"error": f"Chat error: {str(e)}"}, status_code=500)

async def unified_streaming_generator(
    user_message: str, 
    file_ids: List[int], 
    conversation_history: List[Dict[str, str]] = None
) -> AsyncGenerator[str, None]:
    """
    Unified streaming generator with intelligent routing and markdown-aware streaming
    """
    try:
        # Step 1: Analyze intent to determine routing
        intent_analysis = await intent_analyzer.analyze_intent(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=conversation_history
        )
        
        agent_to_use = intent_analysis.get("agent_to_use", "general_assistant")
        class_period = intent_analysis.get("class_period")
        transformed_query = intent_analysis.get("transformed_query", user_message)
        
        print(f"Intent Analysis: {json.dumps(intent_analysis, indent=2)}")
        # Step 2: Route to appropriate assistant with streaming
        if agent_to_use == "rag_assistant":
            # Use RAG assistant streaming with class period filtering
            async for chunk in rag_assistant.get_response_stream(
                semantic_query=transformed_query,
                user_message=user_message,
                file_ids=file_ids,
                class_period=class_period,
                conversation_history=conversation_history
            ):
                yield chunk  # Real streaming chunks from RAG
        else:
            # Use general assistant with simulated streaming
            async for chunk in general_assistant.get_response_stream(
                user_message=user_message,
                file_ids=file_ids,
                conversation_history=conversation_history
            ):
                yield chunk  # Real streaming chunks from general assistant
        
    except Exception as e:
        yield f"Error: {str(e)}"

@router.post("/unified_chat_streaming")
async def unified_chat_streaming_endpoint(request: UnifiedChatRequest):
    """
    Unified teaching assistant streaming endpoint with intelligent routing
    
    Features:
    - Intent analysis for smart routing
    - Real streaming for both general and RAG responses
    - Class period filtering for targeted lesson analysis
    - Markdown-aware streaming output
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
        logger.error(f"Streaming error: {str(e)}")
        return JSONResponse({"error": f"Streaming error: {str(e)}"}, status_code=500)

@router.post("/intent_analysis")
async def intent_analysis_endpoint(request: UnifiedChatRequest):
    """
    Endpoint to analyze intent and get routing decision without generating response
    
    Useful for debugging and understanding routing decisions
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    try:
        intent_analysis = await intent_analyzer.analyze_intent(
            user_message=user_message,
            file_ids=file_ids or [],
            conversation_history=request.conversation_history
        )
        
        return JSONResponse({
            "intent_analysis": intent_analysis,
            "routing_explanation": intent_analyzer.get_routing_explanation(intent_analysis) if hasattr(intent_analyzer, 'get_routing_explanation') else "Analysis completed"
        })
    
    except Exception as e:
        logger.error(f"Intent analysis error: {str(e)}")
        return JSONResponse({"error": f"Intent analysis error: {str(e)}"}, status_code=500)