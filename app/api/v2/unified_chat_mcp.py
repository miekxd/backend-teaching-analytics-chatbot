"""
Unified Chat with MCP Orchestration

This replaces the old unified_chat.py with MCP-enabled orchestration.
The LLM now orchestrates tools intelligently.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Dict, Optional
import logging

from app.models.chat import UnifiedChatRequest, UnifiedResponse
from mcp_implementation.client import MCPClient

router = APIRouter()
logger = logging.getLogger(__name__)

# MCP Client instance
mcp_client = MCPClient()


@router.post("/unified_chat", response_model=UnifiedResponse)
async def unified_chat_endpoint(request: UnifiedChatRequest):
    """
    Unified chat endpoint with MCP orchestration
    
    The LLM intelligently:
    - Decides which tools to use
    - Calls multiple tools if needed
    - Composes responses from tool results
    
    Features:
    - Automatic tool selection
    - Multi-tool composition
    - Streaming support
    - Singapore Teaching Practice framework integration
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    if not file_ids:
        return JSONResponse({"error": "file_ids are required for lesson context"}, status_code=400)
    
    try:
        logger.info(f"📨 unified_chat_endpoint: message='{user_message[:50]}...', files={file_ids}")
        
        # Call MCP client
        result = await mcp_client.chat(
            message=user_message,
            file_ids=file_ids,
            conversation_history=request.conversation_history
        )
        
        # Return in expected format
        return UnifiedResponse(
            response=result["response"],
            needs_graph=result.get("needs_graph", False),
            graph_types=result.get("graph_types", []),
            lesson_filter=result.get("lesson_filter", file_ids),
            area_filter=result.get("area_filter", [])
        )
    
    except Exception as e:
        logger.error(f"❌ unified_chat_endpoint error: {str(e)}", exc_info=True)
        return JSONResponse({"error": f"Chat error: {str(e)}"}, status_code=500)


@router.post("/unified_chat_streaming")
async def unified_chat_streaming_endpoint(request: UnifiedChatRequest):
    """
    Streaming version of unified chat with MCP
    
    Streams:
    1. Graph metadata (if visualization needed)
    2. Response text chunks as generated
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    if not file_ids:
        return JSONResponse({"error": "file_ids are required for lesson context"}, status_code=400)
    
    try:
        logger.info(f"🌊 unified_chat_streaming_endpoint: message='{user_message[:50]}...'")
        
        return StreamingResponse(
            mcp_client.chat_stream(
                message=user_message,
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
        logger.error(f"❌ unified_chat_streaming_endpoint error: {str(e)}", exc_info=True)
        return JSONResponse({"error": f"Streaming error: {str(e)}"}, status_code=500)


@router.post("/intent_analysis")
async def intent_analysis_endpoint(request: UnifiedChatRequest):
    """
    Intent analysis endpoint - for debugging
    
    Shows what the intent analyzer detects without generating response
    """
    
    user_message = request.message.strip()
    file_ids = request.file_ids
    
    if not user_message:
        return JSONResponse({"error": "Message cannot be empty"}, status_code=400)
    
    try:
        from app.services.intent_analyzer import intent_analyzer
        
        intent_analysis = await intent_analyzer.analyze_intent(
            user_message=user_message,
            file_ids=file_ids or [],
            conversation_history=request.conversation_history
        )
        
        return JSONResponse({
            "intent_analysis": intent_analysis,
            "routing_explanation": f"Query will be routed to {intent_analysis.get('agent_to_use')}"
        })
    
    except Exception as e:
        logger.error(f"Intent analysis error: {str(e)}", exc_info=True)
        return JSONResponse({"error": f"Intent analysis error: {str(e)}"}, status_code=500)


@router.get("/mcp/tools")
async def list_mcp_tools():
    """
    Debug endpoint: List all available MCP tools
    
    Useful for:
    - Debugging
    - Documentation
    - Monitoring which tools exist
    """
    try:
        # Get tools from our tool registry
        from mcp_implementation.tools.lesson_analysis import get_lesson_tools
        from mcp_implementation.tools.visualization import get_visualization_tools
        from mcp_implementation.tools.information import get_information_tools
        
        all_tools = []
        all_tools.extend(get_lesson_tools())
        all_tools.extend(get_visualization_tools())
        all_tools.extend(get_information_tools())
        
        tools_info = [
            {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": tool_def.inputSchema
            }
            for tool_def, _ in all_tools
        ]
        
        return {
            "tools": tools_info,
            "count": len(tools_info),
            "mcp_enabled": True
        }
    
    except Exception as e:
        logger.error(f"❌ list_mcp_tools error: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)