"""
Lesson Analysis MCP Tools
"""
from mcp.types import Tool, TextContent
import json
import logging
from typing import Optional, List, Callable, Tuple

logger = logging.getLogger(__name__)


def get_lesson_tools() -> List[Tuple[Tool, Callable]]:
    """
    Get lesson analysis tools
    
    Returns list of (Tool definition, handler function) tuples
    """
    
    # Tool 1: analyze_lesson
    analyze_lesson_tool = Tool(
        name="analyze_lesson",
        description="""Analyze a teaching lesson with intelligent routing between RAG and general assistance.

This is the primary tool for lesson analysis. It automatically:
- Analyzes the query to determine if it needs specific lesson evidence
- Retrieves relevant lesson chunks if needed
- Provides evidence-based or general coaching feedback
- References Singapore Teaching Practice framework

Args:
    message: The teacher's question or request
    file_ids: List of lesson file IDs to analyze
    conversation_history: Previous conversation context (optional)

Examples:
    - "How did my questioning techniques work in the first 10 minutes?"
    - "What went well in this lesson?"
    - "How can I improve student engagement?"
""",
        inputSchema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The teacher's question or request"
                },
                "file_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of lesson file IDs to analyze"
                },
                "conversation_history": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    },
                    "description": "Previous conversation context (optional)"
                }
            },
            "required": ["message", "file_ids"]
        }
    )
    
    async def analyze_lesson_handler(arguments: dict) -> list[TextContent]:
        """Handler for analyze_lesson tool"""
        try:
            message = arguments.get("message")
            file_ids = arguments.get("file_ids", [])
            conversation_history = arguments.get("conversation_history", [])
            
            logger.info(f"🔍 analyze_lesson called: message='{message[:50]}...', files={file_ids}")
            
            # Lazy import to avoid circular dependency
            from app.services.unified import unified_assistant
            
            # Call unified assistant
            response = await unified_assistant.get_response(
                user_message=message,
                file_ids=file_ids,
                conversation_history=conversation_history,
                top_k=5
            )
            
            result = {
                "response": response,
                "file_ids": file_ids,
                "tool": "analyze_lesson"
            }
            
            logger.info(f"✅ analyze_lesson completed: {len(response)} chars")
            return [TextContent(type="text", text=json.dumps(result))]
        
        except Exception as e:
            logger.error(f"❌ analyze_lesson error: {str(e)}", exc_info=True)
            error_result = {
                "error": f"Failed to analyze lesson: {str(e)}",
                "response": "I apologize, but I encountered an error analyzing your lesson. Please try again.",
                "tool": "analyze_lesson"
            }
            return [TextContent(type="text", text=json.dumps(error_result))]
    
    
    # Tool 2: get_lesson_summary
    get_lesson_summary_tool = Tool(
        name="get_lesson_summary",
        description="""Get high-level summaries of one or more lessons.

Retrieves stored lesson summaries without detailed analysis.
Useful for quick overviews or comparing multiple lessons.

Args:
    file_ids: List of lesson file IDs

Examples:
    - "Show me summaries of my last 3 lessons"
    - "What were the main topics in lesson 37?"
""",
        inputSchema={
            "type": "object",
            "properties": {
                "file_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of lesson file IDs"
                }
            },
            "required": ["file_ids"]
        }
    )
    
    async def get_lesson_summary_handler(arguments: dict) -> list[TextContent]:
        """Handler for get_lesson_summary tool"""
        try:
            file_ids = arguments.get("file_ids", [])
            logger.info(f"📋 get_lesson_summary called: files={file_ids}")
            
            from app.db.supabase import get_supabase_client
            supabase = get_supabase_client()
            
            summaries = []
            for file_id in file_ids:
                file_info = supabase.table("files")\
                    .select("stored_filename, data_summary")\
                    .eq("file_id", file_id)\
                    .single()\
                    .execute()
                
                if file_info.data:
                    summaries.append({
                        "file_id": file_id,
                        "filename": file_info.data.get("stored_filename"),
                        "summary": file_info.data.get("data_summary")
                    })
            
            result = {
                "summaries": summaries,
                "tool": "get_lesson_summary"
            }
            
            logger.info(f"✅ get_lesson_summary completed: {len(summaries)} summaries")
            return [TextContent(type="text", text=json.dumps(result))]
        
        except Exception as e:
            logger.error(f"❌ get_lesson_summary error: {str(e)}", exc_info=True)
            error_result = {
                "error": f"Failed to get summaries: {str(e)}",
                "summaries": [],
                "tool": "get_lesson_summary"
            }
            return [TextContent(type="text", text=json.dumps(error_result))]
    
    
    logger.info("✅ Loaded lesson analysis tools: analyze_lesson, get_lesson_summary")
    
    return [
        (analyze_lesson_tool, analyze_lesson_handler),
        (get_lesson_summary_tool, get_lesson_summary_handler)
    ]