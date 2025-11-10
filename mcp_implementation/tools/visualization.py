"""
Visualization MCP Tools
"""
from mcp.types import Tool, TextContent
import json
import logging
from typing import Optional, List, Callable, Tuple

logger = logging.getLogger(__name__)


def get_visualization_tools() -> List[Tuple[Tool, Callable]]:
    """Get visualization tools"""
    
    detect_viz_tool = Tool(
        name="detect_visualization_needs",
        description="""Detect if the user's query requires graphs or visualizations.

Analyzes the query to determine:
- Whether graphs are needed
- What types of graphs (timeline, comparison, distribution, etc.)
- What teaching areas to visualize
- Which lessons to include

Args:
    message: The user's query
    file_ids: Relevant lesson file IDs
    conversation_history: Previous conversation (optional)

Examples:
    - "Show me a graph of my questioning over time"
    - "Compare engagement across my last 3 lessons"
    - "Visualize my lesson pacing"
""",
        inputSchema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The user's query"
                },
                "file_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Relevant lesson file IDs"
                },
                "conversation_history": {
                    "type": "array",
                    "items": {
                        "type": "object"
                    },
                    "description": "Previous conversation (optional)"
                }
            },
            "required": ["message", "file_ids"]
        }
    )
    
    async def detect_viz_handler(arguments: dict) -> list[TextContent]:
        """Handler for detect_visualization_needs"""
        try:
            message = arguments.get("message")
            file_ids = arguments.get("file_ids", [])
            conversation_history = arguments.get("conversation_history", [])
            
            logger.info(f"📊 detect_visualization_needs called: '{message[:50]}...'")
            
            from app.services.intent_analyzer import intent_analyzer
            
            intent = await intent_analyzer.analyze_intent(
                user_message=message,
                file_ids=file_ids,
                conversation_history=conversation_history
            )
            
            result = {
                "needs_graph": intent.get("needs_graph", False),
                "graph_types": intent.get("graph_types", []),
                "lesson_filter": intent.get("lesson_filter", []),
                "area_filter": intent.get("area_filter", []),
                "tool": "detect_visualization_needs"
            }
            
            logger.info(f"✅ Visualization detection: needs_graph={result['needs_graph']}")
            return [TextContent(type="text", text=json.dumps(result))]
        
        except Exception as e:
            logger.error(f"❌ detect_visualization_needs error: {str(e)}", exc_info=True)
            error_result = {
                "needs_graph": False,
                "graph_types": [],
                "error": str(e),
                "tool": "detect_visualization_needs"
            }
            return [TextContent(type="text", text=json.dumps(error_result))]
    
    logger.info("✅ Loaded visualization tools: detect_visualization_needs")
    
    return [(detect_viz_tool, detect_viz_handler)]