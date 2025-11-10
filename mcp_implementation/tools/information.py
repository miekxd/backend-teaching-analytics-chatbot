"""
Information Retrieval MCP Tools
"""
from mcp.types import Tool, TextContent
import json
import logging
from typing import Callable, Tuple, List

logger = logging.getLogger(__name__)


def get_information_tools() -> List[Tuple[Tool, Callable]]:
    """Get information retrieval tools"""
    
    metadata_tool = Tool(
        name="get_lesson_metadata",
        description="""Get metadata about lessons (filename, upload date, duration, etc.)

Retrieves basic information about lesson files without analysis.

Args:
    file_ids: List of lesson file IDs
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
    
    async def metadata_handler(arguments: dict) -> list[TextContent]:
        """Handler for get_lesson_metadata"""
        try:
            file_ids = arguments.get("file_ids", [])
            logger.info(f"📚 get_lesson_metadata called: files={file_ids}")
            
            from app.db.supabase import get_supabase_client
            supabase = get_supabase_client()
            
            metadata_list = []
            for file_id in file_ids:
                file_info = supabase.table("files")\
                    .select("file_id, stored_filename, created_at, data_summary")\
                    .eq("file_id", file_id)\
                    .single()\
                    .execute()
                
                if file_info.data:
                    metadata_list.append({
                        "file_id": file_info.data.get("file_id"),
                        "filename": file_info.data.get("stored_filename"),
                        "uploaded_at": file_info.data.get("created_at"),
                        "summary_preview": file_info.data.get("data_summary", "")[:200]
                    })
            
            result = {
                "metadata": metadata_list,
                "count": len(metadata_list),
                "tool": "get_lesson_metadata"
            }
            
            logger.info(f"✅ get_lesson_metadata completed: {len(metadata_list)} items")
            return [TextContent(type="text", text=json.dumps(result))]
        
        except Exception as e:
            logger.error(f"❌ get_lesson_metadata error: {str(e)}", exc_info=True)
            return [TextContent(type="text", text=json.dumps({
                "error": str(e),
                "metadata": [],
                "tool": "get_lesson_metadata"
            }))]
    
    logger.info("✅ Loaded information tools: get_lesson_metadata")
    
    return [(metadata_tool, metadata_handler)]