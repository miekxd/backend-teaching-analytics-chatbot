"""
Resource Finding MCP Tools

Tools for discovering and managing external teaching resources.
"""
from mcp.types import Tool, TextContent
import json
import logging
from typing import Callable, Tuple, List

from app.resources.resource_finder import resource_finder

logger = logging.getLogger(__name__)


def get_resource_tools() -> List[Tuple[Tool, Callable]]:
    """
    Get resource finding tools
    
    Returns list of (Tool definition, handler function) tuples
    """
    
    # Tool: find_teaching_resources
    find_resources_tool = Tool(
        name="find_teaching_resources",
        description="""Find external teaching resources (papers, videos, articles) for a specific skill or competency.

This tool searches multiple academic and educational sources:
- arXiv: Academic papers on education and teaching
- Semantic Scholar: Research papers with citation counts
- YouTube: Educational videos and tutorials
- Google: Educational articles and websites

The tool automatically:
1. Maps the skill to Singapore Teaching Practice framework (if applicable)
2. Checks cached resources first (fast)
3. Calls external APIs if needed or if user requests fresh results
4. Stores new resources for future use
5. Returns formatted, relevant resources

Use this tool when:
- User asks for "resources", "references", "materials", "examples"
- User wants to learn about a teaching technique
- User needs evidence or research on a topic
- User asks "how to improve" or "suggestions for" a skill

Args:
    skill: The teaching skill or competency (e.g., "questioning techniques", "classroom management", "student engagement")
    force_refresh: Set to true if user explicitly asks for "new", "recent", or "fresh" resources (default: false)

Examples:
    - "Find resources for questioning techniques"
    - "Suggest materials about classroom management"
    - "I need help with student engagement, any resources?"
    - "Find recent articles on formative assessment"
""",
        inputSchema={
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "The teaching skill or competency to find resources for"
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "Force new API calls even if cached results exist (use when user asks for 'new' or 'recent' resources)",
                    "default": False
                }
            },
            "required": ["skill"]
        }
    )
    
    async def find_resources_handler(arguments: dict) -> list[TextContent]:
        """Handler for find_teaching_resources tool"""
        try:
            skill = arguments.get("skill", "")
            force_refresh = arguments.get("force_refresh", False)
            
            if not skill:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Skill parameter is required",
                        "tool": "find_teaching_resources"
                    })
                )]
            
            logger.info(f"🔍 find_teaching_resources called: skill='{skill}', force_refresh={force_refresh}")
            
            # Call resource finder
            resources_data = await resource_finder.find_resources(
                skill=skill,
                force_refresh=force_refresh,
                max_per_source=3
            )
            
            # Format for display
            formatted_output = resource_finder.format_resources_for_display(resources_data)
            
            # Return result
            result = {
                "formatted_response": formatted_output,
                "raw_data": {
                    "skill": resources_data['skill'],
                    "framework_code": resources_data['framework_mapping']['framework_code'],
                    "framework_name": resources_data['framework_mapping']['framework_name'],
                    "total_results": resources_data['total_results'],
                    "from_cache": resources_data['from_cache'],
                    "sources_used": resources_data['sources_used'],
                    "sources_failed": resources_data['sources_failed']
                },
                "tool": "find_teaching_resources"
            }
            
            logger.info(f"✅ find_teaching_resources completed: {resources_data['total_results']} results")
            return [TextContent(type="text", text=json.dumps(result))]
        
        except Exception as e:
            logger.error(f"❌ find_teaching_resources error: {str(e)}", exc_info=True)
            error_result = {
                "error": f"Failed to find resources: {str(e)}",
                "formatted_response": "I apologize, but I encountered an error while searching for resources. Please try again.",
                "tool": "find_teaching_resources"
            }
            return [TextContent(type="text", text=json.dumps(error_result))]
    
    
    logger.info("✅ Loaded resource tools: find_teaching_resources")
    
    return [(find_resources_tool, find_resources_handler)]